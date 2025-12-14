import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np


def _parse_dt(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        # sqlite may return str
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _time_features(dt: datetime) -> np.ndarray:
    # cyclic hour of day
    hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    hour_rad = 2.0 * math.pi * (hour / 24.0)
    # day of week (Mon=0)
    dow = dt.weekday()
    dow_rad = 2.0 * math.pi * (dow / 7.0)
    return np.array([
        1.0,
        math.sin(hour_rad),
        math.cos(hour_rad),
        math.sin(dow_rad),
        math.cos(dow_rad),
    ], dtype=np.float64)


@dataclass
class RidgeModel:
    w: np.ndarray  # shape (d,)

    def predict(self, x: np.ndarray) -> float:
        return float(x @ self.w)


def fit_ridge(X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> RidgeModel:
    # Closed-form ridge: (X^T X + l2 I)^-1 X^T y
    d = X.shape[1]
    XtX = X.T @ X
    reg = l2 * np.eye(d, dtype=np.float64)
    w = np.linalg.solve(XtX + reg, X.T @ y)
    return RidgeModel(w=w)


@dataclass
class LogisticModel:
    w: np.ndarray  # shape (d,)

    def predict_proba(self, x: np.ndarray) -> float:
        z = float(x @ self.w)
        # stable sigmoid
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 1.0,
    lr: float = 0.3,
    steps: int = 600,
) -> LogisticModel:
    n, d = X.shape
    w = np.zeros(d, dtype=np.float64)

    for _ in range(int(steps)):
        z = X @ w
        # sigmoid
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        # gradient of log loss + L2
        grad = (X.T @ (p - y)) / float(n)
        grad += (l2 / float(n)) * w
        w -= lr * grad

    return LogisticModel(w=w)


def build_waittime_dataset(
    rows: List[Tuple],
    rolling_k: int = 5,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """rows: list of (entry_time, dwell_time_seconds) ordered ASC by entry_time."""
    feats: List[np.ndarray] = []
    targets: List[float] = []

    recent: List[float] = []
    for entry_time, dwell in rows:
        dt = _parse_dt(entry_time)
        if dt is None:
            continue
        if dwell is None:
            continue
        dwell = float(dwell)
        if dwell < 0:
            continue

        base = _time_features(dt)
        prev_mean = float(np.mean(recent[-rolling_k:])) if recent else 0.0
        x = np.concatenate([base, np.array([prev_mean], dtype=np.float64)])

        feats.append(x)
        targets.append(dwell)
        recent.append(dwell)

    if not feats:
        return np.zeros((0, 6), dtype=np.float64), np.zeros((0,), dtype=np.float64), 0.0

    X = np.vstack(feats)
    y = np.array(targets, dtype=np.float64)
    last_mean = float(np.mean(recent[-rolling_k:])) if recent else 0.0
    return X, y, last_mean


def build_unauth_dataset(
    rows: List[Tuple],
    rolling_k: int = 30,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """rows: list of (timestamp, alert_type) ordered ASC by timestamp."""
    feats: List[np.ndarray] = []
    labels: List[float] = []

    recent: List[int] = []
    for ts, alert_type in rows:
        dt = _parse_dt(ts)
        if dt is None:
            continue
        label = 1 if str(alert_type).lower() == "unauthorized" else 0

        base = _time_features(dt)
        recent_rate = float(np.mean(recent[-rolling_k:])) if recent else 0.0
        x = np.concatenate([base, np.array([recent_rate], dtype=np.float64)])

        feats.append(x)
        labels.append(float(label))
        recent.append(label)

    if not feats:
        return np.zeros((0, 6), dtype=np.float64), np.zeros((0,), dtype=np.float64), 0.0

    X = np.vstack(feats)
    y = np.array(labels, dtype=np.float64)
    last_rate = float(np.mean(recent[-rolling_k:])) if recent else 0.0
    return X, y, last_rate


def predict_ml(
    *,
    dwell_rows: List[Tuple],
    alert_rows: List[Tuple],
    now: Optional[datetime] = None,
) -> Dict:
    now_dt = now or datetime.now()

    # Wait-time model
    Xw, yw, last_dwell_mean = build_waittime_dataset(dwell_rows)
    wait_pred = 0.0
    wait_meta = {"samples": int(Xw.shape[0]), "model": "ridge"}
    if Xw.shape[0] >= 25:
        model_w = fit_ridge(Xw, yw, l2=5.0)
        x_now = np.concatenate([_time_features(now_dt), np.array([last_dwell_mean], dtype=np.float64)])
        wait_pred = max(0.0, model_w.predict(x_now))
    else:
        # fallback: mean of last available dwell times
        if len(dwell_rows) > 0:
            vals = [float(r[1]) for r in dwell_rows if r and r[1] is not None]
            if vals:
                wait_pred = float(np.mean(vals[-10:]))
        wait_meta["model"] = "fallback_mean"

    # Unauthorized probability model
    Xu, yu, last_unauth_rate = build_unauth_dataset(alert_rows)
    unauth_prob = 0.0
    unauth_meta = {"samples": int(Xu.shape[0]), "model": "logistic"}
    if Xu.shape[0] >= 40 and float(np.sum(yu)) >= 3.0:
        model_u = fit_logistic(Xu, yu, l2=5.0, lr=0.3, steps=700)
        x_now = np.concatenate([_time_features(now_dt), np.array([last_unauth_rate], dtype=np.float64)])
        unauth_prob = float(np.clip(model_u.predict_proba(x_now), 0.0, 1.0))
    else:
        # fallback: recent rate
        if len(alert_rows) > 0:
            labels = [1 if str(r[1]).lower() == "unauthorized" else 0 for r in alert_rows if r and r[1]]
            if labels:
                unauth_prob = float(np.mean(labels[-50:]))
        unauth_meta["model"] = "fallback_rate"

    return {
        "generated_at": now_dt.isoformat(),
        "wait_time": {
            "predicted_seconds": float(wait_pred),
            **wait_meta,
        },
        "unauthorized": {
            "probability": float(unauth_prob),
            **unauth_meta,
        },
    }
