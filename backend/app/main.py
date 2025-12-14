from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
import cv2
import os
import json
from datetime import datetime
from pathlib import Path

# Load environment variables from backend/.env if present
try:
    from dotenv import load_dotenv

    _backend_root = Path(__file__).resolve().parent.parent
    # Prefer backend/.env, but also allow a project-root .env depending on how uvicorn is launched.
    load_dotenv(_backend_root / ".env")
    load_dotenv()  # loads from current working directory if a .env exists
except Exception:
    # python-dotenv is optional; env vars can also be set by the shell
    pass


def _parse_roi_env(name: str):
    """Parse ROI env var formatted as 'x1,y1,x2,y2' with normalized floats [0..1]."""
    raw = os.getenv(name)
    if not raw:
        return None
    try:
        parts = [float(p.strip()) for p in raw.split(",")]
        if len(parts) != 4:
            return None
        x1, y1, x2, y2 = parts
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)
    except Exception:
        return None


def _apply_roi_mask(frame, roi):
    """Black out pixels outside ROI rectangle (keeps same frame size)."""
    if roi is None or frame is None:
        return frame
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = roi
    px1 = int(x1 * w)
    py1 = int(y1 * h)
    px2 = int(x2 * w)
    py2 = int(y2 * h)
    masked = frame.copy()
    # Top
    if py1 > 0:
        masked[0:py1, :] = 0
    # Bottom
    if py2 < h:
        masked[py2:h, :] = 0
    # Left
    if px1 > 0:
        masked[py1:py2, 0:px1] = 0
    # Right
    if px2 < w:
        masked[py1:py2, px2:w] = 0
    return masked


def _filter_detections_to_roi(detections, roi, frame_shape):
    if roi is None:
        return detections
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = roi
    rx1 = x1 * w
    ry1 = y1 * h
    rx2 = x2 * w
    ry2 = y2 * h
    kept = []
    for d in detections:
        bbox = d.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        bx1, by1, bx2, by2 = bbox
        cx = (bx1 + bx2) / 2.0
        cy = (by1 + by2) / 2.0
        if (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2):
            kept.append(d)
    return kept

def _print_twilio_config_status() -> None:
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_whatsapp = os.getenv("TWILIO_WHATSAPP_FROM")
    to_whatsapp = os.getenv("TWILIO_WHATSAPP_TO")
    content_sid = os.getenv("TWILIO_CONTENT_SID")

    missing = [
        name
        for name, value in (
            ("TWILIO_ACCOUNT_SID", account_sid),
            ("TWILIO_AUTH_TOKEN", auth_token),
            ("TWILIO_WHATSAPP_FROM", from_whatsapp),
            ("TWILIO_WHATSAPP_TO", to_whatsapp),
        )
        if not value
    ]

    if missing:
        print(f"📱 Twilio WhatsApp: NOT configured (missing: {', '.join(missing)})")
    else:
        sid_tail = account_sid[-6:] if account_sid else ""
        token_len = len(auth_token) if auth_token else 0
        mode = "template" if content_sid else "body"
        print(
            f"📱 Twilio WhatsApp: configured (sid=...{sid_tail}, token_len={token_len}, mode={mode}, from={from_whatsapp}, to={to_whatsapp})"
        )

# Import Agents & Infrastructure
from app.database import DatabaseManager
from app.agents.gate_agent import GateAgent
from app.agents.auth_agent import AuthAgent
from app.agents.alert_agent import AlertAgent
from app.agents.hazard_agent import HazardAgent
from ultralytics import YOLO
from statistics import mean
from app.forecast_ml import predict_ml

# Initialize FastAPI
app = FastAPI(title="Agentic Hazardous Vehicle Monitoring")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static Files
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
if os.path.exists(data_dir):
    app.mount("/data", StaticFiles(directory=data_dir), name="data")

# --- System Components ---
print("🚀 Initializing Agentic System...")
_print_twilio_config_status()
db = DatabaseManager()

# Models
print("📦 Loading YOLOv8x...")
# Using YOLOv8 Extra Large SEGMENTATION model for masks + detection
model_path = Path(__file__).resolve().parent.parent / 'yolov8x-seg.pt'

# IMPORTANT: tracking state in Ultralytics is not designed to be shared across multiple streams.
# Use one YOLO instance per camera stream so track IDs and timing remain stable.
yolo_model_cam1 = YOLO(str(model_path))
yolo_model_cam2 = YOLO(str(model_path))

# SAM2 Lazy Load
sam_predictor = None
try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    sam_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny")
    print("✓ SAM2 Initialized")
except:
    print("⚠ SAM2 Not Found (Using fallback)")

# WebSocket Manager wrapper for Agent 3
class WebSocketManager:
    def __init__(self):
        self.active_connections = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

ws_manager = WebSocketManager()

# Initialize Agents
gate_agent = GateAgent(db)
auth_agent = AuthAgent(db)
alert_agent = AlertAgent(db, ws_manager)
hazard_agent = HazardAgent()

# Pipeline for vehicle detection (Using existing logic wrapper)
from app.vehicle_detector import VehicleDetector
detector_cam1 = VehicleDetector(yolo_model_cam1, sam_predictor, ocr_backend="none")
detector_cam2 = VehicleDetector(yolo_model_cam2, sam_predictor)

# Optional per-camera region-of-interest to reduce false positives.
# Format: "x1,y1,x2,y2" normalized in [0..1]
roi_cam1 = _parse_roi_env("CAM_01_ROI")
roi_cam2 = _parse_roi_env("CAM_02_ROI")

@app.on_event("startup")
async def startup_event():
    # Start Agents
    await gate_agent.start()
    await auth_agent.start()
    await alert_agent.start()
    await hazard_agent.start()
    
    # Wire up subscriptions (Agents -> AlertAgent)
    gate_agent.subscribe(alert_agent.process_message)
    auth_agent.subscribe(alert_agent.process_message)
    hazard_agent.subscribe(alert_agent.process_message)

    # Optional: one-time Twilio send test at startup.
    # Enable by setting TWILIO_SELF_TEST=1 in your .env
    if os.getenv("TWILIO_SELF_TEST") in ("1", "true", "True", "yes", "YES"):
        await alert_agent._send_whatsapp("Twilio self-test: backend startup message")

@app.on_event("shutdown")
async def shutdown_event():
    await gate_agent.stop()
    await auth_agent.stop()
    await alert_agent.stop()
    await hazard_agent.stop()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        # Dual Video Processing Loop
        path1 = os.path.join(data_dir, 'car.mp4')
        path2 = os.path.join(data_dir, 'gate2.mp4')
        
        cap1 = cv2.VideoCapture(path1)
        cap2 = cv2.VideoCapture(path2)
        
        if not cap1.isOpened() or not cap2.isOpened():
             print("Error opening video feeds")
             return

        frame_idx = 0
        
        while True:
            # Read both cameras
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            # Reset loops independently
            if not ret1:
                cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret1, frame1 = cap1.read()
            if not ret2:
                cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret2, frame2 = cap2.read()

            vehicles_combined = []
            
            # --- PROCESS CAMERA 1 (GATE) ---
            if ret1:
                det_frame1 = _apply_roi_mask(frame1, roi_cam1)
                v1 = detector_cam1.detect_vehicles(det_frame1)
                v1 = _filter_detections_to_roi(v1, roi_cam1, frame1.shape)
                for v in v1: v['camera_id'] = "CAM_01"
                vehicles_combined.extend(v1)
                
                # Send to GateAgent
                await gate_agent.input_queue.put({
                    "type": "frame_detections",
                    "detections": v1,
                    "frame_timestamp": datetime.now().isoformat(),
                    "camera_id": "CAM_01"
                })

            # --- PROCESS CAMERA 2 (PLANT) ---
            if ret2:
                det_frame2 = _apply_roi_mask(frame2, roi_cam2)
                v2 = detector_cam2.detect_vehicles(det_frame2)
                v2 = _filter_detections_to_roi(v2, roi_cam2, frame2.shape)
                for v in v2: v['camera_id'] = "CAM_02"
                vehicles_combined.extend(v2)
                
                # Send to AuthAgent & HazardAgent
                event_payload = {
                    "type": "frame_detections",
                    "detections": v2,
                    "frame_timestamp": datetime.now().isoformat(),
                    "camera_id": "CAM_02"
                }
                await auth_agent.input_queue.put(event_payload)
                await hazard_agent.input_queue.put(event_payload)

            # Send Update to Frontend (Showing Frame 1 for main view, theoretically could toggle)
            # For this demo, let's send combined vehicle data but only one frame's progress?
            # Or just send CAM 1 frame index.
            await websocket.send_json({
                "type": "vehicle_update",
                "vehicles": vehicles_combined,
                "frame": frame_idx
            })
            
            await asyncio.sleep(0.01) # Approx 100 FPS cap (fast forward)
            frame_idx += 1
            
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        print(f"WS Error: {e}")
    finally:
        if 'cap1' in locals(): cap1.release()
        if 'cap2' in locals(): cap2.release()

@app.get("/")
def root():
    return {"status": "Agentic System Running"}


@app.get("/forecast/waittime")
def forecast_waittime(limit: int = 50, camera_id: str = "CAM_01"):
    """Forecast next wait time (seconds) from recent gate dwell times.

    This is a baseline model (robust median/trimmed mean). It can be upgraded later.
    """
    conn = db._get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT dwell_time_seconds
            FROM access_logs
            WHERE camera_id = ?
              AND dwell_time_seconds IS NOT NULL
            ORDER BY entry_time DESC
            LIMIT ?
            """,
            (camera_id, int(limit)),
        )
        rows = [r[0] for r in cur.fetchall() if r and r[0] is not None]
    finally:
        conn.close()

    if not rows:
        return {"camera_id": camera_id, "samples": 0, "predicted_wait_seconds": 0}

    # Trim outliers (top/bottom 10%) then average
    rows_sorted = sorted(float(x) for x in rows if float(x) >= 0)
    n = len(rows_sorted)
    k = max(0, int(n * 0.1))
    trimmed = rows_sorted[k : n - k] if n - 2 * k >= 1 else rows_sorted
    predicted = float(mean(trimmed))
    return {
        "camera_id": camera_id,
        "samples": len(rows_sorted),
        "predicted_wait_seconds": predicted,
        "method": "trimmed_mean_10pct",
    }


@app.get("/forecast/unauthorized")
def forecast_unauthorized(window: int = 100):
    """Estimate probability of next vehicle being unauthorized.

    Baseline: ratio of Unauthorized alerts over last N alerts.
    """
    conn = db._get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT alert_type
            FROM alerts
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (int(window),),
        )
        rows = [r[0] for r in cur.fetchall() if r and r[0]]
    finally:
        conn.close()

    if not rows:
        return {"samples": 0, "unauthorized_probability": 0.0, "method": "alert_ratio"}

    unauth = sum(1 for t in rows if str(t).lower() == "unauthorized")
    prob = unauth / float(len(rows))
    return {
        "samples": len(rows),
        "unauthorized_probability": prob,
        "method": "alert_ratio",
    }


@app.get("/forecast/ml")
def forecast_ml(dwell_limit: int = 500, alerts_limit: int = 1000):
    """ML-style forecast from DB data.

    Returns:
    - predicted next wait time at gate (seconds)
    - probability of an unauthorized alert in the near term
    """
    conn = db._get_conn()
    try:
        cur = conn.cursor()

        # Gate dwell times (CAM_01)
        cur.execute(
            """
            SELECT entry_time, dwell_time_seconds
            FROM access_logs
            WHERE camera_id = 'CAM_01'
              AND dwell_time_seconds IS NOT NULL
            ORDER BY entry_time ASC
            LIMIT ?
            """,
            (int(dwell_limit),),
        )
        dwell_rows = cur.fetchall() or []

        # Alerts history (used for unauthorized probability)
        cur.execute(
            """
            SELECT timestamp, alert_type
            FROM alerts
            WHERE timestamp IS NOT NULL
            ORDER BY timestamp ASC
            LIMIT ?
            """,
            (int(alerts_limit),),
        )
        alert_rows = cur.fetchall() or []
    finally:
        conn.close()

    return predict_ml(dwell_rows=dwell_rows, alert_rows=alert_rows)
