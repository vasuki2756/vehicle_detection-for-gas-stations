# Hazardous Vehicle Monitoring System

A real-time vehicle detection and monitoring system for gas stations and industrial plants. Uses computer vision (YOLOv8 + SAM2) to detect vehicles from camera feeds, classify them, read license plates via OCR, track positions within defined zones, generate alerts, and display everything on a live React dashboard with WhatsApp notifications.

## Features

- **Real-time vehicle detection** from two camera feeds (gate and plant area) using YOLOv8x-seg
- **Segmentation masks** via SAM2 with bounding box fallback
- **License plate recognition** via PaddleOCR or Hugging Face TrOCR with voting stabilization
- **Multi-agent architecture** (GateWatcher, AuthGuard, AlertOrchestrator, HazardTracker) for modular processing
- **Live React dashboard** with video overlay, sortable vehicle table, and alert management
- **Zone monitoring** with configurable ROI per camera
- **Delay detection** for hazardous vehicles
- **Unauthorized vehicle alerts** with DB-backed authorization checking
- **WhatsApp notifications** via Twilio
- **ML forecasting** for gate wait times and unauthorized vehicle probability (ridge/logistic regression)
- **SQLite persistence** for access logs, alerts, and authorized vehicles

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI, WebSockets |
| Frontend | React 18, CSS |
| Object Detection | Ultralytics YOLOv8x-seg |
| Segmentation | SAM2 (facebook/sam2-hiera-tiny) |
| OCR | PaddleOCR / Hugging Face TrOCR |
| ML | NumPy (ridge regression, logistic regression) |
| Database | SQLite |
| Notifications | Twilio WhatsApp API |
| Video | OpenCV |

## Setup

### Backend

1. Navigate to `backend/` and create a virtual environment:
   ```
   python -m venv venv
   ```

2. Activate it and install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and configure Twilio credentials, OCR backend, YOLO params, and ROI settings.

4. Download YOLO model to `backend/yolov8x-seg.pt` (from Ultralytics).

5. Place video files at `backend/data/car.mp4` and `backend/data/gate2.mp4`.

6. Run:
   ```
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

### Frontend

1. Navigate to `frontend/` and install dependencies:
   ```
   npm install
   ```

2. Copy `.env.example` to `.env` and set `REACT_APP_SOCKET_URL=http://localhost:8000`.

3. Start:
   ```
   npm start
   ```
   Opens at `http://localhost:3000`.

## Project Structure

```
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI entry point, WebSocket, REST endpoints
│   │   ├── vehicle_detector.py   # YOLO + SAM2 + OCR detection pipeline
│   │   ├── forecast_ml.py        # ML forecasting
│   │   ├── database.py           # SQLite schema & CRUD
│   │   └── agents/
│   │       ├── base.py           # BaseAgent (async queue, subscribe/emit)
│   │       ├── gate_agent.py     # GateWatcher: dwell time, entry/exit
│   │       ├── auth_agent.py     # AuthGuard: license plate authorization
│   │       ├── alert_agent.py    # AlertOrchestrator: alerts + WhatsApp
│   │       └── hazard_agent.py   # HazardTracker: hazardous vehicle delays
│   ├── data/                     # Video files (car.mp4, gate2.mp4)
│   ├── .env.example
│   └── requirements.txt
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── context/
│   │   │   └── WebSocketContext.js  # WebSocket connection + shared state
│   │   ├── components/
│   │   │   ├── DashboardLayout.js   # Main layout (3-column grid)
│   │   │   ├── VideoFeed.js         # Canvas overlay with vehicle masks
│   │   │   ├── VehicleTable.js      # Sortable/filterable vehicle table
│   │   │   ├── AlertsPanel.js       # Live alert cards with filtering
│   │   │   └── Footer.js            # Summary statistics
│   │   ├── App.js
│   │   └── index.js
│   ├── .env.example
│   └── package.json
```

## Environment Variables

### Backend (.env)

| Variable | Description |
|---|---|
| `TWILIO_ACCOUNT_SID` | Twilio account SID |
| `TWILIO_AUTH_TOKEN` | Twilio auth token |
| `TWILIO_WHATSAPP_NUMBER` | Twilio WhatsApp sender number |
| `ALERT_WHATSAPP_NUMBER` | Recipient WhatsApp number |
| `OCR_BACKEND` | `paddle` or `hf` |
| `HF_API_TOKEN` | Hugging Face API token (for TrOCR) |
| `CAM_01_ROI` | ROI polygon for camera 1 (gate) |
| `CAM_02_ROI` | ROI polygon for camera 2 (plant area) |
| `YOLO_CONF` | YOLO confidence threshold |
| `YOLO_IOU` | YOLO IoU threshold |
| `YOLO_IMGSZ` | YOLO image size |

### Frontend (.env)

| Variable | Description |
|---|---|
| `REACT_APP_SOCKET_URL` | WebSocket URL (default: `http://localhost:8000`) |
