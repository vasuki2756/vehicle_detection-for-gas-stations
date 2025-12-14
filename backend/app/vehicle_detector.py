"""
Vehicle Detection and Tracking System
Uses YOLO for detection, SAM2 for segmentation, PaddleOCR for plate reading
"""
import cv2
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from ultralytics import YOLO
import re
import os
import io
from collections import Counter

try:
    import requests
except ImportError:
    requests = None

try:
    from PIL import Image
except ImportError:
    Image = None

# Try to import SAM2
try:
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("Warning: SAM2 not found. Segmentation will fall back to YOLO masks or boxes.")

# Try to import PaddleOCR
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    print("Warning: PaddleOCR not found. Plate detection will be simulated.")

class VehicleDetector:
    """Main vehicle detection class integrating YOLO, SAM, and PaddleOCR"""
    
    # Chemical truck identifiers (expandable)
    CHEMICAL_KEYWORDS = ['chemical', 'hazmat', 'tanker', 'hazardous', 'gas', 'oil']
    
    # Authorized vehicle database (license plates)
    AUTHORIZED_PLATES = {
        'ABC1234', 'XYZ5678', 'DEF9012', 'GHI3456',
        'JKL7890', 'MNO2345', 'PQR6789', 'STU0123'
    }
    
    def __init__(
        self,
        yolo_model: YOLO,
        sam_predictor=None,
        *,
        ocr_backend: Optional[str] = None,
        hf_token: Optional[str] = None,
        hf_ocr_model: Optional[str] = None,
    ):
        self.yolo = yolo_model
        self.sam = sam_predictor
        self.ocr = None
        self.ocr_initialized = False

        # OCR backend selection
        # - paddle: local PaddleOCR (default)
        # - hf: Hugging Face Inference API (requires HF_TOKEN)
        self.ocr_backend = (ocr_backend or os.getenv("OCR_BACKEND") or "paddle").strip().lower()
        self.hf_ocr_model = hf_ocr_model or os.getenv("HF_OCR_MODEL") or "microsoft/trocr-base-printed"
        self.hf_token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        self._hf_warned_missing_token = False

        # YOLO tuning knobs (env-configurable)
        self.yolo_conf = float(os.getenv("YOLO_CONF", "0.25"))
        self.yolo_iou = float(os.getenv("YOLO_IOU", "0.45"))
        self.yolo_imgsz = int(os.getenv("YOLO_IMGSZ", "960"))

        # Post-detection filtering (reduces false positives)
        self.det_min_area_ratio = float(os.getenv("DETECTION_MIN_AREA_RATIO", "0.0005"))
        self.det_max_area_ratio = float(os.getenv("DETECTION_MAX_AREA_RATIO", "0.50"))
        self.det_max_aspect = float(os.getenv("DETECTION_MAX_ASPECT", "6.0"))
        self.det_min_side = int(os.getenv("DETECTION_MIN_SIDE", "20"))
        self.det_min_texture_std = float(os.getenv("DETECTION_MIN_TEXTURE_STD", "8.0"))
        self.det_min_persist_frames = int(os.getenv("DETECTION_MIN_PERSIST_FRAMES", "2"))
        self.track_seen_count: Dict[int, int] = {}

        # India plate formats (common heuristics)
        # Standard: MH12AB1234, DL1CAB1234, etc.
        # BH-series: 22BH1234AA
        self._re_india_std = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$")
        self._re_india_bh = re.compile(r"^\d{2}BH\d{4}[A-Z]{1,2}$")
        
        # Vehicle tracking information
        self.tracked_vehicles = {}  # track_id -> vehicle_info
        self.plate_history = {}     # track_id -> list of detected plates (for voting)
        self.next_track_id = 1

        # Per-track timing (independent of plate OCR)
        self.track_first_seen = {}  # track_id -> datetime
        
        # Gate monitoring
        self.gate_line_y = None  # Will be set based on frame height
        self.entry_records = {}  # plate -> entry_time

        # Gate dwell timing (for CAM_01 gate processing)
        self.track_gate_first_seen: Dict[int, datetime] = {}
        self.track_gate_last_dwell: Dict[int, int] = {}
        self.track_last_zone: Dict[int, str] = {}
    
    def _init_ocr_if_needed(self):
        """Initialize PaddleOCR on first use"""
        if self.ocr_backend in ("none", "off", "disabled"):
            self.ocr_initialized = True
            return

        if self.ocr_backend != "paddle":
            # HF backend is stateless; nothing to initialize here.
            self.ocr_initialized = True
            return

        if not self.ocr_initialized and self.ocr is None and PADDLE_AVAILABLE:
            try:
                print("Initializing PaddleOCR for plate recognition...")
                # Initialize PaddleOCR (Enabling mkldnn for CPU speedup if possible)
                self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
                self.ocr_initialized = True
                print("PaddleOCR ready")
            except Exception as e:
                print(f"Failed to initialize OCR: {e}")
                self.ocr_initialized = True  # Don't retry repeatedly
        
    def detect_vehicles(self, frame: np.ndarray) -> List[Dict]:
        """
        Main detection pipeline
        Returns list of detected vehicles with all metadata
        """
        if self.gate_line_y is None:
            # Set gate line at 1/3 from top (adjustable)
            self.gate_line_y = frame.shape[0] // 3
        
        results = []
        
        # 1. YOLO Detection & Tracking
        # classes: 2=car, 3=motorcycle, 5=bus, 7=truck
        yolo_results = self.yolo.track(
            frame,
            persist=True,
            classes=[2, 3, 5, 7],
            conf=self.yolo_conf,
            iou=self.yolo_iou,
            imgsz=self.yolo_imgsz,
            verbose=False,
        )
        
        if yolo_results[0].boxes is None or len(yolo_results[0].boxes) == 0:
            return results
            
        # Prepare SAM2 if available - Set image once per frame
        if self.sam is not None and len(yolo_results[0].boxes) > 0:
            try:
                self.sam.set_image(frame)
            except Exception as e:
                print(f"SAM2 set_image error: {e}")
        
        for idx, box in enumerate(yolo_results[0].boxes):
            # Extract YOLO data
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            
            # Track ID
            track_id = int(box.id[0]) if box.id is not None else self.next_track_id
            if box.id is None:
                self.next_track_id += 1

            # Clamp coords to frame bounds
            fh, fw = frame.shape[:2]
            x1 = max(0, min(x1, fw - 1))
            x2 = max(0, min(x2, fw))
            y1 = max(0, min(y1, fh - 1))
            y2 = max(0, min(y2, fh))

            bw = max(0, x2 - x1)
            bh = max(0, y2 - y1)
            if bw < self.det_min_side or bh < self.det_min_side:
                continue

            frame_area = float(fw * fh) if fw > 0 and fh > 0 else 1.0
            area_ratio = (bw * bh) / frame_area
            if area_ratio < self.det_min_area_ratio or area_ratio > self.det_max_area_ratio:
                continue

            # Aspect ratio filter (very long/thin boxes are often false positives)
            if bw > 0 and bh > 0:
                aspect = max(bw / bh, bh / bw)
                if aspect > self.det_max_aspect:
                    continue

            # Texture filter: flat regions are commonly false positives
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            try:
                gray_small = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                # downscale for speed
                if gray_small.shape[0] > 80 or gray_small.shape[1] > 80:
                    gray_small = cv2.resize(gray_small, (80, 80), interpolation=cv2.INTER_AREA)
                if float(np.std(gray_small)) < self.det_min_texture_std:
                    continue
            except Exception:
                # If preprocessing fails, don't drop the detection solely on this check
                pass

            # Require persistence over a few frames to reduce flicker/false positives
            self.track_seen_count[track_id] = self.track_seen_count.get(track_id, 0) + 1
            if self.track_seen_count[track_id] < self.det_min_persist_frames:
                continue

            # Track first-seen time
            if track_id not in self.track_first_seen:
                self.track_first_seen[track_id] = datetime.now()
            
            # 2. Vehicle Classification
            # Get basic crop for classification features
            vehicle_crop = crop
            vehicle_type = self.classify_vehicle_type(vehicle_crop, class_id, confidence)
            
            # 3. License Plate Recognition (PaddleOCR)
            # Only run OCR if we have a decent crop and it's a vehicle of interest
            plate_number = self.get_best_plate(track_id, vehicle_crop)
            
            # 4. Segmentation (SAM2)
            mask = self.get_vehicle_mask(frame, box, yolo_results[0])
            
            # 5. Authorization & Tracking Logic
            # Defer authorization to AuthAgent/DB when running the agentic backend.
            # If plate is unknown, keep authorization as None rather than False to avoid false 'unauthorized' states.
            is_authorized = (plate_number in self.AUTHORIZED_PLATES) if plate_number else None
            
            current_zone, status, time_in_zone, gate_dwell_time = self.track_gate_crossing(track_id, plate_number, y1, y2)
            
            # Build vehicle data object
            vehicle_data = {
                "track_id": f"V{track_id}",
                "vehicle_type": vehicle_type,
                "plate_number": plate_number or "Scanning...",
                "confidence": round(confidence, 2),
                "bbox": [x1, y1, x2, y2],
                "mask": mask,
                "current_zone": current_zone,
                "status": status,
                "is_authorized": is_authorized,
                "time_in_zone": time_in_zone,
                "gate_dwell_time": gate_dwell_time,
                "entry_time": self.track_first_seen.get(track_id, datetime.now()).isoformat()
            }
            
            self.tracked_vehicles[track_id] = vehicle_data
            results.append(vehicle_data)
        
        return results
    
    def get_best_plate(self, track_id: int, vehicle_crop: np.ndarray) -> Optional[str]:
        """
        Run OCR on vehicle crop and use simple voting history to stabilize the plate number.
        """
        if self.ocr_backend in ("none", "off", "disabled"):
            return None

        # Initialize OCR if needed
        self._init_ocr_if_needed()
        
        # 1. Detect Plate on current frame
        current_plate = None
        if vehicle_crop.size > 0:
            # HF backend does not create self.ocr; call read_license_plate directly
            if self.ocr_backend == "hf":
                current_plate = self.read_license_plate(vehicle_crop)
            elif self.ocr:
                current_plate = self.read_license_plate(vehicle_crop)
        
        # 2. Update History
        if track_id not in self.plate_history:
            self.plate_history[track_id] = []
        
        if current_plate:
            self.plate_history[track_id].append(current_plate)
            # Keep last 10 detections
            if len(self.plate_history[track_id]) > 10:
                self.plate_history[track_id].pop(0)
        
        # 3. Vote for best plate
        history = self.plate_history[track_id]
        if not history:
            return None
            
        # Prefer a plate that matches India patterns, then fall back to most common
        counts = Counter(history)
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], -len(kv[0])))
        for plate, _cnt in ranked:
            if self._looks_like_indian_plate(plate):
                return plate
        return ranked[0][0]

    def _looks_like_indian_plate(self, plate: str) -> bool:
        if not plate:
            return False
        p = self._normalize_plate_text(plate)
        return bool(self._re_india_std.match(p) or self._re_india_bh.match(p))

    def _normalize_plate_text(self, text: str) -> str:
        # Uppercase + strip non-alnum
        p = re.sub(r"[^A-Z0-9]", "", str(text).upper())
        # Remove common OCR artifacts
        p = p.replace("IND", "")
        return p

    def read_license_plate(self, vehicle_crop: np.ndarray) -> Optional[str]:
        """
        Read license plate text from vehicle crop.
        Supports PaddleOCR (local) or Hugging Face Inference API (HF).
        Optimized by focusing on the bottom area of the vehicle.
        """
        try:
            h, w = vehicle_crop.shape[:2]
            if h < 20 or w < 20: 
                return None

            # OCR on full vehicle crop is more reliable for different vehicle types
            plate_region = vehicle_crop
            
            # Preprocessing for better OCR
            # 1. Resize if too small (scale up 2x)
            if h < 100:
                scale = 2.0
                plate_region = cv2.resize(plate_region, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
            # 2. Enhance Contrast (Convert to Gray -> CLAHE)
            gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Convert back to BGR for Paddle (it expects 3 channels usually, though it handles gray too)
            plate_region = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            
            # Run PaddleOCR
            if self.ocr_backend == "hf":
                return self._read_license_plate_hf(plate_region)

            if not self.ocr:
                return None

            result = self.ocr.ocr(plate_region, cls=False)
            
            if not result or result[0] is None:
                return None
            
            # Process results: Look for string matching plate patterns
            # e.g., 3-4 letters followed by digits, or similar
            detected_candidates = []
            for line in result[0]:
                text = line[1][0]
                conf = line[1][1]
                
                # Basic cleaning
                clean_text = self._normalize_plate_text(text)
                
                # Filter noise (plates usually > 4 chars)
                if len(clean_text) >= 4 and conf > 0.6:
                    detected_candidates.append((clean_text, float(conf)))
            
            if detected_candidates:
                # Prefer India-plate-looking candidates, then highest confidence, then length
                detected_candidates.sort(
                    key=lambda x: (
                        1 if self._looks_like_indian_plate(x[0]) else 0,
                        x[1],
                        len(x[0]),
                    ),
                    reverse=True,
                )
                return detected_candidates[0][0]
                
            return None
            
        except Exception as e:
            # print(f"OCR error: {e}") # Suppress spam
            return None

    def _read_license_plate_hf(self, plate_region_bgr: np.ndarray) -> Optional[str]:
        """Hugging Face Inference API OCR using a vision->text model (e.g. TrOCR)."""
        if requests is None or Image is None:
            return None

        if not self.hf_token:
            if not self._hf_warned_missing_token:
                print("HF OCR selected but HF_TOKEN is missing; skipping OCR")
                self._hf_warned_missing_token = True
            return None

        try:
            # Convert BGR numpy -> PNG bytes
            rgb = cv2.cvtColor(plate_region_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            payload = buf.getvalue()

            url = f"https://api-inference.huggingface.co/models/{self.hf_ocr_model}"
            headers = {
                "Authorization": f"Bearer {self.hf_token}",
                "Accept": "application/json",
            }

            resp = requests.post(url, headers=headers, data=payload, timeout=20)
            if resp.status_code != 200:
                return None

            data = resp.json()
            # Common response: [{"generated_text": "ABC123"}]
            if isinstance(data, list) and data:
                text = data[0].get("generated_text")
            elif isinstance(data, dict):
                text = data.get("generated_text") or data.get("text")
            else:
                text = None

            if not text:
                return None

            clean_text = self._normalize_plate_text(text)
            if len(clean_text) < 4:
                return None
            return clean_text
        except Exception:
            return None

    def get_vehicle_mask(self, frame: np.ndarray, box, yolo_result) -> List[List[int]]:
        """
        Get vehicle segmentation mask. 
        Prioritizes SAM2 -> YOLO Mask -> Bounding Box fallback
        """
        # 1. SAM2
        if self.sam is not None:
            try:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                input_box = np.array([x1, y1, x2, y2])
                
                # Predict
                masks, scores, _ = self.sam.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False
                )
                
                if len(masks) > 0:
                    best_mask = masks[0].astype(np.uint8)
                    
                    # Convert binary mask to polygon
                    contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest = max(contours, key=cv2.contourArea)
                        # Simplify
                        epsilon = 0.005 * cv2.arcLength(largest, True)
                        approx = cv2.approxPolyDP(largest, epsilon, True)
                        
                        polygon = approx.reshape(-1, 2).tolist()
                        if len(polygon) > 3:
                            return polygon
            except Exception as e:
                print(f"SAM2 seg error: {e}")

        # 2. YOLO Masks
        if yolo_result.masks is not None:
            try:
                # YOLO masks are often relative, need to check format
                # Using the raw xy data if available
                if hasattr(yolo_result.masks[0], 'xy'):
                    # The masks object contains masks for all boxes in the result
                    # We need to find the one corresponding to OUR box. 
                    # Note: yolo_result.masks[i] corresponds to yolo_result.boxes[i]
                    # Since we are iterating boxes, we need the index.
                    # HOWEVER, in the detect_vehicles loop we are iterating boxes. we need the index there.
                    # Update: Let's assume simpler BBox for fallback if SAM fails or complexities arise.
                    pass
            except:
                pass

        # 3. Fallback: Bounding Box
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        return [
            [x1, y1], [x2, y1], [x2, y2], [x1, y2]
        ]

    def classify_vehicle_type(self, vehicle_crop: np.ndarray, class_id: int, confidence: float) -> str:
        """
        Classify vehicle type using YOLO class detection + heuristics
        """
        if class_id == 7:  # Truck
            h, w = vehicle_crop.shape[:2]
            aspect_ratio = w / h if h > 0 else 0
            
            # Heuristic: Tankers often long/bulky. 
            # In a real system, train a specific classifier (e.g. ResNET) on truck types.
            if 0.5 < aspect_ratio < 2.5 and h > 50:
                 return "Hazardous"
            return "Truck"
            
        elif class_id == 5:  # Bus
            return "Worker"
        elif class_id == 3:  # Motorcycle
            return "Motorcycle"
        elif class_id == 2:  # Car
            return "Car"
        
        return "Other"

    def track_gate_crossing(self, track_id: int, plate_number: Optional[str], y1: int, y2: int) -> Tuple[str, str, int, int]:
        """
        Track vehicle crossing through gate logic
        """
        vehicle_cy = (y1 + y2) // 2

        now = datetime.now()
        
        # Simple logic: Top 1/3 is "Outside", Middle is "Gate", Bottom is "Inside"
        # gate_line_y is approx 1/3 of frame
        
        if vehicle_cy < self.gate_line_y - 50:
            current_zone = "Outside"
            status = "approaching"
            time_in_zone = 0

        elif abs(vehicle_cy - self.gate_line_y) <= 50:
            # Crossing gate
            # Ensure first-seen is set for timing even if plate is unknown
            if track_id not in self.track_first_seen:
                self.track_first_seen[track_id] = now

            # Keep legacy plate-based entry record when plate exists
            if plate_number and plate_number not in self.entry_records:
                self.entry_records[plate_number] = now

            current_zone = "Gate Entry"
            status = "in_transit"
            time_in_zone = 0

        else:
            # Inside
            first_seen = self.track_first_seen.get(track_id)
            time_in_zone = int((now - first_seen).total_seconds()) if first_seen else 0
            status = "inside"

            # Rule: Hazardous vehicles > 72 seconds = delayed
            if time_in_zone > 72:
                status = "delayed"
            elif plate_number in self.AUTHORIZED_PLATES:
                status = "stored"  # Just an example status
            elif plate_number and not self.is_authorized(plate_number):
                status = "unauthorized"
            elif not plate_number:
                status = "unverified"

            current_zone = "Inside Facility"

        # Gate dwell calculation based on zone transitions
        prev_zone = self.track_last_zone.get(track_id)
        if prev_zone != current_zone:
            # entering gate zone
            if current_zone == "Gate Entry":
                self.track_gate_first_seen[track_id] = now
            # leaving gate zone
            if prev_zone == "Gate Entry" and current_zone != "Gate Entry":
                start = self.track_gate_first_seen.get(track_id)
                if start:
                    self.track_gate_last_dwell[track_id] = int((now - start).total_seconds())

        self.track_last_zone[track_id] = current_zone

        # live gate dwell time while in gate zone; otherwise last dwell (if exists)
        if current_zone == "Gate Entry":
            start = self.track_gate_first_seen.get(track_id)
            gate_dwell_time = int((now - start).total_seconds()) if start else 0
        else:
            gate_dwell_time = int(self.track_gate_last_dwell.get(track_id, 0) or 0)

        return current_zone, status, time_in_zone, gate_dwell_time

    def _get_time_in_zone(self, plate_number: Optional[str]) -> int:
        if not plate_number or plate_number not in self.entry_records:
            return 0
        return int((datetime.now() - self.entry_records[plate_number]).total_seconds())

    def is_authorized(self, plate_number: Optional[str]) -> bool:
        if not plate_number: return False
        return plate_number in self.AUTHORIZED_PLATES

    def generate_alert(self, vehicle: Dict) -> Optional[Dict]:
        """Generate alerts based on business rules"""
        import uuid
        alert_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Rule 1: Unauthorized inside
        if vehicle['status'] == 'unauthorized' and vehicle['current_zone'] == 'Inside Facility':
            return {
                "id": alert_id, "timestamp": timestamp, "track_id": vehicle["track_id"],
                "plate_number": vehicle["plate_number"], "alert_type": "Unauthorized",
                "message": f"Unauthorized vehicle {vehicle['plate_number']} detected inside."
            }
            
        # Rule 2: Hazardous vehicle delayed
        if vehicle['vehicle_type'] == 'Hazardous' and vehicle['time_in_zone'] > 7200: # 2 hrs
            return {
                "id": alert_id, "timestamp": timestamp, "track_id": vehicle["track_id"],
                "plate_number": vehicle["plate_number"], "alert_type": "Delay",
                "message": f"Hazardous vehicle {vehicle['plate_number']} overstayed (>2h)."
            }
            
        return None

    def cleanup_old_records(self):
        # ... logic to clean entry_records ...
        pass
