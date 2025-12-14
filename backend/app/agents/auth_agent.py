from .base import BaseAgent
from app.database import DatabaseManager

class AuthAgent(BaseAgent):
    """
    Agent 2: AuthGuard
    Responsibilities:
    - Receive vehicle data (with plate)
    - Check authorization in DB
    - Trigger Alert if unauthorized
    """
    def __init__(self, db: DatabaseManager):
        super().__init__("AuthAgent")
        self.db = db
        self.processed_plates = set() # Cache to avoid repeated alerts for same session
        self.processed_unverified = set()  # track_id cache

    async def process_message(self, message):
        if message["type"] == "frame_detections":
            await self._check_authorization(
                detections=message.get("detections", []),
                timestamp=message.get("frame_timestamp")
            )

    async def _check_authorization(self, detections, timestamp=None):
        for det in detections:
            plate = det.get("plate_number")
            track_id = det["track_id"]

            # If plate is missing/unreadable and vehicle is inside, emit an "unverified" alert after a short time.
            # This addresses cases where OCR fails but we still want operator attention.
            plate_missing = (not plate) or ("UNKNOWN" in str(plate)) or ("Scanning" in str(plate))
            if plate_missing and det.get("current_zone") == "Inside Facility":
                try:
                    time_in_zone = float(det.get("time_in_zone") or 0)
                except Exception:
                    time_in_zone = 0

                if time_in_zone >= 3 and track_id not in self.processed_unverified:
                    self.processed_unverified.add(track_id)
                    await self.emit_event("unverified_vehicle", {
                        "track_id": track_id,
                        "plate_number": plate or "UNVERIFIED",
                        "zone": det.get("current_zone"),
                        "timestamp": timestamp,
                        "reason": "Plate unreadable / missing"
                    })

                # Skip DB auth check when plate isn't usable
                if plate_missing:
                    continue
            
            # At this point we have a usable plate string
                
            # Session key to avoid spamming alerts for same vehicle track
            session_key = f"{track_id}_{plate}"
            if session_key in self.processed_plates:
                continue

            # Check DB
            auth_status = self.db.is_authorized(plate)
            
            det["is_authorized"] = auth_status.get("authorized", False)
            det["owner"] = auth_status.get("owner", "Unknown")
            det["vehicle_type"] = auth_status.get("vehicle_type", det.get("vehicle_type"))

            if not det["is_authorized"]:
                print(f"[AuthAgent] 🚨 Unauthorized Detected: {plate}")
                self.processed_plates.add(session_key)

                # Ensure UI reflects authorization state
                det["status"] = "unauthorized"
                
                # Emit critical event
                await self.emit_event("unauthorized_vehicle", {
                    "track_id": track_id,
                    "plate_number": plate,
                    "zone": det.get("current_zone"),
                    "timestamp": timestamp
                })
            else:
                 # Mark valid one too so we don't re-check
                 self.processed_plates.add(session_key)

                 # Optionally mark verified vehicles when inside plant
                 if det.get("current_zone") == "Inside Facility" and det.get("status") in (None, "", "unverified"):
                     det["status"] = "stored"
                 print(f"[AuthAgent] ✅ Verified: {plate} ({det['owner']})")
