import time
from .base import BaseAgent

class HazardAgent(BaseAgent):
    """
    Agent 4: HazardTracker
    Responsibilities:
    - Monitor Hazardous vehicles
    - Track zone entry/exit times
    - Detect delays/overstays
    """
    def __init__(self):
        super().__init__("HazardAgent")
        self.tracking = {} # track_id -> {entry_time, max_allowed}
        
    async def process_message(self, message):
        if message["type"] == "frame_detections":
            await self._monitor_hazardous(message["detections"])

    async def _monitor_hazardous(self, detections):
        for det in detections:
            track_id = det["track_id"]
            v_type = det.get("vehicle_type", "Other")
            
            if v_type == "Hazardous":
                # Start tracking if new
                if track_id not in self.tracking:
                    self.tracking[track_id] = {
                        "entry_time": time.time(),
                        "max_allowed": 7200, # 2 hours in seconds (configurable)
                        "plate": det.get("plate_number")
                    }
                    print(f"[HazardAgent] Tracking Hazardous Load: {track_id}")
                
                # Check status
                data = self.tracking[track_id]
                elapsed = time.time() - data["entry_time"]
                
                # Check for violation
                # NOTE: For demo purposes, let's say 2 hours is too long to wait to see an alert.
                # In real code keep 7200.
                if elapsed > data["max_allowed"]:
                     if not data.get("alerted"):
                         print(f"[HazardAgent] ⚠️ DELAY ALERT: {track_id}")
                         await self.emit_event("hazard_delay", {
                             "track_id": track_id,
                             "plate_number": data["plate"],
                             "elapsed_minutes": elapsed / 60,
                             "severity": "HIGH"
                         })
                         data["alerted"] = True
