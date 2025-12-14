import time
from datetime import datetime
from .base import BaseAgent
from app.database import DatabaseManager

class GateAgent(BaseAgent):
    """
    Agent 1: GateWatcher
    Responsibilities:
    - Detect vehicles at gate
    - Compute dwell time (T_end - T_start)
    - Enforce wait time for next vehicle
    """
    def __init__(self, db: DatabaseManager):
        super().__init__("GateAgent")
        self.db = db
        self.active_vehicles = {} # track_id -> entry_time
        self.last_vehicle_exit_time = 0
        self.required_wait_time = 0

    async def process_message(self, message):
        if message["type"] == "frame_detections":
            await self._process_detections(message["detections"], message["frame_timestamp"])

    async def _process_detections(self, detections, timestamp):
        current_track_ids = set()
        
        for det in detections:
            track_id = det["track_id"]
            current_track_ids.add(track_id)
            
            # Check if entering gate zone (simple y-axis check or polygon check passed from detector)
            # For simplicity, assuming "Gate Entry" zone is identified by detector
            if det.get("current_zone") == "Gate Entry":
                
                # New Arrival
                if track_id not in self.active_vehicles:
                    self.active_vehicles[track_id] = time.time()
                    print(f"[GateAgent] Vehicle {track_id} arrived at gate.")
                    
                    # Log to DB
                    self.db.log_entry(
                        track_id=str(track_id),
                        plate_number=det.get("plate_number", "UNKNOWN"),
                        camera_id="CAM_01",
                        zone="Gate Entry"
                    )
                    
                    await self.emit_event("vehicle_arrived", {
                        "track_id": track_id,
                        "timestamp": datetime.now().isoformat(),
                        "wait_required": self.required_wait_time
                    })

        # Check for Exits (vehicles previously active but not in current frame/zone)
        # Note: In real ByteTrack, we check 'is_lost' or absent for N frames.
        # Here simplifying: if not in list, it left.
        
        left_vehicles = [tid for tid in self.active_vehicles if tid not in current_track_ids]
        
        for tid in left_vehicles:
            start_time = self.active_vehicles.pop(tid)
            end_time = time.time()
            dwell_time = end_time - start_time
            
            # Update system state
            self.last_vehicle_exit_time = end_time
            self.required_wait_time = dwell_time # Next vehicle must wait this long
            
            print(f"[GateAgent] Vehicle {tid} left. Dwell Time: {dwell_time:.2f}s")
            
            # LogDB
            self.db.log_exit_and_dwell(str(tid), dwell_time)
            
            # Emit Event
            await self.emit_event("dwell_time_calculated", {
                "track_id": tid,
                "dwell_time": dwell_time,
                "next_wait_time": self.required_wait_time
            })
