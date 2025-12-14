import sqlite3
import os
from datetime import datetime
from typing import Optional, Dict, List

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plant_security.db")

class DatabaseManager:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Initialize database schema"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # Authorized Vehicles Table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS authorized_vehicles (
            plate_number TEXT PRIMARY KEY,
            vehicle_type TEXT,
            owner_name TEXT,
            authorized_zones TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Access/Dwell Logs
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS access_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id TEXT,
            plate_number TEXT,
            camera_id TEXT,
            entry_time TIMESTAMP,
            exit_time TIMESTAMP,
            dwell_time_seconds REAL,
            zone TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Alerts History
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id TEXT PRIMARY KEY,
            alert_type TEXT,
            message TEXT,
            severity TEXT,
            timestamp TIMESTAMP,
            is_acknowledged BOOLEAN DEFAULT 0
        )
        ''')

        # Seed some data if empty
        cursor.execute("SELECT count(*) FROM authorized_vehicles")
        if cursor.fetchone()[0] == 0:
            print("Seeding database with authorized vehicles...")
            seed_data = [
                ('KL12P7111', 'Car', 'Manager', 'All'),
                ('TN05A1234', 'Worker', 'Staff Bus', 'Gate,Plant'),
                ('MH12AB1234', 'Worker', 'Site Engineer', 'All'),
                ('KA01CD5678', 'Hazardous', 'Chemical Supplier', 'Loading Bay')
            ]
            cursor.executemany(
                "INSERT INTO authorized_vehicles (plate_number, vehicle_type, owner_name, authorized_zones) VALUES (?, ?, ?, ?)",
                seed_data
            )
            
        conn.commit()
        conn.close()

    def is_authorized(self, plate_number: str) -> Dict:
        """Check if vehicle is authorized"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM authorized_vehicles WHERE plate_number = ?", (plate_number,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "authorized": True,
                "plate_number": row[0],
                "vehicle_type": row[1],
                "owner": row[2]
            }
        return {"authorized": False}

    def log_entry(self, track_id: str, plate_number: str, camera_id: str, zone: str):
        """Log vehicle entry"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO access_logs (track_id, plate_number, camera_id, entry_time, zone) VALUES (?, ?, ?, ?, ?)",
            (track_id, plate_number, camera_id, datetime.now(), zone)
        )
        conn.commit()
        conn.close()

    def log_exit_and_dwell(self, track_id: str, dwell_time: float):
        """Update exit time and dwell time for a track_id"""
        conn = self._get_conn()
        cursor = conn.cursor()
        # Find latest open record for this track
        cursor.execute(
            "SELECT id FROM access_logs WHERE track_id = ? AND exit_time IS NULL ORDER BY entry_time DESC LIMIT 1",
            (track_id,)
        )
        row = cursor.fetchone()
        if row:
            record_id = row[0]
            cursor.execute(
                "UPDATE access_logs SET exit_time = ?, dwell_time_seconds = ? WHERE id = ?",
                (datetime.now(), dwell_time, record_id)
            )
            conn.commit()
        conn.close()

    def save_alert(self, alert_data: Dict):
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO alerts (id, alert_type, message, severity, timestamp) VALUES (?, ?, ?, ?, ?)",
                (
                    alert_data["id"],
                    alert_data.get("alert_type"),
                    alert_data.get("message"),
                    alert_data.get("severity", "high"),
                    alert_data.get("timestamp") or datetime.now().isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()
