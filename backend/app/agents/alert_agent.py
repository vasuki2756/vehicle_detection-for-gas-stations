from .base import BaseAgent
import json
import asyncio
import uuid
import os
from app.database import DatabaseManager

class AlertAgent(BaseAgent):
    """
    Agent 3: AlertOrchestrator
    Responsibilities:
    - Aggregate events from Agents 1, 2, 4
    - Format messages
    - Send to WebSocket (Frontend)
    - Send to WhatsApp API
    """
    def __init__(self, db: DatabaseManager, ws_manager):
        super().__init__("AlertAgent")
        self.db = db
        self.ws_manager = ws_manager # Function or Object to send WS messages

    def _new_alert_id(self, prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex}"

    async def process_message(self, message):
         # This agent receives high-level events, not raw frames
         event_type = message.get("type")
         payload = message.get("payload", {})
         source = message.get("source")
         
         if event_type == "unauthorized_vehicle":
             await self._handle_unauthorized(payload)
         elif event_type == "unverified_vehicle":
             await self._handle_unverified(payload)
         elif event_type == "dwell_time_calculated":
             await self._handle_dwell(payload)
         elif event_type == "hazard_delay":
             await self._handle_hazard(payload)
         elif event_type == "vehicle_arrived":
             # Just notify frontend log, no major alert
             await self._notify_frontend("info", f"Vehicle arrived at gate. Wait required: {payload.get('wait_required',0):.1f}s")

    async def _handle_unauthorized(self, data):
        msg = f"Unauthorized Vehicle Detected! Plate: {data['plate_number']} in {data['zone']}"
        alert_pkg = {
            "id": f"unauth_{uuid.uuid4().hex}",
            "alert_type": "Unauthorized",
            "message": msg,
            "severity": "critical",
            "timestamp": data.get("timestamp")
        }
        
        # 1. DB Log
        try:
            self.db.save_alert(alert_pkg)
        except Exception as e:
            print(f"[AlertAgent] Failed to save unauthorized alert: {e}")
        
        # 2. Frontend
        await self._notify_frontend("alert", alert_pkg)
        
        # 3. WhatsApp
        await self._send_whatsapp(msg)

    async def _handle_unverified(self, data):
        msg = f"Unverified Vehicle Detected! Plate unreadable for Track: {data.get('track_id')} in {data.get('zone')}"
        alert_pkg = {
            "id": f"unverified_{uuid.uuid4().hex}",
            "alert_type": "Unverified",
            "message": msg,
            "severity": "warning",
            "timestamp": data.get("timestamp"),
            "track_id": data.get("track_id"),
            "plate_number": data.get("plate_number")
        }
        try:
            self.db.save_alert(alert_pkg)
        except Exception as e:
            print(f"[AlertAgent] Failed to save unverified alert: {e}")
        await self._notify_frontend("alert", alert_pkg)
        await self._send_whatsapp(msg)

    async def _handle_dwell(self, data):
        dwell_time = float(data.get("dwell_time", 0))
        msg = f"Vehicle {data.get('track_id')} Gate Processing Complete. Dwell: {dwell_time:.1f}s"

        alert_pkg = {
            "id": f"dwell_{uuid.uuid4().hex}",
            "alert_type": "Delay",
            "message": msg,
            "severity": "info",
            "timestamp": data.get("timestamp")
        }
        try:
            self.db.save_alert(alert_pkg)
        except Exception as e:
            print(f"[AlertAgent] Failed to save dwell alert: {e}")
        await self._notify_frontend("alert", alert_pkg)

    async def _handle_hazard(self, data):
        msg = f"⚠️ HAZARD DELAY: Vehicle {data['plate_number']} exceed time limit by {int(data['elapsed_minutes'])} min"
        alert_pkg = {
            "id": f"haz_{uuid.uuid4().hex}",
            "alert_type": "Delay",
            "message": msg,
            "severity": "high",
            "timestamp": None
        }
        try:
            self.db.save_alert(alert_pkg)
        except Exception as e:
            print(f"[AlertAgent] Failed to save hazard alert: {e}")
        await self._notify_frontend("alert", alert_pkg)
        await self._send_whatsapp(msg)

    async def _notify_frontend(self, msg_type, data):
        if self.ws_manager:
            if msg_type == "alert":
                await self.ws_manager.broadcast({"type": "alert", **data})
            else:
                 # Generic log or toast
                 await self.ws_manager.broadcast({"type": "log", "message": str(data)})

    async def _send_whatsapp(self, text):
        """Send WhatsApp via Twilio if configured; otherwise fall back to console log."""
        # IMPORTANT: do not hardcode credentials or phone numbers here.
        # Configure via environment variables.
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        from_whatsapp = os.getenv("TWILIO_WHATSAPP_FROM")  # e.g. whatsapp:+14155238886 (sandbox) or your Twilio-enabled sender
        to_whatsapp = os.getenv("TWILIO_WHATSAPP_TO")      # e.g. whatsapp:+9186...

        # Optional Twilio Content Template.
        # Default behavior is to send the *alert text* as the WhatsApp body.
        # To force template sending, set TWILIO_SEND_MODE=template.
        send_mode = (os.getenv("TWILIO_SEND_MODE") or "body").strip().lower()  # body|template
        content_sid = os.getenv("TWILIO_CONTENT_SID")
        content_variables = os.getenv("TWILIO_CONTENT_VARIABLES")  # JSON string e.g. {"1":"12/1","2":"3pm"}

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
            print(f"📱 [WhatsApp:mock] missing env vars: {', '.join(missing)}")
            print(f"📱 [WhatsApp:mock] {text}")
            return

        try:
            from twilio.rest import Client

            client = Client(account_sid, auth_token)

            if send_mode == "template" and content_sid:
                kwargs = {
                    "from_": from_whatsapp,
                    "to": to_whatsapp,
                    "content_sid": content_sid,
                }
                if content_variables:
                    kwargs["content_variables"] = content_variables
                message = client.messages.create(**kwargs)
            else:
                message = client.messages.create(from_=from_whatsapp, to=to_whatsapp, body=text)

            print(f"📱 [WhatsApp:twilio] sent sid={message.sid}")
        except Exception as e:
            print(f"📱 [WhatsApp:twilio] failed: {e}")
