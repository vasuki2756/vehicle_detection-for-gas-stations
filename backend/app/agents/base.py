import asyncio
from typing import Dict, Any, Callable, List

class BaseAgent:
    def __init__(self, name: str):
        self.name = name
        self.input_queue = asyncio.Queue()
        self.subscribers: List[Callable] = []
        self.is_running = False

    async def start(self):
        """Start the agent loop"""
        self.is_running = True
        print(f"[{self.name}] Started.")
        asyncio.create_task(self._process_queue())

    async def stop(self):
        self.is_running = False
        print(f"[{self.name}] Stopped.")

    async def _process_queue(self):
        """Main event loop"""
        while self.is_running:
            try:
                # Wait for message
                message = await self.input_queue.get()
                await self.process_message(message)
                self.input_queue.task_done()
            except Exception as e:
                print(f"[{self.name}] Error processing message: {e}")

    async def process_message(self, message: Dict[str, Any]):
        """Override this method in subclasses"""
        pass

    def subscribe(self, callback: Callable):
        """Other agents can subscribe to this agent's outputs"""
        self.subscribers.append(callback)

    async def emit_event(self, event_type: str, payload: Dict[str, Any]):
        """Emit processed event to subscribers"""
        event = {
            "source": self.name,
            "type": event_type,
            "payload": payload
        }
        for callback in self.subscribers:
            if asyncio.iscoroutinefunction(callback):
                await callback(event)
            else:
                callback(event)

    async def ingest_frame(self, frame, frame_id, metadata=None):
        """Helper to ingest video frames"""
        await self.input_queue.put({
            "type": "frame",
            "frame": frame,
            "frame_id": frame_id,
            "metadata": metadata or {}
        })
