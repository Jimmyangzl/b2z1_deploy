"""WebSocket client for receiving Vive tracker pose in the MuJoCo sim process."""

import asyncio
import json
import os
import sys
import threading
import time

import numpy as np
import websockets

TELEOP_DIR = os.path.dirname(os.path.abspath(__file__))
ACT_DIR = os.path.abspath(os.path.join(TELEOP_DIR, ".."))
if ACT_DIR not in sys.path:
    sys.path.insert(0, ACT_DIR)

from teleop.vr_tracker_core import load_config


class VrWebSocketClient:
    """Background client that keeps the latest tracker pose (queue depth 1)."""

    def __init__(self, host=None, port=None, config_path=None):
        config = load_config(config_path)
        self.host = host or "127.0.0.1"
        self.port = port or int(config.get("websocket_port", 8765))
        self._latest = {
            "connected": False,
            "position": None,
            "orientation": None,
            "Tdes": None,
            "T_init": None,
            "q_init": None,
            "robot_ready": False,
            "holding": False,
            "timestamp": None,
        }
        self._lock = threading.Lock()
        self._thread = None
        self._stop_event = threading.Event()

    @property
    def uri(self):
        return f"ws://{self.host}:{self.port}"

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def get_latest_pose(self):
        with self._lock:
            return dict(self._latest)

    def _set_latest(self, payload):
        with self._lock:
            self._latest = payload

    def _run_loop(self):
        while not self._stop_event.is_set():
            try:
                asyncio.run(self._consume())
            except Exception as exc:
                print(f"VR WebSocket disconnected: {exc}. Retrying in 1s...")
                self._set_latest({
                    "connected": False,
                    "position": None,
                    "orientation": None,
                    "Tdes": None,
                    "T_init": None,
                    "q_init": None,
                    "robot_ready": False,
                    "holding": False,
                    "timestamp": time.time(),
                })
                time.sleep(1.0)

    async def _consume(self):
        async with websockets.connect(self.uri) as websocket:
            print(f"Connected to VR WebSocket at {self.uri}")
            while not self._stop_event.is_set():
                raw = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                data = json.loads(raw)
                position = data.get("position")
                orientation = data.get("orientation")
                Tdes = data.get("Tdes")
                T_init = data.get("T_init")
                q_init = data.get("q_init")
                self._set_latest({
                    "connected": bool(data.get("connected", False)),
                    "position": None if position is None else np.array(position),
                    "orientation": None if orientation is None else np.array(orientation),
                    "Tdes": None
                    if Tdes is None
                    else np.asarray(Tdes, dtype=np.float64).reshape(4, 4),
                    "T_init": None
                    if T_init is None
                    else np.asarray(T_init, dtype=np.float64).reshape(4, 4),
                    "q_init": None if q_init is None else np.asarray(q_init, dtype=np.float64),
                    "robot_ready": bool(data.get("robot_ready", False)),
                    "holding": bool(data.get("holding", False)),
                    "timestamp": data.get("timestamp", time.time()),
                })
