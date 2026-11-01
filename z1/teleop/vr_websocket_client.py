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
    """Background client that keeps the latest tracker pose (queue depth 1).

    Important for 50 Hz teleop: the websocket thread only stores the latest
    *raw* JSON text. ``json.loads`` + NumPy conversion run lazily in
    ``get_latest_pose()`` on the control thread. Parsing every tracker frame
    on the WS thread (often ≫50 Hz) held the GIL and stalled obs/arm after
    a while even when policy ran in another process.
    """

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
        # Preallocated SE3 buffers — updated in place on parse.
        self._Tdes_buf = np.eye(4, dtype=np.float64)
        self._T_init_buf = np.eye(4, dtype=np.float64)
        self._pos_buf = np.zeros(3, dtype=np.float64)
        self._orn_buf = np.zeros(4, dtype=np.float64)
        self._q_init_buf = np.zeros(6, dtype=np.float64)
        self._raw = None
        self._raw_seq = 0
        self._parsed_seq = -1
        self._pose_cache = None
        self._pose_cache_seq = -1
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
        """Return latest pose; parse at most once per new WS message."""
        with self._lock:
            raw = self._raw
            seq = self._raw_seq
        if raw is not None and seq != self._parsed_seq:
            self._parse_raw(raw, seq)
        if self._pose_cache is not None and self._pose_cache_seq == self._parsed_seq:
            return self._pose_cache
        with self._lock:
            pose = {
                "connected": self._latest["connected"],
                "position": None
                if self._latest["position"] is None
                else self._pos_buf.copy(),
                "orientation": None
                if self._latest["orientation"] is None
                else self._orn_buf.copy(),
                "Tdes": None if self._latest["Tdes"] is None else self._Tdes_buf.copy(),
                "T_init": None
                if self._latest["T_init"] is None
                else self._T_init_buf.copy(),
                "q_init": None
                if self._latest["q_init"] is None
                else self._q_init_buf.copy(),
                "robot_ready": self._latest["robot_ready"],
                "holding": self._latest["holding"],
                "timestamp": self._latest["timestamp"],
            }
        self._pose_cache = pose
        self._pose_cache_seq = self._parsed_seq
        return pose

    def _parse_raw(self, raw: str, seq: int) -> None:
        try:
            data = json.loads(raw)
        except Exception:
            return
        position = data.get("position")
        orientation = data.get("orientation")
        Tdes = data.get("Tdes")
        T_init = data.get("T_init")
        q_init = data.get("q_init")

        if position is not None:
            np.copyto(self._pos_buf, np.asarray(position, dtype=np.float64).reshape(3))
        if orientation is not None:
            np.copyto(
                self._orn_buf, np.asarray(orientation, dtype=np.float64).reshape(-1)[:4]
            )
        if Tdes is not None:
            np.copyto(
                self._Tdes_buf, np.asarray(Tdes, dtype=np.float64).reshape(4, 4)
            )
        if T_init is not None:
            np.copyto(
                self._T_init_buf, np.asarray(T_init, dtype=np.float64).reshape(4, 4)
            )
        if q_init is not None:
            q = np.asarray(q_init, dtype=np.float64).reshape(-1)
            n = min(q.size, self._q_init_buf.size)
            self._q_init_buf[:] = 0.0
            self._q_init_buf[:n] = q[:n]

        with self._lock:
            # Drop stale parse if a newer raw arrived while we decoded.
            if seq != self._raw_seq:
                return
            self._latest = {
                "connected": bool(data.get("connected", False)),
                "position": None if position is None else self._pos_buf,
                "orientation": None if orientation is None else self._orn_buf,
                "Tdes": None if Tdes is None else self._Tdes_buf,
                "T_init": None if T_init is None else self._T_init_buf,
                "q_init": None if q_init is None else self._q_init_buf,
                "robot_ready": bool(data.get("robot_ready", False)),
                "holding": bool(data.get("holding", False)),
                "timestamp": data.get("timestamp", time.time()),
            }
            self._parsed_seq = seq

    def _set_disconnected(self) -> None:
        with self._lock:
            self._raw = None
            self._raw_seq += 1
            self._parsed_seq = -1
            self._latest = {
                "connected": False,
                "position": None,
                "orientation": None,
                "Tdes": None,
                "T_init": None,
                "q_init": None,
                "robot_ready": False,
                "holding": False,
                "timestamp": time.time(),
            }

    def _run_loop(self):
        while not self._stop_event.is_set():
            try:
                asyncio.run(self._consume())
            except Exception as exc:
                print(f"VR WebSocket disconnected: {exc}. Retrying in 1s...")
                self._set_disconnected()
                time.sleep(1.0)

    async def _consume(self):
        async with websockets.connect(self.uri) as websocket:
            print(f"Connected to VR WebSocket at {self.uri}")
            while not self._stop_event.is_set():
                raw = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                # Store raw only — no json.loads / NumPy on this thread.
                with self._lock:
                    self._raw = raw
                    self._raw_seq += 1
