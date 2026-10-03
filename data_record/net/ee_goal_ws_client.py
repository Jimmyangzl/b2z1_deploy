"""WebSocket client for WBC ee_goal_local_cart stream."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Optional

import numpy as np

try:
    import websockets
except ImportError as exc:  # pragma: no cover
    raise ImportError("websockets is required for EeGoalWsClient") from exc


class EeGoalWsClient:
    """Background client keeping the latest EE goal from the WBC publisher."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8770):
        self.host = str(host)
        self.port = int(port)
        self._lock = threading.Lock()
        self._latest = {
            "valid": False,
            "ee_goal_local_cart": np.full(3, np.nan, dtype=np.float64),
            "t": 0.0,
            "recv_t": 0.0,
        }
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    @property
    def uri(self) -> str:
        return f"ws://{self.host}:{self.port}"

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, name="ee_goal_ws_client", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def get_latest(self) -> dict:
        with self._lock:
            out = dict(self._latest)
            out["ee_goal_local_cart"] = np.asarray(
                out["ee_goal_local_cart"], dtype=np.float64
            ).copy()
            return out

    def _set_latest(self, payload: dict) -> None:
        with self._lock:
            self._latest = payload

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                asyncio.run(self._consume())
            except Exception as exc:
                print(f"[EeGoalWsClient] disconnected: {exc}. Retrying in 1s...", flush=True)
                self._set_latest(
                    {
                        "valid": False,
                        "ee_goal_local_cart": np.full(3, np.nan, dtype=np.float64),
                        "t": 0.0,
                        "recv_t": time.time(),
                    }
                )
                time.sleep(1.0)

    async def _consume(self) -> None:
        async with websockets.connect(self.uri) as websocket:
            print(f"[EeGoalWsClient] connected to {self.uri}", flush=True)
            while not self._stop.is_set():
                try:
                    raw = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                data = json.loads(raw)
                goal = data.get("ee_goal_local_cart")
                goal_arr = (
                    np.full(3, np.nan, dtype=np.float64)
                    if goal is None
                    else np.asarray(goal, dtype=np.float64).reshape(3)
                )
                self._set_latest(
                    {
                        "valid": bool(data.get("valid", np.all(np.isfinite(goal_arr)))),
                        "ee_goal_local_cart": goal_arr,
                        "t": float(data.get("t", time.time())),
                        "recv_t": time.time(),
                    }
                )
