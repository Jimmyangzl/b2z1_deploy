"""Broadcast latest ee_goal_local_cart over WebSocket for remote data recorders."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Optional, Set

import numpy as np

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
except ImportError as exc:  # pragma: no cover
    raise ImportError("websockets is required for EeGoalWsServer") from exc


class EeGoalWsServer:
    """Background WS server that publishes the latest EE goal at a fixed rate.

    Control loop calls ``update()`` whenever a new observation is available.
    A separate asyncio task broadcasts the keep-latest payload at ``rate_hz``.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8770,
        rate_hz: float = 50.0,
    ):
        self.host = str(host)
        self.port = int(port)
        self.rate_hz = max(1e-3, float(rate_hz))
        self._lock = threading.Lock()
        self._ee_goal = np.full(3, np.nan, dtype=np.float64)
        self._t = 0.0
        self._have_sample = False
        self._clients: Set[WebSocketServerProtocol] = set()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop = threading.Event()

    def update(self, ee_goal_local_cart, timestamp: Optional[float] = None) -> None:
        """Store latest EE goal (robot arm-base / mount frame, 3-vector)."""
        goal = np.asarray(ee_goal_local_cart, dtype=np.float64).reshape(3)
        with self._lock:
            self._ee_goal = goal.copy()
            self._t = float(time.time() if timestamp is None else timestamp)
            self._have_sample = True

    def _snapshot_payload(self) -> dict:
        with self._lock:
            return {
                "t": self._t,
                "ee_goal_local_cart": self._ee_goal.tolist(),
                "valid": bool(self._have_sample and np.all(np.isfinite(self._ee_goal))),
            }

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_thread, name="ee_goal_ws", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._loop = None

    def _run_thread(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._serve())
        except Exception as exc:
            print(f"[EeGoalWsServer] stopped: {exc}", flush=True)
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            loop.close()

    async def _serve(self) -> None:
        async def _handler(websocket: WebSocketServerProtocol) -> None:
            self._clients.add(websocket)
            try:
                await websocket.wait_closed()
            finally:
                self._clients.discard(websocket)

        async with websockets.serve(_handler, self.host, self.port):
            print(
                f"[EeGoalWsServer] listening on ws://{self.host}:{self.port} "
                f"at {self.rate_hz:.1f} Hz",
                flush=True,
            )
            dt = 1.0 / self.rate_hz
            while not self._stop.is_set():
                payload = self._snapshot_payload()
                if self._clients:
                    msg = json.dumps(payload)
                    dead = []
                    for ws in list(self._clients):
                        try:
                            await ws.send(msg)
                        except Exception:
                            dead.append(ws)
                    for ws in dead:
                        self._clients.discard(ws)
                await asyncio.sleep(dt)
