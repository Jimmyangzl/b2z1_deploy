"""Bidirectional ACT WebSocket (ACT ↔ deploy).

ACT → deploy: ``ee_goal_local_cart`` / ``ee_goal_quat_base`` / ``gripper_cmd``
Deploy → ACT: ``type=ee_state`` measured arm-base EE + base_lin_vel (so ACT
does not need the separate EE-goal WS on :8770).
"""

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
    raise ImportError("websockets is required for ActGoalWsServer") from exc


class ActGoalWsServer:
    """Background WS server: ACT goals in; measured EE state out."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8771):
        self.host = str(host)
        self.port = int(port)
        self._lock = threading.Lock()
        self._ee_goal = np.full(3, np.nan, dtype=np.float64)
        self._ee_quat = np.full(4, np.nan, dtype=np.float64)
        self._gripper = float("nan")
        self._t = 0.0
        self._recv_t = 0.0
        self._seq = 0
        self._have = False
        # Measured EE for ACT (deploy → client).
        self._state_ee_pos = np.full(3, np.nan, dtype=np.float64)
        self._state_ee_quat = np.full(4, np.nan, dtype=np.float64)
        self._state_base_lin = np.full(3, np.nan, dtype=np.float64)
        self._state_grip = float("nan")
        self._state_t = 0.0
        self._state_have = False
        self._state_seq = 0
        self._last_broadcast_seq = -1
        self._clients: Set[WebSocketServerProtocol] = set()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop = threading.Event()

    @property
    def has_goal(self) -> bool:
        with self._lock:
            return bool(
                self._have
                and np.all(np.isfinite(self._ee_goal))
                and np.all(np.isfinite(self._ee_quat))
            )

    def get_latest(self) -> dict:
        """Latest ACT goal snapshot for the control loop."""
        with self._lock:
            return {
                "valid": bool(
                    self._have
                    and np.all(np.isfinite(self._ee_goal))
                    and np.all(np.isfinite(self._ee_quat))
                ),
                "ee_goal_local_cart": self._ee_goal.copy(),
                "ee_goal_quat_base": self._ee_quat.copy(),
                "gripper_cmd": float(self._gripper),
                "t": float(self._t),
                "recv_t": float(self._recv_t),
                "seq": int(self._seq),
                "num_clients": len(self._clients),
            }

    def publish_ee_state(
        self,
        ee_pos,
        ee_quat,
        base_lin_vel=None,
        gripper_cmd: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> None:
        """Queue measured arm-base EE for broadcast to ACT clients."""
        pos = np.asarray(ee_pos, dtype=np.float64).reshape(3)
        quat = np.asarray(ee_quat, dtype=np.float64).reshape(4)
        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(quat))):
            return
        n = float(np.linalg.norm(quat))
        if n > 1e-12:
            quat = quat / n
        if base_lin_vel is None:
            vel = np.zeros(3, dtype=np.float64)
        else:
            vel = np.asarray(base_lin_vel, dtype=np.float64).reshape(3)
            if not np.all(np.isfinite(vel)):
                vel = np.zeros(3, dtype=np.float64)
        with self._lock:
            self._state_ee_pos = pos.copy()
            self._state_ee_quat = quat.copy()
            self._state_base_lin = vel.copy()
            self._state_grip = float(gripper_cmd)
            self._state_t = float(time.time() if timestamp is None else timestamp)
            self._state_have = True
            self._state_seq += 1

    def _apply_message(self, data: dict) -> None:
        if data.get("type") == "ee_state":
            return  # server→client only
        cart = data.get("ee_goal_local_cart")
        quat = data.get("ee_goal_quat_base")
        if cart is None or quat is None:
            return
        cart_a = np.asarray(cart, dtype=np.float64).reshape(3)
        quat_a = np.asarray(quat, dtype=np.float64).reshape(4)
        n = float(np.linalg.norm(quat_a))
        quat_a = (
            quat_a / n
            if n > 1e-12
            else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        )
        grip = data.get("gripper_cmd", 0.0)
        try:
            grip_f = float(grip)
        except (TypeError, ValueError):
            grip_f = float("nan")
        t = data.get("t", time.time())
        with self._lock:
            self._ee_goal = cart_a
            self._ee_quat = quat_a
            self._gripper = grip_f
            self._t = float(t)
            self._recv_t = time.time()
            self._seq += 1
            self._have = True

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_thread, name="act_goal_ws", daemon=True
        )
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
            print(f"[ActGoalWsServer] stopped: {exc}", flush=True)
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            except Exception:
                pass
            loop.close()

    def _state_payload(self) -> Optional[str]:
        with self._lock:
            if not self._state_have:
                return None
            if self._state_seq == self._last_broadcast_seq and self._clients:
                # Still re-broadcast periodically so late subscribers get state.
                pass
            self._last_broadcast_seq = self._state_seq
            return json.dumps(
                {
                    "type": "ee_state",
                    "t": float(self._state_t),
                    "ee_pos": self._state_ee_pos.tolist(),
                    "ee_quat": self._state_ee_quat.tolist(),
                    "base_lin_vel": self._state_base_lin.tolist(),
                    "gripper_cmd": float(self._state_grip),
                    "valid": True,
                }
            )

    async def _broadcast_ee_state(self) -> None:
        if not self._clients:
            return
        msg = self._state_payload()
        if msg is None:
            return
        dead = []
        for ws in list(self._clients):
            try:
                await ws.send(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    async def _serve(self) -> None:
        async def _handler(websocket: WebSocketServerProtocol) -> None:
            self._clients.add(websocket)
            print(
                f"[ActGoalWsServer] ACT client connected "
                f"({len(self._clients)} total)",
                flush=True,
            )
            try:
                async for raw in websocket:
                    try:
                        data = json.loads(raw)
                    except Exception:
                        continue
                    if isinstance(data, dict):
                        self._apply_message(data)
            finally:
                self._clients.discard(websocket)
                print(
                    f"[ActGoalWsServer] ACT client disconnected "
                    f"({len(self._clients)} left)",
                    flush=True,
                )

        print(
            f"[ActGoalWsServer] listening on ws://{self.host}:{self.port} "
            "(ACT goals in; measured EE state out)",
            flush=True,
        )
        async with websockets.serve(_handler, self.host, self.port):
            while not self._stop.is_set():
                await self._broadcast_ee_state()
                await asyncio.sleep(0.02)
