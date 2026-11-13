"""VR WebSocket client in a child process (own GIL).

Control process only copies the latest packed pose from shared memory —
json.loads / NumPy parse / websockets asyncio never touch the 50 Hz GIL.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# Packed float64 layout in shared Array (parent reads, child writes).
_IDX_CONNECTED = 0
_IDX_HOLDING = 1
_IDX_ROBOT_READY = 2
_IDX_HAS_POS = 3
_IDX_HAS_ORN = 4
_IDX_HAS_TDES = 5
_IDX_HAS_TINIT = 6
_IDX_HAS_QINIT = 7
_IDX_TIMESTAMP = 8
_IDX_POS = 9  # 3
_IDX_ORN = 12  # 4
_IDX_TDES = 16  # 16
_IDX_TINIT = 32  # 16
_IDX_QINIT = 48  # 6
_POSE_FLOATS = 54


def _pack_pose(dst: np.ndarray, pose: dict) -> None:
    dst[:] = 0.0
    dst[_IDX_CONNECTED] = 1.0 if pose.get("connected") else 0.0
    dst[_IDX_HOLDING] = 1.0 if pose.get("holding") else 0.0
    dst[_IDX_ROBOT_READY] = 1.0 if pose.get("robot_ready") else 0.0
    ts = pose.get("timestamp")
    dst[_IDX_TIMESTAMP] = float(ts) if ts is not None else 0.0

    pos = pose.get("position")
    if pos is not None:
        dst[_IDX_HAS_POS] = 1.0
        dst[_IDX_POS : _IDX_POS + 3] = np.asarray(pos, dtype=np.float64).reshape(3)

    orn = pose.get("orientation")
    if orn is not None:
        dst[_IDX_HAS_ORN] = 1.0
        dst[_IDX_ORN : _IDX_ORN + 4] = np.asarray(orn, dtype=np.float64).reshape(-1)[:4]

    Tdes = pose.get("Tdes")
    if Tdes is not None:
        dst[_IDX_HAS_TDES] = 1.0
        dst[_IDX_TDES : _IDX_TDES + 16] = np.asarray(Tdes, dtype=np.float64).reshape(16)

    T_init = pose.get("T_init")
    if T_init is not None:
        dst[_IDX_HAS_TINIT] = 1.0
        dst[_IDX_TINIT : _IDX_TINIT + 16] = np.asarray(T_init, dtype=np.float64).reshape(16)

    q_init = pose.get("q_init")
    if q_init is not None:
        dst[_IDX_HAS_QINIT] = 1.0
        q = np.asarray(q_init, dtype=np.float64).reshape(-1)
        n = min(q.size, 6)
        dst[_IDX_QINIT : _IDX_QINIT + n] = q[:n]


def _unpack_pose(src: np.ndarray) -> dict:
    pos = None
    if src[_IDX_HAS_POS] > 0.5:
        pos = src[_IDX_POS : _IDX_POS + 3].copy()
    orn = None
    if src[_IDX_HAS_ORN] > 0.5:
        orn = src[_IDX_ORN : _IDX_ORN + 4].copy()
    Tdes = None
    if src[_IDX_HAS_TDES] > 0.5:
        Tdes = src[_IDX_TDES : _IDX_TDES + 16].reshape(4, 4).copy()
    T_init = None
    if src[_IDX_HAS_TINIT] > 0.5:
        T_init = src[_IDX_TINIT : _IDX_TINIT + 16].reshape(4, 4).copy()
    q_init = None
    if src[_IDX_HAS_QINIT] > 0.5:
        q_init = src[_IDX_QINIT : _IDX_QINIT + 6].copy()
    ts = float(src[_IDX_TIMESTAMP])
    return {
        "connected": bool(src[_IDX_CONNECTED] > 0.5),
        "holding": bool(src[_IDX_HOLDING] > 0.5),
        "robot_ready": bool(src[_IDX_ROBOT_READY] > 0.5),
        "timestamp": None if ts <= 0.0 else ts,
        "position": pos,
        "orientation": orn,
        "Tdes": Tdes,
        "T_init": T_init,
        "q_init": q_init,
    }


def _empty_pose() -> dict:
    """Safe hold-last fallback when SHM lock is busy and no cache yet."""
    return {
        "connected": False,
        "holding": False,
        "robot_ready": False,
        "timestamp": None,
        "position": None,
        "orientation": None,
        "Tdes": None,
        "T_init": None,
        "q_init": None,
    }


def _vr_process_entry(
    host: str,
    port: int,
    config_path: Optional[str],
    pose_arr,
    seq,
    stop_evt,
    ready_evt,
) -> None:
    """Child: own GIL — websocket + JSON parse cannot stall control."""
    import os
    import sys
    import time as _time

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    from paths import setup_paths

    setup_paths()

    import numpy as _np
    from teleop.vr_websocket_client import VrWebSocketClient

    client = VrWebSocketClient(host=host, port=port, config_path=config_path)
    client.start()
    ready_evt.set()
    print(
        f"[process_vr] child ready → ws://{client.host}:{client.port}",
        flush=True,
    )

    scratch = _np.zeros(_POSE_FLOATS, dtype=_np.float64)
    last_raw_seq = -1
    while not stop_evt.is_set():
        # Drive parse in this process (not on the control GIL).
        pose = client.get_latest_pose()
        raw_seq = int(getattr(client, "_raw_seq", 0))
        if raw_seq != last_raw_seq:
            last_raw_seq = raw_seq
            _pack_pose(scratch, pose)
            with pose_arr.get_lock():
                _np.frombuffer(pose_arr.get_obj(), dtype=_np.float64)[:] = scratch
            with seq.get_lock():
                seq.value += 1
        _time.sleep(0.001)

    client.stop()


class ProcessVrWorker:
    """Drop-in stand-in for ``VrWebSocketClient`` backed by a child process."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        config_path: Optional[str] = None,
    ):
        import multiprocessing as mp

        self.host = host or "127.0.0.1"
        self.port = int(port if port is not None else 8765)
        self._config_path = config_path
        self._ctx = mp.get_context("spawn")
        self._pose = self._ctx.Array("d", _POSE_FLOATS, lock=True)
        self._seq = self._ctx.Value("i", 0)
        self._stop = self._ctx.Event()
        self._ready = self._ctx.Event()
        self._last_seq = -1
        self._pose_cache: Optional[dict] = None
        self._started = False
        self._proc = self._ctx.Process(
            target=_vr_process_entry,
            args=(
                str(self.host),
                int(self.port),
                config_path,
                self._pose,
                self._seq,
                self._stop,
                self._ready,
            ),
            name="process_vr",
            daemon=True,
        )

    @property
    def uri(self) -> str:
        return f"ws://{self.host}:{self.port}"

    def start(self) -> None:
        if self._started:
            return
        self._stop.clear()
        self._ready.clear()
        self._proc.start()
        self._started = True

    def wait_ready(self, timeout: float = 30.0) -> None:
        if not self._ready.wait(timeout):
            raise TimeoutError(
                f"ProcessVrWorker child not ready within {timeout:.0f}s"
            )

    def stop(self) -> None:
        if not self._started:
            return
        self._stop.set()
        self._proc.join(timeout=2.0)
        if self._proc.is_alive():
            self._proc.terminate()
        self._started = False

    def get_latest_pose(self) -> dict:
        """Cheap shared-memory read — no JSON on the control process.

        Non-blocking: if a lock is busy, return the last cached pose (or an
        empty disconnected pose) so the 50 Hz arm path never waits on the VR child.
        """
        seq_lock = self._seq.get_lock()
        if not seq_lock.acquire(False):
            return self._pose_cache if self._pose_cache is not None else _empty_pose()
        try:
            cur = int(self._seq.value)
        finally:
            seq_lock.release()

        if self._pose_cache is not None and cur == self._last_seq:
            return self._pose_cache

        pose_lock = self._pose.get_lock()
        if not pose_lock.acquire(False):
            return self._pose_cache if self._pose_cache is not None else _empty_pose()
        try:
            buf = np.frombuffer(self._pose.get_obj(), dtype=np.float64).copy()
        finally:
            pose_lock.release()

        pose = _unpack_pose(buf)
        self._pose_cache = pose
        self._last_seq = cur
        return pose
