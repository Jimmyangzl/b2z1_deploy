"""Client for the camera server: WS hello → read frames from shared memory."""

from __future__ import annotations

import asyncio
import json
import struct
import threading
import time
from typing import Optional, Tuple

import numpy as np

from camera.rgb_ws_protocol import HEADER, attach_shared_memory, release_shared_memory

try:
    import websockets
except ImportError as exc:  # pragma: no cover
    raise ImportError("websockets is required for RgbWsClient") from exc

_SEQ_STRUCT = struct.Struct("<q")
_DEFAULT_FRAME_OFF = _SEQ_STRUCT.size


class RgbWsClient:
    """Attach to camera-server SHM after a WS hello (no image bytes on the socket)."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8772,
        width: int = 640,
        height: int = 480,
        connect_timeout_s: float = 8.0,
        stale_s: float = 1.0,
    ):
        self.host = str(host)
        self.port = int(port)
        self.width = int(width)
        self.height = int(height)
        self.connect_timeout_s = float(connect_timeout_s)
        self.stale_s = float(stale_s)
        self.serial = f"ws://{self.host}:{self.port}"

        self._lock = threading.Lock()
        self._rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._t = 0.0
        self._have = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._shm = None
        self._shm_name: Optional[str] = None
        self._frame_off = _DEFAULT_FRAME_OFF
        self._last_seq = -1

    @property
    def uri(self) -> str:
        return f"ws://{self.host}:{self.port}"

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_loop, name="rgb_ws_client", daemon=True
        )
        self._thread.start()
        deadline = time.time() + self.connect_timeout_s
        while time.time() < deadline and not self._stop.is_set():
            with self._lock:
                if self._have:
                    print(
                        f"[RgbWsClient] SHM frames from {self.uri} "
                        f"shm={self._shm_name} {self.width}x{self.height}",
                        flush=True,
                    )
                    return
            time.sleep(0.05)
        print(
            f"[RgbWsClient] no frame yet from {self.uri} "
            "(will keep retrying in background)",
            flush=True,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._close_shm()

    def read(self) -> Tuple[np.ndarray, float]:
        with self._lock:
            if not self._have:
                return (
                    np.zeros((self.height, self.width, 3), dtype=np.uint8),
                    float("nan"),
                )
            age = time.time() - self._t if self._t > 0 else float("inf")
            if age > self.stale_s:
                return (
                    np.zeros((self.height, self.width, 3), dtype=np.uint8),
                    float("nan"),
                )
            return self._rgb.copy(), float(self._t)

    def _close_shm(self) -> None:
        # Attach-only: close + unregister; never unlink (server owns the segment).
        release_shared_memory(self._shm, unlink=False)
        self._shm = None

    def _set_frame(self, rgb: np.ndarray, t: float) -> None:
        h, w, _ = rgb.shape
        with self._lock:
            if h != self.height or w != self.width:
                self.height = int(h)
                self.width = int(w)
            self._rgb = np.asarray(rgb, dtype=np.uint8).copy()
            self._t = float(t)
            self._have = True

    def _poll_shm_once(self) -> bool:
        if self._shm is None:
            return False
        buf = self._shm.buf
        seq1 = _SEQ_STRUCT.unpack_from(buf, 0)[0]
        if seq1 <= 0 or seq1 == self._last_seq:
            return False
        off = self._frame_off
        magic, width, height, ts = HEADER.unpack_from(buf, off)
        if magic != b"RGB1":
            return False
        h, w = int(height), int(width)
        nbytes = h * w * 3
        # Copy pixels out of SHM before verifying seq (avoid torn frame).
        src = np.ndarray(
            nbytes, dtype=np.uint8, buffer=buf, offset=off + HEADER.size
        )
        rgb = src.reshape((h, w, 3)).copy()
        seq2 = _SEQ_STRUCT.unpack_from(buf, 0)[0]
        if seq2 != seq1:
            return False
        self._last_seq = int(seq1)
        self._set_frame(rgb, float(ts))
        return True

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                asyncio.run(self._session())
            except Exception as exc:
                if self._stop.is_set():
                    break
                print(
                    f"[RgbWsClient] disconnected: {exc}. Retrying in 1s...",
                    flush=True,
                )
                self._close_shm()
                self._last_seq = -1
                time.sleep(1.0)

    async def _session(self) -> None:
        async with websockets.connect(
            self.uri,
            max_size=64 * 1024,
            ping_interval=20,
            ping_timeout=20,
        ) as websocket:
            print(f"[RgbWsClient] connected to {self.uri}", flush=True)
            raw = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            if isinstance(raw, bytes):
                raise RuntimeError("expected JSON hello from camera server")
            hello = json.loads(raw)
            if hello.get("type") != "camera_hello" or not hello.get("shm_name"):
                raise RuntimeError(f"bad camera hello: {hello}")
            self._shm_name = str(hello["shm_name"])
            self._frame_off = int(hello.get("frame_offset", _DEFAULT_FRAME_OFF))
            self._close_shm()
            # Attach without owning: unregister so our exit cannot unlink server SHM.
            self._shm = attach_shared_memory(self._shm_name)
            self.serial = f"shm:{self._shm_name}"
            if hello.get("width"):
                self.width = int(hello["width"])
            if hello.get("height"):
                self.height = int(hello["height"])

            # Poll SHM for frames; WS only for keepalive ticks.
            while not self._stop.is_set():
                self._poll_shm_once()
                try:
                    await asyncio.wait_for(websocket.recv(), timeout=0.02)
                except asyncio.TimeoutError:
                    pass
