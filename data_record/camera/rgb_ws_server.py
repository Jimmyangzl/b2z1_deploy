"""Camera server: capture thread → shared memory; tiny WS control plane.

Local recorders should read frames from shared memory after the WS hello
(``RgbWsClient``). The WebSocket only advertises the SHM name and keeps a
heartbeat — it does **not** stream image bytes.

SHM layout (little-endian)::

    seq     : i64   frame sequence (written last; readers poll this)
    payload : RGB1 packed frame (see ``write_rgb_frame_raw_into``)
"""

from __future__ import annotations

import asyncio
import json
import struct
import threading
import time
from multiprocessing.shared_memory import SharedMemory
from typing import Optional, Set

from camera.rgb_ws_protocol import HEADER, release_shared_memory, write_rgb_frame_raw_into

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
except ImportError as exc:  # pragma: no cover
    raise ImportError("websockets is required for RgbWsServer") from exc

_MAX_W, _MAX_H = 1280, 720
_SEQ_STRUCT = struct.Struct("<q")
_FRAME_OFF = _SEQ_STRUCT.size
_SHM_SIZE = _FRAME_OFF + HEADER.size + _MAX_W * _MAX_H * 3


class RgbWsServer:
    """In-process capture into SHM + WS hello/heartbeat (no image bytes on the socket)."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8772,
        rate_hz: float = 30.0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        serial: Optional[str] = None,
        backend: str = "auto",
        jpeg_quality: int = 80,
        exposure_us: float = 1000.0,
        gain: float = 16.0,
    ):
        del jpeg_quality  # kept for CLI compat; images are not sent over WS
        self.host = str(host)
        self.port = int(port)
        self.rate_hz = max(1e-3, float(rate_hz))
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.serial = serial
        self.backend = str(backend)
        self.exposure_us = float(exposure_us)
        self.gain = float(gain)

        self._shm: Optional[SharedMemory] = None
        self._seq = 0
        self._seq_lock = threading.Lock()

        self._clients: Set[WebSocketServerProtocol] = set()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._fatal: Optional[str] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._camera_serial: str = str(serial or backend)

    @property
    def shm_name(self) -> str:
        return "" if self._shm is None else self._shm.name

    @property
    def camera_serial(self) -> str:
        return self._camera_serial

    def _hello_payload(self) -> str:
        return json.dumps(
            {
                "type": "camera_hello",
                "shm_name": self.shm_name,
                "shm_size": _SHM_SIZE,
                "frame_offset": _FRAME_OFF,
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
                "backend": self.backend,
                "transport": "shm",
            }
        )

    def _write_frame(self, rgb, t: float) -> None:
        assert self._shm is not None
        write_rgb_frame_raw_into(self._shm.buf, _FRAME_OFF, rgb, t)
        with self._seq_lock:
            self._seq += 1
            _SEQ_STRUCT.pack_into(self._shm.buf, 0, int(self._seq))

    def _capture_loop(self) -> None:
        from camera.realsense_rgb import RealSenseRGB

        camera = None
        try:
            print(
                f"[RgbWsServer:capture] opening camera backend={self.backend} "
                f"{self.width}x{self.height}@{self.fps} ...",
                flush=True,
            )
            camera = RealSenseRGB(
                width=self.width,
                height=self.height,
                fps=self.fps,
                serial=self.serial,
                backend=self.backend,
                exposure_us=self.exposure_us,
                gain=self.gain,
            )
            camera.start()
            self._camera_serial = str(camera.serial or self.backend)
            print("[RgbWsServer:capture] waiting for first frame ...", flush=True)
            rgb, t = camera.read(wait_ms=5000)
            self._write_frame(rgb, t)
            self._ready.set()
            print(
                f"[RgbWsServer:capture] ready serial={camera.serial} "
                f"{self.width}x{self.height}@{self.fps} shm={self.shm_name}",
                flush=True,
            )

            fail = 0
            last_ok = time.time()
            last_err_log = 0.0
            while not self._stop.is_set():
                try:
                    rgb, t = camera.read(wait_ms=2000)
                    self._write_frame(rgb, t)
                    fail = 0
                    last_ok = time.time()
                except Exception as exc:
                    fail += 1
                    now = time.time()
                    if fail == 1 or (now - last_err_log) >= 2.0:
                        print(
                            f"[RgbWsServer:capture] error ({fail}x, "
                            f"no frame {now - last_ok:.1f}s): {exc}",
                            flush=True,
                        )
                        last_err_log = now
                    if (now - last_ok) >= 8.0:
                        try:
                            print("[RgbWsServer:capture] reopening camera...", flush=True)
                            camera.reopen()
                            fail = 0
                            last_ok = time.time()
                        except Exception as re_exc:
                            print(
                                f"[RgbWsServer:capture] reopen failed: {re_exc}",
                                flush=True,
                            )
                            time.sleep(0.5)
                    else:
                        time.sleep(0.01)
        except BaseException as exc:
            self._fatal = str(exc)
            print(f"[RgbWsServer:capture] fatal: {exc}", flush=True)
            self._ready.set()
        finally:
            try:
                if camera is not None:
                    camera.stop()
            except Exception:
                pass

    def start(self) -> None:
        if self._capture_thread is not None:
            return
        self._stop.clear()
        self._ready.clear()
        self._fatal = None
        self._seq = 0
        self._shm = SharedMemory(create=True, size=_SHM_SIZE)
        _SEQ_STRUCT.pack_into(self._shm.buf, 0, 0)

        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="rgb_ws_capture", daemon=True
        )
        self._capture_thread.start()

        # Poll so a hung open isn't silent for 25s.
        deadline = time.time() + 25.0
        while time.time() < deadline:
            if self._ready.wait(timeout=2.0):
                break
            print(
                "[RgbWsServer] still waiting for first frame "
                "(is another process using the RealSense?)",
                flush=True,
            )
        if not self._ready.is_set():
            self.stop()
            raise RuntimeError(
                "RgbWsServer capture failed to produce frames "
                "(is the RealSense free / on USB3?). "
                "Try: pkill -f camera_ws_server; rm -f /dev/shm/psm_*"
            )
        if self._fatal:
            msg = self._fatal
            self.stop()
            raise RuntimeError(f"RgbWsServer capture failed: {msg}")

        self._ws_thread = threading.Thread(
            target=self._run_ws_thread, name="rgb_ws_server", daemon=True
        )
        self._ws_thread.start()
        print(
            f"[RgbWsServer] ws://{self.host}:{self.port} "
            f"hello/heartbeat only; frames via SHM name={self.shm_name} "
            f"({self.width}x{self.height} @ {self.fps} Hz)",
            flush=True,
        )

    def pump(self) -> bool:
        return (
            self._capture_thread is not None
            and self._capture_thread.is_alive()
            and not self._fatal
        )

    def stop(self) -> None:
        self._stop.set()
        loop = self._loop
        if loop is not None:
            def _halt() -> None:
                for task in asyncio.all_tasks(loop):
                    task.cancel()
                loop.stop()

            try:
                loop.call_soon_threadsafe(_halt)
            except Exception:
                pass
        if self._ws_thread is not None:
            self._ws_thread.join(timeout=3.0)
            self._ws_thread = None
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=5.0)
            self._capture_thread = None
        if self._shm is not None:
            release_shared_memory(self._shm, unlink=True)
            self._shm = None

    def _run_ws_thread(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._serve())
        except (asyncio.CancelledError, RuntimeError):
            pass
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
            try:
                loop.close()
            except Exception:
                pass
            self._loop = None

    async def _serve(self) -> None:
        async with websockets.serve(
            self._handler,
            self.host,
            self.port,
            max_size=64 * 1024,
            ping_interval=20,
            ping_timeout=20,
        ):
            try:
                await self._heartbeat_loop()
            except asyncio.CancelledError:
                return

    async def _handler(self, websocket: WebSocketServerProtocol) -> None:
        self._clients.add(websocket)
        print(
            f"[RgbWsServer] client +1 ({len(self._clients)}) {websocket.remote_address}",
            flush=True,
        )
        try:
            await websocket.send(self._hello_payload())
            await websocket.wait_closed()
        finally:
            self._clients.discard(websocket)
            print(
                f"[RgbWsServer] client -1 ({len(self._clients)})",
                flush=True,
            )

    async def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            if self._clients:
                with self._seq_lock:
                    seq = int(self._seq)
                msg = json.dumps(
                    {
                        "type": "camera_tick",
                        "seq": seq,
                        "shm_name": self.shm_name,
                    }
                )
                dead = []
                for ws in list(self._clients):
                    try:
                        await ws.send(msg)
                    except Exception:
                        dead.append(ws)
                for ws in dead:
                    self._clients.discard(ws)
            try:
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                raise
