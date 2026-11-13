"""Intel RealSense RGB capture at fixed (H, W, C) = (480, 640, 3)."""

from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np


class RealSenseRGB:
    """Grab RGB frames from a RealSense 435i (color stream only).

    Prefers ``pyrealsense2``. If that backend starts but never delivers frames
    (common on USB2 / kernel UVC contention), falls back to OpenCV V4L2.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        serial: Optional[str] = None,
        backend: str = "auto",
        v4l_index: int = -1,
        exposure_us: float = 1000.0,
        gain: float = 16.0,
    ):
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.serial = serial
        self.backend = str(backend)
        # -1 = auto-scan RealSense V4L nodes (do not pin video0).
        self.v4l_index = int(v4l_index)
        # Manual color exposure (µs) / gain — AE is disabled to avoid USB stalls.
        self.exposure_us = float(exposure_us)
        self.gain = float(gain)
        self._pipeline = None
        self._cap = None
        self._active_backend = None
        self._color_sensor = None
        self._last_rgb: Optional[np.ndarray] = None
        self._last_t: float = float("nan")

    def start(self) -> None:
        backend = self.backend.lower()
        if backend in ("auto", "realsense", "rs"):
            try:
                self._start_realsense()
                return
            except Exception as exc:
                if backend != "auto":
                    raise
                print(
                    f"[RealSenseRGB] pyrealsense2 failed ({exc}); "
                    "falling back to V4L2 auto-scan",
                    flush=True,
                )
                self._stop_realsense()
                self.v4l_index = -1
        self._start_v4l()

    def _start_realsense(self) -> None:
        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        # Prefer requested fps; fall back to other D435i color rates.
        # Note: some modes omit 15 Hz (e.g. 320x240 is 60/30/6 only).
        fps_candidates = []
        for f in (self.fps, 30, 15, 6, 60):
            if int(f) not in fps_candidates:
                fps_candidates.append(int(f))
        last_err: Optional[Exception] = None
        for fps in fps_candidates:
            try:
                config.disable_all_streams()
                if self.serial:
                    config.enable_device(self.serial)
                config.enable_stream(
                    rs.stream.color,
                    self.width,
                    self.height,
                    rs.format.bgr8,
                    int(fps),
                )
                profile = pipeline.start(config)
                device = profile.get_device()
                # Prefer a small queue so wait_for_frames isn't starved by a
                # stalled consumer, but allow 2 to absorb brief scheduling gaps.
                try:
                    for sens in device.sensors:
                        if sens.supports(rs.option.frames_queue_size):
                            sens.set_option(rs.option.frames_queue_size, 2)
                except Exception:
                    pass
                # Auto-exposure on D435i color can stall USB for multi-second
                # gaps; pin a moderate manual exposure after warmup below.
                self._color_sensor = None
                try:
                    self._color_sensor = device.first_color_sensor()
                except Exception:
                    self._color_sensor = None
                # Warm up with a few frames so the stream is actually alive.
                got = 0
                for _ in range(20):
                    try:
                        pipeline.wait_for_frames(timeout_ms=5000)
                        got += 1
                        if got >= 3:
                            break
                    except RuntimeError as exc:
                        last_err = exc
                        continue
                if got < 3:
                    pipeline.stop()
                    raise RuntimeError(
                        last_err
                        or f"RealSense color frames never stabilized (got {got}/3)"
                    )
                self.serial = self.serial or str(
                    device.get_info(rs.camera_info.serial_number)
                )
                self.fps = int(fps)
                self._pipeline = pipeline
                self._active_backend = "realsense"
                self._pin_color_exposure()
                print(
                    f"[RealSenseRGB] started backend=realsense serial={self.serial} "
                    f"{self.width}x{self.height}@{self.fps}",
                    flush=True,
                )
                return
            except Exception as exc:
                last_err = exc
                try:
                    pipeline.stop()
                except Exception:
                    pass
        raise RuntimeError(f"pyrealsense2 start failed: {last_err}")

    @staticmethod
    def _list_realsense_v4l_indices() -> list:
        """Return RealSense /dev/video* indices (metadata nodes often fail open)."""
        import glob
        import os

        idxs = []
        for path in sorted(glob.glob("/sys/class/video4linux/video*")):
            try:
                idx = int(os.path.basename(path).replace("video", ""))
                name = open(os.path.join(path, "name"), encoding="utf-8").read().strip()
            except Exception:
                continue
            if "Integrated" in name:
                continue
            if "RealSense" in name or "Intel(R) RealSense" in name:
                idxs.append(idx)
        # D435i color is usually a higher index than depth/IR; try those first.
        return sorted(idxs, reverse=True)

    def _try_open_v4l_index(self, idx: int):
        """Open index and require several stable BGR frames. Return cap or None."""
        import cv2

        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            return None, f"Failed to open V4L2 index {idx}"
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        ok_frames = 0
        last_shape = None
        for _ in range(6):
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if frame.ndim != 3 or frame.shape[2] < 3:
                # Depth/IR nodes often return 1-channel — skip.
                continue
            last_shape = frame.shape
            ok_frames += 1
            if ok_frames >= 3:
                return cap, None
        cap.release()
        return None, (
            f"V4L2 index {idx} unstable/non-RGB "
            f"(ok_frames={ok_frames}, last_shape={last_shape})"
        )

    def _start_v4l(self) -> None:
        candidates = []
        if self.v4l_index >= 0:
            candidates.append(self.v4l_index)
        for idx in self._list_realsense_v4l_indices():
            if idx not in candidates:
                candidates.append(idx)

        last_err: Optional[Exception] = None
        for idx in candidates:
            cap, err = self._try_open_v4l_index(idx)
            if cap is None:
                last_err = RuntimeError(err or f"index {idx} failed")
                continue
            self._cap = cap
            self._active_backend = "v4l"
            self.v4l_index = idx
            self.serial = f"v4l:{idx}"
            print(
                f"[RealSenseRGB] started backend=v4l index={idx} "
                f"{self.width}x{self.height}@{self.fps}",
                flush=True,
            )
            return
        raise RuntimeError(
            f"No RealSense V4L2 device produced stable RGB frames "
            f"(tried {candidates}): {last_err}"
        )

    def _pin_color_exposure(self) -> None:
        """Disable color AE after stream is live (avoids multi-second AE stalls)."""
        import pyrealsense2 as rs

        sens = getattr(self, "_color_sensor", None)
        if sens is None:
            return
        try:
            if sens.supports(rs.option.enable_auto_exposure):
                sens.set_option(rs.option.enable_auto_exposure, 0)
            if sens.supports(rs.option.exposure):
                sens.set_option(rs.option.exposure, float(self.exposure_us))
            if sens.supports(rs.option.gain):
                sens.set_option(rs.option.gain, float(self.gain))
            print(
                f"[RealSenseRGB] color AE off "
                f"exposure={self.exposure_us:.0f}µs gain={self.gain:.0f}",
                flush=True,
            )
        except Exception as exc:
            print(f"[RealSenseRGB] could not pin exposure: {exc}", flush=True)

    def _stop_realsense(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
        self._color_sensor = None

    def stop(self) -> None:
        self._stop_realsense()
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._active_backend = None

    def _resize_bgr(self, bgr: np.ndarray) -> np.ndarray:
        if bgr.shape[0] == self.height and bgr.shape[1] == self.width:
            return bgr
        import cv2

        return cv2.resize(bgr, (self.width, self.height), interpolation=cv2.INTER_AREA)

    def reopen(self, backend: Optional[str] = None) -> None:
        """Stop and start again (optionally forcing a backend)."""
        if backend is not None:
            self.backend = str(backend)
        # Drop a pinned dead V4L index so the next start re-scans.
        if self.backend.lower() in ("auto", "v4l"):
            self.v4l_index = -1
        self.stop()
        self.start()

    def _decode_color_frames(self, frames) -> Tuple[np.ndarray, float]:
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("No RealSense color frame")
        bgr = np.asanyarray(color.get_data())
        bgr = self._resize_bgr(bgr)
        rgb = bgr[:, :, ::-1].copy()
        try:
            t = float(color.get_timestamp()) * 1e-3
        except Exception:
            t = time.time()
        out = rgb.astype(np.uint8, copy=False)
        self._last_rgb = out
        self._last_t = float(t)
        return out, float(t)

    def read(self, wait_ms: int = 5000) -> Tuple[np.ndarray, float]:
        """Return RGB uint8 (H, W, 3) and capture timestamp (seconds).

        Uses a single long ``try_wait_for_frames`` instead of stacked 1s waits,
        which were cascading into false "USB dead" reopen loops on brief stalls.
        """
        if self._active_backend == "realsense":
            assert self._pipeline is not None
            # try_wait returns (ok, frames); prefer it so a timeout is soft.
            try_wait = getattr(self._pipeline, "try_wait_for_frames", None)
            if try_wait is not None:
                ok, frames = try_wait(wait_ms)
                if not ok or frames is None:
                    raise RuntimeError(f"Frame didn't arrive within {int(wait_ms)}")
                return self._decode_color_frames(frames)
            frames = self._pipeline.wait_for_frames(timeout_ms=int(wait_ms))
            return self._decode_color_frames(frames)

        if self._active_backend == "v4l":
            last_err = "No V4L2 color frame"
            for _ in range(4):
                ok, bgr = self._cap.read()
                if ok and bgr is not None and bgr.ndim == 3 and bgr.shape[2] >= 3:
                    bgr = self._resize_bgr(bgr)
                    rgb = bgr[:, :, ::-1].copy()
                    out = rgb.astype(np.uint8, copy=False)
                    self._last_rgb = out
                    self._last_t = time.time()
                    return out, self._last_t
                time.sleep(0.01)
            raise RuntimeError(last_err)

        raise RuntimeError("RealSenseRGB not started")
