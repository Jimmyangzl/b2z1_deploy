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
        v4l_index: int = 0,
    ):
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.serial = serial
        self.backend = str(backend)
        self.v4l_index = int(v4l_index)
        self._pipeline = None
        self._cap = None
        self._active_backend = None

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
                    f"falling back to V4L2 index={self.v4l_index}",
                    flush=True,
                )
                self._stop_realsense()
        self._start_v4l()

    def _start_realsense(self) -> None:
        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        # Prefer requested fps; fall back to 15 if USB2 cannot sustain 30.
        fps_candidates = [self.fps]
        if self.fps != 15:
            fps_candidates.append(15)
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
                # Warm up; require at least one frame (fail fast for USB/UVC issues).
                got = False
                for _ in range(3):
                    try:
                        pipeline.wait_for_frames(timeout_ms=500)
                        got = True
                        break
                    except RuntimeError as exc:
                        last_err = exc
                        continue
                if not got:
                    pipeline.stop()
                    raise RuntimeError(
                        last_err or "RealSense color frames never arrived"
                    )
                device = profile.get_device()
                self.serial = self.serial or str(
                    device.get_info(rs.camera_info.serial_number)
                )
                self.fps = int(fps)
                self._pipeline = pipeline
                self._active_backend = "realsense"
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

    def _start_v4l(self) -> None:
        import cv2
        import glob
        import os

        candidates = []
        if self.v4l_index >= 0:
            candidates.append(self.v4l_index)
        # Prefer Intel RealSense nodes; never default to Integrated Camera.
        for path in sorted(glob.glob("/sys/class/video4linux/video*")):
            try:
                idx = int(os.path.basename(path).replace("video", ""))
                name = open(os.path.join(path, "name"), encoding="utf-8").read().strip()
            except Exception:
                continue
            if "Integrated" in name:
                continue
            if "RealSense" in name or "Intel(R) RealSense" in name:
                if idx not in candidates:
                    candidates.append(idx)

        last_err: Optional[Exception] = None
        for idx in candidates:
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if not cap.isOpened():
                last_err = RuntimeError(f"Failed to open V4L2 index {idx}")
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                last_err = RuntimeError(f"V4L2 index {idx} opened but no frames")
                continue
            self._cap = cap
            self._active_backend = "v4l"
            self.v4l_index = idx
            if not self.serial:
                self.serial = f"v4l:{idx}"
            print(
                f"[RealSenseRGB] started backend=v4l index={idx} "
                f"{self.width}x{self.height}@{self.fps}",
                flush=True,
            )
            return
        raise RuntimeError(
            f"No RealSense V4L2 device produced frames "
            f"(tried {candidates}): {last_err}"
        )

    def _stop_realsense(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None

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

    def read(self) -> Tuple[np.ndarray, float]:
        """Return RGB uint8 (H, W, 3) and capture timestamp (seconds)."""
        if self._active_backend == "realsense":
            frames = self._pipeline.wait_for_frames(timeout_ms=2000)
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
            return rgb.astype(np.uint8, copy=False), t

        if self._active_backend == "v4l":
            ok, bgr = self._cap.read()
            if not ok or bgr is None:
                raise RuntimeError("No V4L2 color frame")
            bgr = self._resize_bgr(bgr)
            rgb = bgr[:, :, ::-1].copy()
            return rgb.astype(np.uint8, copy=False), time.time()

        raise RuntimeError("RealSenseRGB not started")
