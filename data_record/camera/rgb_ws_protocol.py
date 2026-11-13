"""Framing for localhost RGB WebSocket camera stream.

Wire format (little-endian) — JPEG to keep USB capture alive under load::

    magic   : 4s   b\"JPG1\"
    width   : u32  decoded image width
    height  : u32  decoded image height
    t       : f64  capture / wall time (seconds)
    jpeg    : variable-length JPEG bytes (RGB)

Shared-memory capture path still uses raw RGB via ``pack_rgb_frame_raw``.
"""

from __future__ import annotations

import struct
from typing import Any, Tuple

import numpy as np

MAGIC_JPEG = b"JPG1"
MAGIC_RAW = b"RGB1"
HEADER = struct.Struct("<4sIId")  # magic, width, height, t


def _resource_tracker_unregister(shm: Any) -> None:
    """Unregister SHM from CPython's resource_tracker (best-effort).

    Register uses ``shm._name`` (leading ``/`` on Linux). Unregister that
    exact key once — trying both ``name`` and ``_name`` causes KeyError spam
    in the resource_tracker process.
    """
    try:
        from multiprocessing.resource_tracker import unregister
    except Exception:
        return
    key = getattr(shm, "_name", None) or getattr(shm, "name", None)
    if not key:
        return
    try:
        unregister(key, "shared_memory")
    except Exception:
        pass


def attach_shared_memory(name: str) -> Any:
    """Attach to an existing SHM without taking ownership.

    Immediately unregister so a client/viewer exit cannot unlink the server's
    segment (CPython 3.8 resource_tracker bug).
    """
    from multiprocessing.shared_memory import SharedMemory

    shm = SharedMemory(name=name)
    _resource_tracker_unregister(shm)
    return shm


def release_shared_memory(shm: Any, *, unlink: bool = False) -> None:
    """Close a SharedMemory; only the creator should pass ``unlink=True``.

    ``SharedMemory.unlink()`` already unregisters, so we must not unregister
    again after unlink.
    """
    if shm is None:
        return
    try:
        shm.close()
    except Exception:
        pass
    if unlink:
        try:
            shm.unlink()  # unregisters internally
        except Exception:
            pass
        return
    # Attach-only: already unregistered in attach_shared_memory; nothing else.


def pack_rgb_frame_raw(rgb: np.ndarray, timestamp: float) -> bytes:
    """Pack ``(H, W, 3)`` uint8 RGB (allocating). Prefer ``write_rgb_frame_raw_into`` for SHM."""
    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) RGB, got {arr.shape}")
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    h, w, _ = arr.shape
    header = HEADER.pack(MAGIC_RAW, int(w), int(h), float(timestamp))
    return header + arr.tobytes(order="C")


def write_rgb_frame_raw_into(buf, offset: int, rgb: np.ndarray, timestamp: float) -> int:
    """Write RGB1 frame into ``buf`` at ``offset`` without ``tobytes()`` allocations.

    Large per-frame ``tobytes()`` copies were observed to stall librealsense USB
    delivery on this host; numpy writes into the SHM memoryview avoid that.
    Returns bytes written.
    """
    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) RGB, got {arr.shape}")
    h, w, _ = arr.shape
    HEADER.pack_into(buf, offset, MAGIC_RAW, int(w), int(h), float(timestamp))
    nbytes = int(h) * int(w) * 3
    dest = np.ndarray(nbytes, dtype=np.uint8, buffer=buf, offset=offset + HEADER.size)
    if arr.flags["C_CONTIGUOUS"]:
        dest[:] = arr.reshape(-1)
    else:
        dest[:] = np.ascontiguousarray(arr).reshape(-1)
    return HEADER.size + nbytes


def pack_rgb_frame(rgb: np.ndarray, timestamp: float, jpeg_quality: int = 85) -> bytes:
    """Pack ``(H, W, 3)`` uint8 RGB as JPEG for WebSocket."""
    import cv2

    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) RGB, got {arr.shape}")
    bgr = arr[:, :, ::-1]
    ok, enc = cv2.imencode(
        ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    )
    if not ok:
        raise RuntimeError("cv2.imencode JPEG failed")
    h, w, _ = arr.shape
    header = HEADER.pack(MAGIC_JPEG, int(w), int(h), float(timestamp))
    return header + enc.tobytes()


def unpack_rgb_frame(payload: bytes) -> Tuple[np.ndarray, float]:
    """Return ``(rgb HxWx3 uint8, timestamp_s)`` from a WS/SHM payload."""
    if len(payload) < HEADER.size:
        raise ValueError(f"RGB WS frame too short: {len(payload)} bytes")
    magic, width, height, t = HEADER.unpack_from(payload, 0)
    body = payload[HEADER.size :]

    if magic == MAGIC_JPEG:
        import cv2

        arr = np.frombuffer(body, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("JPEG decode failed")
        rgb = bgr[:, :, ::-1].copy()
        return rgb, float(t)

    if magic == MAGIC_RAW:
        need = int(width) * int(height) * 3
        if len(body) < need:
            raise ValueError(f"RAW frame truncated: {len(body)} < {need}")
        rgb = np.frombuffer(body, dtype=np.uint8, count=need).reshape((height, width, 3)).copy()
        return rgb, float(t)

    raise ValueError(f"bad frame magic {magic!r}, expected JPG1 or RGB1")
