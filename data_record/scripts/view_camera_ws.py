#!/usr/bin/env python3
"""Live preview of frames from camera_ws_server (SHM via WS hello).

Terminal 1::

    python scripts/camera_ws_server.py --host 127.0.0.1 --port 8772 --rate 15 \\
      --camera-backend realsense

Terminal 2::

    python scripts/view_camera_ws.py --host 127.0.0.1 --port 8772

Press ``q`` or Esc to quit. On Wayland, Qt windows often flash-close — this
script forces the X11/xcb platform when needed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

# OpenCV Qt + Wayland often opens a window then immediately destroys it.
# Prefer XWayland/xcb before importing cv2.
if os.environ.get("WAYLAND_DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
    os.environ["QT_QPA_PLATFORM"] = "xcb"

DATA_RECORD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DATA_RECORD_ROOT not in sys.path:
    sys.path.insert(0, DATA_RECORD_ROOT)

import cv2
import numpy as np

from camera.rgb_ws_client import RgbWsClient


def _window_alive(win: str) -> bool:
    try:
        prop = cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE)
    except Exception:
        return False
    # -1 means the window is gone on most backends.
    return prop >= 0 and prop != 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview camera_ws_server SHM frames")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8772)
    parser.add_argument("--stale-s", type=float, default=1.0)
    parser.add_argument(
        "--exposure-hint",
        action="store_true",
        help="Print mean brightness tip (dark/bright) every second",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Also write the latest frame to this JPEG path (GUI-less fallback)",
    )
    args = parser.parse_args()

    client = RgbWsClient(
        host=args.host,
        port=args.port,
        connect_timeout_s=10.0,
        stale_s=args.stale_s,
    )
    client.start()

    win = "camera_ws"
    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 640, 480)
    except Exception as exc:
        print(f"[view_camera_ws] OpenCV window failed: {exc}", flush=True)
        print(
            "[view_camera_ws] try:  QT_QPA_PLATFORM=xcb python scripts/view_camera_ws.py ...\n"
            "  or use --save /tmp/cam.jpg and open that file",
            flush=True,
        )
        if not args.save:
            client.stop()
            return 1

    print(
        f"[view_camera_ws] reading {client.uri}  "
        f"QT_QPA_PLATFORM={os.environ.get('QT_QPA_PLATFORM', '')!r} — "
        f"press q/Esc to quit",
        flush=True,
    )

    last_hint = 0.0
    last_save = 0.0
    frames_shown = 0
    try:
        while True:
            rgb, t = client.read()
            if np.isfinite(t):
                bgr = np.ascontiguousarray(rgb[:, :, ::-1])
                label = f"{rgb.shape[1]}x{rgb.shape[0]}  t={t:.3f}"
                cv2.putText(
                    bgr,
                    label,
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                if args.exposure_hint and (time.time() - last_hint) >= 1.0:
                    mean = float(rgb.mean())
                    tip = "OK"
                    if mean < 40:
                        tip = "DARK → raise --exposure-us / --gain"
                    elif mean > 200:
                        tip = "BRIGHT → lower --exposure-us / --gain"
                    print(f"[view_camera_ws] mean={mean:.1f}  {tip}", flush=True)
                    last_hint = time.time()
                if args.save and (time.time() - last_save) >= 0.5:
                    cv2.imwrite(args.save, bgr)
                    last_save = time.time()
                show = bgr
            else:
                show = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    show,
                    "waiting for frames...",
                    (40, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 200, 255),
                    2,
                    cv2.LINE_AA,
                )

            try:
                cv2.imshow(win, show)
                frames_shown += 1
            except Exception as exc:
                print(f"[view_camera_ws] imshow failed: {exc}", flush=True)
                traceback.print_exc()
                if args.save:
                    time.sleep(0.05)
                    continue
                return 1

            if frames_shown == 1:
                print("[view_camera_ws] window up — leave this terminal focused; q/Esc quits", flush=True)

            if not _window_alive(win):
                print(
                    "[view_camera_ws] window closed by UI/backend "
                    "(common on Wayland without xcb). "
                    "Re-run with: QT_QPA_PLATFORM=xcb python scripts/view_camera_ws.py ...",
                    flush=True,
                )
                break

            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                print("[view_camera_ws] quit key", flush=True)
                break
    except KeyboardInterrupt:
        print("\n[view_camera_ws] interrupted", flush=True)
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        client.stop()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
