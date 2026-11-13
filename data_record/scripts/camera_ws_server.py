#!/usr/bin/env python3
"""Run the localhost RealSense camera server (SHM frames + WS hello).

Example::

    cd data_record
    python scripts/camera_ws_server.py --host 127.0.0.1 --port 8772 --rate 30

Then in another terminal::

    python scripts/record_session.py --task pickplace \\
      --camera-ws-host 127.0.0.1 --camera-ws-port 8772 ...
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time

DATA_RECORD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DATA_RECORD_ROOT not in sys.path:
    sys.path.insert(0, DATA_RECORD_ROOT)

import yaml

from camera.rgb_ws_server import RgbWsServer

DEFAULT_CONFIG = os.path.join(DATA_RECORD_ROOT, "config", "record.yaml")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RealSense RGB via shared memory; WebSocket hello/heartbeat only"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to record.yaml")
    parser.add_argument("--host", default=None, help="Bind host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="Bind port (default 8772)")
    parser.add_argument("--rate", type=float, default=None, help="Broadcast rate Hz")
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Camera capture fps (default: same as --rate, else yaml camera.fps)",
    )
    parser.add_argument(
        "--camera-backend",
        choices=("auto", "realsense", "v4l"),
        default="auto",
        help="Capture backend (same as record_session)",
    )
    parser.add_argument("--serial", default=None, help="Optional RealSense serial")
    parser.add_argument(
        "--exposure-us",
        type=float,
        default=None,
        help="Manual color exposure in microseconds (default from yaml / 1000)",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=None,
        help="Manual color gain (default from yaml / 16)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Capture width (default from yaml / 640). Smaller: 424 or 320.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Capture height (default from yaml / 480). Use 240 with 424/320. "
        "Note: 320x240 has no 15 Hz mode (use --rate 30).",
    )
    args = parser.parse_args()

    cfg: dict = {}
    if os.path.isfile(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    cam_cfg = cfg.get("camera") or {}
    ws_cfg = cfg.get("camera_ws") or {}

    host = args.host or ws_cfg.get("host") or "127.0.0.1"
    port = int(args.port if args.port is not None else ws_cfg.get("port", 8772))
    rate = float(args.rate if args.rate is not None else ws_cfg.get("rate_hz", cam_cfg.get("fps", 30)))
    width = int(args.width if args.width is not None else cam_cfg.get("width", 640))
    height = int(args.height if args.height is not None else cam_cfg.get("height", 480))
    # Capture fps must match the librealsense stream — --rate alone used to leave yaml at 30.
    if args.fps is not None:
        fps = int(args.fps)
    else:
        fps = int(round(rate))
    serial = args.serial if args.serial is not None else cam_cfg.get("serial")
    exposure_us = float(
        args.exposure_us
        if args.exposure_us is not None
        else cam_cfg.get("exposure_us", 1000.0)
    )
    gain = float(args.gain if args.gain is not None else cam_cfg.get("gain", 16.0))

    server = RgbWsServer(
        host=host,
        port=port,
        rate_hz=rate,
        width=width,
        height=height,
        fps=fps,
        serial=serial,
        backend=args.camera_backend,
        exposure_us=exposure_us,
        gain=gain,
    )

    stop = {"flag": False}

    def _on_sig(signum, _frame) -> None:
        stop["flag"] = True
        print(f"\nSignal {signum}; stopping camera WS server...", flush=True)

    signal.signal(signal.SIGINT, _on_sig)
    signal.signal(signal.SIGTERM, _on_sig)

    try:
        server.start()
        # Capture runs in a background thread; wait for Ctrl+C.
        while not stop["flag"]:
            time.sleep(0.2)
    except Exception as exc:
        print(f"[camera_ws_server] error: {exc}", flush=True)
        return 1
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
