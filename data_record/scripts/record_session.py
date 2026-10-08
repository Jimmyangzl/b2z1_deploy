#!/usr/bin/env python3
"""Record RealSense RGB + EE goal (WS) + B2 state (DDS) into an HDF5 session file."""

from __future__ import annotations

import argparse
import os
import re
import signal
import sys
import time

import numpy as np
import yaml

DATA_RECORD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DATA_RECORD_ROOT not in sys.path:
    sys.path.insert(0, DATA_RECORD_ROOT)

# Allow importing unitree_sdk2py without installing it.
_B2_SDK = os.path.abspath(
    os.path.join(DATA_RECORD_ROOT, "..", "b2", "unitree_sdk2_python")
)
if os.path.isdir(_B2_SDK) and _B2_SDK not in sys.path:
    sys.path.insert(0, _B2_SDK)

from camera.realsense_rgb import RealSenseRGB
from net.ee_goal_ws_client import EeGoalWsClient
from robot.b2_dds_reader import B2DdsReader
from storage.h5_writer import H5SessionWriter

DEFAULT_CONFIG = os.path.join(DATA_RECORD_ROOT, "config", "record.yaml")
_EPISODE_RE = re.compile(r"^episode_(\d+)\.h5$")


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _sleep_to(deadline: float) -> None:
    remaining = deadline - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)


def _validate_task_name(task: str) -> str:
    task = str(task).strip()
    if not task:
        raise ValueError("task name must be non-empty")
    if task in (".", "..") or "/" in task or "\\" in task:
        raise ValueError(
            f"invalid task name {task!r}: use a single folder name (e.g. task_1)"
        )
    return task


def _next_episode_index(task_dir: str) -> int:
    """Return the next unused episode index under task_dir (0 if empty)."""
    if not os.path.isdir(task_dir):
        return 0
    max_idx = -1
    for name in os.listdir(task_dir):
        m = _EPISODE_RE.match(name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def _resolve_out_path(task: str, out_dir: str, out_override: str | None) -> tuple[str, int | None]:
    """Return (absolute .h5 path, episode index or None if --out override)."""
    if out_override:
        return os.path.abspath(out_override), None

    task = _validate_task_name(task)
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(DATA_RECORD_ROOT, out_dir)
    task_dir = os.path.join(out_dir, task)
    os.makedirs(task_dir, exist_ok=True)
    ep = _next_episode_index(task_dir)
    return os.path.join(task_dir, f"episode_{ep}.h5"), ep


def main() -> int:
    parser = argparse.ArgumentParser(description="B2Z1 data recording session")
    parser.add_argument(
        "--task",
        required=True,
        help="Task name; episodes saved under recordings/<task>/episode_N.h5",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to record.yaml")
    parser.add_argument("--interface", default=None, help="B2 NIC (default from yaml)")
    parser.add_argument("--ee-ws-host", default=None, help="WBC EE-goal WS host")
    parser.add_argument("--ee-ws-port", type=int, default=None, help="WBC EE-goal WS port")
    parser.add_argument("--rate", type=float, default=None, help="Record rate Hz (default 20)")
    parser.add_argument(
        "--out",
        default=None,
        help="Optional explicit .h5 path (skips recordings/<task>/episode_N.h5)",
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Skip RealSense (write black frames) for dry networking tests",
    )
    parser.add_argument(
        "--no-b2",
        action="store_true",
        help="Skip B2 DDS (write NaN vel / identity quat) for camera/WS tests",
    )
    parser.add_argument(
        "--no-ee",
        action="store_true",
        help="Skip EE-goal WebSocket client (write NaN goals)",
    )
    parser.add_argument(
        "--camera-backend",
        choices=("auto", "realsense", "v4l"),
        default="auto",
        help="RGB capture backend (auto falls back to V4L if librealsense fails)",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    net = cfg.get("network") or {}
    rec = cfg.get("record") or {}
    ee_cfg = cfg.get("ee_goal_ws") or {}
    cam_cfg = cfg.get("camera") or {}

    interface = args.interface or net.get("interface", "enp8s0")
    sport_topic = net.get("sport_topic", "rt/lf/sportmodestate")
    rate_hz = float(args.rate if args.rate is not None else rec.get("rate_hz", 20.0))
    ee_host = args.ee_ws_host or ee_cfg.get("host", "127.0.0.1")
    ee_port = int(args.ee_ws_port if args.ee_ws_port is not None else ee_cfg.get("port", 8770))
    stale_s = float(rec.get("ee_goal_stale_s", 0.5))
    width = int(cam_cfg.get("width", 640))
    height = int(cam_cfg.get("height", 480))
    cam_fps = int(cam_cfg.get("fps", 30))
    cam_serial = cam_cfg.get("serial")

    try:
        out_path, episode_idx = _resolve_out_path(
            args.task, rec.get("out_dir", "recordings"), args.out
        )
    except ValueError as exc:
        print(f"[record_session] error: {exc}", flush=True)
        return 2

    dt = 1.0 / max(1e-3, rate_hz)
    stop = {"flag": False}

    def _on_sig(signum, _frame) -> None:
        stop["flag"] = True
        print(f"\nSignal {signum}; stopping recording...", flush=True)

    signal.signal(signal.SIGINT, _on_sig)
    signal.signal(signal.SIGTERM, _on_sig)

    camera = None
    b2 = None
    ee_client = None
    writer = None
    ee_ws_url = f"ws://{ee_host}:{ee_port}"

    try:
        if not args.no_ee:
            ee_client = EeGoalWsClient(host=ee_host, port=ee_port)
            ee_client.start()
            ee_ws_url = ee_client.uri

        if not args.no_b2:
            b2 = B2DdsReader(interface=interface, sport_topic=sport_topic)
            b2.start()

        if not args.no_camera:
            camera = RealSenseRGB(
                width=width,
                height=height,
                fps=cam_fps,
                serial=cam_serial,
                backend=args.camera_backend,
            )
            camera.start()
            cam_serial_used = camera.serial or ""
        else:
            cam_serial_used = "none"

        writer = H5SessionWriter(
            out_path,
            height=height,
            width=width,
            record_rate_hz=rate_hz,
            ee_ws_url=ee_ws_url if not args.no_ee else "disabled",
            interface=interface if not args.no_b2 else "disabled",
            camera_serial=cam_serial_used or "",
        )

        ep_msg = (
            f"episode_{episode_idx}"
            if episode_idx is not None
            else "custom --out"
        )
        print(
            f"[record_session] task={args.task} ({ep_msg})\n"
            f"  writing {out_path} @ {rate_hz:.1f} Hz\n"
            f"  EE WS : {'disabled' if args.no_ee else ee_ws_url}\n"
            f"  B2    : {'disabled' if args.no_b2 else interface}\n"
            f"  camera: {'disabled' if args.no_camera else cam_serial_used}",
            flush=True,
        )

        next_t = time.perf_counter()
        while not stop["flag"]:
            next_t += dt
            wall_t = time.time()

            if camera is not None:
                rgb, _cam_t = camera.read()
            else:
                rgb = np.zeros((height, width, 3), dtype=np.uint8)

            if b2 is not None:
                state = b2.read()
                if state.get("valid"):
                    base_quat = state["base_quat"]
                    base_vel = state["base_lin_vel_local"]
                else:
                    base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
                    base_vel = np.full(3, np.nan, dtype=np.float64)
            else:
                base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
                base_vel = np.full(3, np.nan, dtype=np.float64)

            if ee_client is not None:
                ee = ee_client.get_latest()
                ee_goal = ee["ee_goal_local_cart"]
                if ee.get("valid") and ee.get("t", 0.0) > 0:
                    age = wall_t - float(ee["t"])
                else:
                    age = float("inf")
                if age > stale_s or not ee.get("valid"):
                    ee_goal = np.full(3, np.nan, dtype=np.float64)
                    age = float("nan") if not np.isfinite(age) else age
            else:
                ee_goal = np.full(3, np.nan, dtype=np.float64)
                age = float("nan")

            writer.append(
                timestamp=wall_t,
                rgb=rgb,
                ee_goal_local_cart=ee_goal,
                ee_goal_age_s=age if np.isfinite(age) else float("nan"),
                base_lin_vel_local=base_vel,
                base_quat=base_quat,
            )

            if writer.num_frames % max(1, int(rate_hz)) == 0:
                print(
                    f"[record_session] frames={writer.num_frames} "
                    f"ee={np.array2string(ee_goal, precision=3)} "
                    f"vel={np.array2string(base_vel, precision=3)}",
                    flush=True,
                )

            _sleep_to(next_t)

    except Exception as exc:
        print(f"[record_session] error: {exc}", flush=True)
        return 1
    finally:
        if writer is not None:
            writer.close()
        if camera is not None:
            camera.stop()
        if ee_client is not None:
            ee_client.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
