#!/usr/bin/env python3
"""Convert recorded episode_*.h5 files into observed-EE-relative trajectory chunks.

Output layout under ``recordings/relative_traj/<task>/``:

| Dataset | Shape |
|---------|-------|
| ``timestamp`` | ``(T,)`` |
| ``rgb`` | ``(T, H, W, 3)`` |
| ``ee_goal_cart_chunk`` | ``(T, n, 3)`` |
| ``ee_goal_quat_chunk`` | ``(T, n, 4)`` |
| ``gripper_cmd`` | ``(T, n)`` |

Episodes with any NaN in the gated source fields are reported and skipped.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

DATA_RECORD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DATA_RECORD_ROOT not in sys.path:
    sys.path.insert(0, DATA_RECORD_ROOT)

from transform.relative_traj import (  # noqa: E402
    DEFAULT_ARM_BASE_OFFSET,
    transform_episode_arrays,
)

_EPISODE_RE = re.compile(r"^episode_(\d+)\.h5$")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build observed-EE-relative trajectory chunks from recorded HDF5."
    )
    p.add_argument(
        "--task",
        required=True,
        help="Task name (reads recordings/<task>/, writes recordings/relative_traj/<task>/)",
    )
    p.add_argument(
        "--chunk-n",
        type=int,
        required=True,
        help="Horizon length n for each relative chunk",
    )
    p.add_argument(
        "--episode",
        type=int,
        default=None,
        help="If set, only transform episode_<id>.h5 (for testing)",
    )
    p.add_argument(
        "--arm-offset",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Arm-base mount xyz in robot base (default: 0.15 0 0.10)",
    )
    p.add_argument(
        "--recordings-root",
        default=os.path.join(DATA_RECORD_ROOT, "recordings"),
        help="Root directory containing <task>/episode_*.h5",
    )
    return p.parse_args()


def _list_episodes(task_dir: str, episode: Optional[int]) -> List[str]:
    if episode is not None:
        path = os.path.join(task_dir, f"episode_{episode}.h5")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        return [path]

    paths = []
    for name in sorted(os.listdir(task_dir)):
        m = _EPISODE_RE.match(name)
        if m:
            paths.append(os.path.join(task_dir, name))
    if not paths:
        raise FileNotFoundError(f"No episode_*.h5 under {task_dir}")
    return paths


def _load_episode(path: str) -> Tuple[Dict, dict]:
    with h5py.File(path, "r") as f:
        data = {
            "timestamp": f["timestamp"][:],
            "rgb": f["rgb"][:],
            "ee_goal_local_cart": f["ee_goal_local_cart"][:],
            "ee_goal_quat_base": f["ee_goal_quat_base"][:],
            "ee_pos": f["ee_pos"][:],
            "ee_quat": f["ee_quat"][:],
            "base_lin_vel_local": f["base_lin_vel_local"][:],
            "base_quat": f["base_quat"][:],
            "gripper_cmd": f["gripper_cmd"][:],
        }
        attrs = {k: f.attrs[k] for k in f.attrs}
    return data, attrs


def _write_episode(
    path: str,
    out: dict,
    *,
    src_attrs: dict,
    chunk_n: int,
    dt: float,
    arm_offset: np.ndarray,
    source_path: str,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if os.path.isfile(path):
        os.remove(path)

    rgb = np.asarray(out["rgb"])
    with h5py.File(path, "w") as f:
        for key in (
            "record_rate_hz",
            "ee_ws_url",
            "interface",
            "camera_serial",
            "created_at",
        ):
            if key in src_attrs:
                f.attrs[key] = src_attrs[key]

        f.attrs["num_frames"] = int(out["timestamp"].shape[0])
        f.attrs["chunk_n"] = int(chunk_n)
        f.attrs["dt"] = float(dt)
        f.attrs["arm_base_offset"] = np.asarray(arm_offset, dtype=np.float64)
        f.attrs["source_path"] = os.path.abspath(source_path)
        f.attrs["relative_traj"] = True

        f.create_dataset("timestamp", data=np.asarray(out["timestamp"], dtype=np.float64))
        f.create_dataset(
            "rgb",
            data=rgb,
            compression="lzf",
            chunks=(1,) + tuple(rgb.shape[1:]) if rgb.ndim == 4 else None,
        )
        f.create_dataset(
            "ee_goal_cart_chunk",
            data=np.asarray(out["ee_goal_cart_chunk"], dtype=np.float64),
        )
        f.create_dataset(
            "ee_goal_quat_chunk",
            data=np.asarray(out["ee_goal_quat_chunk"], dtype=np.float64),
        )
        f.create_dataset(
            "gripper_cmd",
            data=np.asarray(out["gripper_cmd"], dtype=np.float64),
        )


def main() -> int:
    args = _parse_args()
    if args.chunk_n < 1:
        print(f"ERROR: --chunk-n must be >= 1, got {args.chunk_n}", flush=True)
        return 2

    arm_offset = (
        np.asarray(args.arm_offset, dtype=np.float64)
        if args.arm_offset is not None
        else DEFAULT_ARM_BASE_OFFSET.copy()
    )

    task_dir = os.path.join(args.recordings_root, args.task)
    out_dir = os.path.join(args.recordings_root, "relative_traj", args.task)
    os.makedirs(out_dir, exist_ok=True)

    episodes = _list_episodes(task_dir, args.episode)
    print(
        f"[make_relative_traj] task={args.task} chunk_n={args.chunk_n} "
        f"arm_offset={arm_offset.tolist()} episodes={len(episodes)}",
        flush=True,
    )

    n_ok = 0
    n_skip = 0
    for src in episodes:
        name = os.path.basename(src)
        dst = os.path.join(out_dir, name)
        print(f"[make_relative_traj] {name} …", flush=True)

        data, attrs = _load_episode(src)
        rate = float(attrs.get("record_rate_hz", 20.0))
        dt = 1.0 / rate

        out = transform_episode_arrays(
            data,
            chunk_n=args.chunk_n,
            record_rate_hz=rate,
            arm_base_offset=arm_offset,
        )
        if out is None:
            print(
                f"[make_relative_traj] SKIP {name}: NaN present — not writing {dst}",
                flush=True,
            )
            n_skip += 1
            continue

        _write_episode(
            dst,
            out,
            src_attrs=attrs,
            chunk_n=args.chunk_n,
            dt=dt,
            arm_offset=arm_offset,
            source_path=src,
        )
        t_len = out["timestamp"].shape[0]
        print(
            f"[make_relative_traj] wrote {dst} "
            f"(T={t_len}, n={args.chunk_n}, dt={dt:.4f})",
            flush=True,
        )
        n_ok += 1

    print(
        f"[make_relative_traj] done: wrote={n_ok} skipped_nan={n_skip}",
        flush=True,
    )
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
