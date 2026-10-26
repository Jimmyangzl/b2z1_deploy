#!/usr/bin/env python3
"""Interactive visualizer for relative-trajectory HDF5 episodes.

Side-by-side RGB + 3D EE chunk in arm-base frame ``F_t``, with a slider over ``t``.
"""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np

DATA_RECORD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DATA_RECORD_ROOT not in sys.path:
    sys.path.insert(0, DATA_RECORD_ROOT)

from transform.relative_traj import quat_rotate  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize relative_traj episode (RGB + 3D EE chunk)."
    )
    p.add_argument("--task", default=None, help="Task name under recordings/relative_traj/")
    p.add_argument("--episode", type=int, default=None, help="Episode index")
    p.add_argument(
        "--path",
        default=None,
        help="Direct path to relative_traj episode_*.h5 (overrides --task/--episode)",
    )
    p.add_argument(
        "--frame-scale",
        type=float,
        default=0.02,
        help="Orientation / arm-base axis length in meters (default: 0.02)",
    )
    p.add_argument(
        "--t0",
        type=int,
        default=0,
        help="Initial scrub index t",
    )
    p.add_argument(
        "--recordings-root",
        default=os.path.join(DATA_RECORD_ROOT, "recordings"),
        help="Recordings root containing relative_traj/<task>/",
    )
    return p.parse_args()


def _resolve_path(args: argparse.Namespace) -> str:
    if args.path:
        path = os.path.abspath(args.path)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        return path
    if args.task is None or args.episode is None:
        raise SystemExit("Provide --path, or both --task and --episode")
    path = os.path.join(
        args.recordings_root,
        "relative_traj",
        args.task,
        f"episode_{args.episode}.h5",
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def _quat_to_axes(q: np.ndarray, scale: float) -> np.ndarray:
    """Return (3, 3) rows = x/y/z axis endpoints relative to origin."""
    ex = quat_rotate(q, np.array([scale, 0.0, 0.0]))
    ey = quat_rotate(q, np.array([0.0, scale, 0.0]))
    ez = quat_rotate(q, np.array([0.0, 0.0, scale]))
    return np.stack([ex, ey, ez], axis=0)


def _draw_frame(ax, origin: np.ndarray, q: np.ndarray, scale: float, alpha: float = 1.0):
    axes = _quat_to_axes(q, scale)
    colors = ("r", "g", "b")
    for i, c in enumerate(colors):
        end = origin + axes[i]
        ax.plot(
            [origin[0], end[0]],
            [origin[1], end[1]],
            [origin[2], end[2]],
            color=c,
            linewidth=1.5,
            alpha=alpha,
        )


def _draw_arm_base_axes(ax, scale: float):
    """RGB = XYZ at F_t origin."""
    origin = np.zeros(3)
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    _draw_frame(ax, origin, identity, scale, alpha=1.0)
    ax.scatter([0], [0], [0], c="k", s=30, depthshade=False, label="arm-base F_t")


def main() -> int:
    args = _parse_args()
    path = _resolve_path(args)

    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.colors import Normalize
        from matplotlib.widgets import Slider
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError as e:
        print(
            "matplotlib is required for visualization: pip install matplotlib",
            flush=True,
        )
        raise SystemExit(1) from e

    with h5py.File(path, "r") as f:
        rgb = f["rgb"][:]
        cart = f["ee_goal_cart_chunk"][:]
        quat = f["ee_goal_quat_chunk"][:]
        grip = f["gripper_cmd"][:]
        ts = f["timestamp"][:]
        chunk_n = int(f.attrs.get("chunk_n", cart.shape[1]))
        dt = float(f.attrs.get("dt", 0.05))
        arm_off = np.asarray(f.attrs.get("arm_base_offset", [0.15, 0.0, 0.1]), dtype=np.float64)

    t_len, n, _ = cart.shape
    if rgb.shape[0] != t_len:
        raise ValueError(f"rgb length {rgb.shape[0]} != cart T {t_len}")

    t0 = int(np.clip(args.t0, 0, t_len - 1))
    scale = float(args.frame_scale)
    if scale <= 0.0:
        raise SystemExit("--frame-scale must be > 0")

    # Fixed 3D limits from whole episode so scrubbing does not jump the view.
    pts = cart.reshape(-1, 3)
    finite = np.isfinite(pts).all(axis=1)
    if not finite.any():
        raise SystemExit("No finite EE positions in episode")
    pts_f = pts[finite]
    lo = pts_f.min(axis=0)
    hi = pts_f.max(axis=0)
    # Include origin (arm-base) and frame tips roughly.
    lo = np.minimum(lo, -scale)
    hi = np.maximum(hi, scale)
    center = 0.5 * (lo + hi)
    half = 0.5 * float(np.max(hi - lo))
    half = max(half, 2.0 * scale, 0.05)
    lim_lo = center - half
    lim_hi = center + half

    grip_finite = grip[np.isfinite(grip)]
    if grip_finite.size:
        gmin, gmax = float(grip_finite.min()), float(grip_finite.max())
    else:
        gmin, gmax = -1.0, 0.0
    if abs(gmax - gmin) < 1e-9:
        gmax = gmin + 1e-6
    norm = Normalize(vmin=gmin, vmax=gmax)
    try:
        cmap = plt.colormaps["coolwarm"]
    except (AttributeError, KeyError):
        cmap = cm.get_cmap("coolwarm")

    fig = plt.figure(figsize=(12, 6))
    fig.canvas.manager.set_window_title(f"relative_traj: {os.path.basename(path)}")
    ax_img = fig.add_axes([0.04, 0.18, 0.40, 0.75])
    ax_3d = fig.add_axes([0.48, 0.18, 0.50, 0.75], projection="3d")
    ax_sl = fig.add_axes([0.15, 0.06, 0.70, 0.04])

    img_artist = ax_img.imshow(rgb[t0])
    ax_img.set_axis_off()
    title_img = ax_img.set_title("")

    slider = Slider(ax_sl, "t", 0, t_len - 1, valinit=t0, valstep=1)

    def _gripper_text(t: int) -> str:
        g0 = grip[t, 0]
        g_last = grip[t, n - 1]
        return (
            f"gripper k=0: {g0:.3f}  k={n-1}: {g_last:.3f}  "
            f"(closed≈{gmax:.2f} … open≈{gmin:.2f})"
        )

    def _set_equal_box():
        ax_3d.set_xlim(lim_lo[0], lim_hi[0])
        ax_3d.set_ylim(lim_lo[1], lim_hi[1])
        ax_3d.set_zlim(lim_lo[2], lim_hi[2])
        try:
            ax_3d.set_box_aspect(
                (
                    lim_hi[0] - lim_lo[0],
                    lim_hi[1] - lim_lo[1],
                    lim_hi[2] - lim_lo[2],
                )
            )
        except Exception:
            pass

    def redraw(t: int):
        t = int(t)
        img_artist.set_data(rgb[t])
        title_img.set_text(
            f"RGB  t={t}/{t_len - 1}  time={ts[t] - ts[0]:.2f}s  "
            f"n={chunk_n} dt={dt:.3f}s\n{_gripper_text(t)}"
        )

        ax_3d.cla()
        _draw_arm_base_axes(ax_3d, scale)

        pos = cart[t]  # (n, 3)
        g = grip[t]
        colors = cmap(norm(g))
        ax_3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], color="0.5", linewidth=1.0, alpha=0.8)
        ax_3d.scatter(
            pos[:, 0],
            pos[:, 1],
            pos[:, 2],
            c=g,
            cmap=cmap,
            norm=norm,
            s=36,
            depthshade=False,
        )
        # Orientation frames (slightly transparent for mid-horizon clutter).
        for k in range(n):
            alpha = 1.0 if k in (0, n - 1) else 0.45
            _draw_frame(ax_3d, pos[k], quat[t, k], scale, alpha=alpha)

        ax_3d.set_xlabel("x (m)")
        ax_3d.set_ylabel("y (m)")
        ax_3d.set_zlabel("z (m)")
        ax_3d.set_title(
            f"EE chunk in arm-base F_t  |  offset={arm_off.tolist()}  |  "
            f"frame_scale={scale:.3f}m"
        )
        _set_equal_box()
        fig.canvas.draw_idle()

    slider.on_changed(lambda val: redraw(val))
    redraw(t0)

    # Colorbar for gripper
    cax = fig.add_axes([0.92, 0.25, 0.015, 0.55])
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("gripper_cmd")

    print(
        f"[viz_relative_traj] {path}  T={t_len} n={n}  "
        f"keys: drag slider / arrows if focus on slider",
        flush=True,
    )
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
