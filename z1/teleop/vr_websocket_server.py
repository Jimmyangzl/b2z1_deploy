"""WebSocket server broadcasting left Vive tracker pose as JSON."""

import argparse
import asyncio
import json
import os
import sys
import time

from typing import Optional

import numpy as np
import websockets

TELEOP_DIR = os.path.dirname(os.path.abspath(__file__))
ACT_DIR = os.path.abspath(os.path.join(TELEOP_DIR, ".."))
if ACT_DIR not in sys.path:
    sys.path.insert(0, ACT_DIR)

from teleop.vr_tracker_core import LeftTrackerReader, load_config
from teleop.z1_robot_prep import Z1RobotPrep, load_arm_init_joint


def quat_wxyz_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert [w, x, y, z] quaternion to a 3x3 rotation matrix."""
    q = np.asarray(quat_wxyz, dtype=np.float64).copy()
    q = q.reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-12:
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        q /= n

    w, x, y, z = q.tolist()
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def make_homogeneous(T_R: np.ndarray, T_p: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from rotation and translation."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(T_R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(T_p, dtype=np.float64).reshape(3)
    return T


def pose_to_message(pose, Tdes: Optional[np.ndarray], robot_state: Optional[dict] = None):
    robot_state = robot_state or {}
    if not pose["connected"]:
        msg = {
            "connected": False,
            "position": None,
            "orientation": None,
            "Tdes": np.eye(4, dtype=np.float64).tolist(),
            "timestamp": time.time(),
        }
        msg.update(robot_state)
        return msg
    msg = {
        "connected": True,
        "position": pose["position"].tolist(),
        "orientation": pose["orientation"].tolist(),
        "timestamp": time.time(),
    }
    if Tdes is not None:
        msg["Tdes"] = np.asarray(Tdes, dtype=np.float64).reshape(4, 4).tolist()
    msg.update(robot_state)
    return msg


def pose_to_T_vr(pose) -> Optional[np.ndarray]:
    """Convert the calibrated tracker pose dict into a 4x4 homogeneous transform."""
    if not pose.get("connected", False):
        return None
    position = np.asarray(pose["position"], dtype=np.float64).reshape(3)
    quat = np.asarray(pose["orientation"], dtype=np.float64).reshape(4)  # wxyz
    R = quat_wxyz_to_rotmat(quat)
    return make_homogeneous(R, position)


async def broadcast_loop(reader, freq, clients, robot_prep, vr_state):
    period = 1.0 / freq
    T_vr0: Optional[np.ndarray] = None
    while True:
        if vr_state.get("reset_tracker_baseline", False):
            T_vr0 = None
            vr_state["reset_tracker_baseline"] = False
            print("[vr_websocket_server] Reset VR tracker baseline for teleop.")

        vr_pose = reader.get_pose()
        Tdes = None
        T_vr = pose_to_T_vr(vr_pose)
        if T_vr is not None:
            if T_vr0 is None:
                T_vr0 = T_vr.copy()
            # Output is "tracker delta transform" relative to initial tracker pose.
            Tdes = np.linalg.inv(T_vr0) @ T_vr

        robot_state = robot_prep.get_state_payload() if robot_prep is not None else {}
        message = json.dumps(pose_to_message(vr_pose, Tdes=Tdes, robot_state=robot_state))
        if clients:
            await asyncio.gather(
                *[client.send(message) for client in list(clients)],
                return_exceptions=True,
            )
        await asyncio.sleep(period)


async def handler(websocket, clients, robot_prep, vr_state):
    clients.add(websocket)
    if len(clients) == 1 and robot_prep is not None:
        robot_prep.release_hold()
        vr_state["reset_tracker_baseline"] = True
    try:
        await websocket.wait_closed()
    finally:
        clients.discard(websocket)


def _start_robot_prep(config: dict) -> Z1RobotPrep:
    move_duration = float(config.get("teleop_prep_move_duration", 3.0))
    connect_timeout = float(config.get("teleop_prep_connect_timeout", 10.0))
    init_joints = load_arm_init_joint(config)
    print(
        "[vr_websocket_server] arm_init_joint:",
        np.array2string(init_joints, precision=4),
        flush=True,
    )
    robot_prep = Z1RobotPrep(
        init_joints=init_joints,
        move_duration=move_duration,
        connect_timeout=connect_timeout,
    )
    robot_prep.prepare_and_hold()
    return robot_prep


async def hold_arm_only_async(config_path):
    """Move Z1 to arm_init_joint and hold; no Vive / WebSocket."""
    config = load_config(config_path)
    robot_prep = _start_robot_prep(config)
    print(
        "[vr_websocket_server] --no-tracking: holding arm at init joints "
        "(no VR tracker / WebSocket). Ctrl+C to stop.",
        flush=True,
    )
    try:
        while True:
            await asyncio.sleep(1.0)
    finally:
        print("[vr_websocket_server] Stopping arm hold...", flush=True)
        robot_prep.release_hold()


async def main_async(host, port, config_path, prep_robot):
    config = load_config(config_path)
    reader = LeftTrackerReader(config_path=config_path)
    reader.calibrate()

    robot_prep = _start_robot_prep(config) if prep_robot else None

    clients = set()
    vr_state = {"reset_tracker_baseline": False}
    freq = float(config.get("freq", 50.0))

    async with websockets.serve(
        lambda ws: handler(ws, clients, robot_prep, vr_state),
        host,
        port,
    ):
        print(f"VR WebSocket server listening on ws://{host}:{port} at {freq} Hz")
        await broadcast_loop(reader, freq, clients, robot_prep, vr_state)


def main():
    parser = argparse.ArgumentParser(description="Broadcast Vive tracker pose over WebSocket")
    parser.add_argument("--config", default=None, help="Path to teleop config.yaml")
    parser.add_argument("--host", default=None, help="WebSocket bind host")
    parser.add_argument("--port", type=int, default=None, help="WebSocket bind port")
    parser.add_argument(
        "--no-robot-prep",
        action="store_true",
        help="Skip moving/holding the Z1 arm (tracker-only mode)",
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Move/hold Z1 at arm_init_joint only; skip Vive tracker and WebSocket",
    )
    args = parser.parse_args()

    if args.no_tracking and args.no_robot_prep:
        parser.error("--no-tracking and --no-robot-prep cannot be used together")

    if args.no_tracking:
        asyncio.run(hold_arm_only_async(args.config))
        return

    config = load_config(args.config)
    host = args.host or config.get("websocket_host", "0.0.0.0")
    port = args.port or int(config.get("websocket_port", 8765))
    prep_robot = not args.no_robot_prep
    asyncio.run(main_async(host, port, args.config, prep_robot))


if __name__ == "__main__":
    main()
