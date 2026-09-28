"""Run b2z1 VR teleoperation in MuJoCo."""

import argparse
import os
import sys
import time

import mujoco.viewer
import torch

ACT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ACT_DIR not in sys.path:
    sys.path.insert(0, ACT_DIR)

from teleop.teleop_sim_env import make_teleop_sim_env
from teleop.vr_websocket_client import VrWebSocketClient


def main():
    parser = argparse.ArgumentParser(description="B2Z1 Vive VR teleoperation in MuJoCo")
    parser.add_argument("--render", action="store_true", help="Launch MuJoCo viewer")
    parser.add_argument("--host", default="127.0.0.1", help="VR WebSocket server host")
    parser.add_argument("--port", type=int, default=8765, help="VR WebSocket server port")
    parser.add_argument(
        "--wbc-ckpt",
        default="wbc_ckpts/b2z1/lambdawbc-walk/model_100000.pt",
        help="Path to WBC checkpoint",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=300.0,
        help="Maximum runtime in seconds",
    )
    args = parser.parse_args()

    os.chdir(ACT_DIR)

    env = make_teleop_sim_env()
    task = env._task
    task.load_wbc(args.wbc_ckpt)

    vr_client = VrWebSocketClient(host=args.host, port=args.port)
    vr_client.start()

    ts = env.reset()
    viewer = None
    if args.render:
        viewer = mujoco.viewer.launch_passive(
            env._physics.model._model,
            env._physics.data._data,
        )

    dt_target = 1.0 / 50.0
    t_start = time.perf_counter()
    print("Teleop sim running. Waiting for VR WebSocket connection...")

    try:
        while True:
            t0 = time.perf_counter()
            if viewer is not None and not viewer.is_running():
                break
            if time.perf_counter() - t_start > args.max_time:
                break

            vr_pose = vr_client.get_latest_pose()
            task.update_vr_goal(vr_pose)

            wbc_action = task.wbc_step(ts.observation)
            ts = env.step(wbc_action.cpu().detach().numpy())

            if viewer is not None:
                viewer.sync()

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, dt_target - elapsed))
    finally:
        vr_client.stop()
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
