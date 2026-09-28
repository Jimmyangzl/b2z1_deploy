"""
Teleoperate the Unitree Z1 arm using a Vive tracker streamed over WebSocket.

Expected WebSocket payload:
  - connected: bool
  - Tdes: 4x4 homogeneous transform (end-effector goal relative to the tracker's initial pose)

This script:
  1) Reads current end-effector pose from the Z1 arm model (forward kinematics).
  2) On the first connected tracker message, stores Tdes0.
  3) For each new Tdes, computes the relative transform:
        T_rel = inv(Tdes0) @ Tdes
     then applies it to the robot's initial end-effector pose:
        T_goal = T_init @ T_rel
  4) Solves IK and sends joints to JOINTCTRL with inverse-dynamics feedforward torque.
"""

import argparse
import os
import sys
import time
from typing import Optional

import numpy as np


def _print_matrix(name: str, T: np.ndarray) -> None:
    print(f"[teleop_z1] {name}:\n{np.array2string(T, precision=4, suppress_small=True)}")


def _vec6(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(6, 1)


def _add_z1_teleop_to_path() -> None:
    """Make `z1/teleop` importable as `teleop.*` from inside `z1_sdk/`."""
    sdk_dir = os.path.dirname(os.path.abspath(__file__))  # .../z1/z1_sdk/teleop_z1
    z1_dir = os.path.abspath(os.path.join(sdk_dir, ".."))  # .../z1/z1_sdk -> .../z1/z1
    repo_z1_dir = os.path.abspath(os.path.join(z1_dir, ".."))  # .../z1
    if repo_z1_dir not in sys.path:
        sys.path.insert(0, repo_z1_dir)


def main() -> None:
    _add_z1_teleop_to_path()

    # Local SDK python bindings
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
    import unitree_arm_interface  # type: ignore

    from teleop.vr_websocket_client import VrWebSocketClient

    parser = argparse.ArgumentParser(description="Teleoperate Z1 arm with Vive tracker (WebSocket Tdes).")
    parser.add_argument("--host", default="127.0.0.1", help="VR WebSocket server host")
    parser.add_argument("--port", type=int, default=8765, help="VR WebSocket server port")
    parser.add_argument("--max-speed", type=float, default=0.15, help="Unused (kept for compatibility)")
    parser.add_argument("--command-freq", type=float, default=100.0, help="How often to send joint IK commands (Hz)")
    parser.add_argument("--gripper-pos", type=float, default=None, help="Gripper position override (rad)")
    parser.add_argument("--wait", action="store_true", help="Unused (kept for compatibility)")
    parser.add_argument(
        "--debug",
        action="store_true",
        default=True,
        help="Print T_init, Tdes, T_rel, T_goal, and hasIK each cycle (default: on)",
    )
    parser.add_argument(
        "--no-debug",
        action="store_false",
        dest="debug",
        help="Disable debug matrix printing",
    )
    args = parser.parse_args()

    arm = unitree_arm_interface.ArmInterface(hasGripper=True)
    armState = unitree_arm_interface.ArmFSMState
    vr_client = VrWebSocketClient(host=args.host, port=args.port)
    vr_client.start()

    try:
        # Robot initial pose (supplied by vr_websocket_server after prep).
        T_init: Optional[np.ndarray] = None
        q_init_ref: Optional[np.ndarray] = None
        armModel = None

        print(
            "Z1 teleop waiting for VR server (and robot prep if enabled)... "
            "(Ctrl+C to quit)"
        )
        wait_start = time.perf_counter()
        while T_init is None:
            prep_msg = vr_client.get_latest_pose()
            if prep_msg.get("T_init") is not None:
                T_init = np.asarray(prep_msg["T_init"], dtype=np.float64).reshape(4, 4)
                if prep_msg.get("q_init") is not None:
                    q_init_ref = np.asarray(prep_msg["q_init"], dtype=np.float64)
                break
            if time.perf_counter() - wait_start > 5.0:
                print(
                    "[teleop_z1] Server did not provide T_init; "
                    "using current robot pose as initial."
                )
                arm.loopOn()
                time.sleep(0.5)
                armModel = arm._ctrlComp.armModel
                q_local = np.asarray(arm.lowstate.getQ(), dtype=np.float64)
                T_init = armModel.forwardKinematics(q_local, 6)
                arm.loopOff()
                break
            time.sleep(0.1)

        if args.debug:
            if q_init_ref is not None:
                print(
                    "[teleop_z1] Using server q_init:",
                    np.array2string(q_init_ref, precision=4, suppress_small=True),
                )
            _print_matrix("T_init", T_init)

        # Start commanding the arm only after the server releases its hold.
        arm.loopOn()
        arm.setWait(False)
        arm.startTrack(armState.JOINTCTRL)

        if armModel is None:
            armModel = arm._ctrlComp.armModel

        dt_target = 1.0 / max(1e-6, float(args.command_freq))
        Tdes0: Optional[np.ndarray] = None
        # Keep gripper fixed (either user-provided or current value at start).
        gripper_pos: Optional[float] = (
            float(args.gripper_pos) if args.gripper_pos is not None else None
        )
        ever_connected = False
        qd_zero = np.zeros(6, dtype=np.float64)
        qdd_zero = np.zeros(6, dtype=np.float64)
        ftip_zero = np.zeros(6, dtype=np.float64)

        print(
            "Z1 teleop running. Move the VR tracker to control the arm. "
            "(Ctrl+C to quit)"
        )

        while True:
            t0 = time.perf_counter()
            msg = vr_client.get_latest_pose()
            vr_active = msg.get("connected", False) and msg.get("Tdes") is not None

            if vr_active:
                ever_connected = True

                Tdes = np.asarray(msg["Tdes"], dtype=np.float64).reshape(4, 4)
                if Tdes0 is None:
                    Tdes0 = Tdes.copy()

                if gripper_pos is None:
                    # Freeze gripper at first valid tracker message.
                    gripper_pos = float(arm.lowstate.getGripperQ())

                # Relative delta from tracker's initial pose
                T_rel = np.linalg.inv(Tdes0) @ Tdes
                T_goal = T_init @ T_rel

                q_seed = np.asarray(arm.lowstate.getQ(), dtype=np.float64).reshape(6)
                hasIK, q_result = armModel.inverseKinematics(T_goal, q_seed, False)

                if args.debug:
                    print("-" * 60)
                    _print_matrix("Tdes", Tdes)
                    _print_matrix("T_rel", T_rel)
                    _print_matrix("T_goal", T_goal)
                    print("[teleop_z1] hasIK:", hasIK)
                    if hasIK:
                        print(
                            "[teleop_z1] q_result:",
                            np.array2string(q_result, precision=4, suppress_small=True),
                        )

                if hasIK:
                    q_cmd = np.asarray(q_result, dtype=np.float64).reshape(6)
                    qd_cmd = qd_zero
                    tau_f = armModel.inverseDynamics(q_cmd, qd_cmd, qdd_zero, ftip_zero)
                    arm.q = q_cmd
                    arm.qd = qd_cmd
                    arm.tau = tau_f
                    arm.setArmCmd(_vec6(q_cmd), _vec6(qd_cmd), _vec6(tau_f))
                    arm.setGripperCmd(float(gripper_pos), 0.0, 0.0)
                    if args.debug:
                        print(
                            "[teleop_z1] tau_f:",
                            np.array2string(np.asarray(tau_f).reshape(6), precision=4, suppress_small=True),
                        )
                else:
                    print("[teleop_z1] IK failed, holding last joint command.")

            elif ever_connected:
                print("[teleop_z1] VR disconnected, switching robot to PASSIVE and exiting.")
                arm.setFsm(armState.PASSIVE)
                break

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, dt_target - elapsed))

    except KeyboardInterrupt:
        pass
    finally:
        # Stop background threads cleanly
        try:
            vr_client.stop()
        except Exception:
            pass
        arm.loopOff()


if __name__ == "__main__":
    main()

