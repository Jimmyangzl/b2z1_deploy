"""Move Z1 to the teleop-ready joint pose and hold until VR takes control."""

import math
import os
import sys
import threading
import time
from typing import Optional

import numpy as np

TELEOP_INIT_JOINTS = np.array(
    [0.0, 0.6, -0.6, 0.0, 0.0, math.pi / 2.0],
    dtype=np.float64,
)


def _default_wbc_yaml_path() -> str:
    teleop_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(
        os.path.join(teleop_dir, "..", "..", "b2", "wbc_deploy", "config", "b2z1_wbc.yaml")
    )


def load_arm_init_joint(config: Optional[dict] = None) -> np.ndarray:
    """Resolve arm init joints; prefer config, else b2z1_wbc.yaml::arm_init_joint."""
    if config is not None and config.get("arm_init_joint") is not None:
        return np.asarray(config["arm_init_joint"], dtype=np.float64).reshape(6)

    wbc_yaml = _default_wbc_yaml_path()
    if os.path.isfile(wbc_yaml):
        try:
            import yaml

            with open(wbc_yaml, "r", encoding="utf-8") as f:
                wbc_cfg = yaml.safe_load(f) or {}
            if wbc_cfg.get("arm_init_joint") is not None:
                return np.asarray(wbc_cfg["arm_init_joint"], dtype=np.float64).reshape(6)
        except Exception as exc:
            print(f"[z1_robot_prep] Failed reading arm_init_joint from {wbc_yaml}: {exc}")

    return TELEOP_INIT_JOINTS.copy()


def _add_z1_sdk_to_path() -> None:
    teleop_dir = os.path.dirname(os.path.abspath(__file__))
    z1_sdk_lib = os.path.join(teleop_dir, "..", "z1_sdk", "lib")
    z1_sdk_lib = os.path.abspath(z1_sdk_lib)
    if z1_sdk_lib not in sys.path:
        sys.path.append(z1_sdk_lib)


def _vec6(values) -> np.ndarray:
    """Return a pybind11-compatible 6x1 joint vector."""
    return np.asarray(values, dtype=np.float64).reshape(6, 1)


class Z1RobotPrep:
    """Drive the arm to TELEOP_INIT_JOINTS, hold, then release for teleop."""

    def __init__(
        self,
        init_joints: Optional[np.ndarray] = None,
        move_duration: float = 3.0,
        hold_rate_hz: float = 50.0,
        connect_timeout: float = 10.0,
    ):
        _add_z1_sdk_to_path()
        import unitree_arm_interface  # type: ignore

        self._unitree_arm_interface = unitree_arm_interface
        self.arm = unitree_arm_interface.ArmInterface(hasGripper=True)
        self.arm_state = unitree_arm_interface.ArmFSMState
        self.q_init = np.asarray(
            load_arm_init_joint() if init_joints is None else init_joints,
            dtype=np.float64,
        ).reshape(6)
        self.move_duration = float(move_duration)
        self.hold_dt = 1.0 / max(1e-6, hold_rate_hz)
        self.connect_timeout = float(connect_timeout)
        self.T_init: Optional[np.ndarray] = None
        self.gripper_init: Optional[float] = None
        self._stop_hold = threading.Event()
        self._hold_thread: Optional[threading.Thread] = None
        self._holding = False
        self._released = False

    def _wait_for_controller(self) -> None:
        """Wait until z1_ctrl responds over UDP with valid joint state."""
        print(
            "[z1_robot_prep] Waiting for z1_ctrl (run ./z1_ctrl first)...",
            flush=True,
        )
        # Real Z1 joints are within ~±π; reject UDP garbage (e.g. 1e23).
        max_abs_q = 10.0
        arm = self.arm
        deadline = time.time() + self.connect_timeout
        while time.time() < deadline:
            time.sleep(0.5)
            q = np.asarray(arm.lowstate.getQ(), dtype=np.float64).reshape(6)
            if (
                np.all(np.isfinite(q))
                and np.max(np.abs(q)) > 1e-4
                and np.max(np.abs(q)) <= max_abs_q
            ):
                print(
                    "[z1_robot_prep] Connected to z1_ctrl. q:",
                    np.array2string(q, precision=4, suppress_small=True),
                    flush=True,
                )
                return
            if np.any(~np.isfinite(q)) or np.max(np.abs(q)) > max_abs_q:
                print(
                    "[z1_robot_prep] Waiting for valid arm state "
                    "(z1_ctrl must be linked to the arm). "
                    f"got q={np.array2string(q, precision=3)}",
                    flush=True,
                )
        raise RuntimeError(
            "[z1_robot_prep] Timed out waiting for valid Z1 state.\n"
            "  1) Power on the arm and check the Ethernet link.\n"
            "  2) Host NIC should reach IP in z1_controller/config/config.xml "
            "(default 192.168.123.110:8881).\n"
            "  3) ./z1_ctrl must stop printing 'connect with z1_arm wait time out' "
            "before running this script."
        )

    def _move_to_init_joints(self) -> None:
        """Ramp joints in JOINTCTRL instead of blocking MoveJ."""
        arm = self.arm
        arm.setWait(False)
        arm.startTrack(self.arm_state.JOINTCTRL)

        q_start = np.asarray(arm.lowstate.getQ(), dtype=np.float64).reshape(6)
        q_target = self.q_init.copy()
        gripper_pos = float(arm.lowstate.getGripperQ())
        dt = float(arm._ctrlComp.dt)
        steps = max(1, int(self.move_duration / dt))
        qd_cmd = (q_target - q_start) / max(self.move_duration, dt)

        print(
            "[z1_robot_prep] Moving to teleop init joints:",
            np.array2string(q_target, precision=4, suppress_small=True),
            flush=True,
        )
        print(
            "[z1_robot_prep] From q:",
            np.array2string(q_start, precision=4, suppress_small=True),
            flush=True,
        )

        tau_zero = _vec6(np.zeros(6, dtype=np.float64))
        for step in range(1, steps + 1):
            alpha = step / steps
            q_cmd = q_start * (1.0 - alpha) + q_target * alpha
            q_cmd_vec = _vec6(q_cmd)
            qd_cmd_vec = _vec6(qd_cmd)
            arm.q = q_cmd
            arm.qd = qd_cmd
            # Python binding requires q, qd, and tau (C++ default is not exposed).
            arm.setArmCmd(q_cmd_vec, qd_cmd_vec, tau_zero)
            arm.setGripperCmd(gripper_pos, 0.0, 0.0)
            time.sleep(dt)

        # Hold target briefly so lowstate catches up.
        qd_zero = _vec6(np.zeros(6, dtype=np.float64))
        q_target_vec = _vec6(q_target)
        for _ in range(20):
            arm.q = q_target
            arm.qd = np.zeros(6, dtype=np.float64)
            arm.setArmCmd(q_target_vec, qd_zero, tau_zero)
            arm.setGripperCmd(gripper_pos, 0.0, 0.0)
            time.sleep(dt)

        q_final = np.asarray(arm.lowstate.getQ(), dtype=np.float64)
        print(
            "[z1_robot_prep] Reached q:",
            np.array2string(q_final, precision=4, suppress_small=True),
            flush=True,
        )
        self.gripper_init = gripper_pos

    def prepare_and_hold(self) -> None:
        """Move to init joints, compute T_init, and hold until release_hold()."""
        arm = self.arm
        arm_model = arm._ctrlComp.armModel
        arm.loopOn()
        self._wait_for_controller()
        self._move_to_init_joints()

        self.T_init = arm_model.forwardKinematics(self.q_init, 6)
        arm.startTrack(self.arm_state.JOINTCTRL)
        print(
            "[z1_robot_prep] Teleop init T_init:\n",
            np.array2string(self.T_init, precision=4, suppress_small=True),
            flush=True,
        )
        self._holding = True
        self._stop_hold.clear()
        self._hold_thread = threading.Thread(target=self._hold_loop, daemon=True)
        self._hold_thread.start()
        print(
            "[z1_robot_prep] Holding init joints until VR teleop connects.",
            flush=True,
        )

    def _hold_loop(self) -> None:
        q_init_vec = _vec6(self.q_init)
        qd_zero = _vec6(np.zeros(6, dtype=np.float64))
        tau_zero = _vec6(np.zeros(6, dtype=np.float64))
        while not self._stop_hold.is_set():
            self.arm.q = self.q_init
            self.arm.qd = np.zeros(6, dtype=np.float64)
            self.arm.setArmCmd(q_init_vec, qd_zero, tau_zero)
            self.arm.setGripperCmd(float(self.gripper_init), 0.0, 0.0)
            time.sleep(self.hold_dt)

    def release_hold(self) -> None:
        """Stop holding joints and stop this process from commanding the arm."""
        if self._released:
            return
        self._released = True
        self._stop_hold.set()
        if self._hold_thread is not None:
            self._hold_thread.join(timeout=2.0)
        self._holding = False
        self.arm.loopOff()
        print("[z1_robot_prep] Released arm hold.", flush=True)

    def get_state_payload(self) -> dict:
        """Fields to include in websocket JSON for the teleop client."""
        return {
            "robot_ready": self.T_init is not None,
            "q_init": None if self.T_init is None else self.q_init.tolist(),
            "T_init": None if self.T_init is None else self.T_init.tolist(),
            "holding": self._holding and not self._released,
        }
