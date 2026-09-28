#!/usr/bin/env python3
"""Deploy B2Z1 WBC on the real robot. Motor commands gated behind --execute-actions."""

import os
import signal
import sys
import threading
import time
from typing import Optional

import numpy as np

WBC_DEPLOY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WBC_DEPLOY_ROOT not in sys.path:
    sys.path.insert(0, WBC_DEPLOY_ROOT)

from paths import setup_paths

setup_paths()

import unitree_legged_const as b2_const
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

from state.leg_dof_mapping import motor_legs_to_sim, sim_legs_to_motor
from commands.vel_keyboard import VelKeyboardController
from control.incremental_ik import incremental_ik_step
from safety.leg_shutdown import leg_joint_dict_to_array, run_leg_soft_shutdown
from utils.math_utils import pos_quat_to_homogeneous
from wbc_runtime import build_arg_parser, load_runtime, print_obs_summary, run_policy


def _print_command_mode(config: dict) -> None:
    cmd = config.get("commands", {})
    use_cmd = bool(cmd.get("use_cmd", False))
    print(f"  use_cmd        : {use_cmd}")
    if use_cmd:
        print(
            "  commands       : "
            f"vx={cmd.get('cmd_vx', 0.0)} "
            f"vy={cmd.get('cmd_vy', 0.0)} "
            f"yaw={cmd.get('cmd_yaw', 0.0)}"
        )
    else:
        print("  locomotion     : dvel_b_local from VR / EE goal")


def _vec6(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(6, 1)


def init_deploy_joint_from_config(config: dict) -> np.ndarray:
    """Build 12-dim sim-order leg targets from init_deploy_joint (FL, FR, RL, RR)."""
    return leg_joint_dict_to_array(config, "init_deploy_joint")


class B2LowCmdWriter:
    """500 Hz rt/lowcmd publisher for leg position targets."""

    def __init__(self, kp: float, kd: float, initial_targets_sim: Optional[np.ndarray] = None):
        self.kp = kp
        self.kd = kd
        self.crc = CRC()
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        if initial_targets_sim is not None:
            self._targets = sim_legs_to_motor(initial_targets_sim)
        else:
            self._targets = np.zeros(12, dtype=np.float64)
        self._lock = threading.Lock()
        self._init_low_cmd()
        self.publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.publisher.Init()
        self._thread = RecurrentThread(interval=0.002, target=self._write, name="wbc_lowcmd")
        self._thread.Start()

    def stop(self) -> None:
        """Stop the 500 Hz writer thread."""
        if self._thread is not None:
            self._thread.Wait(timeout=1.0)
            self._thread = None

    def _init_low_cmd(self) -> None:
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01
            self.low_cmd.motor_cmd[i].q = b2_const.PosStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = b2_const.VelStopF
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def set_gains(self, kp: float, kd: float) -> None:
        self.kp = kp
        self.kd = kd

    def set_leg_targets(self, q_targets_sim: np.ndarray) -> None:
        """Accept leg targets in URDF/sim order; store in B2 motor order for lowcmd."""
        with self._lock:
            self._targets = sim_legs_to_motor(q_targets_sim)

    def get_leg_targets_sim(self) -> np.ndarray:
        """Current commanded leg targets in URDF/sim order."""
        with self._lock:
            return motor_legs_to_sim(self._targets)

    def _write(self) -> None:
        with self._lock:
            targets = self._targets.copy()
        for i in range(12):
            self.low_cmd.motor_cmd[i].q = float(targets[i])
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kp = self.kp
            self.low_cmd.motor_cmd[i].kd = self.kd
            self.low_cmd.motor_cmd[i].tau = 0.0
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)


class Z1ArmWriter:
    """Send Z1 joint commands via unitree_arm_interface.

    Deploy always uses teleop law ``T_goal = T_init @ inv(Tdes0) @ Tdes`` (``tdes``).
    ``wbc`` (VrPoseMapper + phase_variable home) is retained only for experiments.
    """

    def __init__(
        self,
        z1_reader,
        arm_home_joint: np.ndarray,
        arm_mount_in_base: np.ndarray,
        control_dt: float = 0.02,
        home_interp_s: float = 2.0,
        control_mode: str = "tdes",
        elbow_limit: Optional[tuple] = None,
    ):
        if control_mode not in ("wbc", "tdes"):
            raise ValueError(f"control_mode must be 'wbc' or 'tdes', got {control_mode!r}")
        self.arm_home_joint = np.asarray(arm_home_joint, dtype=np.float64).reshape(6)
        self.arm_mount_in_base = np.asarray(arm_mount_in_base, dtype=np.float64).reshape(3)
        self.control_dt = float(control_dt)
        self.home_interp_s = max(1e-3, float(home_interp_s))
        self.control_mode = control_mode
        if elbow_limit is None:
            self._elbow_limit = (-2.0, 0.0)
        else:
            lo, hi = float(elbow_limit[0]), float(elbow_limit[1])
            if lo > hi:
                raise ValueError(f"elbow_limit lower > upper: {elbow_limit}")
            self._elbow_limit = (lo, hi)
        self._arm = z1_reader.arm
        self._arm_model = z1_reader.arm_model
        self._ready = False
        self._qd_zero = np.zeros(6, dtype=np.float64)
        self._qdd_zero = np.zeros(6, dtype=np.float64)
        self._ftip_zero = np.zeros(6, dtype=np.float64)
        self._gripper_pos = None
        self._homing = False
        self._home_q_start = np.zeros(6, dtype=np.float64)
        self._home_alpha = 0.0
        self._q_hold = None
        self._T_init: Optional[np.ndarray] = None
        self._Tdes0: Optional[np.ndarray] = None
        self._last_status: dict = {
            "vr_connected": False,
            "holding": False,
            "Tdes0_locked": False,
            "has_ik": None,
            "tdes_trans_norm": 0.0,
            "mode": "idle",
        }

    def _apply_joint_limits(self, q: np.ndarray) -> np.ndarray:
        """Clip commanded joints (elbow index 2 → configured range)."""
        q = np.asarray(q, dtype=np.float64).reshape(6).copy()
        q[2] = float(np.clip(q[2], self._elbow_limit[0], self._elbow_limit[1]))
        return q

    def _ensure_ready(self) -> bool:
        if self._ready:
            return True
        if self._arm is None or self._arm_model is None:
            return False
        import unitree_arm_interface

        self._arm_state = unitree_arm_interface.ArmFSMState
        # loopOn() is started by Z1StateReader after VR prep release.
        self._arm.setWait(False)
        self._arm.startTrack(self._arm_state.JOINTCTRL)
        self._ready = True
        return True

    def stop(self) -> None:
        self._ready = False
        self._homing = False
        self._home_alpha = 0.0

    def status_line(self) -> str:
        s = self._last_status
        has_ik = s.get("has_ik")
        ik_s = "?" if has_ik is None else str(bool(has_ik))
        return (
            f"vr_connected={s.get('vr_connected')} holding={s.get('holding')} "
            f"Tdes0={s.get('Tdes0_locked')} |Tdes|={s.get('tdes_trans_norm'):.3f} "
            f"hasIK={ik_s} mode={s.get('mode')}"
        )

    def _goal_pose_arm_frame_wbc(self, robot, vr_goal) -> np.ndarray:
        """Build 4x4 EE goal in arm-base frame from VrPoseMapper goal."""
        goal_pos_base = np.asarray(vr_goal.get_goal_position(robot.ee_pos), dtype=np.float64)
        goal_quat = getattr(vr_goal, "ee_quat_goal_base", None)
        if goal_quat is None:
            goal_quat = getattr(vr_goal, "ee_quat_goal_world", None)
        if goal_quat is None:
            goal_quat = robot.ee_quat
        goal_pos_arm = goal_pos_base - self.arm_mount_in_base
        return pos_quat_to_homogeneous(goal_pos_arm, goal_quat)

    def _solve_ik(self, T_goal: np.ndarray, q_seed: np.ndarray) -> Optional[np.ndarray]:
        """SDK analytical IK (used by experimental ``wbc`` arm mode only)."""
        has_ik, q_result = self._arm_model.inverseKinematics(T_goal, q_seed, False)
        self._last_status["has_ik"] = bool(has_ik)
        if not has_ik:
            return None
        return np.asarray(q_result, dtype=np.float64).reshape(6)

    def _solve_ik_incremental(self, T_goal: np.ndarray, q_meas: np.ndarray) -> Optional[np.ndarray]:
        """Sim-style incremental IK: δq from damped LS on CalcJacobian, q_goal = q + δq."""
        T_ee = np.asarray(
            self._arm_model.forwardKinematics(q_meas, 6),
            dtype=np.float64,
        ).reshape(4, 4)
        J = np.asarray(self._arm_model.CalcJacobian(q_meas), dtype=np.float64)
        q_goal, _dpose = incremental_ik_step(J, T_ee, T_goal, q_meas)
        self._last_status["has_ik"] = q_goal is not None
        return q_goal

    def _vr_pose_dict(self, vr_goal) -> dict:
        client = getattr(vr_goal, "client", None)
        if client is None:
            return {}
        return client.get_latest_pose()

    def _ensure_T_init(self, vr_pose: dict, q_meas: np.ndarray) -> np.ndarray:
        if self._T_init is not None:
            return self._T_init
        if vr_pose.get("T_init") is not None:
            self._T_init = np.asarray(vr_pose["T_init"], dtype=np.float64).reshape(4, 4).copy()
            print(
                "[Z1ArmWriter] Using server T_init for teleop arm control.",
                flush=True,
            )
            return self._T_init
        self._T_init = np.asarray(
            self._arm_model.forwardKinematics(q_meas, 6),
            dtype=np.float64,
        ).reshape(4, 4)
        print(
            "[Z1ArmWriter] No server T_init; using current FK as T_init.",
            flush=True,
        )
        return self._T_init

    def _cmd_from_tdes(self, vr_goal, q_meas: np.ndarray) -> np.ndarray:
        """Teleop law: T_goal = T_init @ inv(Tdes0) @ Tdes."""
        vr_pose = self._vr_pose_dict(vr_goal)
        connected = bool(vr_pose.get("connected", False))
        holding = bool(vr_pose.get("holding", False))
        Tdes = vr_pose.get("Tdes")
        self._last_status.update(
            {
                "vr_connected": connected,
                "holding": holding,
                "Tdes0_locked": self._Tdes0 is not None,
                "tdes_trans_norm": (
                    0.0
                    if Tdes is None
                    else float(np.linalg.norm(np.asarray(Tdes)[:3, 3]))
                ),
                "mode": "hold",
                "has_ik": None,
            }
        )

        T_init = self._ensure_T_init(vr_pose, q_meas)

        if not connected or Tdes is None:
            self._last_status["mode"] = "hold_no_vr"
            return self._q_hold if self._q_hold is not None else q_meas.copy()

        Tdes = np.asarray(Tdes, dtype=np.float64).reshape(4, 4)
        if self._Tdes0 is None:
            self._Tdes0 = Tdes.copy()
            self._last_status["Tdes0_locked"] = True
            print("[Z1ArmWriter] Locked Tdes0 (teleop baseline).", flush=True)

        T_rel = np.linalg.inv(self._Tdes0) @ Tdes
        T_goal = T_init @ T_rel
        q_ik = self._solve_ik_incremental(T_goal, q_meas)
        if q_ik is None:
            self._last_status["mode"] = "hold_ik_fail"
            return self._q_hold if self._q_hold is not None else q_meas.copy()
        self._last_status["mode"] = "tdes_inc_ik"
        return q_ik

    def _cmd_from_wbc(
        self,
        robot,
        vr_goal,
        phase_variable: float,
        q_meas: np.ndarray,
    ) -> np.ndarray:
        if phase_variable > 0.5:
            if not self._homing:
                self._homing = True
                self._home_q_start = q_meas.copy()
                self._home_alpha = 0.0
            self._home_alpha = min(1.0, self._home_alpha + self.control_dt / self.home_interp_s)
            a = self._home_alpha
            return (1.0 - a) * self._home_q_start + a * self.arm_home_joint

        self._homing = False
        self._home_alpha = 0.0
        T_goal = self._goal_pose_arm_frame_wbc(robot, vr_goal)
        q_ik = self._solve_ik(T_goal, q_meas)
        if q_ik is None:
            return self._q_hold if self._q_hold is not None else q_meas.copy()
        return q_ik

    def send(
        self,
        gripper_q: float,
        robot,
        vr_goal,
        obs_builder,
        phase_variable: float = 0.0,
    ) -> None:
        del obs_builder
        if not self._ensure_ready():
            return
        if self._gripper_pos is None:
            self._gripper_pos = gripper_q

        q_meas = np.asarray(robot.arm_q, dtype=np.float64).reshape(6)
        if self._q_hold is None:
            self._q_hold = q_meas.copy()

        if self.control_mode == "tdes":
            q_cmd = self._cmd_from_tdes(vr_goal, q_meas)
        else:
            q_cmd = self._cmd_from_wbc(robot, vr_goal, phase_variable, q_meas)

        q_cmd = self._apply_joint_limits(q_cmd)
        self._q_hold = q_cmd.copy()
        qd_cmd = self._qd_zero
        tau_f = self._arm_model.inverseDynamics(q_cmd, qd_cmd, self._qdd_zero, self._ftip_zero)
        # Match teleop_z1_vive_ws / z1_robot_prep: assign arm.q/qd/tau then setArmCmd.
        self._arm.q = q_cmd
        self._arm.qd = qd_cmd
        self._arm.tau = tau_f
        self._arm.setArmCmd(_vec6(q_cmd), _vec6(qd_cmd), _vec6(tau_f))
        self._arm.setGripperCmd(float(self._gripper_pos), 0.0, 0.0)

def leg_targets_from_action(action: np.ndarray, default_leg: np.ndarray, action_scale: list) -> np.ndarray:
    scale = np.asarray(action_scale[:12], dtype=np.float64)
    return default_leg[:12] + scale * action[:12]


def _sleep_dt(dt: float, shutdown: dict) -> None:
    end = time.perf_counter() + max(0.0, dt)
    while not shutdown["stop"] and time.perf_counter() < end:
        time.sleep(min(0.05, end - time.perf_counter()))


def run_init_deploy_sequence(
    *,
    config: dict,
    aggregator,
    vr_goal,
    obs_builder,
    policy,
    lowcmd_writer: B2LowCmdWriter,
    z1_writer: Optional[Z1ArmWriter],
    q_start: np.ndarray,
    q_init: np.ndarray,
    rate: float,
    dt: float,
    shutdown: dict,
) -> bool:
    """Ramp to init_deploy_joint, hold while printing obs, then confirm policy.

    Returns True if user confirms policy motor output; False to keep holding init pose.
    """
    ramp_s = float(config.get("init_deploy_ramp_s", 5.0))
    hold_s = float(config.get("init_deploy_hold_s", 5.0))
    use_cmd = bool(config["commands"].get("use_cmd"))
    print_every = max(1, int(rate))

    print(
        f"\nInit deploy: ramp {ramp_s:.1f}s → hold {hold_s:.1f}s at init_deploy_joint\n"
        f"  target (sim FL/FR/RL/RR): {np.array2string(q_init, precision=3)}"
    )

    # --- ramp ---
    ramp_steps = max(1, int(ramp_s * rate))
    for i in range(ramp_steps):
        if shutdown["stop"]:
            return False
        t0 = time.perf_counter()
        alpha = (i + 1) / ramp_steps
        q_cmd = (1.0 - alpha) * q_start + alpha * q_init
        lowcmd_writer.set_leg_targets(q_cmd)

        robot = aggregator.read()
        if robot.valid:
            obs_pack = obs_builder.build(robot, vr_goal)
            action = run_policy(policy, obs_pack["obs_vector"])
            obs_builder.update_action_history(action)
            if z1_writer is not None:
                z1_writer.send(
                    gripper_q=robot.gripper_q,
                    robot=robot,
                    vr_goal=vr_goal,
                    obs_builder=obs_builder,
                    phase_variable=obs_pack["phase_variable"],
                )
            if i % print_every == 0 or i == ramp_steps - 1:
                print(f"\n=== init ramp {i + 1}/{ramp_steps} (alpha={alpha:.2f}) ===")
                print_obs_summary(i, obs_pack, action, use_cmd=use_cmd)

        _sleep_dt(dt - (time.perf_counter() - t0), shutdown)

    lowcmd_writer.set_leg_targets(q_init)
    print(f"\nInit ramp done; holding init_deploy_joint for {hold_s:.1f}s...")

    # --- hold + print obs ---
    hold_steps = max(1, int(hold_s * rate))
    for i in range(hold_steps):
        if shutdown["stop"]:
            return False
        t0 = time.perf_counter()
        lowcmd_writer.set_leg_targets(q_init)

        robot = aggregator.read()
        if robot.valid:
            obs_pack = obs_builder.build(robot, vr_goal)
            action = run_policy(policy, obs_pack["obs_vector"])
            obs_builder.update_action_history(action)
            if z1_writer is not None:
                z1_writer.send(
                    gripper_q=robot.gripper_q,
                    robot=robot,
                    vr_goal=vr_goal,
                    obs_builder=obs_builder,
                    phase_variable=obs_pack["phase_variable"],
                )
            if i % print_every == 0 or i == hold_steps - 1:
                print(f"\n=== init hold {i + 1}/{hold_steps} ===")
                print_obs_summary(i, obs_pack, action, use_cmd=use_cmd)

        _sleep_dt(dt - (time.perf_counter() - t0), shutdown)

    if shutdown["stop"]:
        return False

    print(
        "\nInit hold complete. Robot is standing at init_deploy_joint.\n"
        "Type YES to start sending policy leg actions to the robot.\n"
        "Anything else keeps holding init_deploy_joint."
    )
    # Temporarily restore default SIGINT so Ctrl+C during input works cleanly.
    prev_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    try:
        confirm = input("Confirm policy output [YES/no]: ")
    except (EOFError, KeyboardInterrupt):
        print("\nNo confirm; soft-shutdown on exit.")
        shutdown["stop"] = True
        return False
    finally:
        signal.signal(signal.SIGINT, prev_handler)

    if confirm.strip() != "YES":
        print("Keeping legs at init_deploy_joint (policy commands disabled).")
        return False

    print("Confirmed. Switching to policy gains and applying policy actions.")
    lowcmd_writer.set_gains(config["control"]["kp"], config["control"]["kd"])
    return True


def _make_z1_writer(args, aggregator, config, arm_home, dt, *, control_mode: str) -> Z1ArmWriter:
    limits = config.get("arm_joint_limits") or {}
    elbow = limits.get("elbow", [-2.0, 0.0])
    return Z1ArmWriter(
        aggregator.z1,
        arm_home,
        arm_mount_in_base=np.array(
            config["robot"].get(
                "policy_arm_base_offset",
                config["robot"]["z1_base_in_robot_base"],
            ),
            dtype=np.float64,
        ),
        control_dt=dt,
        home_interp_s=float(config.get("arm_home_interp_s", 2.0)),
        control_mode=control_mode,
        elbow_limit=(float(elbow[0]), float(elbow[1])),
    )


def main() -> None:
    parser = build_arg_parser("Deploy B2Z1 WBC on real robot")
    parser.add_argument(
        "--execute-actions",
        action="store_true",
        help="Actually send B2 lowcmd and Z1 arm commands (default: dry run)",
    )
    parser.add_argument(
        "--legs-only",
        action="store_true",
        help="With --execute-actions, command B2 legs only (skip Z1 arm lowcmd)",
    )
    parser.add_argument(
        "--arm-only",
        action="store_true",
        help=(
            "Test arm VR teleop only: leave B2 sport mode alone (no lowcmd), "
            "command Z1 with T_goal=T_init@inv(Tdes0)@Tdes; still runs WBC obs/policy prints"
        ),
    )
    parser.add_argument(
        "--vel_keyboard",
        action="store_true",
        help="After policy starts, update cmd_vx/vy/yaw via keyboard (w/s/a/d/q/e/space)",
    )
    args = parser.parse_args()

    if args.arm_only and args.legs_only:
        print("Error: --arm-only and --legs-only cannot be used together.")
        return
    if args.arm_only and args.no_vr:
        print("Error: --arm-only requires VR (Tdes path). Do not pass --no-vr.")
        return

    if args.arm_only:
        print("WARNING: --arm-only will command the Z1 arm.")
        print("B2 legs stay in sport mode (no ReleaseMode / no lowcmd).")
        print("Arm uses teleop law: T_goal = T_init @ inv(Tdes0) @ Tdes.")
        confirm = input("Type YES to start arm teleop (WBC obs still printed): ")
        if confirm.strip() != "YES":
            print("Aborted.")
            return
    elif args.execute_actions:
        print("WARNING: --execute-actions will command the real robot.")
        print("You will first ramp/hold to init_deploy_joint, then confirm policy output.")
        confirm = input("Type YES to take low-level control and move to init pose: ")
        if confirm.strip() != "YES":
            print("Aborted.")
            return

    # Arm-only: never release sport. Full execute: defer release until lowcmd ready.
    defer_release = bool(args.execute_actions or args.arm_only)
    if args.arm_only:
        args.no_release_mode = True

    config, aggregator, vr_goal, obs_builder, policy, rate, dt = load_runtime(
        args, defer_release=defer_release
    )
    if policy is None:
        print(
            "Error: checkpoint required. Set policy.checkpoint in b2z1_wbc.yaml "
            "or pass --checkpoint /path/to/model.pt"
        )
        return

    default_dof = obs_builder.default_dof_pos
    action_scale = config["control"]["action_scale"]
    arm_home = np.array(
        config.get("arm_init_joint", config["arm_home_joint"]),
        dtype=np.float64,
    )
    q_init = init_deploy_joint_from_config(config)

    lowcmd_writer = None
    z1_writer = None
    vel_keyboard = None
    # Dry run and arm-only: always evaluate policy for printing (legs not applied in arm-only).
    apply_policy = (not args.execute_actions) or args.arm_only
    shutdown = {"stop": False}

    def _request_shutdown(signum, _frame) -> None:
        shutdown["stop"] = True
        print(f"\nSignal {signum} received; stopping...", flush=True)

    signal.signal(signal.SIGINT, _request_shutdown)
    signal.signal(signal.SIGTERM, _request_shutdown)

    if args.arm_only:
        print("Starting Z1 arm writer (teleop Tdes mode); sport mode untouched...")
        z1_writer = _make_z1_writer(
            args, aggregator, config, arm_home, dt, control_mode="tdes"
        )
        apply_policy = False  # never send policy leg targets
    elif args.execute_actions:
        print("Waiting for valid B2 lowstate before starting lowcmd...")
        deadline = time.time() + 30.0
        q_start = None
        while time.time() < deadline and not shutdown["stop"]:
            robot = aggregator.read()
            if robot.valid:
                q_start = np.asarray(robot.leg_q[:12], dtype=np.float64).copy()
                break
            time.sleep(0.05)
        if shutdown["stop"]:
            vr_goal.stop()
            aggregator.close()
            return
        if q_start is None:
            print("Error: no valid B2 state within 30 s; aborting.")
            vr_goal.stop()
            aggregator.close()
            return

        print(f"  current leg_q (sim): {np.array2string(q_start, precision=3)}")
        print(f"  init_deploy_joint  : {np.array2string(q_init, precision=3)}")
        startup_kp = float(config["control"].get("startup_kp", 1000.0))
        startup_kd = float(config["control"].get("startup_kd", 10.0))
        print("Starting lowcmd writer (sport mode still active)...")
        lowcmd_writer = B2LowCmdWriter(
            kp=startup_kp,
            kd=startup_kd,
            initial_targets_sim=q_start,
        )
        time.sleep(0.1)
        if not args.no_release_mode:
            print("Releasing sport mode...")
            aggregator.release_motion_mode()
        if not args.legs_only:
            print("Starting Z1 arm writer (teleop Tdes mode)...")
            z1_writer = _make_z1_writer(
                args, aggregator, config, arm_home, dt, control_mode="tdes"
            )

        apply_policy = run_init_deploy_sequence(
            config=config,
            aggregator=aggregator,
            vr_goal=vr_goal,
            obs_builder=obs_builder,
            policy=policy,
            lowcmd_writer=lowcmd_writer,
            z1_writer=z1_writer,
            q_start=q_start,
            q_init=q_init,
            rate=rate,
            dt=dt,
            shutdown=shutdown,
        )

    if args.vel_keyboard and apply_policy and not args.arm_only:
        obs_builder.use_cmd = True
        config.setdefault("commands", {})["use_cmd"] = True

        def _on_vel(vx: float, vy: float, yaw: float) -> None:
            obs_builder.set_commands(vx, vy, yaw)

        vel_keyboard = VelKeyboardController(
            config.get("keyboard_vel", {}),
            on_update=_on_vel,
        )
        vel_keyboard.start()
    elif args.vel_keyboard and args.arm_only:
        print("[vel_keyboard] skipped (--arm-only; no leg policy commands).")
    elif args.vel_keyboard and not apply_policy:
        print("[vel_keyboard] skipped (policy output not confirmed).")

    print("\nB2Z1 WBC deploy")
    print(f"  execute_actions: {args.execute_actions}")
    print(f"  arm_only       : {args.arm_only}")
    print(f"  apply_policy   : {apply_policy and not args.arm_only}")
    print(f"  vel_keyboard   : {bool(vel_keyboard is not None)}")
    if args.execute_actions and not args.arm_only:
        print(f"  legs_only      : {args.legs_only}")
        print(
            f"  gains          : "
            f"startup Kp={config['control'].get('startup_kp', 1000.0)} "
            f"Kd={config['control'].get('startup_kd', 10.0)}; "
            f"policy Kp={config['control']['kp']} Kd={config['control']['kd']}"
        )
        if config.get("shutdown_joint") is not None:
            print(
                f"  soft_shutdown  : ramp {float(config.get('shutdown_ramp_s', 3.0)):.1f}s "
                f"+ hold {float(config.get('shutdown_hold_s', 0.5)):.1f}s → damping"
            )
    if args.arm_only:
        print("  arm_control    : Tdes (T_goal = T_init @ inv(Tdes0) @ Tdes)")
        print("  legs           : sport mode left alone")
    elif args.execute_actions and not args.legs_only:
        print("  arm_control    : Tdes (T_goal = T_init @ inv(Tdes0) @ Tdes)")
    print(f"  checkpoint     : {args.checkpoint}")
    print(f"  rate           : {rate} Hz")
    if args.no_vr:
        print("  VR server      : disabled (--no-vr); EE goal held at current pose")
    else:
        print(f"  VR server      : {config['vr']['host']}:{config['vr']['port']}")
    _print_command_mode(config)
    print("Press Ctrl+C to stop.\n")

    step = 0
    try:
        while not shutdown["stop"]:
            t0 = time.perf_counter()
            robot = aggregator.read()
            if not robot.valid:
                time.sleep(0.05)
                continue

            obs_pack = obs_builder.build(robot, vr_goal)
            action = run_policy(policy, obs_pack["obs_vector"])
            obs_builder.update_action_history(action)

            if apply_policy and not args.arm_only:
                leg_q_target = leg_targets_from_action(action, default_dof, action_scale)
            else:
                leg_q_target = q_init

            # Arm Tdes teleop whenever Z1 writer is active (--arm-only or full deploy).
            if z1_writer is not None:
                z1_writer.send(
                    gripper_q=robot.gripper_q,
                    robot=robot,
                    vr_goal=vr_goal,
                    obs_builder=obs_builder,
                    phase_variable=obs_pack["phase_variable"],
                )

            if args.verbose or step % max(1, int(rate)) == 0:
                print_obs_summary(
                    step,
                    obs_pack,
                    action,
                    use_cmd=bool(obs_builder.use_cmd or config["commands"].get("use_cmd")),
                )
                if args.arm_only:
                    print("  leg_cmd        : sport mode (no lowcmd; --arm-only)")
                elif args.execute_actions and not apply_policy:
                    print("  leg_cmd        : holding init_deploy_joint (policy not applied)")
                if z1_writer is not None:
                    print("  arm_cmd        : Tdes incremental IK → JOINTCTRL")
                    print(f"  arm_teleop     : {z1_writer.status_line()}")
                if vel_keyboard is not None:
                    print(
                        f"  keyboard_cmd   : "
                        f"vx={obs_builder.commands[0]:.2f} "
                        f"vy={obs_builder.commands[1]:.2f} "
                        f"yaw={obs_builder.commands[2]:.2f}"
                    )

            if args.execute_actions and not args.arm_only and lowcmd_writer is not None:
                lowcmd_writer.set_leg_targets(leg_q_target)

            step += 1
            _sleep_dt(dt - (time.perf_counter() - t0), shutdown)
    except KeyboardInterrupt:
        shutdown["stop"] = True
        print("\nStopped.")
    finally:
        if vel_keyboard is not None:
            vel_keyboard.stop()
        if z1_writer is not None:
            z1_writer.stop()
        if lowcmd_writer is not None:
            try:
                q_shutdown = leg_joint_dict_to_array(config, "shutdown_joint")
                run_leg_soft_shutdown(
                    lowcmd_writer=lowcmd_writer,
                    aggregator=aggregator,
                    q_goal=q_shutdown,
                    ramp_s=float(config.get("shutdown_ramp_s", 3.0)),
                    hold_s=float(config.get("shutdown_hold_s", 0.5)),
                    rate_hz=float(rate),
                    kp=float(config.get("shutdown_kp", config["control"]["kp"])),
                    kd=float(config.get("shutdown_kd", config["control"]["kd"])),
                )
            except Exception as exc:
                print(f"[safety] Soft shutdown failed ({exc}); stopping lowcmd.", flush=True)
            print("Stopping B2 lowcmd writer (damping)...", flush=True)
            lowcmd_writer.stop()
        vr_goal.stop()
        aggregator.close()


if __name__ == "__main__":
    main()
