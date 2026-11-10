#!/usr/bin/env python3
"""Deploy B2Z1 WBC on the real robot. Motor commands gated behind --execute-actions."""

import os

# Must run before NumPy/PyTorch import: default OpenBLAS=32 / GOMP=24 threads on this
# machine. Tiny 50 Hz control ops then oversubscribe the CPU next to SteamVR and the
# policy wall-clock spikes to 100–300 ms even though MLP compute is <1 ms.
for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_k, "1")

import signal
import sys
import threading
import time
from typing import Optional, Tuple

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
from commands.vel_keyboard import DeployHotkeys
from control.dof_limits import clip_dof_pos, load_leg_arm_limit_arrays
from control.gains import GainLike, as_gain_vector
from control.incremental_ik import incremental_ik_step
from safety.leg_shutdown import leg_joint_dict_to_array, run_leg_soft_shutdown
from utils.math_utils import pos_quat_to_homogeneous, rotmat_to_quaternion
from obs.process_obs_worker import ProcessObsWorker
from wbc_runtime import (
    ProcessPolicyWorker,
    build_arg_parser,
    load_runtime,
    print_obs_summary,
)


def _maybe_publish_ee(publisher, obs_pack: dict, gripper_cmd=None, z1_writer=None) -> None:
    """Publish recorder stream. EE goal prefers the arm IK target (arm-base)."""
    if publisher is None:
        return
    goal = obs_pack.get("ee_goal_local_cart")
    goal_quat = obs_pack.get("ee_goal_quat_base")
    if z1_writer is not None:
        ik_pos, ik_quat = z1_writer.ee_goal_arm_pose()
        if ik_pos is not None:
            goal = ik_pos
            goal_quat = ik_quat
    if goal is None:
        return
    publisher.update(
        goal,
        gripper_cmd=gripper_cmd,
        base_lin_vel=obs_pack.get("base_lin_vel"),
        ee_goal_quat_base=goal_quat,
        ee_pos=obs_pack.get("ee_pos"),
        ee_quat=obs_pack.get("ee_quat"),
    )


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

    def __init__(
        self,
        kp: GainLike,
        kd: GainLike,
        initial_targets_sim: Optional[np.ndarray] = None,
        leg_lo: Optional[np.ndarray] = None,
        leg_hi: Optional[np.ndarray] = None,
    ):
        # Store gains in B2 motor order (same indexing as motor_cmd[0:12]).
        self.kp = sim_legs_to_motor(as_gain_vector(kp, 12, "leg kp"))
        self.kd = sim_legs_to_motor(as_gain_vector(kd, 12, "leg kd"))
        if leg_lo is None or leg_hi is None:
            leg_lo, leg_hi, _, _ = load_leg_arm_limit_arrays()
        self._leg_lo = np.asarray(leg_lo, dtype=np.float64).reshape(12)
        self._leg_hi = np.asarray(leg_hi, dtype=np.float64).reshape(12)
        self.crc = CRC()
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        if initial_targets_sim is not None:
            q0 = clip_dof_pos(initial_targets_sim, self._leg_lo, self._leg_hi)
            self._targets = sim_legs_to_motor(q0)
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

    def set_gains(self, kp: GainLike, kd: GainLike) -> None:
        """Set leg gains; scalar or length-12 list in sim (FL/FR/RL/RR) order."""
        with self._lock:
            self.kp = sim_legs_to_motor(as_gain_vector(kp, 12, "leg kp"))
            self.kd = sim_legs_to_motor(as_gain_vector(kd, 12, "leg kd"))

    def set_leg_targets(self, q_targets_sim: np.ndarray) -> None:
        """Accept leg targets in URDF/sim order; clamp then store in B2 motor order."""
        q = clip_dof_pos(q_targets_sim, self._leg_lo, self._leg_hi)
        with self._lock:
            self._targets = sim_legs_to_motor(q)

    def get_leg_targets_sim(self) -> np.ndarray:
        """Current commanded leg targets in URDF/sim order."""
        with self._lock:
            return motor_legs_to_sim(self._targets)

    def _write(self) -> None:
        with self._lock:
            targets = self._targets.copy()
            kp = self.kp.copy()
            kd = self.kd.copy()
        for i in range(12):
            self.low_cmd.motor_cmd[i].q = float(targets[i])
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kp = float(kp[i])
            self.low_cmd.motor_cmd[i].kd = float(kd[i])
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
        arm_lo: Optional[np.ndarray] = None,
        arm_hi: Optional[np.ndarray] = None,
        fsm_mode: str = "jointctrl",
        kp: GainLike = 20.0,
        kd: GainLike = 2000.0,
        gripper_kp: float = 20.0,
        gripper_kd: float = 2000.0,
    ):
        if control_mode not in ("wbc", "tdes", "act"):
            raise ValueError(
                f"control_mode must be 'wbc', 'tdes', or 'act', got {control_mode!r}"
            )
        fsm_mode = str(fsm_mode).lower().strip()
        if fsm_mode not in ("jointctrl", "lowcmd"):
            raise ValueError(f"fsm_mode must be 'jointctrl' or 'lowcmd', got {fsm_mode!r}")
        self.arm_home_joint = np.asarray(arm_home_joint, dtype=np.float64).reshape(6)
        self.arm_mount_in_base = np.asarray(arm_mount_in_base, dtype=np.float64).reshape(3)
        self.control_dt = float(control_dt)
        self.home_interp_s = max(1e-3, float(home_interp_s))
        self.control_mode = control_mode
        self.fsm_mode = fsm_mode
        # Per-joint Z1 gains (Unitree command units). Applied only in LOWCMD.
        self.kp = as_gain_vector(kp, 6, "arm kp")
        self.kd = as_gain_vector(kd, 6, "arm kd")
        self.gripper_kp = float(gripper_kp)
        self.gripper_kd = float(gripper_kd)
        if arm_lo is None or arm_hi is None:
            _, _, arm_lo, arm_hi = load_leg_arm_limit_arrays()
        self._arm_lo = np.asarray(arm_lo, dtype=np.float64).reshape(6)
        self._arm_hi = np.asarray(arm_hi, dtype=np.float64).reshape(6)
        self.arm_home_joint = clip_dof_pos(self.arm_home_joint, self._arm_lo, self._arm_hi)
        self._arm = z1_reader.arm
        self._arm_model = z1_reader.arm_model
        self._ready = False
        self._qd_zero = np.zeros(6, dtype=np.float64)
        self._qdd_zero = np.zeros(6, dtype=np.float64)
        self._ftip_zero = np.zeros(6, dtype=np.float64)
        # Binary gripper: closed=0, open=-1 (space toggles). Observed polarity.
        self.gripper_closed = 0.0
        self.gripper_open = -1.0
        self._gripper_pos = float(self.gripper_closed)
        self._gripper_lock = threading.Lock()
        self._homing = False
        self._home_q_start = np.zeros(6, dtype=np.float64)
        self._home_alpha = 0.0
        self._q_hold = None
        self._T_init: Optional[np.ndarray] = None
        self._Tdes0: Optional[np.ndarray] = None
        # Last EE goal pose used for IK, in arm-base / FK frame (4x4).
        self._last_T_goal: Optional[np.ndarray] = None
        self._status_tick = 0
        self._last_status: dict = {
            "vr_connected": False,
            "holding": False,
            "Tdes0_locked": False,
            "has_ik": None,
            "tdes_trans_norm": 0.0,
            "trel_trans_norm": 0.0,
            "pos_err": 0.0,
            "rot_err": 0.0,
            "q_err": 0.0,
            "loop_ms": 0.0,
            "mode": "idle",
            "fsm": fsm_mode,
        }

    def _apply_joint_limits(self, q: np.ndarray) -> np.ndarray:
        """Clip commanded Z1 joints to hard_dof_pos_limit."""
        return clip_dof_pos(q, self._arm_lo, self._arm_hi)

    def toggle_gripper(self) -> float:
        """Toggle commanded gripper between closed (0) and open (-1).

        Called from the hotkey thread — do **not** print here. A ``print(...,
        flush=True)`` on a background thread holds the GIL / can block on the
        TTY and stalls both the 50 Hz arm loop and async policy (seen as
        policy wait=100ms+ with cpu≪wall, right next to ``[gripper]`` lines).
        Gripper state is already visible on the 1 Hz ``arm_teleop`` status.
        """
        with self._gripper_lock:
            if abs(self._gripper_pos - self.gripper_closed) < 1e-6:
                self._gripper_pos = float(self.gripper_open)
            else:
                self._gripper_pos = float(self.gripper_closed)
            return float(self._gripper_pos)

    def gripper_command(self) -> float:
        with self._gripper_lock:
            return float(self._gripper_pos)

    def _store_T_goal(self, T_goal: np.ndarray) -> None:
        self._last_T_goal = np.asarray(T_goal, dtype=np.float64).reshape(4, 4).copy()

    def ee_goal_arm_pose(self):
        """Return ``(pos, quat_wxyz)`` of the last IK EE goal in arm-base, or ``(None, None)``."""
        if self._last_T_goal is None:
            return None, None
        T = self._last_T_goal
        pos = np.asarray(T[:3, 3], dtype=np.float64).copy()
        quat = rotmat_to_quaternion(np.asarray(T[:3, :3], dtype=np.float64))
        n = float(np.linalg.norm(quat))
        if n > 0.0:
            quat = quat / n
        return pos, quat

    def set_gains(
        self,
        kp: GainLike,
        kd: GainLike,
        gripper_kp: Optional[float] = None,
        gripper_kd: Optional[float] = None,
    ) -> None:
        """Update arm PD gains (applied immediately if already in LOWCMD)."""
        self.kp = as_gain_vector(kp, 6, "arm kp")
        self.kd = as_gain_vector(kd, 6, "arm kd")
        if gripper_kp is not None:
            self.gripper_kp = float(gripper_kp)
        if gripper_kd is not None:
            self.gripper_kd = float(gripper_kd)
        if self._ready and self.fsm_mode == "lowcmd" and self._arm is not None:
            self._apply_arm_gains()

    def _apply_arm_gains(self) -> None:
        lowcmd = self._arm._ctrlComp.lowcmd
        lowcmd.setControlGain(self.kp.tolist(), self.kd.tolist())
        if hasattr(lowcmd, "setGripperGain"):
            lowcmd.setGripperGain(self.gripper_kp, self.gripper_kd)

    def _ensure_ready(self) -> bool:
        if self._ready:
            return True
        if self._arm is None or self._arm_model is None:
            return False
        import unitree_arm_interface

        self._arm_state = unitree_arm_interface.ArmFSMState
        # loopOn() is started by Z1StateReader after VR prep release.
        self._arm.setWait(False)
        if self.fsm_mode == "lowcmd":
            # Unitree requires PASSIVE → LOWCMD; gains / tau apply only in LOWCMD.
            if hasattr(self._arm, "setFsmLowcmd"):
                self._arm.setFsmLowcmd()
            else:
                self._arm.setFsm(self._arm_state.PASSIVE)
                self._arm.setFsm(self._arm_state.LOWCMD)
            self._apply_arm_gains()
            print(
                "[Z1ArmWriter] LOWCMD gains kp="
                f"{np.array2string(self.kp, precision=1)} "
                f"kd={np.array2string(self.kd, precision=1)} "
                f"gripper=({self.gripper_kp:.1f},{self.gripper_kd:.1f})",
                flush=True,
            )
        else:
            self._arm.startTrack(self._arm_state.JOINTCTRL)
            print(
                "[Z1ArmWriter] JOINTCTRL (arm kp/kd from yaml ignored; "
                "set arm_control.mode: lowcmd to use custom gains).",
                flush=True,
            )
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
        g = self.gripper_command()
        g_s = "closed" if abs(g - self.gripper_closed) < 1e-6 else "open"
        return (
            f"fsm={self.fsm_mode} vr_connected={s.get('vr_connected')} "
            f"holding={s.get('holding')} "
            f"Tdes0={s.get('Tdes0_locked')} |Tdes|={s.get('tdes_trans_norm'):.3f} "
            f"|T_rel|={float(s.get('trel_trans_norm') or 0.0):.3f} "
            f"hasIK={ik_s} |e_pos|={float(s.get('pos_err') or 0.0):.3f} "
            f"|e_rot|={float(s.get('rot_err') or 0.0):.3f} "
            f"|dq|={float(s.get('q_err') or 0.0):.3f} "
            f"loop={float(s.get('loop_ms') or 0.0):.0f}ms "
            f"mode={s.get('mode')} gripper={g_s}({g:.1f})"
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
        """SDK analytical IK. ``has_ik`` mirrors teleop_z1 MoveL gating."""
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
        q_goal, dpose = incremental_ik_step(J, T_ee, T_goal, q_meas)
        self._last_status["pos_err"] = float(np.linalg.norm(dpose[3:6]))
        self._last_status["rot_err"] = float(np.linalg.norm(dpose[0:3]))
        return q_goal

    def _pose_error_norms(self, T_goal: np.ndarray, q_meas: np.ndarray) -> Tuple[float, float]:
        T_ee = np.asarray(
            self._arm_model.forwardKinematics(q_meas, 6),
            dtype=np.float64,
        ).reshape(4, 4)
        dpos = float(np.linalg.norm(T_goal[:3, 3] - T_ee[:3, 3]))
        # Cheap rotation magnitude via orientation_error path inside incremental helper.
        from utils.math_utils import orientation_error, rotmat_to_quaternion

        qg = rotmat_to_quaternion(T_goal[:3, :3])
        qe = rotmat_to_quaternion(T_ee[:3, :3])
        qe = qe / max(float(np.linalg.norm(qe)), 1e-12)
        drot = float(np.linalg.norm(orientation_error(qg, qe)))
        return dpos, drot

    def _vr_pose_dict(self, vr_goal) -> dict:
        client = getattr(vr_goal, "client", None)
        if client is None:
            return {}
        return client.get_latest_pose()

    def _lock_teleop_baselines(self, Tdes: np.ndarray, q_meas: np.ndarray) -> None:
        """Engage teleop like teleop_z1: T_init = FK(now), Tdes0 = Tdes(now).

        Do NOT use z1_robot_prep's server T_init here. That pose is from prep
        time; locking it with a later Tdes0 makes T_goal = T_init_prep @ T_rel
        so "draw back to zero" aims at the prep pose while the arm may be
        elsewhere — feels permanently stuck off the tracker null.
        """
        self._T_init = np.asarray(
            self._arm_model.forwardKinematics(q_meas, 6),
            dtype=np.float64,
        ).reshape(4, 4)
        self._Tdes0 = np.asarray(Tdes, dtype=np.float64).reshape(4, 4).copy()
        self._last_status["Tdes0_locked"] = True
        print(
            "[Z1ArmWriter] Locked teleop baselines at current EE "
            "(T_init=FK(q_meas), Tdes0=Tdes).",
            flush=True,
        )

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

        if not connected or Tdes is None:
            # Keep baselines across brief dropouts so drawing the tracker back
            # still reduces T_rel toward I and pulls the arm with it.
            self._last_status["mode"] = "hold_no_vr"
            return self._q_hold if self._q_hold is not None else q_meas.copy()

        Tdes = np.asarray(Tdes, dtype=np.float64).reshape(4, 4)
        if self._Tdes0 is None or self._T_init is None:
            self._lock_teleop_baselines(Tdes, q_meas)

        T_rel = np.linalg.inv(self._Tdes0) @ Tdes
        T_goal = self._T_init @ T_rel
        self._store_T_goal(T_goal)
        self._last_status["trel_trans_norm"] = float(
            np.linalg.norm(T_rel[:3, 3])
        )
        # Pose-error norms are display-only; skip most ticks (FK every cycle
        # is unnecessary for JOINTCTRL).
        if (self._status_tick % 50) == 0:
            pos_err, rot_err = self._pose_error_norms(T_goal, q_meas)
            self._last_status["pos_err"] = pos_err
            self._last_status["rot_err"] = rot_err
        self._status_tick += 1

        # Match teleop_z1 when hasIK. When analytical fails, do NOT keep full-pose
        # incremental (that walks elbow→0 and then "draw back" cannot recover).
        # Retry with goal position + *current* EE orientation (position-only
        # teleop). Low-manip poses with hasIK=True are still accepted unchanged.
        # Cap per-tick joint step so pos_only → full-pose re-engage (or a
        # distant analytical solution) cannot snap |dq|~0.4+ in one packet.
        dq_max = 0.25

        q_ana = self._solve_ik(T_goal, q_meas)
        if q_ana is not None:
            q_cmd = q_meas + np.clip(q_ana - q_meas, -dq_max, dq_max)
            self._last_status["mode"] = "tdes_ana"
        else:
            T_ee = np.asarray(
                self._arm_model.forwardKinematics(q_meas, 6),
                dtype=np.float64,
            ).reshape(4, 4)
            T_pos_only = T_ee.copy()
            T_pos_only[:3, 3] = T_goal[:3, 3]
            q_pos = self._solve_ik(T_pos_only, q_meas)
            if q_pos is not None:
                # has_ik left False from full-pose fail; position-only is OK.
                self._last_status["has_ik"] = False
                q_cmd = q_meas + np.clip(q_pos - q_meas, -dq_max, dq_max)
                self._last_status["mode"] = "tdes_pos_only"
            else:
                # Last resort: incremental on position-only target (drot≈0).
                q_ik = self._solve_ik_incremental(T_pos_only, q_meas)
                if q_ik is None:
                    self._last_status["mode"] = "hold_ik_fail"
                    return q_meas.copy()
                q_cmd = q_meas + np.clip(q_ik - q_meas, -dq_max, dq_max)
                self._last_status["mode"] = "tdes_inc_pos"

        self._last_status["q_err"] = float(np.linalg.norm(q_cmd - q_meas))
        return q_cmd

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
        self._store_T_goal(T_goal)
        q_ik = self._solve_ik(T_goal, q_meas)
        if q_ik is None:
            return self._q_hold if self._q_hold is not None else q_meas.copy()
        return q_ik

    def _cmd_from_act(self, act_goal, q_meas: np.ndarray) -> np.ndarray:
        """Arm-base IK on ACT ``ee_goal_local_cart`` / ``ee_goal_quat_base``.

        Holds last joint command until the first valid ACT goal arrives; then
        holds last goal between ACT updates (provider already hold-last).
        """
        latest = None
        if act_goal is not None and hasattr(act_goal, "latest_arm_goal"):
            latest = act_goal.latest_arm_goal()
        self._last_status.update(
            {
                "vr_connected": False,
                "holding": latest is None,
                "mode": "hold_no_act" if latest is None else "act",
                "has_ik": None,
            }
        )
        if latest is None:
            return self._q_hold if self._q_hold is not None else q_meas.copy()

        cart = np.asarray(latest["ee_goal_local_cart"], dtype=np.float64).reshape(3)
        quat = np.asarray(latest["ee_goal_quat_base"], dtype=np.float64).reshape(4)
        qn = float(np.linalg.norm(quat))
        quat = (
            quat / qn
            if qn > 1e-12
            else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        )
        T_goal = pos_quat_to_homogeneous(cart, quat)
        self._store_T_goal(T_goal)
        if (self._status_tick % 50) == 0:
            pos_err, rot_err = self._pose_error_norms(T_goal, q_meas)
            self._last_status["pos_err"] = pos_err
            self._last_status["rot_err"] = rot_err
        self._status_tick += 1

        dq_max = 0.25
        q_ana = self._solve_ik(T_goal, q_meas)
        if q_ana is not None:
            q_cmd = q_meas + np.clip(q_ana - q_meas, -dq_max, dq_max)
            self._last_status["mode"] = "act_ana"
        else:
            T_ee = np.asarray(
                self._arm_model.forwardKinematics(q_meas, 6),
                dtype=np.float64,
            ).reshape(4, 4)
            T_pos_only = T_ee.copy()
            T_pos_only[:3, 3] = T_goal[:3, 3]
            q_pos = self._solve_ik(T_pos_only, q_meas)
            if q_pos is not None:
                self._last_status["has_ik"] = False
                q_cmd = q_meas + np.clip(q_pos - q_meas, -dq_max, dq_max)
                self._last_status["mode"] = "act_pos_only"
            else:
                q_ik = self._solve_ik_incremental(T_pos_only, q_meas)
                if q_ik is None:
                    self._last_status["mode"] = "hold_ik_fail"
                    return q_meas.copy()
                q_cmd = q_meas + np.clip(q_ik - q_meas, -dq_max, dq_max)
                self._last_status["mode"] = "act_inc_pos"
        self._last_status["q_err"] = float(np.linalg.norm(q_cmd - q_meas))
        return q_cmd

    def send(
        self,
        gripper_q: float,
        robot,
        vr_goal,
        obs_builder,
        phase_variable: float = 0.0,
    ) -> None:
        del obs_builder  # unused on hot path
        if not self._ensure_ready():
            return

        t_send0 = time.perf_counter()
        q_meas = np.asarray(robot.arm_q, dtype=np.float64).reshape(6)
        if self._q_hold is None:
            self._q_hold = q_meas.copy()

        t_goal_before = self._last_T_goal
        if self.control_mode == "tdes":
            q_cmd = self._cmd_from_tdes(vr_goal, q_meas)
            # teleop_z1 does not apply training hard_dof clips on arm cmd.
            grip_cmd = self.gripper_command()
        elif self.control_mode == "act":
            q_cmd = self._cmd_from_act(vr_goal, q_meas)
            q_cmd = self._apply_joint_limits(q_cmd)
            # Prefer ACT gripper; fall back to space-latch / explicit arg.
            grip_cmd = getattr(vr_goal, "gripper_cmd", None)
            if grip_cmd is None or not np.isfinite(float(grip_cmd)):
                grip_cmd = (
                    float(gripper_q)
                    if gripper_q is not None and np.isfinite(float(gripper_q))
                    else self.gripper_command()
                )
            else:
                grip_cmd = float(grip_cmd)
                with self._gripper_lock:
                    self._gripper_pos = grip_cmd
        else:
            q_cmd = self._cmd_from_wbc(robot, vr_goal, phase_variable, q_meas)
            q_cmd = self._apply_joint_limits(q_cmd)
            grip_cmd = self.gripper_command()
        t_ik = time.perf_counter()

        self._q_hold = q_cmd.copy()
        # If this step did not compute a new T_goal (hold / homing), use FK of q_cmd.
        if self._last_T_goal is t_goal_before or self._last_T_goal is None:
            self._store_T_goal(
                np.asarray(
                    self._arm_model.forwardKinematics(q_cmd, 6),
                    dtype=np.float64,
                ).reshape(4, 4)
            )
        qd_cmd = self._qd_zero
        tau_f = self._arm_model.inverseDynamics(q_cmd, qd_cmd, self._qdd_zero, self._ftip_zero)
        # Match teleop_z1_vive_ws / z1_robot_prep: assign arm.q/qd/tau then setArmCmd.
        self._arm.q = q_cmd
        self._arm.qd = qd_cmd
        self._arm.tau = tau_f
        self._arm.setArmCmd(_vec6(q_cmd), _vec6(qd_cmd), _vec6(tau_f))
        self._arm.setGripperCmd(float(grip_cmd), 0.0, 0.0)
        t_cmd = time.perf_counter()
        self._last_status["arm_ik_ms"] = (t_ik - t_send0) * 1000.0
        self._last_status["arm_cmd_ms"] = (t_cmd - t_ik) * 1000.0

def leg_targets_from_action(
    action: np.ndarray,
    default_leg: np.ndarray,
    action_scale: list,
    *,
    clip_actions: float = 20.0,
) -> np.ndarray:
    scale = np.asarray(action_scale[:12], dtype=np.float64)
    act = np.asarray(action[:12], dtype=np.float64)
    clip = float(clip_actions)
    if clip > 0:
        act = np.clip(act, -clip, clip)
    return default_leg[:12] + scale * act


def _sleep_dt(dt: float, shutdown: dict) -> None:
    end = time.perf_counter() + max(0.0, dt)
    while not shutdown["stop"] and time.perf_counter() < end:
        time.sleep(min(0.05, end - time.perf_counter()))


def _vr_pose_for_obs(vr_goal) -> Optional[dict]:
    client = getattr(vr_goal, "client", None)
    if client is None:
        return None
    return client.get_latest_pose()


def run_z1_act_arm_prep(config: dict) -> None:
    """Move Z1 to arm_init_joint and release (same role as vr_websocket_server prep).

    Uses a temporary ArmInterface; must run before aggregator.start_z1_loop().
    """
    import sys as _sys

    teleop_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "z1", "teleop"
        )
    )
    act_dir = os.path.dirname(teleop_dir)
    if act_dir not in _sys.path:
        _sys.path.insert(0, act_dir)
    from teleop.z1_robot_prep import Z1RobotPrep, load_arm_init_joint

    move_duration = float(config.get("teleop_prep_move_duration", 3.0))
    connect_timeout = float(config.get("teleop_prep_connect_timeout", 10.0))
    # Prefer WBC yaml arm_init_joint; fall back to teleop loader.
    init_joints = np.asarray(
        config.get("arm_init_joint")
        if config.get("arm_init_joint") is not None
        else load_arm_init_joint(config),
        dtype=np.float64,
    ).reshape(6)
    print(
        "[act-ee] Z1 prep → arm_init_joint "
        f"{np.array2string(init_joints, precision=4)} "
        f"(duration={move_duration:.1f}s)",
        flush=True,
    )
    prep = Z1RobotPrep(
        init_joints=init_joints,
        move_duration=move_duration,
        connect_timeout=connect_timeout,
    )
    prep.prepare_and_hold()
    # Brief settle at init, then release so deploy ArmInterface can loopOn.
    time.sleep(0.3)
    prep.release_hold()
    print("[act-ee] Z1 prep released; deploy will take JOINTCTRL.", flush=True)


def run_init_deploy_sequence(
    *,
    config: dict,
    aggregator,
    vr_goal,
    obs_builder,
    async_obs: ProcessObsWorker,
    async_policy: ProcessPolicyWorker,
    lowcmd_writer: B2LowCmdWriter,
    z1_writer: Optional[Z1ArmWriter],
    q_start: np.ndarray,
    q_init: np.ndarray,
    rate: float,
    dt: float,
    shutdown: dict,
    ee_goal_publisher=None,
    hotkeys: Optional[DeployHotkeys] = None,
) -> bool:
    """Ramp to init_deploy_joint, hold while printing obs, then confirm policy.

    Returns True if user confirms policy motor output; False to keep holding init pose.
    Obs + policy run in child processes (legs stay on ramp/hold until confirm).
    """
    del obs_builder  # history / build live in async_obs child
    ramp_s = float(config.get("init_deploy_ramp_s", 5.0))
    hold_s = float(config.get("init_deploy_hold_s", 5.0))
    use_cmd = bool(config["commands"].get("use_cmd"))
    clip_actions = float(config.get("control", {}).get("clip_actions", 20.0))
    print_every = max(1, int(rate))
    action = np.zeros(12, dtype=np.float64)

    print(
        f"\nInit deploy: ramp {ramp_s:.1f}s → hold {hold_s:.1f}s at init_deploy_joint\n"
        f"  target (sim FL/FR/RL/RR): {np.array2string(q_init, precision=3)}"
    )

    def _tick_obs_policy(robot, *, debug: bool):
        nonlocal action
        if hasattr(vr_goal, "latest_arm_goal"):
            vr_goal.update(robot.ee_pos, robot.ee_quat, robot.base_quat)
            async_obs.set_act_goal(vr_goal.latest_arm_goal())
        obs_pack = async_obs.step(
            robot, _vr_pose_for_obs(vr_goal), debug=debug
        )
        action = async_policy.step(
            obs_pack["obs_vector"],
            action,
            clip_actions=clip_actions,
            on_action=async_obs.push_action,
        )
        return obs_pack, action

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
            do_print = i % print_every == 0 or i == ramp_steps - 1
            obs_pack, action = _tick_obs_policy(robot, debug=do_print)
            if z1_writer is not None:
                z1_writer.send(
                    gripper_q=robot.gripper_q,
                    robot=robot,
                    vr_goal=vr_goal,
                    obs_builder=None,
                    phase_variable=obs_pack["phase_variable"],
                )
            _maybe_publish_ee(
                ee_goal_publisher,
                obs_pack,
                None if z1_writer is None else z1_writer.gripper_command(),
                z1_writer=z1_writer,
            )
            if do_print:
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
            do_print = i % print_every == 0 or i == hold_steps - 1
            obs_pack, action = _tick_obs_policy(robot, debug=do_print)
            if z1_writer is not None:
                z1_writer.send(
                    gripper_q=robot.gripper_q,
                    robot=robot,
                    vr_goal=vr_goal,
                    obs_builder=None,
                    phase_variable=obs_pack["phase_variable"],
                )
            _maybe_publish_ee(
                ee_goal_publisher,
                obs_pack,
                None if z1_writer is None else z1_writer.gripper_command(),
                z1_writer=z1_writer,
            )
            if do_print:
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
    # Hotkeys own stdin in cbreak and would steal YES keystrokes — pause first.
    if hotkeys is not None:
        hotkeys.pause_for_input()
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
        if hotkeys is not None:
            hotkeys.resume_after_input()

    if confirm.strip() != "YES":
        print(
            f"Keeping legs at init_deploy_joint (policy commands disabled). "
            f"Got {confirm!r}; need exact YES."
        )
        return False

    print("Confirmed. Switching to policy gains and applying policy actions.")
    # Drop init ramp/hold policy leftovers so step-0 obs matches a clean episode start.
    async_obs.reset()
    print("  obs/action history reset after confirm (clean last_leg_actions).")
    lowcmd_writer.set_gains(config["control"]["kp"], config["control"]["kd"])
    return True


def _format_gains(label: str, kp: GainLike, kd: GainLike, n: int) -> str:
    kp_v = as_gain_vector(kp, n, label + " kp")
    kd_v = as_gain_vector(kd, n, label + " kd")
    if np.allclose(kp_v, kp_v[0]) and np.allclose(kd_v, kd_v[0]):
        return f"{label} Kp={kp_v[0]:.1f} Kd={kd_v[0]:.1f}"
    return (
        f"{label} Kp={np.array2string(kp_v, precision=1)} "
        f"Kd={np.array2string(kd_v, precision=1)}"
    )


def _make_z1_writer(args, aggregator, config, arm_home, dt, *, control_mode: str) -> Z1ArmWriter:
    arm_cfg = config.get("arm_control") or {}
    _, _, arm_lo, arm_hi = load_leg_arm_limit_arrays(config)
    # SDK LOWCMD defaults (Unitree command units).
    default_arm_kp = [20.0, 30.0, 30.0, 20.0, 15.0, 10.0]
    default_arm_kd = [2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0]
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
        arm_lo=arm_lo,
        arm_hi=arm_hi,
        fsm_mode=str(arm_cfg.get("mode", "jointctrl")),
        kp=arm_cfg.get("kp", default_arm_kp),
        kd=arm_cfg.get("kd", default_arm_kd),
        gripper_kp=float(arm_cfg.get("gripper_kp", 20.0)),
        gripper_kd=float(arm_cfg.get("gripper_kd", 2000.0)),
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
        help="After policy starts, update cmd_vx/vy/yaw via keyboard (w/s/a/d/q/e; 0=stop)",
    )
    parser.add_argument(
        "--act-ee",
        action="store_true",
        help=(
            "EE goals from ACT over WebSocket (not VR). Starts ActGoalWsServer; "
            "Z1 uses arm-base IK on ee_goal_local_cart/quat; runs Z1 arm_init prep "
            "(no vr_websocket_server)."
        ),
    )
    parser.add_argument(
        "--act-goal-ws-host",
        type=str,
        default="0.0.0.0",
        help="Bind host for ACT→deploy goal WebSocket (default 0.0.0.0)",
    )
    parser.add_argument(
        "--act-goal-ws-port",
        type=int,
        default=8771,
        help="Port for ACT→deploy goal WebSocket (default 8771)",
    )
    args = parser.parse_args()

    if args.arm_only and args.legs_only:
        print("Error: --arm-only and --legs-only cannot be used together.")
        return
    if args.arm_only and args.no_vr:
        print("Error: --arm-only requires VR (Tdes path). Do not pass --no-vr.")
        return
    if args.act_ee and args.arm_only:
        print("Error: --act-ee cannot be combined with --arm-only (VR Tdes).")
        return
    if args.act_ee and args.legs_only:
        print("Error: --act-ee requires Z1 arm control (do not pass --legs-only).")
        return
    # online_infer needs measured EE on :8770; always enable with --act-ee.
    if args.act_ee and not getattr(args, "ee_goal_ws", False):
        args.ee_goal_ws = True
        print("[act-ee] auto-enabling --ee-goal-ws (measured EE stream for ACT).")

    if args.act_ee:
        args.no_vr = True
        print("WARNING: --act-ee will command Z1 from ACT WebSocket goals.")
        print("  Arm: arm-base IK on ee_goal_local_cart / ee_goal_quat_base + gripper.")
        print("  Prep: move to arm_init_joint (like vr_websocket_server), no VR server.")
        print(
            f"  ACT goal WS: ws://{args.act_goal_ws_host}:{args.act_goal_ws_port}"
        )
        confirm = input("Type YES to start ACT-EE deploy: ")
        if confirm.strip() != "YES":
            print("Aborted.")
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

    # Arm-only: never release sport. Full execute / ACT-EE: defer until writers ready.
    defer_release = bool(args.execute_actions or args.arm_only or args.act_ee)
    if args.arm_only:
        args.no_release_mode = True

    # Inference always runs in ProcessPolicyWorker (child GIL). Do not load Torch
    # into the control process.
    config, aggregator, vr_goal, obs_builder, policy, rate, dt = load_runtime(
        args, defer_release=defer_release, load_policy=False
    )
    if not args.checkpoint:
        print(
            "Error: checkpoint required. Set policy.checkpoint in b2z1_wbc.yaml "
            "or pass --checkpoint /path/to/model.pt"
        )
        return
    del policy  # always None when load_policy=False

    act_goal_server = None
    ee_goal_publisher = None
    ee_pump_stop = threading.Event()
    ee_pump_thread = None

    def _start_ee_goal_ws() -> None:
        nonlocal ee_goal_publisher
        if ee_goal_publisher is not None or not getattr(args, "ee_goal_ws", False):
            return
        from streaming.ee_goal_ws_server import EeGoalWsServer

        stream_cfg = config.get("streaming") or {}
        ee_host = args.ee_goal_ws_host or stream_cfg.get("host", "0.0.0.0")
        ee_port = int(
            args.ee_goal_ws_port
            if args.ee_goal_ws_port is not None
            else stream_cfg.get("port", 8770)
        )
        ee_rate = float(
            args.ee_goal_ws_rate
            if args.ee_goal_ws_rate is not None
            else stream_cfg.get("rate_hz", 50.0)
        )
        ee_goal_publisher = EeGoalWsServer(host=ee_host, port=ee_port, rate_hz=ee_rate)
        ee_goal_publisher.start()

    def _publish_measured_ee(robot) -> None:
        """Push measured arm-base EE (+ base vel) for ACT / data_record clients."""
        if robot is None or not robot.valid:
            return
        ee_pos = np.asarray(robot.ee_pos, dtype=np.float64).reshape(3)
        ee_quat = np.asarray(robot.ee_quat, dtype=np.float64).reshape(4)
        if not (np.all(np.isfinite(ee_pos)) and np.all(np.isfinite(ee_quat))):
            return
        base_lin = np.asarray(
            robot.base_lin_vel_local, dtype=np.float64
        ).reshape(3)
        grip = float(robot.gripper_q)
        # Primary ACT path: measured EE on the same WS as goals (:8771).
        if act_goal_server is not None:
            act_goal_server.publish_ee_state(
                ee_pos,
                ee_quat,
                base_lin_vel=base_lin,
                gripper_cmd=grip,
            )
        if ee_goal_publisher is not None:
            ee_goal_publisher.update(
                ee_pos,
                gripper_cmd=grip,
                base_lin_vel=base_lin,
                ee_goal_quat_base=ee_quat,
                ee_pos=ee_pos,
                ee_quat=ee_quat,
            )

    def _ee_pump_loop() -> None:
        """Keep EE WS valid while policy/obs workers load (ACT can connect early)."""
        while not ee_pump_stop.is_set():
            try:
                _publish_measured_ee(aggregator.read())
            except Exception:
                pass
            time.sleep(0.02)

    if args.act_ee:
        from streaming.act_goal_ws_server import ActGoalWsServer
        from vr_goal.act_goal_provider import ActGoalProvider

        run_z1_act_arm_prep(config)
        aggregator.start_z1_loop()
        act_goal_server = ActGoalWsServer(
            host=str(args.act_goal_ws_host),
            port=int(args.act_goal_ws_port),
        )
        act_goal_server.start()
        # Optional :8770 + measured-EE pump (also feeds act_goal_server ee_state).
        _start_ee_goal_ws()
        ee_pump_thread = threading.Thread(
            target=_ee_pump_loop, name="ee_ws_pump", daemon=True
        )
        ee_pump_thread.start()
        arm_mount = config["robot"].get(
            "policy_arm_base_offset",
            config["robot"].get("z1_base_in_robot_base", [0.03, 0.0, 0.09]),
        )
        vr_goal = ActGoalProvider(act_goal_server, arm_mount_in_base=arm_mount)
        print(
            f"[act-ee] ACT duplex WS ws://{args.act_goal_ws_host}:{args.act_goal_ws_port} "
            "(goals in; measured EE out). Arm tracks ACT after policy YES.",
            flush=True,
        )

    default_dof = obs_builder.default_dof_pos
    action_scale = config["control"]["action_scale"]
    clip_actions = float(config["control"].get("clip_actions", 20.0))
    leg_lo, leg_hi, arm_lo, arm_hi = load_leg_arm_limit_arrays(config)
    arm_home = clip_dof_pos(
        np.array(
            config.get("arm_init_joint", config["arm_home_joint"]),
            dtype=np.float64,
        ),
        arm_lo,
        arm_hi,
    )
    q_init = clip_dof_pos(init_deploy_joint_from_config(config), leg_lo, leg_hi)

    lowcmd_writer = None
    z1_writer = None
    hotkeys = None
    # Dry run: evaluate policy (no lowcmd). execute: gated by init confirm.
    # arm-only: never send policy leg targets.
    apply_policy = not args.execute_actions
    shutdown = {"stop": False}

    _n_thr = int((config.get("policy") or {}).get("num_threads", 1) or 1)
    _dev = str((config.get("policy") or {}).get("device", "cpu"))
    async_policy = ProcessPolicyWorker(
        checkpoint=str(args.checkpoint),
        device=_dev,
        num_threads=_n_thr,
    )
    async_policy.start()
    print(
        f"[policy] process worker starting (device={_dev}, threads={_n_thr})...",
        flush=True,
    )
    try:
        async_policy.wait_ready(timeout=120.0)
    except TimeoutError as exc:
        print(f"Error: {exc}")
        ee_pump_stop.set()
        async_policy.stop()
        if act_goal_server is not None:
            act_goal_server.stop()
        if ee_goal_publisher is not None:
            ee_goal_publisher.stop()
        vr_goal.stop()
        aggregator.close()
        return
    print(
        "[policy] process worker ready (separate GIL; control process has no Torch).",
        flush=True,
    )

    async_obs = ProcessObsWorker(
        config,
        control_dt=dt,
        use_vr=(not bool(args.no_vr)) and (not bool(args.act_ee)),
        use_act=bool(args.act_ee),
    )
    async_obs.start()
    print("[obs] process worker starting...", flush=True)
    try:
        async_obs.wait_ready(timeout=60.0)
    except TimeoutError as exc:
        print(f"Error: {exc}")
        ee_pump_stop.set()
        async_obs.stop()
        async_policy.stop()
        if act_goal_server is not None:
            act_goal_server.stop()
        if ee_goal_publisher is not None:
            ee_goal_publisher.stop()
        vr_goal.stop()
        aggregator.close()
        return
    print(
        "[obs] process worker ready (ObservationBuilder in child; own GIL).",
        flush=True,
    )

    def _request_shutdown(signum, _frame) -> None:
        shutdown["stop"] = True
        print(f"\nSignal {signum} received; stopping...", flush=True)

    signal.signal(signal.SIGINT, _request_shutdown)
    signal.signal(signal.SIGTERM, _request_shutdown)

    # Non-ACT paths still start EE WS here (after workers).
    _start_ee_goal_ws()

    def _start_gripper_hotkeys() -> None:
        nonlocal hotkeys
        if hotkeys is not None or z1_writer is None:
            return
        hotkeys = DeployHotkeys(
            config.get("keyboard_vel", {}),
            enable_vel=False,
            on_gripper_toggle=z1_writer.toggle_gripper,
        )
        hotkeys.start()

    if args.arm_only:
        print("Starting Z1 arm writer (teleop Tdes mode); sport mode untouched...")
        z1_writer = _make_z1_writer(
            args, aggregator, config, arm_home, dt, control_mode="tdes"
        )
        _start_gripper_hotkeys()
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
            async_obs.stop()
            async_policy.stop()
            if act_goal_server is not None:
                act_goal_server.stop()
            vr_goal.stop()
            aggregator.close()
            return
        if q_start is None:
            print("Error: no valid B2 state within 30 s; aborting.")
            async_obs.stop()
            async_policy.stop()
            if act_goal_server is not None:
                act_goal_server.stop()
            vr_goal.stop()
            aggregator.close()
            return

        print(f"  current leg_q (sim): {np.array2string(q_start, precision=3)}")
        print(f"  init_deploy_joint  : {np.array2string(q_init, precision=3)}")
        startup_kp = config["control"].get("startup_kp", 1000.0)
        startup_kd = config["control"].get("startup_kd", 10.0)
        print("Starting lowcmd writer (sport mode still active)...")
        lowcmd_writer = B2LowCmdWriter(
            kp=startup_kp,
            kd=startup_kd,
            initial_targets_sim=q_start,
            leg_lo=leg_lo,
            leg_hi=leg_hi,
        )
        time.sleep(0.1)
        if not args.no_release_mode:
            print("Releasing sport mode...")
            aggregator.release_motion_mode()
        if not args.legs_only:
            arm_mode = "act" if args.act_ee else "tdes"
            print(
                f"Starting Z1 arm writer "
                f"({'ACT arm-base IK' if arm_mode == 'act' else 'teleop Tdes'} mode)..."
            )
            z1_writer = _make_z1_writer(
                args, aggregator, config, arm_home, dt, control_mode=arm_mode
            )
            if not args.act_ee:
                _start_gripper_hotkeys()

        apply_policy = run_init_deploy_sequence(
            config=config,
            aggregator=aggregator,
            vr_goal=vr_goal,
            obs_builder=obs_builder,
            async_obs=async_obs,
            async_policy=async_policy,
            lowcmd_writer=lowcmd_writer,
            z1_writer=z1_writer,
            q_start=q_start,
            q_init=q_init,
            rate=rate,
            dt=dt,
            shutdown=shutdown,
            ee_goal_publisher=ee_goal_publisher,
            hotkeys=hotkeys,
        )
        # ACT EE tracking only after policy YES (arm holds init through ramp/hold).
        if args.act_ee and apply_policy and hasattr(vr_goal, "set_enabled"):
            vr_goal.set_enabled(True)
        elif args.act_ee and not apply_policy:
            print(
                "[act-ee] policy not confirmed — ACT EE goals stay disabled; "
                "arm holds init.",
                flush=True,
            )
    elif args.act_ee:
        # Dry-run / observe: no init YES — enable ACT EE immediately.
        print("Starting Z1 arm writer (ACT arm-base IK; legs not commanded)...")
        z1_writer = _make_z1_writer(
            args, aggregator, config, arm_home, dt, control_mode="act"
        )
        if hasattr(vr_goal, "set_enabled"):
            vr_goal.set_enabled(True)

    # Enable velocity keys on the same stdin listener (space remains gripper-only).
    if args.vel_keyboard and apply_policy and not args.arm_only:
        obs_builder.use_cmd = True
        config.setdefault("commands", {})["use_cmd"] = True

        def _on_vel(vx: float, vy: float, yaw: float) -> None:
            obs_builder.set_commands(vx, vy, yaw)
            async_obs.set_commands(vx, vy, yaw)

        if hotkeys is None:
            hotkeys = DeployHotkeys(
                config.get("keyboard_vel", {}),
                enable_vel=True,
                on_vel_update=_on_vel,
                on_gripper_toggle=(
                    z1_writer.toggle_gripper if z1_writer is not None else None
                ),
            )
            hotkeys.start()
        else:
            hotkeys.set_velocity_enabled(True, on_vel_update=_on_vel)
    elif args.vel_keyboard and args.arm_only:
        print("[hotkeys] velocity skipped (--arm-only; no leg policy commands).")
    elif args.vel_keyboard and not apply_policy:
        print("[hotkeys] velocity skipped (policy output not confirmed).")

    print("\nB2Z1 WBC deploy")
    print(f"  execute_actions: {args.execute_actions}")
    print(f"  arm_only       : {args.arm_only}")
    print(f"  act_ee         : {args.act_ee}")
    # Policy net always runs via ProcessPolicyWorker. apply_policy only gates
    # whether leg targets are sent to lowcmd (never for --arm-only).
    print(
        f"  policy_eval    : process workers every tick "
        f"(obs child + policy device="
        f"{(config.get('policy') or {}).get('device', 'cpu')}, "
        f"threads={(config.get('policy') or {}).get('num_threads', 1)})"
    )
    print(
        f"  apply_leg_cmd  : {bool(apply_policy and not args.arm_only)} "
        f"({'sport mode; actions discarded' if args.arm_only else 'lowcmd from policy' if apply_policy else 'hold init'})"
    )
    print(
        f"  hotkeys        : "
        f"gripper={'on' if (z1_writer is not None and not args.act_ee) else 'off'} "
        f"vel={'on' if (hotkeys is not None and hotkeys.enable_vel) else 'off'}"
    )
    if args.execute_actions and not args.arm_only:
        print(f"  legs_only      : {args.legs_only}")
        print(
            f"  gains          : "
            f"{_format_gains('startup', config['control'].get('startup_kp', 1000.0), config['control'].get('startup_kd', 10.0), 12)}; "
            f"{_format_gains('policy', config['control']['kp'], config['control']['kd'], 12)}"
        )
        arm_cfg = config.get("arm_control") or {}
        if not args.legs_only:
            print(
                f"  arm_gains      : mode={arm_cfg.get('mode', 'jointctrl')} "
                f"{_format_gains('arm', arm_cfg.get('kp', [20, 30, 30, 20, 15, 10]), arm_cfg.get('kd', 2000), 6)}"
            )
        if config.get("shutdown_joint") is not None:
            print(
                f"  soft_shutdown  : ramp {float(config.get('shutdown_ramp_s', 3.0)):.1f}s "
                f"+ hold {float(config.get('shutdown_hold_s', 0.5)):.1f}s → damping"
            )
    if args.arm_only:
        print("  arm_control    : Tdes (T_goal = T_init @ inv(Tdes0) @ Tdes)")
        print("  legs           : sport mode left alone")
        arm_cfg = config.get("arm_control") or {}
        print(
            f"  arm_gains      : mode={arm_cfg.get('mode', 'jointctrl')} "
            f"{_format_gains('arm', arm_cfg.get('kp', [20, 30, 30, 20, 15, 10]), arm_cfg.get('kd', 2000), 6)}"
        )
    elif args.act_ee:
        print(
            "  arm_control    : ACT arm-base IK "
            "(ee_goal_local_cart / ee_goal_quat_base + gripper)"
        )
        print(
            f"  ACT goal WS    : ws://{args.act_goal_ws_host}:{args.act_goal_ws_port} "
            "(online_infer → deploy)"
        )
    elif args.execute_actions and not args.legs_only:
        print("  arm_control    : Tdes (T_goal = T_init @ inv(Tdes0) @ Tdes)")
    print(f"  checkpoint     : {args.checkpoint}")
    print(f"  rate           : {rate} Hz")
    print(
        "  hard_dof_limit : legs+arm clamped to control.hard_dof_pos_limit "
        f"(elbow {arm_lo[2]:.2f}..{arm_hi[2]:.2f})"
    )
    if args.act_ee:
        print("  VR server      : disabled (--act-ee; no vr_websocket_server)")
    elif args.no_vr:
        print("  VR server      : disabled (--no-vr); EE goal held at current pose")
    else:
        print(f"  VR server      : {config['vr']['host']}:{config['vr']['port']}")
    if ee_goal_publisher is not None:
        print(
            f"  EE goal WS     : ws://{ee_goal_publisher.host}:{ee_goal_publisher.port} "
            f"@ {ee_goal_publisher.rate_hz:.1f} Hz"
        )
    _print_command_mode(config)
    print("Press Ctrl+C to stop.\n")

    step = 0
    max_work_ms = 0.0
    max_stages_ms = {}
    action = np.zeros(12, dtype=np.float64)
    # Alloc churn is reduced; disable GC in the control process so rare full
    # collections cannot hold the GIL for 50–150 ms mid-run. Policy child has
    # its own GC. RSS should stay flat (was ~450MB steady, not growing).
    import gc as _gc
    import resource as _resource

    from policy.constants import num_observations as _num_observations

    _gc.collect()
    _gc.disable()
    _last_lag_note_t = 0.0
    print_hz_divider = max(1, int(rate))
    # Arm-only test: skip WBC obs on the 50 Hz teleop hot path; rebuild at 1 Hz
    # for prints and feed the child a cached vector at 25 Hz for load only.
    # Full control: fresh obs every tick (no cache).
    _load_sim_obs = np.zeros(_num_observations, dtype=np.float64)
    # robot.ee_pos is robot-base (mount + FK); recorded/IK goals are arm-base.
    _arm_mount = np.asarray(
        config["robot"].get(
            "policy_arm_base_offset",
            config["robot"]["z1_base_in_robot_base"],
        ),
        dtype=np.float64,
    ).reshape(3)
    _last_obs_pack = {
        "phase_variable": 0.0,
        "obs_vector": _load_sim_obs,
        "slices": {},
        "ee_goal_world": np.zeros(3),
        "ee_goal_base": np.zeros(3),
        "use_cmd": False,
        "commands": np.zeros(3),
        "dvel_b_local": np.zeros(3),
        "base_lin_vel": np.zeros(3, dtype=np.float64),
        "ee_pos": np.full(3, np.nan, dtype=np.float64),
        "ee_quat": np.full(4, np.nan, dtype=np.float64),
    }
    # Main loop publishes EE; stop the startup pump to avoid double-read.
    ee_pump_stop.set()
    try:
        while not shutdown["stop"]:
            t0 = time.perf_counter()
            robot = aggregator.read()
            t_read = time.perf_counter()
            if not robot.valid:
                time.sleep(0.05)
                continue

            do_print = step % print_hz_divider == 0
            policy_stage_ms = 0.0
            leg_q_target = q_init

            if args.arm_only:
                if z1_writer is not None:
                    z1_writer.send(
                        gripper_q=robot.gripper_q,
                        robot=robot,
                        vr_goal=vr_goal,
                        obs_builder=None,
                        phase_variable=0.0,
                    )
                t_arm = time.perf_counter()

                # Arm-only: obs child at 25 Hz for load; arm never waits on build.
                obs_ms = 0.0
                if do_print or step % 2 == 0:
                    t_ob0 = time.perf_counter()
                    obs_pack = async_obs.step(
                        robot,
                        _vr_pose_for_obs(vr_goal),
                        debug=bool(do_print and args.verbose),
                    )
                    _load_sim_obs = np.asarray(
                        obs_pack["obs_vector"], dtype=np.float64
                    ).reshape(-1)
                    _last_obs_pack = obs_pack
                    obs_ms = (time.perf_counter() - t_ob0) * 1000.0
                    action = async_policy.step(
                        _load_sim_obs,
                        action,
                        clip_actions=clip_actions,
                        on_action=async_obs.push_action,
                    )
                    policy_stage_ms = float(async_policy.last_infer_ms)
                else:
                    obs_pack = _last_obs_pack

                # Refresh vel/measured EE every tick. Subtract mount so ee_pos
                # is arm-base (matches IK goal), not robot-base.
                pub_pack = dict(obs_pack)
                pub_pack["base_lin_vel"] = np.asarray(
                    robot.base_lin_vel_local, dtype=np.float64
                ).copy()
                pub_pack["ee_pos"] = (
                    np.asarray(robot.ee_pos, dtype=np.float64).reshape(3)
                    - _arm_mount
                )
                pub_pack["ee_quat"] = np.asarray(
                    robot.ee_quat, dtype=np.float64
                ).reshape(4).copy()
                _maybe_publish_ee(
                    ee_goal_publisher,
                    pub_pack,
                    None if z1_writer is None else z1_writer.gripper_command(),
                    z1_writer=z1_writer,
                )

                work_ms = (time.perf_counter() - t0) * 1000.0
                stages_ms = {
                    "read": (t_read - t0) * 1000.0,
                    "arm": (t_arm - t_read) * 1000.0,
                    "obs": obs_ms,
                    "policy": policy_stage_ms,
                    "infer": policy_stage_ms,
                    "obs_build": float(async_obs.last_build_ms),
                }
                teleop_ms = stages_ms["read"] + stages_ms["arm"]
            else:
                # Full control: obs + policy in children (non-blocking take).
                # Parent only packs state / applies last-ready action.
                if args.act_ee:
                    vr_goal.update(
                        robot.ee_pos, robot.ee_quat, robot.base_quat
                    )
                    async_obs.set_act_goal(vr_goal.latest_arm_goal())
                t_ob0 = time.perf_counter()
                obs_pack = async_obs.step(
                    robot, _vr_pose_for_obs(vr_goal), debug=do_print
                )
                t_obs = time.perf_counter()

                action = async_policy.step(
                    obs_pack["obs_vector"],
                    action,
                    clip_actions=clip_actions,
                    on_action=async_obs.push_action,
                )
                policy_stage_ms = float(async_policy.last_infer_ms)
                if apply_policy:
                    leg_q_target = leg_targets_from_action(
                        action, default_dof, action_scale, clip_actions=clip_actions
                    )
                t_policy_pre = time.perf_counter()

                if z1_writer is not None:
                    z1_writer.send(
                        gripper_q=robot.gripper_q,
                        robot=robot,
                        vr_goal=vr_goal,
                        obs_builder=None,
                        phase_variable=obs_pack["phase_variable"],
                    )
                t_arm = time.perf_counter()

                # Measured EE → ACT (:8771) and optional data_record EE WS (:8770).
                _publish_measured_ee(robot)
                pub_pack = dict(obs_pack)
                pub_pack["base_lin_vel"] = np.asarray(
                    robot.base_lin_vel_local, dtype=np.float64
                ).copy()
                pub_pack["ee_pos"] = (
                    np.asarray(robot.ee_pos, dtype=np.float64).reshape(3)
                    - _arm_mount
                )
                pub_pack["ee_quat"] = np.asarray(
                    robot.ee_quat, dtype=np.float64
                ).reshape(4).copy()
                _maybe_publish_ee(
                    ee_goal_publisher,
                    pub_pack,
                    None if z1_writer is None else z1_writer.gripper_command(),
                    z1_writer=z1_writer,
                )
                if args.execute_actions and lowcmd_writer is not None:
                    lowcmd_writer.set_leg_targets(leg_q_target)

                work_ms = (time.perf_counter() - t0) * 1000.0
                stages_ms = {
                    "read": (t_read - t0) * 1000.0,
                    # Parent IPC submit/take only (child build overlapped).
                    "obs": (t_obs - t_ob0) * 1000.0,
                    "policy": (t_policy_pre - t_obs) * 1000.0,
                    "arm": (t_arm - t_policy_pre) * 1000.0,
                    "infer": policy_stage_ms,
                    "obs_build": float(async_obs.last_build_ms),
                }
                teleop_ms = work_ms

            if teleop_ms >= max_work_ms:
                max_work_ms = teleop_ms
                max_stages_ms = dict(stages_ms)
            if z1_writer is not None:
                # For arm-only, loop_ms = arm-critical path (read+arm), not obs.
                z1_writer._last_status["loop_ms"] = (
                    teleop_ms if args.arm_only else work_ms
                )

            # Cap print rate: at most 1 Hz for status (do_print computed pre-build).
            if do_print:
                if args.verbose or not args.arm_only:
                    print_obs_summary(
                        step,
                        obs_pack,
                        action,
                        use_cmd=bool(
                            obs_builder.use_cmd or config["commands"].get("use_cmd")
                        ),
                    )
                    if args.arm_only:
                        print("  leg_cmd        : sport mode (no lowcmd; --arm-only)")
                        print(
                            "  policy         : process-worker "
                            f"(last_infer={policy_stage_ms:.2f}ms, legs not applied)"
                        )
                    else:
                        print(
                            "  policy         : process-worker "
                            f"(last_infer={async_policy.last_infer_ms:.2f}ms "
                            f"wait={async_policy.last_wait_ms:.2f}ms)"
                        )
                        if args.execute_actions and not apply_policy:
                            print(
                                "  leg_cmd        : holding init_deploy_joint "
                                "(policy not applied)"
                            )
                if z1_writer is not None:
                    print(f"  arm_teleop     : {z1_writer.status_line()}")
                overrun = step > 0 and max_work_ms > dt * 1000.0 * 1.5
                if overrun:
                    rss_mb = _resource.getrusage(
                        _resource.RUSAGE_SELF
                    ).ru_maxrss / 1024.0
                    if z1_writer is not None:
                        ik_ms = float(z1_writer._last_status.get("arm_ik_ms") or 0.0)
                        cmd_ms = float(z1_writer._last_status.get("arm_cmd_ms") or 0.0)
                        print(
                            f"  arm_lag_hint   : now={teleop_ms:.0f}ms "
                            f"max={max_work_ms:.0f}ms > dt={dt*1000.0:.0f}ms "
                            f"(read={max_stages_ms.get('read', 0):.1f} "
                            f"arm={max_stages_ms.get('arm', 0):.1f} "
                            f"ik={ik_ms:.1f} cmd={cmd_ms:.1f} | "
                            f"obs_ipc={max_stages_ms.get('obs', 0):.1f} "
                            f"obs_build={max_stages_ms.get('obs_build', 0):.1f} "
                            f"pol_ipc={max_stages_ms.get('policy', 0):.1f} "
                            f"infer={max_stages_ms.get('infer', policy_stage_ms):.1f} ms) "
                            f"rss={rss_mb:.0f}MB."
                        )
                    else:
                        print(
                            f"  loop_lag_hint  : now={teleop_ms:.0f}ms "
                            f"max={max_work_ms:.0f}ms > dt={dt*1000.0:.0f}ms "
                            f"(read={max_stages_ms.get('read', 0):.1f} "
                            f"obs_ipc={max_stages_ms.get('obs', 0):.1f} "
                            f"obs_build={max_stages_ms.get('obs_build', 0):.1f} "
                            f"pol_ipc={max_stages_ms.get('policy', 0):.1f} "
                            f"infer={max_stages_ms.get('infer', policy_stage_ms):.1f} ms) "
                            f"rss={rss_mb:.0f}MB."
                        )
                    now_t = time.perf_counter()
                    if now_t - _last_lag_note_t >= 5.0:
                        _last_lag_note_t = now_t
                        print(
                            f"  policy_timing  : wall={async_policy.last_infer_ms:.2f}ms "
                            f"cpu={async_policy.last_cpu_ms:.2f}ms "
                            f"wait={async_policy.last_wait_ms:.2f}ms "
                            f"(child process; overlapped with control)."
                        )
                max_work_ms = 0.0
                max_stages_ms = {}
                if z1_writer is not None:
                    mode = z1_writer._last_status.get("mode")
                    if mode == "hold_no_vr":
                        print(
                            "  arm_stuck_hint: VR connected=False or Tdes=None "
                            "(tracker/websocket)."
                        )
                    elif mode == "tdes_pos_only":
                        print(
                            "  arm_note       : full-pose hasIK=False; tracking "
                            "position only (keep current EE orientation)."
                        )
                    elif mode == "tdes_inc_pos":
                        print(
                            "  arm_note       : full+pos IK failed; incremental "
                            "on position-only target."
                        )
                if hotkeys is not None and hotkeys.enable_vel:
                    print(
                        f"  keyboard_cmd   : "
                        f"vx={async_obs._commands[0]:.2f} "
                        f"vy={async_obs._commands[1]:.2f} "
                        f"yaw={async_obs._commands[2]:.2f}"
                    )

            step += 1
            _sleep_dt(dt - (time.perf_counter() - t0), shutdown)
    except KeyboardInterrupt:
        shutdown["stop"] = True
        print("\nStopped.")
    finally:
        _gc.enable()
        if async_obs is not None:
            async_obs.stop()
        if async_policy is not None:
            async_policy.stop()
        if act_goal_server is not None:
            act_goal_server.stop()
        if hotkeys is not None:
            hotkeys.stop()
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
                    kp=config.get("shutdown_kp", config["control"]["kp"]),
                    kd=config.get("shutdown_kd", config["control"]["kd"]),
                )
            except Exception as exc:
                print(f"[safety] Soft shutdown failed ({exc}); stopping lowcmd.", flush=True)
            print("Stopping B2 lowcmd writer (damping)...", flush=True)
            lowcmd_writer.stop()
        ee_pump_stop.set()
        if ee_goal_publisher is not None:
            ee_goal_publisher.stop()
        vr_goal.stop()
        aggregator.close()


if __name__ == "__main__":
    main()
