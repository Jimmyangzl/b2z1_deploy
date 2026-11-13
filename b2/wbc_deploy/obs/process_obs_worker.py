"""ObservationBuilder in a child process (own GIL).

Control process packs robot + VR pose into shared memory; child builds the
WBC obs vector. Parent uses last-ready obs (non-blocking) so a slow build
cannot stretch the 50 Hz send loop.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import numpy as np

from policy.constants import num_observations, num_proprio
from vr_goal.process_vr_client import _POSE_FLOATS, _pack_pose, _unpack_pose

# --- input layout (float64) ---
_IN_VALID = 0
_IN_ARM_CONN = 1
_IN_GRIPPER = 2
_IN_TIMESTAMP = 3
_IN_BASE_QUAT = 4  # 4
_IN_BASE_LIN = 8  # 3
_IN_BASE_ANG = 11  # 3
_IN_LEG_Q = 14  # 12
_IN_LEG_DQ = 26  # 12
_IN_ARM_Q = 38  # 6
_IN_ARM_DQ = 44  # 6
_IN_EE_POS = 50  # 3
_IN_EE_QUAT = 53  # 4
_IN_FOOT = 57  # 4
_IN_GRAV = 61  # 3
_IN_HAS_JAC = 64
_IN_JAC = 65  # 36 (6x6)
_IN_ROBOT_END = 101
_IN_VR = _IN_ROBOT_END  # 54
_IN_DEBUG = _IN_VR + _POSE_FLOATS
_IN_HAS_ACTION = _IN_DEBUG + 1
_IN_RESET = _IN_HAS_ACTION + 1
_IN_USE_CMD = _IN_RESET + 1
_IN_CMD = _IN_USE_CMD + 1  # 3
_IN_ACTION = _IN_CMD + 3  # 12
_IN_HAS_ACT = _IN_ACTION + 12
_IN_ACT_CART = _IN_HAS_ACT + 1  # 3 arm-base
_IN_ACT_QUAT = _IN_ACT_CART + 3  # 4
_IN_ACT_GRIP = _IN_ACT_QUAT + 4
_IN_FLOATS = _IN_ACT_GRIP + 1  # 183

# --- output layout ---
_OUT_OBS = 0  # num_observations
_OUT_PROPRIO = _OUT_OBS + num_observations  # num_proprio
_OUT_PHASE = _OUT_PROPRIO + num_proprio
_OUT_EE_GOAL_LOCAL = _OUT_PHASE + 1  # 3
_OUT_EE_GOAL_QUAT = _OUT_EE_GOAL_LOCAL + 3  # 4
_OUT_EE_GOAL_BASE = _OUT_EE_GOAL_QUAT + 4  # 3
_OUT_COMMANDS = _OUT_EE_GOAL_BASE + 3  # 3
_OUT_DVEL = _OUT_COMMANDS + 3  # 3
_OUT_BASE_LIN = _OUT_DVEL + 3  # 3
_OUT_EE_POS = _OUT_BASE_LIN + 3  # 3
_OUT_EE_QUAT = _OUT_EE_POS + 3  # 4
_OUT_USE_CMD = _OUT_EE_QUAT + 4
_OUT_DIST = _OUT_USE_CMD + 1
_OUT_MANIP = _OUT_DIST + 1
_OUT_BUILD_MS = _OUT_MANIP + 1
_OUT_HAS_DEBUG = _OUT_BUILD_MS + 1
_OUT_FLOATS = _OUT_HAS_DEBUG + 1


class _PoseClient:
    """Minimal client duck-type for VrGoalProvider inside the obs child."""

    def __init__(self):
        self._pose = {
            "connected": False,
            "holding": False,
            "robot_ready": False,
            "timestamp": None,
            "position": None,
            "orientation": None,
            "Tdes": None,
            "T_init": None,
            "q_init": None,
        }

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def set_pose(self, pose: dict) -> None:
        self._pose = pose

    def get_latest_pose(self) -> dict:
        return self._pose


class _ActGoalSnapshot:
    """In-child goal provider fed each tick from packed ACT arm-base goals.

    Must duck-type ``VrGoalProvider`` enough for ``ObservationBuilder``
    (``update``, ``get_goal_position``, ``is_active``, ``ee_quat_goal_base``).
    """

    def __init__(self, arm_mount_in_base: np.ndarray):
        self.arm_mount_in_base = np.asarray(arm_mount_in_base, dtype=np.float64).reshape(3)
        self.curr_ee_goal_cart_base = None
        self.ee_goal_orn_quat = None
        self.ee_pos_goal_base = None
        self.ee_quat_goal_base = None
        self._cart_arm = None
        self._quat = None
        self._have = False

    @property
    def is_active(self) -> bool:
        # Match ActGoalProvider: RFM/dvel only when a real ACT goal is present.
        return bool(self._have)

    def set_arm_goal(self, cart_arm: np.ndarray, quat: np.ndarray) -> None:
        self._cart_arm = np.asarray(cart_arm, dtype=np.float64).reshape(3).copy()
        self._quat = np.asarray(quat, dtype=np.float64).reshape(4).copy()
        self._have = True

    def update(self, ee_pos, ee_quat, base_quat) -> None:
        del base_quat
        if not self._have:
            ee_pos = np.asarray(ee_pos, dtype=np.float64).reshape(3)
            ee_quat = np.asarray(ee_quat, dtype=np.float64).reshape(4)
            self.curr_ee_goal_cart_base = ee_pos.copy()
            self.ee_goal_orn_quat = ee_quat.copy()
            self.ee_pos_goal_base = ee_pos.copy()
            self.ee_quat_goal_base = ee_quat.copy()
            return
        goal_base = self._cart_arm + self.arm_mount_in_base
        self.curr_ee_goal_cart_base = goal_base.copy()
        self.ee_goal_orn_quat = self._quat.copy()
        self.ee_pos_goal_base = goal_base.copy()
        self.ee_quat_goal_base = self._quat.copy()

    def get_goal_position(self, fallback_ee_pos):
        if self.curr_ee_goal_cart_base is None:
            return np.asarray(fallback_ee_pos, dtype=np.float64).copy()
        return self.curr_ee_goal_cart_base.copy()

    def reset(self) -> None:
        self.curr_ee_goal_cart_base = None
        self.ee_goal_orn_quat = None
        self.ee_pos_goal_base = None
        self.ee_quat_goal_base = None
        self._cart_arm = None
        self._quat = None
        self._have = False


def _pack_robot(dst: np.ndarray, robot) -> None:
    dst[_IN_VALID] = 1.0 if robot.valid else 0.0
    dst[_IN_ARM_CONN] = 1.0 if robot.arm_connected else 0.0
    dst[_IN_GRIPPER] = float(robot.gripper_q)
    dst[_IN_TIMESTAMP] = float(robot.timestamp)
    dst[_IN_BASE_QUAT : _IN_BASE_QUAT + 4] = np.asarray(
        robot.base_quat, dtype=np.float64
    ).reshape(4)
    dst[_IN_BASE_LIN : _IN_BASE_LIN + 3] = np.asarray(
        robot.base_lin_vel_local, dtype=np.float64
    ).reshape(3)
    dst[_IN_BASE_ANG : _IN_BASE_ANG + 3] = np.asarray(
        robot.base_ang_vel_local, dtype=np.float64
    ).reshape(3)
    dst[_IN_LEG_Q : _IN_LEG_Q + 12] = np.asarray(robot.leg_q, dtype=np.float64).reshape(
        12
    )
    dst[_IN_LEG_DQ : _IN_LEG_DQ + 12] = np.asarray(
        robot.leg_dq, dtype=np.float64
    ).reshape(12)
    dst[_IN_ARM_Q : _IN_ARM_Q + 6] = np.asarray(robot.arm_q, dtype=np.float64).reshape(6)
    dst[_IN_ARM_DQ : _IN_ARM_DQ + 6] = np.asarray(
        robot.arm_dq, dtype=np.float64
    ).reshape(6)
    dst[_IN_EE_POS : _IN_EE_POS + 3] = np.asarray(robot.ee_pos, dtype=np.float64).reshape(
        3
    )
    dst[_IN_EE_QUAT : _IN_EE_QUAT + 4] = np.asarray(
        robot.ee_quat, dtype=np.float64
    ).reshape(4)
    dst[_IN_FOOT : _IN_FOOT + 4] = np.asarray(
        robot.foot_contacts, dtype=np.float64
    ).reshape(4)
    dst[_IN_GRAV : _IN_GRAV + 3] = np.asarray(
        robot.projected_gravity, dtype=np.float64
    ).reshape(3)
    jac = robot.jacobian
    if jac is not None:
        j = np.asarray(jac, dtype=np.float64).reshape(-1)
        dst[_IN_HAS_JAC] = 1.0
        n = min(j.size, 36)
        dst[_IN_JAC : _IN_JAC + n] = j[:n]
        if n < 36:
            dst[_IN_JAC + n : _IN_JAC + 36] = 0.0
    else:
        dst[_IN_HAS_JAC] = 0.0
        dst[_IN_JAC : _IN_JAC + 36] = 0.0


def _unpack_robot(src: np.ndarray):
    from state.robot_state import RobotState

    jac = None
    if src[_IN_HAS_JAC] > 0.5:
        jac = src[_IN_JAC : _IN_JAC + 36].reshape(6, 6).copy()
    return RobotState(
        base_quat=src[_IN_BASE_QUAT : _IN_BASE_QUAT + 4].copy(),
        base_lin_vel_local=src[_IN_BASE_LIN : _IN_BASE_LIN + 3].copy(),
        base_ang_vel_local=src[_IN_BASE_ANG : _IN_BASE_ANG + 3].copy(),
        leg_q=src[_IN_LEG_Q : _IN_LEG_Q + 12].copy(),
        leg_dq=src[_IN_LEG_DQ : _IN_LEG_DQ + 12].copy(),
        arm_q=src[_IN_ARM_Q : _IN_ARM_Q + 6].copy(),
        arm_dq=src[_IN_ARM_DQ : _IN_ARM_DQ + 6].copy(),
        gripper_q=float(src[_IN_GRIPPER]),
        ee_pos=src[_IN_EE_POS : _IN_EE_POS + 3].copy(),
        ee_quat=src[_IN_EE_QUAT : _IN_EE_QUAT + 4].copy(),
        foot_contacts=(src[_IN_FOOT : _IN_FOOT + 4] > 0.5),
        projected_gravity=src[_IN_GRAV : _IN_GRAV + 3].copy(),
        jacobian=jac,
        arm_connected=bool(src[_IN_ARM_CONN] > 0.5),
        valid=bool(src[_IN_VALID] > 0.5),
        timestamp=float(src[_IN_TIMESTAMP]),
    )


def _named_slices(proprio: np.ndarray, use_cmd: bool) -> Dict[str, np.ndarray]:
    idx = 0
    locomotion_name = "commands" if use_cmd else "dvel_b_local"
    names = [
        ("body_rp", 2),
        ("base_lin_vel", 3),
        ("base_ang_vel", 3),
        ("dof_pos_err", 18),
        ("dof_vel", 18),
        ("last_leg_actions", 12),
        ("foot_contacts", 4),
        ("projected_gravity", 3),
        ("ee_goal_local_cart", 3),
        (locomotion_name, 3),
        ("manip_det_pred", 1),
        ("gait_indices", 1),
        ("clock_inputs", 4),
    ]
    out = {}
    for name, size in names:
        out[name] = proprio[idx : idx + size].copy()
        idx += size
    return out


def _obs_process_entry(
    config_json: str,
    control_dt: float,
    use_vr: bool,
    use_act: bool,
    in_arr,
    out_arr,
    seq,
    done_seq,
    stop_evt,
    ready_evt,
) -> None:
    import os
    import sys
    import time as _time

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    from paths import setup_paths

    setup_paths()

    import numpy as _np
    from obs.observation_builder import ObservationBuilder
    from vr_goal.vr_goal_provider import NullVrGoalProvider, VrGoalProvider

    config = json.loads(config_json)
    # Cap BLAS in obs child (manip / numpy).
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = "1"

    builder = ObservationBuilder(config, control_dt=float(control_dt))
    builder.reset()
    arm_mount = np.asarray(
        config["robot"].get(
            "policy_arm_base_offset",
            config["robot"].get("z1_base_in_robot_base", [0.03, 0.0, 0.09]),
        ),
        dtype=np.float64,
    ).reshape(3)
    pose_client = None
    if use_act:
        vr_goal = _ActGoalSnapshot(arm_mount)
    elif use_vr:
        pose_client = _PoseClient()
        vr_goal = VrGoalProvider(
            arm_mount_in_base=arm_mount,
            use_process_client=False,
            client=pose_client,
        )
        # No websocket in this process — poses arrive via shared memory.
        vr_goal._started = True
    else:
        vr_goal = NullVrGoalProvider()

    ready_evt.set()
    print("[process_obs] child ready (ObservationBuilder).", flush=True)

    in_scratch = _np.zeros(_IN_FLOATS, dtype=_np.float64)
    out_scratch = _np.zeros(_OUT_FLOATS, dtype=_np.float64)
    last = -1
    in_lock = in_arr.get_lock()
    while not stop_evt.is_set():
        with seq.get_lock():
            cur = int(seq.value)
        if cur == last:
            _time.sleep(0.0005)
            continue
        # Non-blocking read: if parent holds the lock, retry without advancing
        # last so we still consume this seq once the lock is free.
        if not in_lock.acquire(False):
            _time.sleep(0.0005)
            continue
        try:
            in_scratch[:] = _np.frombuffer(in_arr.get_obj(), dtype=_np.float64)
        finally:
            in_lock.release()
        last = cur

        if in_scratch[_IN_RESET] > 0.5:
            builder.reset()
            if hasattr(vr_goal, "reset"):
                vr_goal.reset()

        builder.use_cmd = bool(in_scratch[_IN_USE_CMD] > 0.5)
        builder.commands[:] = in_scratch[_IN_CMD : _IN_CMD + 3]

        if in_scratch[_IN_HAS_ACTION] > 0.5:
            builder.update_action_history(
                in_scratch[_IN_ACTION : _IN_ACTION + 12],
                clip_actions=None,
            )

        try:
            robot = _unpack_robot(in_scratch)
            if pose_client is not None:
                pose_client.set_pose(
                    _unpack_pose(in_scratch[_IN_VR : _IN_VR + _POSE_FLOATS])
                )
            if use_act and isinstance(vr_goal, _ActGoalSnapshot):
                if in_scratch[_IN_HAS_ACT] > 0.5:
                    vr_goal.set_arm_goal(
                        in_scratch[_IN_ACT_CART : _IN_ACT_CART + 3],
                        in_scratch[_IN_ACT_QUAT : _IN_ACT_QUAT + 4],
                    )

            debug = bool(in_scratch[_IN_DEBUG] > 0.5)
            t0 = _time.perf_counter()
            pack = builder.build(robot, vr_goal, debug=debug)
            build_ms = (_time.perf_counter() - t0) * 1000.0

            out_scratch[:] = 0.0
            obs_vec = _np.asarray(pack["obs_vector"], dtype=_np.float64).reshape(-1)
            out_scratch[_OUT_OBS : _OUT_OBS + num_observations] = obs_vec[
                :num_observations
            ]
            # Proprio = first num_proprio of unclipped layout; recover from obs if
            # clipped similarly for display (slices are approximate under clip).
            if "proprio" in pack:
                proprio = _np.asarray(pack["proprio"], dtype=_np.float64).reshape(-1)
            else:
                proprio = obs_vec[:num_proprio]
            out_scratch[_OUT_PROPRIO : _OUT_PROPRIO + num_proprio] = proprio[
                :num_proprio
            ]
            out_scratch[_OUT_PHASE] = float(pack["phase_variable"])
            out_scratch[_OUT_EE_GOAL_LOCAL : _OUT_EE_GOAL_LOCAL + 3] = _np.asarray(
                pack["ee_goal_local_cart"], dtype=_np.float64
            ).reshape(3)
            out_scratch[_OUT_EE_GOAL_QUAT : _OUT_EE_GOAL_QUAT + 4] = _np.asarray(
                pack["ee_goal_quat_base"], dtype=_np.float64
            ).reshape(4)
            goal_base = pack.get("ee_goal_base", pack.get("ee_goal_world"))
            if goal_base is None:
                goal_base = pack["ee_goal_local_cart"]
            out_scratch[_OUT_EE_GOAL_BASE : _OUT_EE_GOAL_BASE + 3] = _np.asarray(
                goal_base, dtype=_np.float64
            ).reshape(3)
            out_scratch[_OUT_COMMANDS : _OUT_COMMANDS + 3] = _np.asarray(
                pack["commands"], dtype=_np.float64
            ).reshape(3)
            out_scratch[_OUT_DVEL : _OUT_DVEL + 3] = _np.asarray(
                pack["dvel_b_local"], dtype=_np.float64
            ).reshape(3)
            out_scratch[_OUT_BASE_LIN : _OUT_BASE_LIN + 3] = _np.asarray(
                pack["base_lin_vel"], dtype=_np.float64
            ).reshape(3)
            out_scratch[_OUT_EE_POS : _OUT_EE_POS + 3] = _np.asarray(
                pack["ee_pos"], dtype=_np.float64
            ).reshape(3)
            out_scratch[_OUT_EE_QUAT : _OUT_EE_QUAT + 4] = _np.asarray(
                pack["ee_quat"], dtype=_np.float64
            ).reshape(4)
            out_scratch[_OUT_USE_CMD] = 1.0 if pack.get("use_cmd") else 0.0
            out_scratch[_OUT_DIST] = float(pack.get("dist_base_ee", 0.0) or 0.0)
            out_scratch[_OUT_MANIP] = float(pack.get("manip_det", 0.0) or 0.0)
            out_scratch[_OUT_BUILD_MS] = float(build_ms)
            out_scratch[_OUT_HAS_DEBUG] = 1.0 if debug else 0.0

            with out_arr.get_lock():
                _np.frombuffer(out_arr.get_obj(), dtype=_np.float64)[:] = out_scratch
            with done_seq.get_lock():
                done_seq.value = cur
        except Exception as exc:
            # Never silently die — parent would keep serving a zero obs stub.
            import traceback

            print(
                f"[process_obs] BUILD FAILED (seq={cur}): {exc}\n"
                f"{traceback.format_exc()}",
                flush=True,
            )
            _time.sleep(0.05)


class ProcessObsWorker:
    """Async obs build in a child process."""

    def __init__(
        self,
        config: dict,
        control_dt: float,
        *,
        use_vr: bool = True,
        use_act: bool = False,
    ):
        import multiprocessing as mp

        self._ctx = mp.get_context("spawn")
        self._in = self._ctx.Array("d", _IN_FLOATS, lock=True)
        self._out = self._ctx.Array("d", _OUT_FLOATS, lock=True)
        self._seq = self._ctx.Value("i", 0)
        self._done_seq = self._ctx.Value("i", 0)
        self._stop = self._ctx.Event()
        self._ready = self._ctx.Event()
        self._last_taken = 0
        self._pending_action: Optional[np.ndarray] = None
        self._pending_reset = False
        self._pending_act_goal: Optional[Dict[str, Any]] = None
        self._use_act = bool(use_act)
        self._use_cmd = bool(config.get("commands", {}).get("use_cmd", False))
        self._commands = np.array(
            [
                float(config.get("commands", {}).get("cmd_vx", 0.0)),
                float(config.get("commands", {}).get("cmd_vy", 0.0)),
                float(config.get("commands", {}).get("cmd_yaw", 0.0)),
            ],
            dtype=np.float64,
        )
        self.last_build_ms = 0.0
        self._last_pack: Optional[Dict[str, Any]] = None
        self._submit_count = 0
        self._warned_stale = False
        self._proc = self._ctx.Process(
            target=_obs_process_entry,
            args=(
                json.dumps(config),
                float(control_dt),
                bool(use_vr) and not bool(use_act),
                bool(use_act),
                self._in,
                self._out,
                self._seq,
                self._done_seq,
                self._stop,
                self._ready,
            ),
            name="process_obs",
            daemon=True,
        )

    def start(self) -> None:
        self._proc.start()

    def wait_ready(self, timeout: float = 60.0) -> None:
        if not self._ready.wait(timeout):
            raise TimeoutError(
                f"ProcessObsWorker child not ready within {timeout:.0f}s"
            )

    def stop(self) -> None:
        self._stop.set()
        self._proc.join(timeout=2.0)
        if self._proc.is_alive():
            self._proc.terminate()

    def set_commands(self, vx: float, vy: float, yaw: float) -> None:
        self._commands[:] = (float(vx), float(vy), float(yaw))
        self._use_cmd = True

    def reset(self) -> None:
        self._pending_reset = True

    def push_action(self, action: np.ndarray) -> None:
        """Queue leg action for history update on the next child build."""
        self._pending_action = (
            np.asarray(action, dtype=np.float64).reshape(-1)[:12].copy()
        )

    def set_act_goal(self, act_goal: Optional[dict]) -> None:
        """Arm-base ACT goal for the next obs build (hold-last if None)."""
        self._pending_act_goal = act_goal

    def submit(self, robot, vr_pose: Optional[dict] = None, *, debug: bool = False) -> bool:
        """Pack robot state into SHM. Non-blocking: returns False if lock busy.

        Pending action/reset are only cleared after a successful publish so a
        contended tick does not drop them.
        """
        buf = np.zeros(_IN_FLOATS, dtype=np.float64)
        _pack_robot(buf, robot)
        if vr_pose is not None:
            _pack_pose(buf[_IN_VR : _IN_VR + _POSE_FLOATS], vr_pose)
        buf[_IN_DEBUG] = 1.0 if debug else 0.0
        buf[_IN_USE_CMD] = 1.0 if self._use_cmd else 0.0
        buf[_IN_CMD : _IN_CMD + 3] = self._commands
        has_action = self._pending_action is not None
        if has_action:
            buf[_IN_HAS_ACTION] = 1.0
            buf[_IN_ACTION : _IN_ACTION + 12] = self._pending_action
        do_reset = bool(self._pending_reset)
        if do_reset:
            buf[_IN_RESET] = 1.0
        if self._use_act and self._pending_act_goal is not None:
            g = self._pending_act_goal
            buf[_IN_HAS_ACT] = 1.0
            buf[_IN_ACT_CART : _IN_ACT_CART + 3] = np.asarray(
                g["ee_goal_local_cart"], dtype=np.float64
            ).reshape(3)
            buf[_IN_ACT_QUAT : _IN_ACT_QUAT + 4] = np.asarray(
                g["ee_goal_quat_base"], dtype=np.float64
            ).reshape(4)
            buf[_IN_ACT_GRIP] = float(g.get("gripper_cmd", 0.0))

        in_lock = self._in.get_lock()
        if not in_lock.acquire(False):
            return False
        try:
            np.frombuffer(self._in.get_obj(), dtype=np.float64)[:] = buf
        finally:
            in_lock.release()

        seq_lock = self._seq.get_lock()
        if not seq_lock.acquire(False):
            # Input written but seq not bumped; next successful submit overwrites.
            return False
        try:
            self._seq.value += 1
        finally:
            seq_lock.release()

        if has_action:
            self._pending_action = None
        if do_reset:
            self._pending_reset = False
        return True

    def take(self) -> Optional[Dict[str, Any]]:
        """Non-blocking SHM read. Returns None if lock busy or no new pack."""
        done_lock = self._done_seq.get_lock()
        if not done_lock.acquire(False):
            return None
        try:
            done = int(self._done_seq.value)
        finally:
            done_lock.release()
        if done <= self._last_taken:
            return None

        out_lock = self._out.get_lock()
        if not out_lock.acquire(False):
            # Do not advance _last_taken — retry this pack next tick.
            return None
        try:
            out = np.frombuffer(self._out.get_obj(), dtype=np.float64).copy()
        finally:
            out_lock.release()
        self._last_taken = done
        self.last_build_ms = float(out[_OUT_BUILD_MS])
        obs_vector = out[_OUT_OBS : _OUT_OBS + num_observations].copy()
        proprio = out[_OUT_PROPRIO : _OUT_PROPRIO + num_proprio].copy()
        use_cmd = bool(out[_OUT_USE_CMD] > 0.5)
        pack: Dict[str, Any] = {
            "obs_vector": obs_vector,
            "proprio": proprio,
            "phase_variable": float(out[_OUT_PHASE]),
            "ee_goal_local_cart": out[
                _OUT_EE_GOAL_LOCAL : _OUT_EE_GOAL_LOCAL + 3
            ].copy(),
            "ee_goal_quat_base": out[_OUT_EE_GOAL_QUAT : _OUT_EE_GOAL_QUAT + 4].copy(),
            "ee_goal_base": out[_OUT_EE_GOAL_BASE : _OUT_EE_GOAL_BASE + 3].copy(),
            "ee_goal_world": out[_OUT_EE_GOAL_BASE : _OUT_EE_GOAL_BASE + 3].copy(),
            "commands": out[_OUT_COMMANDS : _OUT_COMMANDS + 3].copy(),
            "dvel_b_local": out[_OUT_DVEL : _OUT_DVEL + 3].copy(),
            "base_lin_vel": out[_OUT_BASE_LIN : _OUT_BASE_LIN + 3].copy(),
            "ee_pos": out[_OUT_EE_POS : _OUT_EE_POS + 3].copy(),
            "ee_quat": out[_OUT_EE_QUAT : _OUT_EE_QUAT + 4].copy(),
            "use_cmd": use_cmd,
            "dist_base_ee": float(out[_OUT_DIST]),
            "manip_det": float(out[_OUT_MANIP]),
            "slices": _named_slices(proprio, use_cmd),
        }
        self._last_pack = pack
        return pack

    def step(
        self, robot, vr_pose: Optional[dict] = None, *, debug: bool = False
    ) -> Dict[str, Any]:
        """Submit and return newest ready pack (or last pack / empty stub)."""
        if self.submit(robot, vr_pose, debug=debug):
            self._submit_count += 1
        ready = self.take()
        if ready is not None:
            self._warned_stale = False
            return ready
        if self._last_pack is not None:
            return self._last_pack
        if self._submit_count >= 25 and not self._warned_stale:
            alive = self._proc.is_alive() if self._proc is not None else False
            print(
                "[process_obs] CRITICAL: no obs pack after "
                f"{self._submit_count} submits (child_alive={alive}). "
                "Serving zero stub — check [process_obs] BUILD FAILED logs.",
                flush=True,
            )
            self._warned_stale = True
        return {
            "obs_vector": np.zeros(num_observations, dtype=np.float64),
            "phase_variable": 0.0,
            "ee_goal_local_cart": np.zeros(3),
            "ee_goal_quat_base": np.full(4, np.nan),
            "ee_goal_base": np.zeros(3),
            "ee_goal_world": np.zeros(3),
            "commands": self._commands.copy(),
            "dvel_b_local": np.zeros(3),
            "base_lin_vel": np.zeros(3),
            "ee_pos": np.full(3, np.nan),
            "ee_quat": np.full(4, np.nan),
            "use_cmd": self._use_cmd,
            "dist_base_ee": 0.0,
            "manip_det": 0.0,
            "slices": {},
        }
