"""In-process ObservationBuilder adapter (no child / no SHM IPC).

Same call surface as ``ProcessObsWorker`` so deploy can swap without
rewriting the control loop. Build runs on the control GIL; typical cost
is ~2–7 ms, which fits the 20 ms tick and avoids 40–70 ms SHM lock stalls.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np

from obs.observation_builder import ObservationBuilder
from policy.constants import num_observations


class InProcessObsWorker:
    """Thin wrapper: ``step`` / ``push_action`` / ``set_commands`` / ``reset``."""

    def __init__(self, builder: ObservationBuilder, vr_goal: Any):
        self._builder = builder
        self._vr_goal = vr_goal
        self.last_build_ms = 0.0
        self._last_pack: Optional[Dict[str, Any]] = None
        self._submit_count = 0
        self._warned_stale = False

    @property
    def _commands(self) -> np.ndarray:
        return self._builder.commands

    def start(self) -> None:
        return

    def wait_ready(self, timeout: float = 60.0) -> None:
        del timeout
        return

    def stop(self) -> None:
        return

    def set_commands(self, vx: float, vy: float, yaw: float) -> None:
        self._builder.set_commands(vx, vy, yaw)

    def reset(self) -> None:
        self._builder.reset()

    def push_action(self, action: np.ndarray) -> None:
        self._builder.update_action_history(action, clip_actions=None)

    def set_act_goal(self, act_goal: Optional[dict]) -> None:
        """No-op: ACT goals live on ``vr_goal`` (ActGoalProvider) already."""
        del act_goal

    def submit(self, robot, vr_pose: Optional[dict] = None, *, debug: bool = False) -> bool:
        del vr_pose
        self.step(robot, None, debug=debug)
        return True

    def take(self) -> Optional[Dict[str, Any]]:
        return self._last_pack

    def step(
        self, robot, vr_pose: Optional[dict] = None, *, debug: bool = False
    ) -> Dict[str, Any]:
        del vr_pose  # parent ``vr_goal`` already has ProcessVr / ACT state
        self._submit_count += 1
        t0 = time.perf_counter()
        try:
            pack = self._builder.build(robot, self._vr_goal, debug=debug)
        except Exception as exc:
            import traceback

            print(
                f"[inprocess_obs] BUILD FAILED: {exc}\n{traceback.format_exc()}",
                flush=True,
            )
            if self._last_pack is not None:
                return self._last_pack
            if self._submit_count >= 25 and not self._warned_stale:
                print(
                    "[inprocess_obs] CRITICAL: serving zero stub after build failures.",
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
                "commands": self._builder.commands.copy(),
                "dvel_b_local": np.zeros(3),
                "base_lin_vel": np.zeros(3),
                "ee_pos": np.full(3, np.nan),
                "ee_quat": np.full(4, np.nan),
                "use_cmd": self._builder.use_cmd,
                "dist_base_ee": 0.0,
                "manip_det": 0.0,
                "slices": {},
            }
        self.last_build_ms = (time.perf_counter() - t0) * 1000.0
        self._last_pack = pack
        self._warned_stale = False
        return pack
