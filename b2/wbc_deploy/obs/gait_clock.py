"""Gait phase clock inputs for WBC observations."""

from typing import Tuple

import numpy as np


class GaitClock:
    def __init__(self, dt: float, cfg: dict):
        self.dt = dt
        self.frequencies = float(cfg["frequencies"])
        self.phases = float(cfg["phases"])
        self.offsets = float(cfg["offsets"])
        self.bounds = float(cfg["bounds"])
        self.durations = float(cfg["durations"])
        self.gait_indices = np.array([0.0])
        self.clock_inputs = np.zeros(4, dtype=np.float64)
        self.foot_indices = np.zeros((4, 1), dtype=np.float64)

    def step(
        self,
        locomotion_cmd: np.ndarray,
        *,
        use_cmd: bool = False,
        lin_vel_x_clip: float = 0.1,
        lin_vel_y_clip: float = 0.1,
        ang_vel_yaw_clip: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray]:
        del use_cmd  # same clip thresholds for commands and dvel_b_local
        self.gait_indices = np.remainder(self.gait_indices + self.dt * self.frequencies, 1.0)
        walking = self._walking_cmd_mask(
            locomotion_cmd,
            lin_vel_x_clip=lin_vel_x_clip,
            lin_vel_y_clip=lin_vel_y_clip,
            ang_vel_yaw_clip=ang_vel_yaw_clip,
        )
        if not walking:
            self.gait_indices[:] = 0.0

        foot_indices = [
            self.gait_indices + self.phases + self.offsets + self.bounds,
            self.gait_indices + self.offsets,
            self.gait_indices + self.bounds,
            self.gait_indices + self.phases,
        ]
        processed = []
        for idxs in foot_indices:
            idx_arr = np.remainder(idxs, 1.0).copy()
            stance = np.remainder(idx_arr, 1) < self.durations
            swing = np.remainder(idx_arr, 1) > self.durations
            idx_arr[stance] = np.remainder(idx_arr[stance], 1) * (0.5 / self.durations)
            idx_arr[swing] = 0.5 + (np.remainder(idx_arr[swing], 1) - self.durations) * (
                0.5 / (1.0 - self.durations)
            )
            processed.append(idx_arr)

        self.foot_indices = np.column_stack(processed)
        self.clock_inputs[0] = np.sin(2 * np.pi * processed[0])[0]
        self.clock_inputs[1] = np.sin(2 * np.pi * processed[1])[0]
        self.clock_inputs[2] = np.sin(2 * np.pi * processed[2])[0]
        self.clock_inputs[3] = np.sin(2 * np.pi * processed[3])[0]
        return self.gait_indices.copy(), self.clock_inputs.copy()

    @staticmethod
    def _walking_cmd_mask(
        cmd: np.ndarray,
        *,
        lin_vel_x_clip: float,
        lin_vel_y_clip: float,
        ang_vel_yaw_clip: float,
    ) -> bool:
        """True if locomotion command is large enough to advance gait (cmd or dvel)."""
        return (
            abs(cmd[0]) > lin_vel_x_clip
            or abs(cmd[1]) > lin_vel_y_clip
            or abs(cmd[2]) > ang_vel_yaw_clip
        )
