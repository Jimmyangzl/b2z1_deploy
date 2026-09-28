"""Manipulability estimate from Z1 arm Jacobian."""

from typing import Optional

import numpy as np


class ManipulabilityTracker:
    def __init__(self, hist_len: int = 20, forecast_steps: int = 5):
        self.hist_len = hist_len
        self.forecast_steps = forecast_steps
        self.manip_det = 0.0
        self.manip_det_hist = 0.0005 * np.ones(hist_len, dtype=np.float64)
        self.manip_det_pred = np.zeros(forecast_steps, dtype=np.float64)

    def update(self, jacobian: Optional[np.ndarray]) -> float:
        if jacobian is None:
            self.manip_det = 0.0
        else:
            A = jacobian @ jacobian.T
            self.manip_det = float(np.linalg.det(A))
        self.manip_det_hist = np.append(self.manip_det_hist[1:], self.manip_det)
        self._predict()
        return self.manip_det

    def _predict(self) -> None:
        level, trend = self._double_exponential_smoothing(self.manip_det_hist)
        self.manip_det_pred = np.array(
            [level[-1] + (i + 1) * trend[-1] for i in range(self.forecast_steps)],
            dtype=np.float64,
        )

    @staticmethod
    def _double_exponential_smoothing(x, alpha=0.5):
        seq_len = len(x)
        level = np.zeros(seq_len)
        trend_component = np.zeros(seq_len)
        level[0] = x[0]
        if seq_len > 1:
            trend_component[0] = x[1] - x[0]
        for t in range(1, seq_len):
            level[t] = alpha * x[t] + (1 - alpha) * (level[t - 1] + trend_component[t - 1])
            trend_component[t] = alpha * (level[t] - level[t - 1]) + (1 - alpha) * trend_component[t - 1]
        return level, trend_component
