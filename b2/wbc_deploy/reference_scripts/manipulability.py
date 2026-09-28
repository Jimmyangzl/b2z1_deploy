import torch
from isaacgym.torch_utils import *


class ManipulabilityTracker:
    """Tracks arm manipulability metrics and short-horizon forecasts."""

    def __init__(self, env):
        self.env = env

    @property
    def device(self):
        return self.env.device

    @property
    def num_envs(self):
        return self.env.num_envs

    def init_buffers(self):
        self.manip_det = torch.zeros(self.num_envs, 1, device=self.device)
        self.manip_det_pred = torch.zeros(self.num_envs, 5, device=self.device)
        self.manip_det_hist = 0.0005 * torch.ones(self.num_envs, 20, device=self.device)
        self.manip_xy = torch.zeros(self.num_envs, 1, device=self.device)
        self.manip_xy_pred = torch.zeros(self.num_envs, 5, device=self.device)
        self.manip_xy_hist = 0.8 * torch.ones(self.num_envs, 20, device=self.device)
        self.manip_yaw = torch.zeros(self.num_envs, 1, device=self.device)
        self.manip_yaw_pred = torch.zeros(self.num_envs, 5, device=self.device)
        self.manip_yaw_hist = 1.4 * torch.ones(self.num_envs, 20, device=self.device)
        self.yaw_direction = torch.zeros(self.num_envs, 1, device=self.device)

    def reset(self, env_ids):
        self.manip_det_pred[env_ids, :] = 0.0005
        self.manip_det_hist[env_ids, :] = 0.0005
        self.manip_xy_pred[env_ids, :] = 0.8
        self.manip_xy_hist[env_ids, :] = 0.8
        self.manip_yaw_pred[env_ids, :] = 1.4
        self.manip_yaw_hist[env_ids, :] = 1.4

    def compute(self):
        env = self.env
        A = torch.bmm(env.ee_j_eef, env.ee_j_eef.transpose(1, 2))
        self.manip_det = torch.det(A).unsqueeze(-1)

        dpos_world = (env.curr_ee_goal_cart_world - env.ee_pos).clone()
        dpos_world_xy_norm = dpos_world[:, :2] / torch.norm(dpos_world[:, :2], p=2, dim=1, keepdim=True)
        yaw = env.base_yaw_euler[:, 2]
        _, _, yaw_d = euler_from_quat(env.ee_goal_orn_quat)
        yaw_diff = torch.atan2(torch.sin(yaw_d - yaw), torch.cos(yaw_d - yaw))
        self.yaw_direction[:, 0] = torch.sign(yaw_diff)

        u_xy = torch.zeros(A.shape[0], 6, device=A.device)
        u_yaw = torch.zeros(A.shape[0], 6, device=A.device)
        u_xy[:, :2] = dpos_world_xy_norm
        u_yaw[:, 5] = self.yaw_direction[:, 0]

        Au_xy = torch.bmm(u_xy.unsqueeze(1), A)
        uTAu_xy = torch.bmm(Au_xy, u_xy.unsqueeze(2))
        Au_yaw = torch.bmm(u_yaw.unsqueeze(1), A)
        uTAu_yaw = torch.bmm(Au_yaw, u_yaw.unsqueeze(2))

        self.manip_xy = torch.sqrt(uTAu_xy.clamp(min=0)).reshape(-1, 1)
        self.manip_yaw = torch.sqrt(uTAu_yaw.clamp(min=0)).reshape(-1, 1)

    def update_history(self):
        self.manip_det_hist = torch.cat([self.manip_det_hist[:, 1:], self.manip_det], dim=1)
        self.manip_xy_hist = torch.cat([self.manip_xy_hist[:, 1:], self.manip_xy], dim=1)
        self.manip_yaw_hist = torch.cat([self.manip_yaw_hist[:, 1:], self.manip_yaw], dim=1)

    @staticmethod
    def double_exponential_smoothing(x, alpha=0.5):
        batch_size, seq_len = x.shape
        level = torch.zeros_like(x)
        trend_component = torch.zeros_like(x)
        level[:, 0] = x[:, 0]
        if seq_len > 1:
            trend_component[:, 0] = x[:, 1] - x[:, 0]
        for t in range(1, seq_len):
            level[:, t] = alpha * x[:, t] + (1 - alpha) * (level[:, t - 1] + trend_component[:, t - 1])
            trend_component[:, t] = alpha * (level[:, t] - level[:, t - 1]) + (1 - alpha) * trend_component[:, t - 1]
        return level, trend_component

    def predict(self, forecast_steps=5):
        level, trend = self.double_exponential_smoothing(self.manip_det_hist)
        last_level = level[:, -1].clone()
        last_trend = trend[:, -1].clone()
        self.manip_det_pred = torch.stack(
            [last_level + (i + 1) * last_trend for i in range(forecast_steps)], dim=1,
        )

        level, trend = self.double_exponential_smoothing(self.manip_xy_hist)
        last_level = level[:, -1].clone()
        last_trend = trend[:, -1].clone()
        self.manip_xy_pred = torch.stack(
            [last_level + (i + 1) * last_trend for i in range(forecast_steps)], dim=1,
        )

        level, trend = self.double_exponential_smoothing(self.manip_yaw_hist)
        last_level = level[:, -1].clone()
        last_trend = trend[:, -1].clone()
        self.manip_yaw_pred = torch.stack(
            [last_level + (i + 1) * last_trend for i in range(forecast_steps)], dim=1,
        )
