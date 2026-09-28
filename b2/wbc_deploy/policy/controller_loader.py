"""Load B2Z1 WBC checkpoint for real-robot inference."""

import inspect
import os
import sys

import torch

# Ensure vendored rsl_rl is importable before other imports.
_DEPLOY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RSL_RL_ROOT = os.path.join(_DEPLOY_ROOT, "third_party", "rsl_rl")
if _RSL_RL_ROOT not in sys.path:
    sys.path.insert(0, _RSL_RL_ROOT)

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic

from policy.constants import history_len, num_actions, num_observations, num_priv, num_proprio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseConfig:
    def __init__(self) -> None:
        self.init_member_classes(self)

    @staticmethod
    def init_member_classes(obj):
        for key in dir(obj):
            if key == "__class__":
                continue
            var = getattr(obj, key)
            if inspect.isclass(var):
                i_var = var()
                setattr(obj, key, i_var)
                BaseConfig.init_member_classes(i_var)


class B2Z1WBCRoughCfg(BaseConfig):
    class policy:
        continue_from_last_std = True
        init_std = [[0.8, 1.0, 1.0] * 4]
        actor_hidden_dims = [128]
        critic_hidden_dims = [128]
        activation = "elu"
        output_tanh = False
        leg_control_head_hidden_dims = [128, 128]
        arm_control_head_hidden_dims = [128, 128]
        priv_encoder_dims = [64, 20]
        num_leg_actions = 12
        num_arm_actions = 0
        adaptive_arm_gains = False
        adaptive_arm_gains_scale = 10.0

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.0
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 2e-4
        schedule = "fixed"
        gamma = 0.99
        lam = 0.95
        desired_kl = None
        max_grad_norm = 1.0
        min_policy_std = [[0.15, 0.25, 0.25] * 4]
        mixing_schedule = [1.0, 0, 3000]
        torque_supervision_schedule = [0.0, 1000, 1000]
        adaptive_arm_gains = False
        dagger_update_freq = 20
        priv_reg_coef_schedual = [0, 0.1, 3000, 7000]

    class runner:
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24
        max_iterations = 20000
        save_interval = 200
        experiment_name = "b2z1_v2"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        val = getattr(obj, key)
        if isinstance(val, list):
            element = [class_to_dict(item) for item in val]
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


class WBCLoader:
    def __init__(self, train_cfg, log_dir=None, device="cpu", summarize=False):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        actor_critic_class = eval(self.cfg["policy_class_name"])
        actor_critic: ActorCritic = actor_critic_class(
            num_proprio,
            num_proprio,
            num_actions,
            **self.policy_cfg,
            num_priv=num_priv,
            num_hist=history_len,
            num_prop=num_proprio,
        ).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"])
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        if summarize:
            try:
                from torchinfo import summary

                summary(self.alg.actor_critic)
            except ImportError:
                pass

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device, weights_only=False)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer and "optimizer_state_dict" in loaded_dict:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict.get("iter", 0)
        return loaded_dict.get("infos")

    def get_inference_policy(self, device=None, stochastic=False):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        if not stochastic:
            return self.alg.actor_critic.act_inference
        return self.alg.actor_critic.act
