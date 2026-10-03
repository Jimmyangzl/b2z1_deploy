"""Shared WBC deploy runtime helpers."""

from typing import Optional

import argparse
import os
import sys
import time

import numpy as np
import yaml

from paths import WBC_DEPLOY_ROOT, setup_paths

setup_paths()

from obs.observation_builder import ObservationBuilder
from policy.constants import num_observations
from state.aggregator import RobotStateAggregator, load_config
from vr_goal.vr_goal_provider import NullVrGoalProvider, VrGoalProvider


DEFAULT_CONFIG = os.path.join(WBC_DEPLOY_ROOT, "config", "b2z1_wbc.yaml")


def build_arg_parser(description: str, *, require_checkpoint: bool = False) -> argparse.ArgumentParser:
    del require_checkpoint  # validated in load_runtime / deploy script instead
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to b2z1_wbc.yaml",
    )
    parser.add_argument("--interface", default=None, help="B2 network interface override")
    parser.add_argument("--vr-host", default=None, help="VR WebSocket host override")
    parser.add_argument("--vr-port", type=int, default=None, help="VR WebSocket port override")
    parser.add_argument("--rate", type=float, default=None, help="Control loop rate (Hz)")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="WBC checkpoint (.pt); overrides policy.checkpoint in config",
    )
    parser.add_argument(
        "--no-release-mode",
        action="store_true",
        help="Skip MotionSwitcher release (use if already in low-level mode)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print full observation vector")
    parser.add_argument(
        "--use-cmd",
        action="store_true",
        help="Override config: use manual base velocity commands in policy obs",
    )
    parser.add_argument(
        "--no-use-cmd",
        action="store_true",
        help="Override config: use dvel_b_local from VR / EE goal in policy obs",
    )
    parser.add_argument("--cmd-vx", type=float, default=None, help="Override commands.cmd_vx (m/s)")
    parser.add_argument("--cmd-vy", type=float, default=None, help="Override commands.cmd_vy (m/s)")
    parser.add_argument("--cmd-yaw", type=float, default=None, help="Override commands.cmd_yaw (rad/s)")
    parser.add_argument(
        "--no-vr",
        action="store_true",
        help="Disable VR WebSocket; hold EE goal at current pose (NullVrGoalProvider)",
    )
    parser.add_argument(
        "--ee-goal-ws",
        action="store_true",
        help="Broadcast ee_goal_local_cart over WebSocket for data_record clients",
    )
    parser.add_argument(
        "--ee-goal-ws-host",
        default=None,
        help="EE-goal WS bind host (default: streaming.host or 0.0.0.0)",
    )
    parser.add_argument(
        "--ee-goal-ws-port",
        type=int,
        default=None,
        help="EE-goal WS bind port (default: streaming.port or 8770)",
    )
    parser.add_argument(
        "--ee-goal-ws-rate",
        type=float,
        default=None,
        help="EE-goal WS broadcast rate Hz (default: streaming.rate_hz or 50)",
    )
    return parser


def parse_base_args(description: str, argv=None) -> argparse.Namespace:
    return build_arg_parser(description).parse_args(argv)


def resolve_checkpoint_path(config: dict, args: argparse.Namespace) -> Optional[str]:
    """CLI --checkpoint overrides policy.checkpoint from yaml."""
    path = args.checkpoint
    if not path:
        policy_cfg = config.get("policy", {})
        path = policy_cfg.get("checkpoint")
    if path is None or str(path).strip() == "":
        return None

    path = os.path.expanduser(str(path))
    if not os.path.isabs(path):
        config_dir = os.path.dirname(os.path.abspath(args.config))
        path = os.path.normpath(os.path.join(config_dir, path))
    return os.path.abspath(path)


def apply_command_overrides(config: dict, args: argparse.Namespace) -> None:
    if "commands" not in config:
        config["commands"] = {}
    cmd_cfg = config["commands"]
    if getattr(args, "use_cmd", False):
        cmd_cfg["use_cmd"] = True
    if getattr(args, "no_use_cmd", False):
        cmd_cfg["use_cmd"] = False
    if getattr(args, "cmd_vx", None) is not None:
        cmd_cfg["cmd_vx"] = float(args.cmd_vx)
    if getattr(args, "cmd_vy", None) is not None:
        cmd_cfg["cmd_vy"] = float(args.cmd_vy)
    if getattr(args, "cmd_yaw", None) is not None:
        cmd_cfg["cmd_yaw"] = float(args.cmd_yaw)


def _wait_vr_prep_released(vr_goal, timeout_s: float = 15.0) -> None:
    """Block until websocket has a message and prep is no longer holding.

    Connecting as the first client triggers ``z1_robot_prep.release_hold()``.
    We must not call ArmInterface.loopOn() until that finishes.
    """
    client = getattr(vr_goal, "client", None)
    if client is None:
        return
    deadline = time.time() + timeout_s
    saw_msg = False
    while time.time() < deadline:
        pose = client.get_latest_pose()
        ts = pose.get("timestamp")
        if ts is not None:
            saw_msg = True
            # holding=False once prep released (or no prep / already free).
            if not bool(pose.get("holding", False)):
                print(
                    "[load_runtime] VR websocket up; arm prep released "
                    f"(robot_ready={bool(pose.get('robot_ready', False))}, "
                    f"tracker_connected={bool(pose.get('connected', False))}).",
                    flush=True,
                )
                return
        time.sleep(0.05)
    if saw_msg:
        print(
            "[load_runtime] WARNING: VR still reports holding=True after "
            f"{timeout_s:.0f}s; starting Z1 loop anyway.",
            flush=True,
        )
    else:
        print(
            "[load_runtime] WARNING: no VR websocket messages within "
            f"{timeout_s:.0f}s; starting Z1 loop anyway "
            "(is vr_websocket_server running without --no-tracking?).",
            flush=True,
        )


def load_runtime(args: argparse.Namespace, *, defer_release: bool = False):
    config = load_config(args.config)
    if args.interface:
        config["network"]["interface"] = args.interface
    if args.vr_host:
        config["vr"]["host"] = args.vr_host
    if args.vr_port:
        config["vr"]["port"] = args.vr_port
    apply_command_overrides(config, args)
    rate = float(args.rate if args.rate is not None else config["control"]["rate_hz"])
    dt = 1.0 / rate

    use_vr = not getattr(args, "no_vr", False)
    # With VR + robot prep: defer Z1 UDP until the websocket client connects
    # (prep releases on first client). Matches teleop_z1_vive_ws startup order.
    defer_z1_loop = use_vr

    aggregator = RobotStateAggregator(config)
    release_now = not args.no_release_mode and not defer_release
    aggregator.initialize(release_motion_mode=release_now, start_z1_loop=not defer_z1_loop)

    if not use_vr:
        vr_goal = NullVrGoalProvider()
    else:
        vr_goal = VrGoalProvider(
            host=config["vr"]["host"],
            port=int(config["vr"]["port"]),
        )
        vr_goal.start()
        _wait_vr_prep_released(vr_goal)
        aggregator.start_z1_loop()

    obs_builder = ObservationBuilder(config, control_dt=dt)
    obs_builder.reset()

    checkpoint_path = resolve_checkpoint_path(config, args)
    args.checkpoint = checkpoint_path

    policy = None
    if checkpoint_path:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        import torch

        from policy.controller_loader import (
            B2Z1WBCRoughCfg,
            WBCLoader,
            class_to_dict,
            device,
        )

        loader_cfg = class_to_dict(B2Z1WBCRoughCfg())
        wbcloader = WBCLoader(train_cfg=loader_cfg, device=str(device))
        wbcloader.load(checkpoint_path, load_optimizer=False)
        policy = wbcloader.get_inference_policy(device=device)

    return config, aggregator, vr_goal, obs_builder, policy, rate, dt


def run_policy(policy, obs_vector: np.ndarray) -> np.ndarray:
    import torch

    from policy.controller_loader import device

    obs_tensor = torch.tensor(obs_vector, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        action = policy(obs_tensor, hist_encoding=True)
    return action.squeeze(0).cpu().numpy()


def format_slice(name: str, values: np.ndarray, *, precision: int = 3) -> str:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return (
        f"{name:22s} [{flat.size:2d}] "
        f"{np.array2string(flat, precision=precision, suppress_small=False)}"
    )


def print_obs_summary(step: int, obs_pack: dict, action=None, *, use_cmd: bool = False) -> None:
    print(f"\n=== step {step} ===")
    for name, values in obs_pack["slices"].items():
        # manip ~1e-4; default precision=3 prints as 0.
        prec = 6 if name == "manip_det_pred" else 3
        print(format_slice(name, values, precision=prec))
    obs_vec = obs_pack["obs_vector"]
    print(f"{'obs_total':22s} [{obs_vec.size:2d}] norm={np.linalg.norm(obs_vec):.3f} (expected {num_observations})")
    print(f"{'ee_goal_base':22s}     {np.array2string(obs_pack.get('ee_goal_base', obs_pack['ee_goal_world']), precision=3)}")
    print(f"{'phase_variable':22s}     {obs_pack['phase_variable']:.3f}")
    if "manip_det" in obs_pack:
        print(
            f"{'manip_det (raw)':22s}     {float(obs_pack['manip_det']):.6e}  "
            f"pred={np.array2string(np.asarray(obs_pack.get('manip_det_pred', [])), precision=6, suppress_small=False)}"
        )
    if obs_pack.get("use_cmd"):
        print(
            f"{'commands_raw':22s}     "
            f"{np.array2string(obs_pack['commands'], precision=3)} [vx vy yaw]"
        )
    else:
        print(
            f"{'dvel_b_local_raw':22s}     "
            f"{np.array2string(obs_pack['dvel_b_local'], precision=3)}"
        )
    if action is not None:
        leg = np.asarray(action, dtype=np.float64).reshape(-1)[:12]
        print(
            f"{'action (legs)':22s} [{leg.size:2d}] "
            f"{np.array2string(leg, precision=3, suppress_small=True)}"
        )
        if leg.size < np.asarray(action).size:
            print(f"{'action (extra)':22s}     unexpected dims beyond 12 (legs-only policy)")
