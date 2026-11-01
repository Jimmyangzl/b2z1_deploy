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
        "--policy-device",
        default=None,
        choices=("cpu", "cuda"),
        help="Policy inference device (default: policy.device in yaml, cpu). "
        "Prefer cpu when SteamVR uses the same GPU — cuda can stall 100–300 ms.",
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


def resolve_policy_device(config: dict, args: argparse.Namespace):
    """CLI --policy-device overrides policy.device (default cpu)."""
    import torch

    from policy.controller_loader import resolve_torch_device

    spec = getattr(args, "policy_device", None)
    if not spec:
        spec = (config.get("policy") or {}).get("device", "cpu")
    return resolve_torch_device(spec)


def configure_blas_threadpools(n: int = 1) -> int:
    """Cap OpenBLAS/OpenMP via env + threadpoolctl (no Torch import)."""
    import os

    n = max(1, int(n))
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = str(n)
    try:
        import threadpoolctl

        threadpoolctl.threadpool_limits(limits=n)
        pools = {
            (info.get("user_api"), info.get("internal_api")): info.get("num_threads")
            for info in threadpoolctl.threadpool_info()
        }
        print(f"[load_runtime] threadpools capped to {n}: {pools}", flush=True)
    except Exception as exc:
        print(f"[load_runtime] threadpoolctl unavailable ({exc}); env only.", flush=True)
    return n


def configure_torch_inference_threads(config: dict) -> int:
    """Cap *all* compute threadpools for 50 Hz batch-1 control.

    Root cause of "policy=250 ms" on CPU: the MLP itself needs ~0.3 ms of CPU.
    Wall-clock spikes are the process sitting runnable/waiting while the machine
    is oversubscribed — default OpenBLAS uses ~all cores (32 here) and Torch
    OpenMP defaults to ~all cores (24). SteamVR + those pools → scheduler
    stalls that land inside ``run_policy`` and get misread as slow inference.

    Returns the configured per-library thread count.
    """
    import torch

    n = int((config.get("policy") or {}).get("num_threads", 1) or 1)
    n = configure_blas_threadpools(n)
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already set in this process
    return n


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
        arm_mount = config["robot"].get(
            "policy_arm_base_offset",
            config["robot"].get("z1_base_in_robot_base", [0.03, 0.0, 0.09]),
        )
        vr_goal = VrGoalProvider(
            host=config["vr"]["host"],
            port=int(config["vr"]["port"]),
            arm_mount_in_base=arm_mount,
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

        # --arm-only runs policy in a child process; skip loading Torch in the
        # teleop process (keeps GIL / OpenMP out of the 50 Hz arm path).
        if getattr(args, "arm_only", False):
            print(
                "[load_runtime] skip in-process policy load (--arm-only; "
                "ProcessPolicyWorker child loads the checkpoint).",
                flush=True,
            )
            # Cap BLAS only — do not import Torch into the teleop process.
            n = int((config.get("policy") or {}).get("num_threads", 1) or 1)
            configure_blas_threadpools(n)
        else:
            import policy.controller_loader as controller_loader
            from policy.controller_loader import B2Z1WBCRoughCfg, WBCLoader, class_to_dict

            policy_device = resolve_policy_device(config, args)
            controller_loader.device = policy_device
            n_threads = configure_torch_inference_threads(config)
            print(
                f"[load_runtime] policy device={policy_device} "
                f"torch_num_threads={n_threads}",
                flush=True,
            )

            loader_cfg = class_to_dict(B2Z1WBCRoughCfg())
            wbcloader = WBCLoader(train_cfg=loader_cfg, device=str(policy_device))
            wbcloader.load(checkpoint_path, load_optimizer=False)
            policy = wbcloader.get_inference_policy(device=policy_device)

    return config, aggregator, vr_goal, obs_builder, policy, rate, dt


# Reused H2D/D2H buffers — avoids per-tick torch.tensor alloc + reduces CUDA stalls.
_OBS_BUF = None
_OBS_DEV = None
_ACT_HOST = None
_RUN_POLICY_LOCK = None
# Last infer timing: wall = perf_counter, cpu = thread_time (this thread only).
# If wall >> cpu, the stall is scheduling/GIL wait — not MLP flops.
LAST_POLICY_TIMING = {"wall_ms": 0.0, "cpu_ms": 0.0, "wait_ms": 0.0}


def _policy_lock():
    import threading

    global _RUN_POLICY_LOCK
    if _RUN_POLICY_LOCK is None:
        _RUN_POLICY_LOCK = threading.Lock()
    return _RUN_POLICY_LOCK


def run_policy(policy, obs_vector: np.ndarray) -> np.ndarray:
    import torch

    from policy.constants import num_actions, num_observations
    from policy.controller_loader import device

    global _OBS_BUF, _OBS_DEV, _ACT_HOST
    t_wall0 = time.perf_counter()
    t_cpu0 = time.thread_time()
    with _policy_lock():
        use_cuda = device.type == "cuda"
        if (
            _OBS_BUF is None
            or _OBS_BUF.shape[-1] != num_observations
            or _OBS_DEV is None
            or _OBS_DEV.device != device
        ):
            _OBS_BUF = torch.empty(
                1, num_observations, dtype=torch.float32, pin_memory=use_cuda
            )
            _OBS_DEV = torch.empty(
                1, num_observations, dtype=torch.float32, device=device
            )
            _ACT_HOST = torch.empty(
                1, num_actions, dtype=torch.float32, pin_memory=use_cuda
            )

        flat = np.asarray(obs_vector, dtype=np.float32).reshape(-1)
        if flat.shape[0] != num_observations:
            raise ValueError(f"obs length {flat.shape[0]} != {num_observations}")
        np.copyto(_OBS_BUF.numpy(), flat.reshape(1, -1))
        _OBS_DEV.copy_(_OBS_BUF, non_blocking=use_cuda)
        with torch.inference_mode():
            action = policy(_OBS_DEV, hist_encoding=True)
            act = action.reshape(1, -1)
            if act.shape[-1] != num_actions:
                act = act[:, :num_actions]
            if use_cuda:
                _ACT_HOST.copy_(act, non_blocking=False)
                out = _ACT_HOST.squeeze(0).numpy().copy()
            else:
                out = act.squeeze(0).detach().cpu().numpy()
    wall_ms = (time.perf_counter() - t_wall0) * 1000.0
    cpu_ms = (time.thread_time() - t_cpu0) * 1000.0
    LAST_POLICY_TIMING["wall_ms"] = wall_ms
    LAST_POLICY_TIMING["cpu_ms"] = cpu_ms
    LAST_POLICY_TIMING["wait_ms"] = max(0.0, wall_ms - cpu_ms)
    return out


def warmup_policy(policy, n: int = 50) -> None:
    """Run dummy forwards so CUDA context / cuDNN stay hot before control starts."""
    import torch

    from policy.constants import num_observations
    from policy.controller_loader import device

    for _ in range(max(1, int(n))):
        run_policy(policy, np.zeros(num_observations, dtype=np.float64))
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _policy_process_entry(
    checkpoint: str,
    device_str: str,
    num_threads: int,
    obs_arr,
    act_arr,
    wall_ms,
    cpu_ms,
    wait_ms,
    seq,
    done_seq,
    stop_evt,
) -> None:
    """Child process: own interpreter/GIL — cannot stall the teleop process."""
    import os
    import time as _time

    # Match parent RT thread caps before NumPy/Torch init in this process.
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = str(max(1, int(num_threads)))

    # Child needs deploy package on sys.path (spawn starts fresh).
    import sys

    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)
    from paths import setup_paths

    setup_paths()

    import numpy as _np
    import policy.controller_loader as controller_loader
    from policy.constants import num_actions, num_observations
    from policy.controller_loader import (
        B2Z1WBCRoughCfg,
        WBCLoader,
        class_to_dict,
        resolve_torch_device,
    )

    device = resolve_torch_device(device_str)
    controller_loader.device = device
    configure_torch_inference_threads({"policy": {"num_threads": num_threads}})

    loader = WBCLoader(train_cfg=class_to_dict(B2Z1WBCRoughCfg()), device=str(device))
    loader.load(checkpoint, load_optimizer=False)
    policy = loader.get_inference_policy(device=device)
    warmup_policy(policy, n=20)

    n_obs = int(num_observations)
    n_act = int(num_actions)
    last = -1
    while not stop_evt.is_set():
        with seq.get_lock():
            cur = int(seq.value)
        if cur == last:
            _time.sleep(0.001)
            continue
        last = cur
        with obs_arr.get_lock():
            obs = _np.frombuffer(obs_arr.get_obj(), dtype=_np.float64).copy()
        if obs.shape[0] != n_obs:
            continue
        action = run_policy(policy, obs)
        act = _np.asarray(action, dtype=_np.float64).reshape(-1)[:n_act]
        with act_arr.get_lock():
            _np.frombuffer(act_arr.get_obj(), dtype=_np.float64)[:n_act] = act
        wall_ms.value = float(LAST_POLICY_TIMING["wall_ms"])
        cpu_ms.value = float(LAST_POLICY_TIMING["cpu_ms"])
        wait_ms.value = float(LAST_POLICY_TIMING["wait_ms"])
        with done_seq.get_lock():
            done_seq.value = cur


class ProcessPolicyWorker:
    """Policy in a *separate process* (own GIL) for --arm-only load simulation.

    A background *thread* still shares the teleop GIL with obs/VR/hotkeys, so
    SteamVR + torch/numpy can make ``wait≫cpu`` and also stall ``obs``/``arm``
    on the main thread. A child process cannot hold this process's GIL.
    """

    def __init__(self, checkpoint: str, device: str = "cpu", num_threads: int = 1):
        import multiprocessing as mp

        from policy.constants import num_actions, num_observations

        # spawn: safe with CUDA / avoids forking threads from DDS/Z1.
        self._ctx = mp.get_context("spawn")
        self._n_obs = int(num_observations)
        self._n_act = int(num_actions)
        self._obs = self._ctx.Array("d", self._n_obs, lock=True)
        self._act = self._ctx.Array("d", self._n_act, lock=True)
        self._wall_ms = self._ctx.Value("d", 0.0)
        self._cpu_ms = self._ctx.Value("d", 0.0)
        self._wait_ms = self._ctx.Value("d", 0.0)
        self._seq = self._ctx.Value("i", 0)
        self._done_seq = self._ctx.Value("i", 0)
        self._stop = self._ctx.Event()
        self._last_taken = 0
        self._proc = self._ctx.Process(
            target=_policy_process_entry,
            args=(
                str(checkpoint),
                str(device),
                int(num_threads),
                self._obs,
                self._act,
                self._wall_ms,
                self._cpu_ms,
                self._wait_ms,
                self._seq,
                self._done_seq,
                self._stop,
            ),
            name="process_policy",
            daemon=True,
        )
        self.infer_count = 0

    @property
    def last_infer_ms(self) -> float:
        return float(self._wall_ms.value)

    @property
    def last_cpu_ms(self) -> float:
        return float(self._cpu_ms.value)

    @property
    def last_wait_ms(self) -> float:
        return float(self._wait_ms.value)

    def start(self) -> None:
        self._proc.start()

    def stop(self) -> None:
        self._stop.set()
        self._proc.join(timeout=2.0)
        if self._proc.is_alive():
            self._proc.terminate()

    def submit(self, obs_vector: np.ndarray) -> None:
        flat = np.asarray(obs_vector, dtype=np.float64).reshape(-1)
        if flat.shape[0] != self._n_obs:
            raise ValueError(f"obs length {flat.shape[0]} != {self._n_obs}")
        with self._obs.get_lock():
            np.frombuffer(self._obs.get_obj(), dtype=np.float64)[:] = flat
        with self._seq.get_lock():
            self._seq.value += 1

    def take_action(self) -> Optional[np.ndarray]:
        with self._done_seq.get_lock():
            done = int(self._done_seq.value)
        if done <= self._last_taken:
            return None
        self._last_taken = done
        self.infer_count += 1
        with self._act.get_lock():
            return np.frombuffer(self._act.get_obj(), dtype=np.float64).copy()


# Back-compat alias (thread worker was GIL-coupled; prefer ProcessPolicyWorker).
AsyncPolicyWorker = ProcessPolicyWorker


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
    if "dist_base_ee" in obs_pack:
        print(f"{'dist_base_ee':22s}     {float(obs_pack['dist_base_ee']):.3f}")
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
