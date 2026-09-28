#!/usr/bin/env python3
"""Print WBC observations from the real B2Z1 robot without sending motor commands."""

import os
import sys
import time

WBC_DEPLOY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WBC_DEPLOY_ROOT not in sys.path:
    sys.path.insert(0, WBC_DEPLOY_ROOT)

from paths import setup_paths

setup_paths()

from state.leg_dof_mapping import leg_mapping_diagnostic
from wbc_runtime import build_arg_parser, format_slice, load_runtime, print_obs_summary, run_policy


def _print_command_mode(config: dict) -> None:
    cmd = config.get("commands", {})
    use_cmd = bool(cmd.get("use_cmd", False))
    print(f"  use_cmd    : {use_cmd}")
    if use_cmd:
        print(
            "  commands   : "
            f"vx={cmd.get('cmd_vx', 0.0)} "
            f"vy={cmd.get('cmd_vy', 0.0)} "
            f"yaw={cmd.get('cmd_yaw', 0.0)}"
        )
    else:
        print("  locomotion : dvel_b_local from VR / EE goal")


def main() -> None:
    parser = build_arg_parser("Print B2Z1 WBC observations (dry run, no motor commands)")
    args = parser.parse_args()
    config, aggregator, vr_goal, obs_builder, policy, rate, dt = load_runtime(args)

    print("B2Z1 WBC observation monitor (dry run — no motor commands)")
    print(f"  interface : {config['network']['interface']}")
    print(f"  sport topic: {aggregator.b2.sport_topic_in_use}")
    if args.no_vr:
        print("  VR server  : disabled (--no-vr); EE goal held at current pose")
        print("  arm obs    : defaults when Z1 feedback missing or out of range")
    else:
        print(f"  VR server  : {config['vr']['host']}:{config['vr']['port']}")
    print(f"  rate       : {rate} Hz")
    _print_command_mode(config)
    if policy is not None:
        print(f"  checkpoint : {args.checkpoint}")
    print("Press Ctrl+C to stop.\n")

    step = 0
    try:
        while True:
            t0 = time.perf_counter()
            robot = aggregator.read()
            if not robot.valid:
                print("[print_obs] waiting for valid robot state...")
                time.sleep(0.1)
                continue

            if step == 0 and robot.leg_q_motor is not None:
                diag = leg_mapping_diagnostic(robot.leg_q_motor, robot.leg_q)
                print(
                    "Leg DOF mapping check: "
                    f"motor FR hip={diag['motor_FR_hip']:.4f}, FL hip={diag['motor_FL_hip']:.4f} | "
                    f"sim FL hip={diag['sim_FL_hip']:.4f}, FR hip={diag['sim_FR_hip']:.4f} | "
                    f"swap_detected={diag['swap_detected']} "
                    "(True means remapping looks wrong)"
                )

            obs_pack = obs_builder.build(robot, vr_goal)
            action = None
            if policy is not None:
                action = run_policy(policy, obs_pack["obs_vector"])
                obs_builder.update_action_history(action)

            if args.verbose:
                print(f"\n=== step {step} FULL OBS ===")
                print(obs_pack["obs_vector"])
            else:
                print_obs_summary(step, obs_pack, action)

            step += 1
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, dt - elapsed))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        vr_goal.stop()
        aggregator.close()


if __name__ == "__main__":
    main()
