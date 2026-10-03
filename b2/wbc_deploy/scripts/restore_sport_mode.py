#!/usr/bin/env python3
"""Re-enable B2 default (AI/sport) control after WBC lowcmd / soft shutdown.

After ``deploy_wbc.py --execute-actions``, MotionSwitcher is released and the
handheld remote cannot command the robot until a mode is selected again.
``SelectMode('ai')`` restores that; optional ``--recovery-stand`` then asks
the sport service to stand up from a lying pose.

Usage (from ``b2/wbc_deploy``)::

    python scripts/restore_sport_mode.py --interface enp8s0
    python scripts/restore_sport_mode.py --interface enp8s0 --recovery-stand
"""

from __future__ import annotations

import argparse
import os
import sys
import time

WBC_DEPLOY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WBC_DEPLOY_ROOT not in sys.path:
    sys.path.insert(0, WBC_DEPLOY_ROOT)

from paths import setup_paths

setup_paths()

from unitree_sdk2py.b2.sport.sport_client import SportClient
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

# Prefer the alias that works on this B2; fall back through Unitree example names.
_MODE_CANDIDATES = ("ai", "sport_mode", "ai_sport", "normal", "advanced")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Restore B2 AI/sport mode after WBC deploy (for handheld remote)."
    )
    parser.add_argument(
        "--interface",
        default="enp8s0",
        help="Network interface to the robot (default: enp8s0)",
    )
    parser.add_argument(
        "--mode",
        default=None,
        help="MotionSwitcher mode/alias (default: try ai, then sport_mode, …)",
    )
    parser.add_argument(
        "--recovery-stand",
        action="store_true",
        help="After SelectMode succeeds, call SportClient.RecoveryStand()",
    )
    parser.add_argument(
        "--settle-s",
        type=float,
        default=1.0,
        help="Seconds to wait after SelectMode before RecoveryStand (default: 1.0)",
    )
    args = parser.parse_args()

    ChannelFactoryInitialize(0, args.interface)

    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()

    print("CheckMode before:", msc.CheckMode())

    candidates = (args.mode,) if args.mode else _MODE_CANDIDATES
    selected = None
    for name in candidates:
        if not name:
            continue
        code, _ = msc.SelectMode(name)
        print(f"SelectMode({name!r}): {code}")
        if code == 0:
            selected = name
            break

    code, mode = msc.CheckMode()
    print("CheckMode after:", code, mode)
    active = (mode or {}).get("name") or ""
    if code != 0 or not active:
        print(
            "ERROR: no motion mode active. Ensure deploy/lowcmd is fully stopped, "
            "then retry or use the Unitree app.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: mode {active!r} active (selected via {selected!r}). Handheld remote should work.")

    if not args.recovery_stand:
        return 0

    if args.settle_s > 0:
        time.sleep(float(args.settle_s))

    sport = SportClient()
    sport.SetTimeout(10.0)
    sport.Init()
    ret = sport.RecoveryStand()
    print(f"RecoveryStand: {ret}")
    if ret != 0:
        print(
            "WARNING: RecoveryStand failed (sport RPC). Mode is still active — "
            "stand with the handheld remote if needed.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
