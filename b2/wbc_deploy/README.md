# B2Z1 Whole-Body Controller — Real Robot Deploy

Python deploy stack for running the B2Z1 lambda-WBC policy on the real robot with Vive VR end-effector goals.

## Prerequisites

- B2 on the same network; set `network.interface` in [`config/b2z1_wbc.yaml`](config/b2z1_wbc.yaml)
- Z1: `./z1_ctrl` running; `unitree_arm_interface` available under `z1/z1_sdk/lib`
- VR WebSocket server running (see `z1/teleop/vr_websocket_server.py`)
- Python deps: `pip install -r requirements.txt`
- WBC checkpoint (`.pt`) — path passed via `--checkpoint`

## Run order

1. Start `./z1_ctrl` on the robot PC
2. Start VR WebSocket server (optionally with robot prep)
3. **Dry run** — verify observations:

```bash
cd b2/wbc_deploy
python scripts/print_observations.py --interface enp8s0 --vr-host 127.0.0.1 --vr-port 8765
```

4. Optional: add checkpoint to also print policy actions (still no motor output):

```bash
python scripts/print_observations.py --interface enp8s0 --checkpoint /path/to/model_XXXXX.pt
```

5. Full deploy (dry run by default):

```bash
python scripts/deploy_wbc.py --interface enp8s0 --checkpoint /path/to/model_XXXXX.pt
```

Without VR (EE goal = current pose; `use_cmd` from yaml or CLI):

```bash
python scripts/deploy_wbc.py --interface enp8s0 --no-vr --no-release-mode
python scripts/deploy_wbc.py --interface enp8s0 --no-vr --use-cmd --cmd-vx 0.3
```

## Starting from a standing pose

When you want low-level WBC control **without** the robot collapsing first:

1. **Put the robot in sport stand** on the handheld controller (normal standing mode).
2. Start `./z1_ctrl` and confirm observations look sane (`print_observations.py --no-vr`).
3. Run deploy **without** `--no-release-mode`. Sport mode is released **immediately before** lowcmd starts (not during policy load). Startup uses high `startup_kp` / `startup_kd` (default 1000 / 10, like `b2_stand_example`).
4. On `--execute-actions`, legs **ramp** to `init_deploy_joint` (yaml) over `init_deploy_ramp_s` (default 5 s), then **hold** for `init_deploy_hold_s` while printing obs. You then type `YES` to apply policy actions (or anything else to keep holding `init_deploy_joint`).

Dry run while still in sport stand (no motors):

```bash
python scripts/deploy_wbc.py --interface enp8s0 --no-vr --no-release-mode --checkpoint /path/to/model.pt
```

Low-level deploy from standing:

```bash
python scripts/deploy_wbc.py --interface enp8s0 --no-vr --checkpoint /path/to/model.pt --execute-actions
# optional: legs only
python scripts/deploy_wbc.py --interface enp8s0 --no-vr --execute-actions --legs-only
```

If you are **already** in low-level mode (`rt/lowcmd` active, sport released), add `--no-release-mode` to skip the motion-switcher handshake.

**Tips:** Check step-0 leg actions in a dry run before `--execute-actions`. If the robot is soft or droops after the hold period, try higher `control.kp` / `control.kd` in yaml (official stand example uses Kp≈1000; deploy defaults are lower for WBC tracking).

## Observation layout (843 dims)

Matches `controller_loader.py` with gait commands (`num_proprio=75`):

| Block | Dims | Description |
|-------|------|-------------|
| proprio | 75 | Current step (body rp, velocities, joints, contacts, EE goal, RFM, gait) |
| priv | 18 | Nominal zeros at deploy (mass/friction/motor strength) |
| history | 750 | 10 stacked proprio frames |

**Actions:** policy outputs **12 leg actions** only (`num_arm_actions=0`). Arm pose is commanded via IK on the robot (same as sim `arm.process_actions` for legs-only). Use a legs-only checkpoint; the previous 20-dim (leg+arm head) `model_50000.pt` will not load.

## Validation checklist

1. Robot standing, VR idle → `dvel_b_local` ≈ 0, obs stable
2. Move VR tracker → `ee_goal_local_cart` changes; goal tracks EE on connect
3. Lift feet → `foot_contacts` bits toggle
4. Compare `dof_pos_err` with known posture
5. Only then use `--execute-actions`

## Configuration

Edit [`config/b2z1_wbc.yaml`](config/b2z1_wbc.yaml) for:

- Network interface and sport topic (`rt/lf/sportmodestate` vs `rt/sportmodestate`)
- Default joint angles and arm base offset
- Control gains and action scales
- VR host/port

## Module overview

| Module | Role |
|--------|------|
| `state/b2_reader.py` | B2 `rt/lowstate` + sport mode state (leg q/dq remapped to URDF order) |
| `state/leg_dof_mapping.py` | B2 motor FR/FL/RR/RL ↔ URDF FL/FR/RL/RR permutation |
| `state/z1_reader.py` | Z1 joints, FK EE pose, Jacobian |
| `vr_goal/vr_goal_provider.py` | VR → world EE goal via `VrPoseMapper` |
| `obs/observation_builder.py` | Build 843-dim policy input |
| `policy/controller_loader.py` | Load checkpoint / run inference |
| `scripts/print_observations.py` | Dry-run observation monitor |
| `scripts/deploy_wbc.py` | Policy loop + optional motor commands |
