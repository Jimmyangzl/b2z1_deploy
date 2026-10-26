# Data record — RealSense RGB + EE goal + B2 state

Records synchronized sessions to HDF5 on the **recording laptop**:

- RGB from Intel RealSense D435i → `(N, 480, 640, 3)` uint8
- `ee_goal_local_cart` from WBC WebSocket publisher
- B2 `base_quat` + `base_lin_vel_local` via DDS on the robot Ethernet

## Topology

| Laptop | Role |
|--------|------|
| VR / Z1 | Existing VR WebSocket + `z1_ctrl` |
| WBC | `deploy_wbc.py --ee-goal-ws` (streams EE goal @ 50 Hz default) |
| Recording | This package + RealSense USB + B2 NIC (`enp8s0`) |

## Setup (recording laptop)

```bash
cd ~/unitree_deploy/b2z1_deploy/data_record
pip install -r requirements.txt
# Ensure unitree_sdk2py is importable (script adds ../b2/unitree_sdk2_python automatically).
```

Edit [`config/record.yaml`](config/record.yaml) for interface, WBC host IP, and rates.

## Run order

1. Power robot; start VR server + `z1_ctrl` on the VR laptop.
2. On the **WBC** laptop:

```bash
cd ~/unitree_deploy/b2z1_deploy/b2/wbc_deploy
python scripts/deploy_wbc.py --interface enp8s0 --ee-goal-ws \
  --ee-goal-ws-host 0.0.0.0 --ee-goal-ws-port 8770 --ee-goal-ws-rate 50
# add --execute-actions when ready for motors
```

3. On the **recording** laptop (same robot Ethernet as B2):

```bash
cd ~/unitree_deploy/b2z1_deploy/data_record
python scripts/record_session.py \
  --task task_1 \
  --interface enp8s0 \
  --ee-ws-host <WBC_LAPTOP_IP> \
  --ee-ws-port 8770 \
  --rate 20
```

`--task` is required. Each run creates the next episode under
`recordings/<task>/episode_0.h5`, `episode_1.h5`, … (folder is created if missing).
Optional `--out path.h5` overrides the auto episode path.

Ctrl+C stops cleanly and flushes the HDF5 file.

### Dry tests without hardware

```bash
# Camera + WS only (no DDS):
python scripts/record_session.py --task dry_run --no-b2 --ee-ws-host 127.0.0.1 --rate 5

# Networking only (black frames):
python scripts/record_session.py --task dry_run --no-camera --no-b2 --ee-ws-host 127.0.0.1 --rate 5

# Image only:
python scripts/record_session.py --task task_1 --no-b2 --no-ee --rate 20
```

## HDF5 layout

| Dataset | Shape | Notes |
|---------|-------|-------|
| `timestamp` | `(N,)` | Recorder wall time (s) |
| `rgb` | `(N, 480, 640, 3)` | RGB uint8 |
| `ee_goal_local_cart` | `(N, 3)` | Arm-base position. NaN if stale/missing |
| `ee_goal_quat_base` | `(N, 4)` | EE target quaternion, wxyz, robot base frame. NaN if stale/missing |
| `ee_goal_age_s` | `(N,)` | Age of EE sample vs wall time |
| `gripper_cmd` | `(N,)` | Z1 gripper command: `0.2` closed, `-1` open. NaN if missing/stale |
| `base_lin_vel_local` | `(N, 3)` | WBC `base_lin_vel` (sport velocity in the base frame). NaN if that stream is stale |
| `base_quat` | `(N, 4)` | IMU wxyz |

File attrs: `record_rate_hz`, `ee_ws_url`, `interface`, `camera_serial`, `created_at`, `num_frames`.

## Relative trajectory transform

Converts each recorded frame into a horizon of poses expressed in the **arm-base
mount frame at time `t`**. Base motion is recovered by integrating
`base_lin_vel_local * dt` (`dt = 1 / record_rate_hz`) with `base_quat` rotations.
Past the episode end, the last relative pose is repeated (stationary pad).

```bash
cd ~/unitree_deploy/b2z1_deploy/data_record
# Single episode (e.g. validate):
python scripts/make_relative_traj.py --task test --chunk-n 20 --episode 1

# All episodes under recordings/<task>/:
python scripts/make_relative_traj.py --task test --chunk-n 20

# Optional arm-mount override (default 0.15 0 0.10):
python scripts/make_relative_traj.py --task test --chunk-n 20 --arm-offset 0.15 0 0.10
```

Writes `recordings/relative_traj/<task>/episode_*.h5` (overwrites if present).
Episodes with any NaN in `ee_goal_local_cart`, `ee_goal_quat_base`,
`base_lin_vel_local`, `base_quat`, or `gripper_cmd` are reported and skipped.

| Dataset | Shape | Notes |
|---------|-------|-------|
| `timestamp` | `(T,)` | Copied from source |
| `rgb` | `(T, H, W, 3)` | Copied unchanged |
| `ee_goal_cart_chunk` | `(T, n, 3)` | EE goal in arm-base frame at `t` |
| `ee_goal_quat_chunk` | `(T, n, 4)` | EE quat wxyz in arm-base frame at `t` |
| `gripper_cmd` | `(T, n)` | Sliced/padded gripper command |

Extra attrs: `chunk_n`, `dt`, `arm_base_offset`, `source_path`, `relative_traj`.

### Visualize a relative episode

Interactive matplotlib UI: RGB | 3D EE chunk in `F_t` (arm-base axes at origin,
orientation frames, markers colored by `gripper_cmd`). Slider scrubs `t`.

```bash
cd ~/unitree_deploy/b2z1_deploy/data_record
python scripts/viz_relative_traj.py --task test --episode 1
# or:
python scripts/viz_relative_traj.py --path recordings/relative_traj/test/episode_1.h5
# optional: --frame-scale 0.02  --t0 100
```
