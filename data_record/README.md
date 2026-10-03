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
  --interface enp8s0 \
  --ee-ws-host <WBC_LAPTOP_IP> \
  --ee-ws-port 8770 \
  --rate 20 \
  --out recordings/session.h5
```

Ctrl+C stops cleanly and flushes the HDF5 file.

### Dry tests without hardware

```bash
# Camera + WS only (no DDS):
python scripts/record_session.py --no-b2 --ee-ws-host 127.0.0.1 --rate 5

# Networking only (black frames):
python scripts/record_session.py --no-camera --no-b2 --ee-ws-host 127.0.0.1 --rate 5
```

## HDF5 layout

| Dataset | Shape | Notes |
|---------|-------|-------|
| `timestamp` | `(N,)` | Recorder wall time (s) |
| `rgb` | `(N, 480, 640, 3)` | RGB uint8 |
| `ee_goal_local_cart` | `(N, 3)` | NaN if stale/missing |
| `ee_goal_age_s` | `(N,)` | Age of EE sample vs wall time |
| `base_lin_vel_local` | `(N, 3)` | From sport velocity |
| `base_quat` | `(N, 4)` | IMU wxyz |

File attrs: `record_rate_hz`, `ee_ws_url`, `interface`, `camera_serial`, `created_at`, `num_frames`.
