---
title: NPZ Data Schema – speed_sweep_friction
---

# NPZ file schema

Files live at:
```
sweep_friction_results/data/<robot>/<method>/data_<mult>.npz          # baseline
sweep_friction_results/data/<robot>/<method>/data_<mult>_all.npz      # all other methods
```

## Keys

| Key | Shape | Unit | Notes |
|-----|-------|------|-------|
| `force_error` | `(N,)` | N | Force Z error (actual − desired) |
| `position_error` | `(N,)` | m | Scalar EE position error |
| `actual_positions` | `(N, 3)` | m | EE position [X, Y, Z] |
| `desired_positions` | `(N, 3)` | m | Desired EE position [X, Y, Z] |
| `actual_velocitys` | `(N, 3)` | m/s | EE velocity [X, Y, Z] |
| `desired_velocitys` | `(N, 3)` | m/s | Desired EE velocity [X, Y, Z] |
| `multiplier` | scalar | — | Angular speed multiplier used for this run |
| `angular_speed_rad_s` | scalar | rad/s | ω = multiplier × π |
| `ee_linear_speed_m_s` | scalar | m/s | v = r × ω, r = 0.1 m  (**absent in baseline files**) |

## Sizes

- `dt = 0.001 s`
- Trajectory type 2 (−60° → +60°): ~667 samples per run
- Baseline files are longer (~834 samples) because the approach phase is included

## Robots

| Key | Description |
|-----|-------------|
| `fr3_friction` | FR3 with surface friction model |
| `fr3_jointf_surff` | FR3 with joint friction + surface friction |

## Methods

| Key | Description |
|-----|-------------|
| `baseline` | No force control, position-only |
| `ff` | Feedforward force control |
| `ff_pi` | Feedforward + PI force control |
| `pd` | PD force control |
| `paper` | HFDC (paper method) |
| `paper_pi` | HFDC + PI |

## Notes

- No burn-in skip is needed for this cylinder experiment (unlike angular_speed_sweep which skips 1 s).
- The circular trajectory lies in the **Y-Z plane** (cylinder axis along X). Use `actual_positions[:, 1]` and `actual_positions[:, 2]` for 2D tracking plots.
