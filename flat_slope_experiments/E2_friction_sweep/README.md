# E2 Friction Sweep Experiments

Four sub-experiments studying how surface and joint friction affect HFDC performance.
All runs use a fixed angular speed ω = 2π rad/s and skip the first 1 s for metrics.

---

## Sub-experiments overview

| Sub-exp | Folder | Variable | Fixed conditions |
|---------|--------|----------|-----------------|
| E2a | `combined/` | Surface μ ∈ [0.1, 1.0] | slope 30°, no joint friction |
| E2b | `surface_comparison/` | Flat vs slope 30°, μ = 0 or 0.7 | no joint friction |
| E2c | `joint_friction_comparison/` | Joint friction on/off × surface friction on/off | flat + slope 30° |
| E2d | `tracking/` | Time-series at specific μ values | HFDC with vs without friction |

---

## E2a — Combined friction sweep (`combined/`)

**Question:** How do force and position errors scale with surface friction μ?

**Data source:** `friction_sweep/sweep_friction.sh` (original), `E2_friction_sweep/sweep_baseline.sh` (baseline, new)

**Data directories:**
```
data/paper/       data_0.1.npz … data_1.0.npz   (HFDC)
data/paper_pi/    data_0.1.npz … data_1.0.npz   (HFDC+PI)
data/baseline/    data_0.1.npz … data_1.0.npz   (Baseline — needs collection)
```

**Simulation parameters:**
- Robot: `fr3_friction`, `--surface-friction μ` (varies)
- `--angular-speed 6.283` (2π), slope default (30°)

**Collect baseline data:**
```bash
bash flat_slope_experiments/E2_friction_sweep/sweep_baseline.sh
```

**Plot:**
```bash
python flat_slope_experiments/E2_friction_sweep/combined/plot.py
```

**Outputs:** `combined/plots/force_combined.png`, `position_combined.png`

---

## E2b — Surface comparison (`surface_comparison/`)

**Question:** How does performance differ between flat and slope 30°, with and without surface friction?

**Data source:** `friction_sweep/run_surface_comparison.sh`

**Data directories:**
```
data/surface_comparison/
  paper/
    flat_frictionless/          data_0.0_all.npz   (fr3, slope 0°)
    flat_friction_0.7/          data_0.7_all.npz   (fr3_friction, slope 0°, μ=0.7)
    slope30_frictionless/       data_0.0_all.npz   (fr3, slope 30°)
    slope30_friction_0.7/       data_0.7_all.npz   (fr3_friction, slope 30°, μ=0.7)
  paper_pi/                     (same 4 conditions)
```

**Design:** 2 methods × 2 surfaces × 2 friction conditions = 8 runs

**Collect data:**
```bash
bash friction_sweep/run_surface_comparison.sh
```

**Plot:**
```bash
python flat_slope_experiments/E2_friction_sweep/surface_comparison/plot.py
```

**Outputs:** `surface_comparison/plots/surface_force_error.png`, `surface_position_error.png`

---

## E2c — Joint friction comparison (`joint_friction_comparison/`)

**Question:** Does adding joint friction (robot internal) hurt performance, and does it interact with surface friction?

**Data source:**
- New runs: `friction_sweep/run_joint_friction_comparison.sh`
- Reuses E2b data for the no-joint-friction baseline conditions

**Data directories:**
```
data/joint_friction_comparison/
  paper/
    flat_jointf_no_surff/       data_0.0_all.npz   (fr3_jointf, slope 0°)
    flat_jointf_surff_0.7/      data_0.7_all.npz   (fr3_jointf_surff, slope 0°, μ=0.7)
    slope30_jointf_no_surff/    data_0.0_all.npz   (fr3_jointf, slope 30°)
    slope30_jointf_surff_0.7/   data_0.7_all.npz   (fr3_jointf_surff, slope 30°, μ=0.7)
  paper_pi/                     (same 4 conditions)

data/surface_comparison/        (reused from E2b as no-joint-friction reference)
```

**Design:** 2 methods × 2 surfaces × 2 joint-friction × 2 surface-friction = 8 bar-chart groups

**Collect data:**
```bash
bash friction_sweep/run_joint_friction_comparison.sh
```

**Plot:**
```bash
python flat_slope_experiments/E2_friction_sweep/joint_friction_comparison/plot.py
```

**Outputs:**
- `joint_friction_comparison/plots/joint_friction_force_error.png`
- `joint_friction_comparison/plots/joint_friction_position_error.png`
- `joint_friction_comparison/plots/timeseries_hfdc_flat.png`
- `joint_friction_comparison/plots/timeseries_hfdc_slope.png`

---

## E2d — Tracking time-series (`tracking/`)

**Question:** What do the force and position trajectories look like at a given friction value?

**Data source:** Reuses `data/paper/` and `data/paper_wo_surface_friction/` from E2a.

**Data directories:**
```
data/paper/                          data_<mu>.npz    (HFDC with friction)
data/paper_wo_surface_friction/      data_0.0.npz     (HFDC, no surface friction)
data/paper_pi/                       data_<mu>.npz    (commented out)
data/paper_pi_wo_surface_friction/   data_0.0.npz     (commented out)
```

**No separate data collection needed** — shares data with E2a.

**Plot:**
```bash
python flat_slope_experiments/E2_friction_sweep/tracking/plot.py --friction 0.3 0.7
```

**Outputs:** `tracking/plots/tracking_friction_<mu>/force_tracking.png`, `position_tracking_xyz.png`

---

## Data collection summary

| What to run | Script | Missing data |
|-------------|--------|-------------|
| E2a baseline sweep | `flat_slope_experiments/E2_friction_sweep/sweep_baseline.sh` | `data/baseline/` |
| E2b surface comparison | `friction_sweep/run_surface_comparison.sh` | already collected |
| E2c joint friction | `friction_sweep/run_joint_friction_comparison.sh` | already collected |
| E2d tracking | — (reuses E2a data) | — |

## Plot all at once

```bash
bash flat_slope_experiments/plot_all.sh
```
