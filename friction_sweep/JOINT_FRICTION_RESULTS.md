# Joint Friction Comparison — Experiment Results

**ω = 2π rad/s** (v ≈ 0.628 m/s) · **F_d = −8 N** · metrics skip first **1 s** of each trial

---

## Scripts

| Role | File |
|------|------|
| Run experiments | [`friction_sweep/run_joint_friction_comparison.sh`](run_joint_friction_comparison.sh) |
| Bar chart plot | [`friction_sweep/plot_joint_friction_comparison.py`](plot_joint_friction_comparison.py) |
| Time-series plot | [`friction_sweep/plot_joint_friction_timeseries.py`](plot_joint_friction_timeseries.py) |
| Baseline data (no joint friction) | [`friction_sweep/run_surface_comparison.sh`](run_surface_comparison.sh) |

---

## Plots

### Bar charts (mean ± std across all 4 surface conditions)

| Plot | File |
|------|------|
| Force error | [joint_friction_force_error.png](plots/joint_friction_comparison/joint_friction_force_error.png) |
| Position error | [joint_friction_position_error.png](plots/joint_friction_comparison/joint_friction_position_error.png) |

### Time-series (first 2 s shown)

| Plot | File |
|------|------|
| HFDC — Flat surface | [timeseries_hfdc_flat.png](plots/joint_friction_comparison/timeseries_hfdc_flat.png) |
| HFDC+PI — Flat surface | [timeseries_hfdc_pi_flat.png](plots/joint_friction_comparison/timeseries_hfdc_pi_flat.png) |
| HFDC — Slope 30° | [timeseries_hfdc_slope.png](plots/joint_friction_comparison/timeseries_hfdc_slope.png) |
| HFDC+PI — Slope 30° | [timeseries_hfdc_pi_slope.png](plots/joint_friction_comparison/timeseries_hfdc_pi_slope.png) |

---

## Results

> **Trace key**
> - *No joint, no surf.* — baseline: no joint friction, frictionless surface (μ=0)
> - *Joint, no surf.* — joint friction added, frictionless surface (μ=0)
> - *No joint, surf.* — baseline: no joint friction, surface friction μ=0.7
> - *Joint + surf.* — joint friction added, surface friction μ=0.7

---

### HFDC — Flat surface

| Trace | Force mean (N) | Force max (N) | Pos mean (mm) | Pos max (mm) |
|-------|---------------|---------------|---------------|--------------|
| No joint, no surf. (μ=0) | 0.2573 | 0.6018 | 4.118 | 5.986 |
| Joint, no surf. (μ=0) | 1.7479 | 8.0000 | 8.474 | 13.326 |
| No joint, surf. (μ=0.7) | 0.8797 | 2.5948 | 11.552 | 13.161 |
| Joint + surf. (μ=0.7) | 1.5687 | 3.7738 | 13.740 | 17.428 |

---

### HFDC+PI — Flat surface

| Trace | Force mean (N) | Force max (N) | Pos mean (mm) | Pos max (mm) |
|-------|---------------|---------------|---------------|--------------|
| No joint, no surf. (μ=0) | 0.0835 | 0.1793 | 4.121 | 5.994 |
| Joint, no surf. (μ=0) | 0.7383 | 8.0000 | 8.384 | 12.729 |
| No joint, surf. (μ=0.7) | 0.3078 | 1.2579 | 11.537 | 12.888 |
| Joint + surf. (μ=0.7) | 0.4600 | 1.3387 | 13.645 | 17.235 |

---

### HFDC — Slope 30°

| Trace | Force mean (N) | Force max (N) | Pos mean (mm) | Pos max (mm) |
|-------|---------------|---------------|---------------|--------------|
| No joint, no surf. (μ=0) | 0.2262 | 0.5673 | 3.940 | 5.355 |
| Joint, no surf. (μ=0) | 6.2544 | 8.0000 | 12.787 | 17.123 |
| No joint, surf. (μ=0.7) | 0.8839 | 2.5195 | 10.681 | 12.124 |
| Joint + surf. (μ=0.7) | 1.5210 | 8.0000 | 12.556 | 15.503 |

---

### HFDC+PI — Slope 30°

| Trace | Force mean (N) | Force max (N) | Pos mean (mm) | Pos max (mm) |
|-------|---------------|---------------|---------------|--------------|
| No joint, no surf. (μ=0) | 0.0735 | 0.1690 | 3.944 | 5.365 |
| Joint, no surf. (μ=0) | 3.6630 | 8.5256 | 22.742 | 39.167 |
| No joint, surf. (μ=0.7) | 0.3006 | 1.2349 | 10.651 | 12.105 |
| Joint + surf. (μ=0.7) | 0.5019 | 1.5050 | 12.388 | 15.039 |

---

## Key Observations

- **Joint friction alone (no surface friction)** causes the largest degradation on slope 30°: HFDC force mean rises from 0.23 N to 6.25 N; HFDC+PI from 0.07 N to 3.66 N with position error reaching 39 mm max.
- **On flat surface**, joint friction raises HFDC force mean ~6.8× (0.26 → 1.75 N) and HFDC+PI ~8.8× (0.08 → 0.74 N).
- **Combined joint + surface friction (μ=0.7)** is less severe than joint friction alone on slope: force mean of HFDC 1.52 N vs 6.25 N — surface friction appears to suppress some of the erratic contact-loss behaviour.
- **HFDC+PI consistently outperforms HFDC** under combined friction on both flat (0.46 vs 1.57 N mean) and slope (0.50 vs 1.52 N mean), though the advantage narrows with joint friction.
- **Position error** follows the same trend: joint friction roughly doubles position error in frictionless conditions, and combined friction raises it to ~13–14 mm mean on flat and ~12 mm on slope.
