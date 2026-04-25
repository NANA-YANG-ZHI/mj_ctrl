#!/usr/bin/env bash
# Cylinder surface: friction robot speed sweep, all methods.
#
# Grid:
#   Speeds  : 0.1π to 3.0π rad/s  (step 0.1π, 30 values)
#   Robots  : fr3_friction, fr3_jointf_surff
#   Methods : baseline, ff, ff_pi, pd, paper, paper_pi
#   Total   : 30 × 2 × 6 = 360 jobs
#
# Data saved to:
#   sweep_friction_results/data/<robot>/<method>/data_<mult>*.npz
#
# Usage (from workspace root):
#   bash cylinder_experiments/speed_sweep_friction/run_speed_sweep_friction.sh
#
# Environment overrides:
#   NUM_WORKERS    parallel jobs                  (default: 10)
#   SKIP_SECONDS   burn-in seconds for collect    (default: 1.0)
#   TRAJECTORY     1 = 0°→60°, 2 = −60°→60°      (default: 1)
#   FORCE_DESIRED  desired contact force N         (default: -10.0)

set -euo pipefail
export LC_NUMERIC=C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
COLLECT_SCRIPT="${SCRIPT_DIR}/collect_results.py"
PLOT_SCRIPT="${SCRIPT_DIR}/plot_comparison.py"

# ── Configuration ─────────────────────────────────────────────────────────────
NUM_WORKERS="${NUM_WORKERS:-10}"
SKIP_SECONDS="${SKIP_SECONDS:-0.0}"
TRAJECTORY="${TRAJECTORY:-2}"
FORCE_DESIRED="${FORCE_DESIRED:--10.0}"

ROBOTS=(fr3_friction fr3_jointf_surff)

# method_name : script_type ("baseline"|"hybrid") : force_method : use_pi (0|1)
METHODS=(
    "baseline:baseline:baseline:0"
    "ff:hybrid:feedforward:0"
    "ff_pi:hybrid:feedforward:1"
    "pd:hybrid:pd:0"
    "paper:hybrid:paper:0"
    "paper_pi:hybrid:paper:1"
)

OUTPUT_DIR="${SCRIPT_DIR}/sweep_friction_results"
DATA_BASE="${OUTPUT_DIR}/data"
mkdir -p "${DATA_BASE}"

# ── Worker: one (robot, speed, method) triple ────────────────────────────────


# ── Collect NPZ → CSV ─────────────────────────────────────────────────────────
echo "Collecting results..."
RESULTS_CSV="${OUTPUT_DIR}/results.csv"
python3 "${COLLECT_SCRIPT}" \
    "${DATA_BASE}" \
    --output "${RESULTS_CSV}" \
    --skip-seconds "${SKIP_SECONDS}"

# ── Plot ──────────────────────────────────────────────────────────────────────
echo "Plotting..."
PLOTS_DIR="${OUTPUT_DIR}/plots"
python3 "${PLOT_SCRIPT}" "${RESULTS_CSV}" "${PLOTS_DIR}"
echo "Done. Results in ${OUTPUT_DIR}"
