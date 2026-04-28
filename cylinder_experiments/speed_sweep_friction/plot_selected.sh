#!/usr/bin/env bash
# Generate selected speed_sweep_friction plots.
#
# Usage (from workspace root):
#   bash cylinder_experiments/speed_sweep_friction/plot_selected.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

DATA_BASE="${SCRIPT_DIR}/sweep_friction_results/data"
RESULTS_CSV="${SCRIPT_DIR}/sweep_friction_results/results.csv"
PLOTS_DIR="${SCRIPT_DIR}/sweep_friction_results/plots"
TRAJ_DIR="${PLOTS_DIR}/traj"

TRAJ_SCRIPT="${SCRIPT_DIR}/plot_trajectory_and_force.py"
CMP_SCRIPT="${SCRIPT_DIR}/plot_comparison.py"
COLLECT="${SCRIPT_DIR}/collect_results.py"

mkdir -p "$TRAJ_DIR" "$PLOTS_DIR"

# ── 1. Collect NPZ → CSV ──────────────────────────────────────────────────────
echo "[1/4] Collecting results..."
python "$COLLECT" "$DATA_BASE" --output "$RESULTS_CSV" --skip-seconds 1.0

# ── 2. comparison_force (default grid mode, also produces comparison_yz) ─────
echo "[2/4] comparison_force plots..."
for robot in fr3_friction fr3_jointf_surff; do
    for mult in 0.1 0.5 1.0; do
        echo "  robot=${robot}  mult=${mult}"
        python "$TRAJ_SCRIPT" --robot "$robot" --multiplier "$mult" --output-dir "$TRAJ_DIR"
    done
done

# ── 3. overlay_yz ─────────────────────────────────────────────────────────────
echo "[3/4] overlay_yz plots..."
for robot in fr3_friction fr3_jointf_surff; do
    for mult in 0.1 0.5 1.0; do
        echo "  robot=${robot}  mult=${mult}"
        python "$TRAJ_SCRIPT" --robot "$robot" --multiplier "$mult" --output-dir "$TRAJ_DIR" --overlay
    done
done

# ── 4. force/position error combined ─────────────────────────────────────────
echo "[4/4] force/position error combined plots..."
python "$CMP_SCRIPT" "$RESULTS_CSV" "$PLOTS_DIR" --max-multiplier 999

echo ""
echo "Done. Plots in ${PLOTS_DIR}"
