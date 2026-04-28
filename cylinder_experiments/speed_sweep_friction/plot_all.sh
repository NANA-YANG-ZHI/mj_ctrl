#!/usr/bin/env bash
# Generate all speed_sweep_friction plots.
#
# Usage (from workspace root):
#   bash cylinder_experiments/speed_sweep_friction/plot_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

DATA_BASE="${SCRIPT_DIR}/sweep_friction_results/data"
RESULTS_CSV="${SCRIPT_DIR}/sweep_friction_results/results.csv"
PLOTS_DIR="${SCRIPT_DIR}/sweep_friction_results/plots"
TRAJ_DIR="${PLOTS_DIR}/traj"

TRAJ_SCRIPT="${SCRIPT_DIR}/plot_trajectory_and_force.py"
POS_SCRIPT="${SCRIPT_DIR}/plot_positions_vs_time.py"
CMP_SCRIPT="${SCRIPT_DIR}/plot_comparison.py"
COLLECT="${SCRIPT_DIR}/collect_results.py"

mkdir -p "$TRAJ_DIR" "$PLOTS_DIR"

# ── 1. Collect NPZ → CSV ──────────────────────────────────────────────────────
echo "[1/4] Collecting results..."
python3 "$COLLECT" "$DATA_BASE" --output "$RESULTS_CSV" --skip-seconds 1.0

# ── 2. comparison_force + comparison_yz (default mode) ───────────────────────
echo "[2/4] Trajectory & force comparison grids..."
for robot in fr3_friction fr3_jointf_surff; do
    for mult in 0.1 0.5 1.0; do
        echo "  plot_trajectory_and_force  robot=${robot}  mult=${mult}"
        python3 "$TRAJ_SCRIPT" --robot "$robot" --multiplier "$mult" --output-dir "$TRAJ_DIR"
    done
done

# ── 3. overlay_yz ─────────────────────────────────────────────────────────────
echo "[3/4] Overlay trajectory plots..."
for robot in fr3_friction fr3_jointf_surff; do
    for mult in 0.1 0.5 1.0; do
        echo "  overlay  robot=${robot}  mult=${mult}"
        python3 "$TRAJ_SCRIPT" --robot "$robot" --multiplier "$mult" --output-dir "$TRAJ_DIR" --overlay
    done
done

# ── 4. pos_time plots ─────────────────────────────────────────────────────────
echo "[4/4] Position-vs-time plots..."
python3 "$POS_SCRIPT" --robot fr3_friction --multiplier 0.1 --method baseline --output-dir "$TRAJ_DIR"
python3 "$POS_SCRIPT" --robot fr3_friction --multiplier 0.5 --method baseline --output-dir "$TRAJ_DIR"
python3 "$POS_SCRIPT" --robot fr3_friction --multiplier 0.1 --method paper    --output-dir "$TRAJ_DIR"

# ── 5. force/position error combined & max ────────────────────────────────────
echo "[5/5] Summary comparison plots..."
python3 "$CMP_SCRIPT" "$RESULTS_CSV" "$PLOTS_DIR" --max-multiplier 999

echo ""
echo "Done. Plots in ${PLOTS_DIR}"
