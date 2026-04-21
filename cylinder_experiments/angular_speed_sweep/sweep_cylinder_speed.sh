#!/usr/bin/env bash
# Cylinder surface angular-speed sweep.
#
# Sweeps omega from 0.1 to 1.6 rad/s (step 0.1) for four robot configurations.
# Saves full time-series NPZ data per run, then collects results and plots.
#
# Usage (from workspace root):
#   bash cylinder_experiments/angular_speed_sweep/sweep_cylinder_speed.sh
#
# Environment overrides:
#   FORCE_CONTROL_METHOD   paper | pd | feedforward  (default: paper)
#   NUM_WORKERS            parallel jobs              (default: 8)
#   SKIP_SECONDS           burn-in seconds for collect script (default: 1.0)
#   TRAJECTORY             1 = 0°→75°, 2 = −75°→75°  (default: 1)
#   FORCE_DESIRED          desired contact force N     (default: -10.0)

set -euo pipefail
export LC_NUMERIC=C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
COLLECT_SCRIPT="${SCRIPT_DIR}/collect_cylinder_results.py"
PLOT_SCRIPT="${SCRIPT_DIR}/plot_cylinder_speed_sweep.py"

# ── Configuration ─────────────────────────────────────────────────────────────
FORCE_CONTROL_METHOD="${FORCE_CONTROL_METHOD:-paper}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SKIP_SECONDS="${SKIP_SECONDS:-1.0}"
TRAJECTORY="${TRAJECTORY:-1}"
FORCE_DESIRED="${FORCE_DESIRED:--10.0}"

ROBOTS=(fr3 fr3_friction fr3_jointf fr3_jointf_surff)

# Angular speeds 0.1 … 1.6 rad/s, step 0.1  (16 values)
OMEGAS=()
for i in $(seq 1 16); do
    OMEGAS+=( "$(python3 -c "print(f'{$i * 0.1:.1f}')")" )
done

OUTPUT_DIR="${SCRIPT_DIR}/sweep_speed_results/${FORCE_CONTROL_METHOD}"
DATA_BASE="${OUTPUT_DIR}/data"
mkdir -p "${DATA_BASE}"

# ── Worker: one (robot, omega) pair ──────────────────────────────────────────
run_one() {
    local robot="$1"
    local omega="$2"
    local job_id="$3"

    local data_dir="${DATA_BASE}/${robot}"
    mkdir -p "$data_dir"

    echo "[${job_id}] START  robot=${robot}  omega=${omega} rad/s"

    python3 "${REPO_DIR}/run_approach_then_hybrid_mujoco.py" \
        --cylinder \
        --robot                 "${robot}" \
        --angular-speed         "${omega}" \
        --trajectory            "${TRAJECTORY}" \
        --force-desired         "${FORCE_DESIRED}" \
        --force-control-method  "${FORCE_CONTROL_METHOD}" \
        --multiplier            "${omega}" \
        --save-data \
        --data-dir              "${data_dir}" \
        --headless \
        > "${data_dir}/log_${omega}.txt" 2>&1

    echo "[${job_id}] DONE   robot=${robot}  omega=${omega}  → ${data_dir}/data_${omega}_all.npz"
}

export -f run_one
export DATA_BASE REPO_DIR FORCE_CONTROL_METHOD TRAJECTORY FORCE_DESIRED

# ── Dispatch jobs ─────────────────────────────────────────────────────────────
total=$(( ${#ROBOTS[@]} * ${#OMEGAS[@]} ))
echo "Launching ${total} jobs  (${#ROBOTS[@]} robots × ${#OMEGAS[@]} speeds, ${NUM_WORKERS} parallel)"
echo ""

active=0
job_id=0

for robot in "${ROBOTS[@]}"; do
    for omega in "${OMEGAS[@]}"; do
        job_id=$(( job_id + 1 ))
        run_one "$robot" "$omega" "$job_id" &
        active=$(( active + 1 ))

        if [ "$active" -ge "$NUM_WORKERS" ]; then
            wait -n 2>/dev/null || wait
            active=$(( active - 1 ))
        fi
    done
done

wait
echo ""
echo "All ${total} jobs finished."

# ── Collect NPZ → CSV ─────────────────────────────────────────────────────────
echo "Collecting results..."
RESULTS_CSV="${OUTPUT_DIR}/sweep_results.csv"
python3 "${COLLECT_SCRIPT}" \
    "${DATA_BASE}" \
    --output "${RESULTS_CSV}" \
    --skip-seconds "${SKIP_SECONDS}"

# ── Plot ──────────────────────────────────────────────────────────────────────
echo "Plotting..."
python3 "${PLOT_SCRIPT}" "${RESULTS_CSV}" "${OUTPUT_DIR}"
echo "Done. Results in ${OUTPUT_DIR}"
