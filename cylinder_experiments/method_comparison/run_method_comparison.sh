#!/usr/bin/env bash
# Cylinder surface: force-control method comparison sweep.
#
# Grid:
#   Speeds  : 0.3, 0.6, 0.9, 1.2, 1.6  rad/s
#   Methods : ff, ff_pi, pd, paper, paper_pi
#   Robots  : fr3, fr3_friction, fr3_jointf, fr3_jointf_surff
#   Total   : 5 × 5 × 4 = 100 jobs
#
# Data saved to:  sweep_method_results/data/<robot>/<method>/data_<speed>_all.npz
# Run collect + plot scripts after all jobs finish.
#
# Usage (from workspace root):
#   bash cylinder_experiments/method_comparison/run_method_comparison.sh
#
# Environment overrides:
#   NUM_WORKERS    parallel jobs         (default: 10)
#   SKIP_SECONDS   burn-in for collect   (default: 1.0)
#   TRAJECTORY     1 = 0°→75°, 2 = −75°→75°  (default: 1)
#   FORCE_DESIRED  desired contact force  (default: -10.0)

set -euo pipefail
export LC_NUMERIC=C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
COLLECT_SCRIPT="${SCRIPT_DIR}/collect_method_comparison.py"
PLOT_SCRIPT="${SCRIPT_DIR}/plot_method_comparison.py"

# ── Configuration ─────────────────────────────────────────────────────────────
NUM_WORKERS="${NUM_WORKERS:-10}"
SKIP_SECONDS="${SKIP_SECONDS:-1.0}"
TRAJECTORY="${TRAJECTORY:-1}"
FORCE_DESIRED="${FORCE_DESIRED:--10.0}"

ROBOTS=(fr3 fr3_friction fr3_jointf fr3_jointf_surff)
SPEEDS=(0.3 0.6 0.9 1.2 1.6)

# method_name : force_control_method : use_pi (0|1)
METHODS=(
    "ff:feedforward:0"
    "ff_pi:feedforward:1"
    "pd:pd:0"
    "paper:paper:0"
    "paper_pi:paper:1"
)

OUTPUT_DIR="${SCRIPT_DIR}/sweep_method_results"
DATA_BASE="${OUTPUT_DIR}/data"
mkdir -p "${DATA_BASE}"

# ── Worker ────────────────────────────────────────────────────────────────────
run_one() {
    local robot="$1"
    local speed="$2"
    local method_name="$3"
    local force_method="$4"
    local use_pi="$5"
    local job_id="$6"

    local data_dir="${DATA_BASE}/${robot}/${method_name}"
    mkdir -p "$data_dir"

    local PI_FLAG=""
    [ "$use_pi" = "1" ] && PI_FLAG="--use-pi"

    local CMD=(python3 "${REPO_DIR}/run_approach_then_hybrid_mujoco.py"
        --cylinder
        --robot                 "${robot}"
        --angular-speed         "${speed}"
        --trajectory            "${TRAJECTORY}"
        --force-desired         "${FORCE_DESIRED}"
        --force-control-method  "${force_method}"
        --multiplier            "${speed}"
        --save-data
        --data-dir              "${data_dir}"
        --headless
    )
    [ -n "$PI_FLAG" ] && CMD+=("$PI_FLAG")

    echo "[${job_id}] START  robot=${robot}  method=${method_name}  omega=${speed} rad/s"
    echo "[${job_id}] CMD    ${CMD[*]}"

    "${CMD[@]}" > "${data_dir}/log_${speed}.txt" 2>&1

    echo "[${job_id}] DONE   robot=${robot}  method=${method_name}  omega=${speed}  → ${data_dir}/data_${speed}_all.npz"
}

export -f run_one
export DATA_BASE REPO_DIR TRAJECTORY FORCE_DESIRED

# ── Dispatch ──────────────────────────────────────────────────────────────────
total=$(( ${#ROBOTS[@]} * ${#SPEEDS[@]} * ${#METHODS[@]} ))
echo "Launching ${total} jobs  (${#ROBOTS[@]} robots × ${#SPEEDS[@]} speeds × ${#METHODS[@]} methods, ${NUM_WORKERS} parallel)"
echo ""

active=0
job_id=0

for robot in "${ROBOTS[@]}"; do
    for speed in "${SPEEDS[@]}"; do
        for cfg in "${METHODS[@]}"; do
            IFS=':' read -r method_name force_method use_pi <<< "$cfg"
            job_id=$(( job_id + 1 ))
            run_one "$robot" "$speed" "$method_name" "$force_method" "$use_pi" "$job_id" &
            active=$(( active + 1 ))

            if [ "$active" -ge "$NUM_WORKERS" ]; then
                wait -n 2>/dev/null || wait
                active=$(( active - 1 ))
            fi
        done
    done
done

wait
echo ""
echo "All ${total} jobs finished."

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
