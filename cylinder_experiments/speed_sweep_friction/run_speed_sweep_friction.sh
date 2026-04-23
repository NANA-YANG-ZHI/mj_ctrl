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
#   TRAJECTORY     1 = 0°→75°, 2 = −75°→75°      (default: 1)
#   FORCE_DESIRED  desired contact force N         (default: -10.0)

set -euo pipefail
export LC_NUMERIC=C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
COLLECT_SCRIPT="${SCRIPT_DIR}/collect_results.py"
PLOT_SCRIPT="${SCRIPT_DIR}/plot_comparison.py"

# ── Configuration ─────────────────────────────────────────────────────────────
NUM_WORKERS="${NUM_WORKERS:-10}"
SKIP_SECONDS="${SKIP_SECONDS:-1.0}"
TRAJECTORY="${TRAJECTORY:-1}"
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
run_one() {
    local robot="$1"
    local mult="$2"        # e.g. "0.1"
    local omega="$3"       # e.g. "0.31415..."
    local method_name="$4"
    local script_type="$5" # "baseline" or "hybrid"
    local force_method="$6"
    local use_pi="$7"
    local job_id="$8"

    local data_dir="${DATA_BASE}/${robot}/${method_name}"
    mkdir -p "$data_dir"

    echo "[${job_id}] START  robot=${robot}  method=${method_name}  omega=${omega} rad/s"

    if [ "$script_type" = "baseline" ]; then
        python3 "${REPO_DIR}/run_baseline_mujoco.py" \
            --cylinder \
            --robot                 "${robot}" \
            --angular-speed         "${omega}" \
            --trajectory            "${TRAJECTORY}" \
            --force-desired         "${FORCE_DESIRED}" \
            --multiplier            "${mult}" \
            --save-data \
            --data-dir              "${data_dir}" \
            --headless \
            > "${data_dir}/log_${mult}.txt" 2>&1
    else
        local PI_FLAG=""
        [ "$use_pi" = "1" ] && PI_FLAG="--use-pi"

        local CMD=(python3 "${REPO_DIR}/run_approach_then_hybrid_mujoco.py"
            --cylinder
            --robot                 "${robot}"
            --angular-speed         "${omega}"
            --trajectory            "${TRAJECTORY}"
            --force-desired         "${FORCE_DESIRED}"
            --force-control-method  "${force_method}"
            --multiplier            "${mult}"
            --save-data
            --data-dir              "${data_dir}"
            --headless
        )
        [ -n "$PI_FLAG" ] && CMD+=("$PI_FLAG")
        "${CMD[@]}" > "${data_dir}/log_${mult}.txt" 2>&1
    fi

    echo "[${job_id}] DONE   robot=${robot}  method=${method_name}  omega=${omega}"
}

export -f run_one
export DATA_BASE REPO_DIR TRAJECTORY FORCE_DESIRED

# ── Dispatch jobs ─────────────────────────────────────────────────────────────
n_robots=${#ROBOTS[@]}
n_speeds=30
n_methods=${#METHODS[@]}
total=$(( n_robots * n_speeds * n_methods ))

echo "Launching ${total} jobs  (${n_robots} robots × ${n_speeds} speeds × ${n_methods} methods, ${NUM_WORKERS} parallel)"
echo ""

active=0
job_id=0

for robot in "${ROBOTS[@]}"; do
    for i in $(seq 1 30); do
        mult=$(python3 -c "print(f'{$i * 0.1:.1f}')")
        omega=$(python3 -c "import math; print(math.pi * $i * 0.1)")
        for cfg in "${METHODS[@]}"; do
            IFS=':' read -r method_name script_type force_method use_pi <<< "$cfg"
            job_id=$(( job_id + 1 ))
            run_one "$robot" "$mult" "$omega" "$method_name" "$script_type" "$force_method" "$use_pi" "$job_id" &
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
