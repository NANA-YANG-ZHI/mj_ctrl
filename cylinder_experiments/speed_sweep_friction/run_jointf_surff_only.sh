#!/usr/bin/env bash
# Rerun speed sweep for fr3_jointf_surff only (30 speeds × 6 methods = 180 jobs),
# then collect results and regenerate all plots.
#
# Usage (from workspace root):
#   bash cylinder_experiments/speed_sweep_friction/run_jointf_surff_only.sh
#
# Environment overrides:
#   NUM_WORKERS    parallel jobs                  (default: 10)
#   SKIP_SECONDS   burn-in seconds for collect    (default: 1.0)
#   TRAJECTORY     1 = 0°→60°, 2 = −60°→60°      (default: 2)
#   FORCE_DESIRED  desired contact force N         (default: -10.0)
#   ONLY_BASELINE  set to 1 to rerun baseline only (default: 0)

set -euo pipefail
export LC_NUMERIC=C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
COLLECT_SCRIPT="${SCRIPT_DIR}/collect_results.py"
TRAJ_SCRIPT="${SCRIPT_DIR}/plot_trajectory_and_force.py"
CMP_SCRIPT="${SCRIPT_DIR}/plot_comparison.py"

# ── Configuration ─────────────────────────────────────────────────────────────
NUM_WORKERS="${NUM_WORKERS:-10}"
SKIP_SECONDS="${SKIP_SECONDS:-1.0}"
TRAJECTORY="${TRAJECTORY:-2}"
FORCE_DESIRED="${FORCE_DESIRED:--10.0}"

ROBOT="fr3_jointf_surff"

ALL_METHODS=(
    "baseline:baseline:baseline:0"
    "ff:hybrid:feedforward:0"
    "ff_pi:hybrid:feedforward:1"
    "pd:hybrid:pd:0"
    "paper:hybrid:paper:0"
    "paper_pi:hybrid:paper:1"
)
if [ "${ONLY_BASELINE:-0}" = "1" ]; then
    METHODS=("baseline:baseline:baseline:0")
else
    METHODS=("${ALL_METHODS[@]}")
fi

OUTPUT_DIR="${SCRIPT_DIR}/sweep_friction_results"
DATA_BASE="${OUTPUT_DIR}/data"
mkdir -p "${DATA_BASE}"

# ── Worker: one (speed, method) pair ─────────────────────────────────────────
run_one() {
    local robot="$1"
    local mult="$2"
    local omega="$3"
    local method_name="$4"
    local script_type="$5"
    local force_method="$6"
    local use_pi="$7"
    local job_id="$8"

    local data_dir="${DATA_BASE}/${robot}/${method_name}"
    mkdir -p "$data_dir"

    echo "[${job_id}] START  method=${method_name}  omega=${omega} rad/s"

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

    echo "[${job_id}] DONE   method=${method_name}  omega=${omega}"
}

export -f run_one
export DATA_BASE REPO_DIR TRAJECTORY FORCE_DESIRED

# ── Dispatch jobs ─────────────────────────────────────────────────────────────
n_speeds=30
n_methods=${#METHODS[@]}
total=$(( n_speeds * n_methods ))

echo "Launching ${total} jobs  (${n_speeds} speeds × ${n_methods} methods, ${NUM_WORKERS} parallel)  robot=${ROBOT}"
echo ""

active=0
job_id=0

for i in $(seq 1 30); do
    mult=$(python3 -c "print(f'{$i * 0.1:.1f}')")
    omega=$(python3 -c "import math; print(math.pi * $i * 0.1)")
    for cfg in "${METHODS[@]}"; do
        IFS=':' read -r method_name script_type force_method use_pi <<< "$cfg"
        job_id=$(( job_id + 1 ))
        run_one "$ROBOT" "$mult" "$omega" "$method_name" "$script_type" "$force_method" "$use_pi" "$job_id" &
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
RESULTS_CSV="${OUTPUT_DIR}/results.csv"
python3 "${COLLECT_SCRIPT}" \
    "${DATA_BASE}" \
    --output "${RESULTS_CSV}" \
    --skip-seconds "${SKIP_SECONDS}"

# ── Plot ──────────────────────────────────────────────────────────────────────
PLOTS_DIR="${OUTPUT_DIR}/plots"
TRAJ_DIR="${PLOTS_DIR}/traj"
mkdir -p "$TRAJ_DIR" "$PLOTS_DIR"

echo "Plotting trajectory & force grids..."
for mult in 0.1 0.5 1.0; do
    python3 "$TRAJ_SCRIPT" --robot "$ROBOT" --multiplier "$mult" --output-dir "$TRAJ_DIR"
    python3 "$TRAJ_SCRIPT" --robot "$ROBOT" --multiplier "$mult" --output-dir "$TRAJ_DIR" --overlay
done

echo "Plotting summary comparison..."
python3 "$CMP_SCRIPT" "$RESULTS_CSV" "$PLOTS_DIR" --max-multiplier 999

echo ""
echo "Done. Results in ${OUTPUT_DIR}"
