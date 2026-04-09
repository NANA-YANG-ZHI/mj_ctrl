#!/usr/bin/env bash
# Sweep surface sliding friction coefficient from 0.1 to 1.0 in steps of 0.1 (10 values).
# Runs run_approach_then_hybrid_mujoco.py headless for each value,
# collects avg |force Z error|, variance, avg position error, variance,
# then generates a summary plot.
# Runs NUM_WORKERS simulations in parallel.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/.."
PLOT_SCRIPT="${SCRIPT_DIR}/plot_friction_sweep.py"

# ---------------------------------------------------------------
# Configure the sweep here
# Results and sweep plot will be saved to:
#   friction_sweep/plots/<SWEEP_NAME>/
#
# FORCE_CONTROL_METHOD: "paper" | "pd" | "feedforward"
# USE_PI: "true" to add PI correction, "false" otherwise
# ANGULAR_SPEED: fixed angular speed in rad/s (default: pi*2)
# NUM_WORKERS: how many simulations to run in parallel
# ---------------------------------------------------------------
SWEEP_NAME="${SWEEP_NAME:-paper}"
FORCE_CONTROL_METHOD="${FORCE_CONTROL_METHOD:-paper}"
USE_PI="${USE_PI:-false}"
KP_FORCE="${KP_FORCE:-}"
KD_FORCE="${KD_FORCE:-}"
KI_FORCE="${KI_FORCE:-}"
NUM_WORKERS="${NUM_WORKERS:-10}"
SKIP_SECONDS="${SKIP_SECONDS:-1.0}"
ANGULAR_SPEED="${ANGULAR_SPEED:-6.283185307179586}"  # pi*2

OUTPUT_DIR="${SCRIPT_DIR}/plots/${SWEEP_NAME}"
TMP_DIR="${OUTPUT_DIR}/tmp_results"
DATA_DIR="${OUTPUT_DIR}/data"
RESULTS_CSV="${OUTPUT_DIR}/sweep_results.csv"
mkdir -p "${OUTPUT_DIR}" "${TMP_DIR}" "${DATA_DIR}"

# ---------------------------------------------------------------
# Worker function: runs one friction value, writes result to temp file.
# ---------------------------------------------------------------
run_one() {
    local i=$1

    local MU
    MU=$(python3 -c "print(f'{$i * 0.1:.1f}')")

    echo "[worker ${i}] surface_friction mu = ${MU}"

    local PI_FLAG=""
    [ "${USE_PI}" = "true" ] && PI_FLAG="--use-pi"

    local KP_FLAG="" KD_FLAG="" KI_FLAG=""
    [ -n "${KP_FORCE}" ] && KP_FLAG="--kp-force ${KP_FORCE}"
    [ -n "${KD_FORCE}" ] && KD_FLAG="--kd-force ${KD_FORCE}"
    [ -n "${KI_FORCE}" ] && KI_FLAG="--ki-force ${KI_FORCE}"

    local OUTPUT
    OUTPUT=$(python3 "${REPO_DIR}/run_approach_then_hybrid_mujoco.py" \
        --robot fr3_friction \
        --surface-friction "${MU}" \
        --headless \
        --angular-speed "${ANGULAR_SPEED}" \
        --force-control-method "${FORCE_CONTROL_METHOD}" \
        --multiplier 2.0 \
        --run-id "${MU}" \
        --skip-seconds "${SKIP_SECONDS}" \
        --save-data \
        --data-dir "${DATA_DIR}" \
        ${PI_FLAG} \
        ${KP_FLAG} \
        ${KD_FLAG} \
        ${KI_FLAG} \
        2>&1)

    # Extract metrics
    local AVG_FORCE_ERROR VAR_FORCE_ERROR AVG_POS_ERROR VAR_POS_ERROR
    AVG_FORCE_ERROR=$(echo "$OUTPUT" | grep "AVG_FORCE_Z_ERROR:" | tail -1 | awk '{print $2}')
    VAR_FORCE_ERROR=$(echo "$OUTPUT" | grep "VAR_FORCE_Z_ERROR:" | tail -1 | awk '{print $2}')
    AVG_POS_ERROR=$(echo "$OUTPUT"   | grep "AVG_POSITION_ERROR:" | tail -1 | awk '{print $2}')
    VAR_POS_ERROR=$(echo "$OUTPUT"   | grep "VAR_POSITION_ERROR:" | tail -1 | awk '{print $2}')
    [ -z "$AVG_FORCE_ERROR" ] && AVG_FORCE_ERROR="nan"
    [ -z "$VAR_FORCE_ERROR" ] && VAR_FORCE_ERROR="nan"
    [ -z "$AVG_POS_ERROR"   ] && AVG_POS_ERROR="nan"
    [ -z "$VAR_POS_ERROR"   ] && VAR_POS_ERROR="nan"

    echo "${MU},${AVG_FORCE_ERROR},${VAR_FORCE_ERROR},${AVG_POS_ERROR},${VAR_POS_ERROR}" \
        > "${TMP_DIR}/result_${i}.csv"

    echo "[worker ${i}] done — mu=${MU}  avg_force_z_error=${AVG_FORCE_ERROR}  var=${VAR_FORCE_ERROR}  avg_pos_error=${AVG_POS_ERROR}  var_pos=${VAR_POS_ERROR}"
}

export -f run_one
export OUTPUT_DIR TMP_DIR DATA_DIR REPO_DIR FORCE_CONTROL_METHOD USE_PI KP_FORCE KD_FORCE KI_FORCE SKIP_SECONDS ANGULAR_SPEED

# ---------------------------------------------------------------
# Dispatch workers with a simple job-pool (no GNU parallel needed)
# ---------------------------------------------------------------
echo "Starting friction sweep with NUM_WORKERS=${NUM_WORKERS} (10 values: mu=0.1 to 1.0)..."
active_jobs=0

for i in $(seq 1 10); do
    run_one "$i" &
    active_jobs=$(( active_jobs + 1 ))

    if [ "$active_jobs" -ge "$NUM_WORKERS" ]; then
        wait -n 2>/dev/null || wait
        active_jobs=$(( active_jobs - 1 ))
    fi
done

wait
echo ""
echo "All workers finished. Merging results..."

# ---------------------------------------------------------------
# Merge temp files in order (i=1..10 → mu=0.1..1.0)
# ---------------------------------------------------------------
echo "friction_coeff,avg_force_z_error,var_force_z_error,avg_position_error,var_position_error" > "${RESULTS_CSV}"
for i in $(seq 1 10); do
    cat "${TMP_DIR}/result_${i}.csv" >> "${RESULTS_CSV}"
done
rm -rf "${TMP_DIR}"

echo ""
echo "============================================================"
echo "Sweep complete. Results saved to ${RESULTS_CSV}"
echo "Generating summary plot..."
echo "============================================================"

python3 "${PLOT_SCRIPT}" "${RESULTS_CSV}" "${OUTPUT_DIR}"
