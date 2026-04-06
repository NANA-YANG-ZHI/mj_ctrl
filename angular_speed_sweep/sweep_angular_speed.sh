#!/usr/bin/env bash
# Sweep angular_speed from pi*0.1 to pi*5.0 in steps of 0.1,
# run run_approach_then_hybrid_mujoco.py headless for each,
# collect avg force Z error, then plot angular_speed vs avg error.
# Saves individual simulation plots only at multipliers: 0.1, 0.5, 1.0, 1.5, ...
# Runs NUM_WORKERS simulations in parallel.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/.."
PLOT_SCRIPT="${SCRIPT_DIR}/plot_angular_speed_sweep.py"

# ---------------------------------------------------------------
# Configure the sweep here
# Results and sweep plot will be saved to:
#   angular_speed_sweep/plots/<SWEEP_NAME>/
# Individual speed plots:
#   angular_speed_sweep/plots/<SWEEP_NAME>/speed_<mult>pi/
#
# FORCE_CONTROL_METHOD: "paper" | "pd" | "feedforward"
# USE_PI: "true" to add PI correction on top of the method, "false" otherwise
# NUM_WORKERS: how many simulations to run in parallel
# ---------------------------------------------------------------
SWEEP_NAME="pd"
FORCE_CONTROL_METHOD="pd"
USE_PI="false"
NUM_WORKERS=10

OUTPUT_DIR="${SCRIPT_DIR}/plots/${SWEEP_NAME}"
TMP_DIR="${OUTPUT_DIR}/tmp_results"
RESULTS_CSV="${OUTPUT_DIR}/sweep_results.csv"
mkdir -p "${OUTPUT_DIR}" "${TMP_DIR}"

# Angular speed multipliers that trigger saving individual plots
SAVE_PLOT_MULTIPLIERS="0.1 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0"
# SAVE_PLOT_MULTIPLIERS="0.1 0.5 1.0"

# ---------------------------------------------------------------
# Worker function: runs one speed, writes result to a temp file.
# Called in a subshell via & so must not share state.
# ---------------------------------------------------------------
run_one() {
    local i=$1

    local MULTIPLIER
    MULTIPLIER=$(python3 -c "print(f'{$i * 0.1:.1f}')")
    local ANGULAR_SPEED
    ANGULAR_SPEED=$(python3 -c "import math; print(math.pi * $i * 0.1)")
    local EE_LINEAR_SPEED
    EE_LINEAR_SPEED=$(python3 -c "import math; print(0.1 * math.pi * $i * 0.1)")

    # Check if this multiplier is in the save-plot list
    local SAVE_FLAG=""
    local PLOT_DIR_FLAG=""
    local SPEED_DIR=""
    for m in $SAVE_PLOT_MULTIPLIERS; do
        if [ "$MULTIPLIER" = "$m" ]; then
            SPEED_DIR="${OUTPUT_DIR}/speed_${MULTIPLIER}pi"
            SAVE_FLAG="--save-plots"
            PLOT_DIR_FLAG="--plot-dir ${SPEED_DIR}"
            break
        fi
    done

    echo "[worker ${i}] angular_speed = pi * ${MULTIPLIER} = ${ANGULAR_SPEED} rad/s  (v = ${EE_LINEAR_SPEED} m/s)"

    local PI_FLAG=""
    [ "${USE_PI}" = "true" ] && PI_FLAG="--use-pi"

    local OUTPUT
    OUTPUT=$(python3 "${REPO_DIR}/run_approach_then_hybrid_mujoco.py" \
        --headless \
        --angular-speed "${ANGULAR_SPEED}" \
        --force-control-method "${FORCE_CONTROL_METHOD}" \
        ${PI_FLAG} \
        ${SAVE_FLAG} \
        ${PLOT_DIR_FLAG} \
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

    # Write to per-run temp file (no shared file = no race condition)
    echo "${MULTIPLIER},${ANGULAR_SPEED},${EE_LINEAR_SPEED},${AVG_FORCE_ERROR},${VAR_FORCE_ERROR},${AVG_POS_ERROR},${VAR_POS_ERROR}" \
        > "${TMP_DIR}/result_${i}.csv"

    echo "[worker ${i}] done — avg_force_z_error=${AVG_FORCE_ERROR}  var=${VAR_FORCE_ERROR}  avg_pos_error=${AVG_POS_ERROR}  var=${VAR_POS_ERROR}"
}

export -f run_one
export OUTPUT_DIR TMP_DIR REPO_DIR FORCE_CONTROL_METHOD USE_PI SAVE_PLOT_MULTIPLIERS

# ---------------------------------------------------------------
# Dispatch workers with a simple job-pool (no GNU parallel needed)
# ---------------------------------------------------------------
echo "Starting sweep with NUM_WORKERS=${NUM_WORKERS} (50 speeds total)..."
active_jobs=0

for i in $(seq 1 50); do
    run_one "$i" &
    active_jobs=$(( active_jobs + 1 ))

    # When the pool is full, wait for any one job to finish before launching the next
    if [ "$active_jobs" -ge "$NUM_WORKERS" ]; then
        wait -n 2>/dev/null || wait   # wait -n requires bash 4.3+; fall back to wait
        active_jobs=$(( active_jobs - 1 ))
    fi
done

# Wait for all remaining jobs
wait
echo ""
echo "All workers finished. Merging results..."

# ---------------------------------------------------------------
# Merge temp files in order (i=1..50 → multiplier 0.1..5.0)
# ---------------------------------------------------------------
echo "multiplier,angular_speed_rad_s,ee_linear_speed_m_s,avg_force_z_error,var_force_z_error,avg_position_error,var_position_error" > "${RESULTS_CSV}"
for i in $(seq 1 50); do
    cat "${TMP_DIR}/result_${i}.csv" >> "${RESULTS_CSV}"
done
rm -rf "${TMP_DIR}"

echo ""
echo "============================================================"
echo "Sweep complete. Results saved to ${RESULTS_CSV}"
echo "Generating summary plot..."
echo "============================================================"

python3 "${PLOT_SCRIPT}" "${RESULTS_CSV}" "${OUTPUT_DIR}"
