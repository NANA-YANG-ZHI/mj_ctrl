#!/usr/bin/env bash
# Sweep angular_speed on a 30-degree slope.
# Same structure as sweep_angular_speed.sh but passes --slope-angle 30
# and uses a slope-specific SWEEP_NAME so results land in a separate directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/.."
PLOT_SCRIPT="${SCRIPT_DIR}/plot_angular_speed_sweep.py"

SWEEP_NAME="${SWEEP_NAME:-slope30_feedforward_pi}"
FORCE_CONTROL_METHOD="${FORCE_CONTROL_METHOD:-feedforward}"
USE_PI="${USE_PI:-true}"
KP_FORCE="${KP_FORCE:-2.0}"
KD_FORCE="${KD_FORCE:-}"
KI_FORCE="${KI_FORCE:-5.0}"
NUM_WORKERS="${NUM_WORKERS:-10}"
SKIP_SECONDS="${SKIP_SECONDS:-1.0}"
SLOPE_ANGLE="${SLOPE_ANGLE:-30.0}"

OUTPUT_DIR="${SCRIPT_DIR}/plots/${SWEEP_NAME}"
TMP_DIR="${OUTPUT_DIR}/tmp_results"
DATA_DIR="${OUTPUT_DIR}/data"
RESULTS_CSV="${OUTPUT_DIR}/sweep_results.csv"
mkdir -p "${OUTPUT_DIR}" "${TMP_DIR}" "${DATA_DIR}"

SAVE_PLOT_MULTIPLIERS="0.1 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0"

run_one() {
    local i=$1

    local MULTIPLIER
    MULTIPLIER=$(python3 -c "print(f'{$i * 0.1:.1f}')")
    local ANGULAR_SPEED
    ANGULAR_SPEED=$(python3 -c "import math; print(math.pi * $i * 0.1)")
    local EE_LINEAR_SPEED
    EE_LINEAR_SPEED=$(python3 -c "import math; print(0.1 * math.pi * $i * 0.1)")

    local SAVE_FLAG=""
    local PLOT_DIR_FLAG=""
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

    local KP_FLAG="" KD_FLAG="" KI_FLAG=""
    [ -n "${KP_FORCE}" ] && KP_FLAG="--kp-force ${KP_FORCE}"
    [ -n "${KD_FORCE}" ] && KD_FLAG="--kd-force ${KD_FORCE}"
    [ -n "${KI_FORCE}" ] && KI_FLAG="--ki-force ${KI_FORCE}"

    local OUTPUT
    OUTPUT=$(python3 "${REPO_DIR}/run_approach_then_hybrid_mujoco.py" \
        --headless \
        --angular-speed "${ANGULAR_SPEED}" \
        --force-control-method "${FORCE_CONTROL_METHOD}" \
        --multiplier "${MULTIPLIER}" \
        --skip-seconds "${SKIP_SECONDS}" \
        --save-data \
        --data-dir "${DATA_DIR}" \
        --slope-angle "${SLOPE_ANGLE}" \
        ${PI_FLAG} \
        ${KP_FLAG} \
        ${KD_FLAG} \
        ${KI_FLAG} \
        ${SAVE_FLAG} \
        ${PLOT_DIR_FLAG} \
        2>&1)

    local AVG_FORCE_ERROR VAR_FORCE_ERROR AVG_POS_ERROR VAR_POS_ERROR
    AVG_FORCE_ERROR=$(echo "$OUTPUT" | grep "AVG_FORCE_Z_ERROR:" | tail -1 | awk '{print $2}')
    VAR_FORCE_ERROR=$(echo "$OUTPUT" | grep "VAR_FORCE_Z_ERROR:" | tail -1 | awk '{print $2}')
    AVG_POS_ERROR=$(echo "$OUTPUT"   | grep "AVG_POSITION_ERROR:" | tail -1 | awk '{print $2}')
    VAR_POS_ERROR=$(echo "$OUTPUT"   | grep "VAR_POSITION_ERROR:" | tail -1 | awk '{print $2}')
    [ -z "$AVG_FORCE_ERROR" ] && AVG_FORCE_ERROR="nan"
    [ -z "$VAR_FORCE_ERROR" ] && VAR_FORCE_ERROR="nan"
    [ -z "$AVG_POS_ERROR"   ] && AVG_POS_ERROR="nan"
    [ -z "$VAR_POS_ERROR"   ] && VAR_POS_ERROR="nan"

    echo "${MULTIPLIER},${ANGULAR_SPEED},${EE_LINEAR_SPEED},${AVG_FORCE_ERROR},${VAR_FORCE_ERROR},${AVG_POS_ERROR},${VAR_POS_ERROR}" \
        > "${TMP_DIR}/result_${i}.csv"

    echo "[worker ${i}] done — avg_force_z_error=${AVG_FORCE_ERROR}  var=${VAR_FORCE_ERROR}  avg_pos_error=${AVG_POS_ERROR}  var=${VAR_POS_ERROR}"
}

export -f run_one
export OUTPUT_DIR TMP_DIR DATA_DIR REPO_DIR FORCE_CONTROL_METHOD USE_PI KP_FORCE KD_FORCE KI_FORCE SKIP_SECONDS SLOPE_ANGLE SAVE_PLOT_MULTIPLIERS

echo "Starting slope-30 sweep with NUM_WORKERS=${NUM_WORKERS} (50 speeds total)..."
active_jobs=0

for i in $(seq 1 50); do
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
