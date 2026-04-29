#!/usr/bin/env bash
# Sweep angular_speed from pi*0.1 to pi*5.0 in steps of 0.1 using the
# Baseline controller on a 30-degree slope surface.
#
# Results saved to:
#   flat_slope_experiments/E1_angular_speed_sweep/data/slope30/baseline/
#
# Usage (from workspace root):
#   bash flat_slope_experiments/E1_angular_speed_sweep/sweep_baseline_slope30.sh
#
# Environment overrides:
#   NUM_WORKERS    parallel jobs           (default: 10)
#   SKIP_SECONDS   burn-in seconds        (default: 1.0)
#   FORCE_DESIRED  desired force N        (default: -8.0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/../.."

# ── Configuration ─────────────────────────────────────────────────────────────
NUM_WORKERS="${NUM_WORKERS:-10}"
SKIP_SECONDS="${SKIP_SECONDS:-1.0}"
SLOPE_ANGLE="30.0"
FORCE_DESIRED="${FORCE_DESIRED:--8.0}"

DATA_DIR="${SCRIPT_DIR}/data/slope30/baseline"
TMP_DIR="${DATA_DIR}/tmp_results"
RESULTS_CSV="${DATA_DIR}/sweep_results.csv"
mkdir -p "${DATA_DIR}" "${TMP_DIR}"

# ── Worker: one speed ─────────────────────────────────────────────────────────
run_one() {
    local i=$1

    local MULTIPLIER
    MULTIPLIER=$(python3 -c "print(f'{$i * 0.1:.1f}')")
    local ANGULAR_SPEED
    ANGULAR_SPEED=$(python3 -c "import math; print(math.pi * $i * 0.1)")
    local EE_LINEAR_SPEED
    EE_LINEAR_SPEED=$(python3 -c "import math; print(0.1 * math.pi * $i * 0.1)")

    echo "[worker ${i}] angular_speed = pi * ${MULTIPLIER} = ${ANGULAR_SPEED} rad/s  (v = ${EE_LINEAR_SPEED} m/s)"

    local OUTPUT
    OUTPUT=$(python3 "${REPO_DIR}/run_baseline_mujoco.py" \
        --headless \
        --angular-speed  "${ANGULAR_SPEED}" \
        --slope-angle    "${SLOPE_ANGLE}" \
        --force-desired  "${FORCE_DESIRED}" \
        --multiplier     "${MULTIPLIER}" \
        --skip-seconds   "${SKIP_SECONDS}" \
        --save-data \
        --data-dir       "${DATA_DIR}" \
        2>&1)

    local AVG_FORCE_ERROR VAR_FORCE_ERROR AVG_POS_ERROR VAR_POS_ERROR
    AVG_FORCE_ERROR=$(echo "$OUTPUT" | grep "AVG_FORCE_ERROR"    | tail -1 | awk '{print $NF}')
    VAR_FORCE_ERROR=$(echo "$OUTPUT" | grep "VAR_FORCE_ERROR"    | tail -1 | awk '{print $NF}')
    AVG_POS_ERROR=$(echo "$OUTPUT"   | grep "AVG_POSITION_ERROR" | tail -1 | awk '{print $NF}')
    VAR_POS_ERROR=$(echo "$OUTPUT"   | grep "VAR_POSITION_ERROR" | tail -1 | awk '{print $NF}')
    [ -z "$AVG_FORCE_ERROR" ] && AVG_FORCE_ERROR="nan"
    [ -z "$VAR_FORCE_ERROR" ] && VAR_FORCE_ERROR="nan"
    [ -z "$AVG_POS_ERROR"   ] && AVG_POS_ERROR="nan"
    [ -z "$VAR_POS_ERROR"   ] && VAR_POS_ERROR="nan"

    echo "${MULTIPLIER},${ANGULAR_SPEED},${EE_LINEAR_SPEED},${AVG_FORCE_ERROR},${VAR_FORCE_ERROR},${AVG_POS_ERROR},${VAR_POS_ERROR}" \
        > "${TMP_DIR}/result_${i}.csv"

    echo "[worker ${i}] done — avg_force_error=${AVG_FORCE_ERROR}  avg_pos_error=${AVG_POS_ERROR}"
}

export -f run_one
export DATA_DIR TMP_DIR REPO_DIR SLOPE_ANGLE FORCE_DESIRED SKIP_SECONDS

# ── Dispatch workers ──────────────────────────────────────────────────────────
echo "Starting baseline slope30 sweep  (50 speeds, NUM_WORKERS=${NUM_WORKERS})..."
echo "Output: ${DATA_DIR}"
echo ""
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

# ── Merge temp files in order ─────────────────────────────────────────────────
echo "multiplier,angular_speed_rad_s,ee_linear_speed_m_s,avg_force_z_error,var_force_z_error,avg_position_error,var_position_error" > "${RESULTS_CSV}"
for i in $(seq 1 50); do
    cat "${TMP_DIR}/result_${i}.csv" >> "${RESULTS_CSV}"
done
rm -rf "${TMP_DIR}"

echo ""
echo "Sweep complete."
echo "Data:    ${DATA_DIR}/data_*.npz"
echo "Results: ${RESULTS_CSV}"
