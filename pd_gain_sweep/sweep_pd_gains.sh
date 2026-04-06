#!/usr/bin/env bash
# 2-D grid search over PD gains (Kp_force x Kd_force).
# Runs run_approach_then_hybrid_mujoco.py headless for every (Kp, Kd) pair,
# collects steady-state force/position error metrics, then plots a heatmap.
# Runs NUM_WORKERS simulations in parallel.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/.."
PLOT_SCRIPT="${SCRIPT_DIR}/plot_pd_gain_sweep.py"

# ---------------------------------------------------------------
# Configure the sweep here
# Results saved to: pd_gain_sweep/plots/<SWEEP_NAME>/
#
# FORCE_CONTROL_METHOD: "pd"
# ANGULAR_SPEED: fixed speed for all runs (rad/s)
# SKIP_SECONDS: seconds of data to skip (transient) when computing metrics
# KP_VALUES / KD_VALUES: space-separated lists of gain values to sweep
# NUM_WORKERS: parallel simulations
# ---------------------------------------------------------------
SWEEP_NAME="pd_gain_sweep"
FORCE_CONTROL_METHOD="pd"
ANGULAR_SPEED="6.283185307179586"   # pi*2 rad/s
SKIP_SECONDS="1.0"

KP_VALUES="0.5 1.0 2.0 3.0 5.0 8.0"
KD_VALUES="0.5 1.0 2.0 3.0 5.0 8.0"
NUM_WORKERS=10

OUTPUT_DIR="${SCRIPT_DIR}/plots/${SWEEP_NAME}"
TMP_DIR="${OUTPUT_DIR}/tmp_results"
RESULTS_CSV="${OUTPUT_DIR}/sweep_results.csv"
mkdir -p "${OUTPUT_DIR}" "${TMP_DIR}"

# ---------------------------------------------------------------
# Worker function: runs one (Kp, Kd) pair, writes result to a temp file.
# ---------------------------------------------------------------
run_one() {
    local idx=$1
    local KP=$2
    local KD=$3

    echo "[worker ${idx}] Kp=${KP}  Kd=${KD}"

    local OUTPUT
    OUTPUT=$(python3 "${REPO_DIR}/run_approach_then_hybrid_mujoco.py" \
        --headless \
        --force-control-method "${FORCE_CONTROL_METHOD}" \
        --angular-speed "${ANGULAR_SPEED}" \
        --kp-force "${KP}" \
        --kd-force "${KD}" \
        --skip-seconds "${SKIP_SECONDS}" \
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

    echo "${KP},${KD},${AVG_FORCE_ERROR},${VAR_FORCE_ERROR},${AVG_POS_ERROR},${VAR_POS_ERROR}" \
        > "${TMP_DIR}/result_${idx}.csv"

    echo "[worker ${idx}] done — Kp=${KP} Kd=${KD}  avg_force_error=${AVG_FORCE_ERROR}  var=${VAR_FORCE_ERROR}"
}

export -f run_one
export TMP_DIR REPO_DIR FORCE_CONTROL_METHOD ANGULAR_SPEED SKIP_SECONDS

# ---------------------------------------------------------------
# Build full list of (Kp, Kd) pairs and dispatch workers
# ---------------------------------------------------------------
KP_ARR=($KP_VALUES)
KD_ARR=($KD_VALUES)

total=$(( ${#KP_ARR[@]} * ${#KD_ARR[@]} ))
echo "Starting PD gain sweep: ${#KP_ARR[@]} Kp values × ${#KD_ARR[@]} Kd values = ${total} runs  (NUM_WORKERS=${NUM_WORKERS})"

idx=0
active_jobs=0

for KP in "${KP_ARR[@]}"; do
    for KD in "${KD_ARR[@]}"; do
        run_one "$idx" "$KP" "$KD" &
        active_jobs=$(( active_jobs + 1 ))
        idx=$(( idx + 1 ))

        if [ "$active_jobs" -ge "$NUM_WORKERS" ]; then
            wait -n 2>/dev/null || wait
            active_jobs=$(( active_jobs - 1 ))
        fi
    done
done

wait
echo ""
echo "All workers finished. Merging results..."

# ---------------------------------------------------------------
# Merge temp files in order
# ---------------------------------------------------------------
echo "kp,kd,avg_force_z_error,var_force_z_error,avg_position_error,var_position_error" > "${RESULTS_CSV}"
for i in $(seq 0 $(( total - 1 ))); do
    f="${TMP_DIR}/result_${i}.csv"
    [ -f "$f" ] && cat "$f" >> "${RESULTS_CSV}"
done
rm -rf "${TMP_DIR}"

echo ""
echo "============================================================"
echo "Sweep complete. Results saved to ${RESULTS_CSV}"
echo "Generating heatmap..."
echo "============================================================"

python3 "${PLOT_SCRIPT}" "${RESULTS_CSV}" "${OUTPUT_DIR}"
