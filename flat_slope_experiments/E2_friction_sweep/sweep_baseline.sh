#!/usr/bin/env bash
# Sweep surface friction coefficient from 0.1 to 1.0 (step 0.1) using the
# Baseline controller at a fixed angular speed (mirrors sweep_friction.sh).
#
# Results saved to:
#   flat_slope_experiments/E2_friction_sweep/data/baseline/
#
# Usage (from workspace root):
#   bash flat_slope_experiments/E2_friction_sweep/sweep_baseline.sh
#
# Environment overrides:
#   NUM_WORKERS    parallel jobs              (default: 10)
#   SKIP_SECONDS   burn-in seconds            (default: 1.0)
#   ANGULAR_SPEED  fixed angular speed rad/s  (default: pi*2 ≈ 6.283)
#   FORCE_DESIRED  desired contact force N    (default: -8.0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/../.."

# ── Configuration ─────────────────────────────────────────────────────────────
NUM_WORKERS="${NUM_WORKERS:-10}"
SKIP_SECONDS="${SKIP_SECONDS:-1.0}"
FORCE_DESIRED="${FORCE_DESIRED:--8.0}"
# Match the fixed speed used for paper/paper_pi runs (default: pi*2)
ANGULAR_SPEED="${ANGULAR_SPEED:-$(python3 -c "import math; print(math.pi * 2)")}"

DATA_DIR="${SCRIPT_DIR}/data/baseline"
TMP_DIR="${DATA_DIR}/tmp_results"
RESULTS_CSV="${DATA_DIR}/sweep_results.csv"
mkdir -p "${DATA_DIR}" "${TMP_DIR}"

# ── Worker: one friction coefficient ─────────────────────────────────────────
run_one() {
    local i=$1

    local MU
    MU=$(python3 -c "print(f'{$i * 0.1:.1f}')")

    echo "[worker ${i}] surface_friction mu = ${MU}"

    local OUTPUT
    OUTPUT=$(python3 "${REPO_DIR}/run_baseline_mujoco.py" \
        --headless \
        --robot            fr3_friction \
        --surface-friction "${MU}" \
        --angular-speed    "${ANGULAR_SPEED}" \
        --force-desired    "${FORCE_DESIRED}" \
        --multiplier       "${MU}" \
        --skip-seconds     "${SKIP_SECONDS}" \
        --save-data \
        --data-dir         "${DATA_DIR}" \
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

    echo "${MU},${AVG_FORCE_ERROR},${VAR_FORCE_ERROR},${AVG_POS_ERROR},${VAR_POS_ERROR}" \
        > "${TMP_DIR}/result_${i}.csv"

    echo "[worker ${i}] done — mu=${MU}  avg_force_error=${AVG_FORCE_ERROR}  avg_pos_error=${AVG_POS_ERROR}"
}

export -f run_one
export DATA_DIR TMP_DIR REPO_DIR ANGULAR_SPEED FORCE_DESIRED SKIP_SECONDS

# ── Dispatch workers ──────────────────────────────────────────────────────────
echo "Starting baseline friction sweep (10 coefficients, NUM_WORKERS=${NUM_WORKERS})..."
echo "Angular speed: ${ANGULAR_SPEED} rad/s"
echo "Output: ${DATA_DIR}"
echo ""
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

# ── Merge temp files in order (mu 0.1 → 1.0) ─────────────────────────────────
echo "friction_coeff,avg_force_z_error,var_force_z_error,avg_position_error,var_position_error" > "${RESULTS_CSV}"
for i in $(seq 1 10); do
    cat "${TMP_DIR}/result_${i}.csv" >> "${RESULTS_CSV}"
done
rm -rf "${TMP_DIR}"

echo ""
echo "Sweep complete."
echo "Data:    ${DATA_DIR}/data_*.npz  (data_0.1.npz .. data_1.0.npz)"
echo "Results: ${RESULTS_CSV}"
