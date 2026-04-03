#!/usr/bin/env bash
# Sweep angular_speed from pi*0.1 to pi*5.0 in steps of 0.1,
# run run_approach_then_hybrid_mujoco.py headless for each,
# collect avg force Z error, then plot angular_speed vs avg error.
# Saves individual simulation plots only at multipliers: 0.1, 0.5, 1.0, 1.5, ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLOT_SCRIPT="${SCRIPT_DIR}/plot_angular_speed_sweep.py"

# ---------------------------------------------------------------
# Configure the output subfolder name here
# Results and sweep plot will be saved to:
#   angular_speed_sweep_plots/<SWEEP_NAME>/
# ---------------------------------------------------------------
SWEEP_NAME="default"

OUTPUT_DIR="${SCRIPT_DIR}/angular_speed_sweep_plots/${SWEEP_NAME}"
mkdir -p "${OUTPUT_DIR}"
RESULTS_CSV="${OUTPUT_DIR}/sweep_results.csv"

# Angular speed multipliers that trigger saving individual plots
SAVE_PLOT_MULTIPLIERS="0.1 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0"

echo "multiplier,angular_speed_rad_s,ee_linear_speed_m_s,avg_force_z_error,avg_position_error" > "${RESULTS_CSV}"

for i in $(seq 1 50); do
    # multiplier = i * 0.1 (e.g. 1->0.1, 5->0.5, 10->1.0, ..., 50->5.0)
    MULTIPLIER=$(python3 -c "print(f'{$i * 0.1:.1f}')")
    ANGULAR_SPEED=$(python3 -c "import math; print(math.pi * $i * 0.1)")
    # EE linear speed = circle_radius * angular_speed (radius = 0.1 m)
    EE_LINEAR_SPEED=$(python3 -c "import math; print(0.1 * math.pi * $i * 0.1)")

    # Check if this multiplier is in the save-plot list
    SAVE_FLAG=""
    PLOT_DIR_FLAG=""
    SPEED_DIR=""
    for m in $SAVE_PLOT_MULTIPLIERS; do
        if [ "$MULTIPLIER" = "$m" ]; then
            SPEED_DIR="${OUTPUT_DIR}/speed_${MULTIPLIER}pi"
            SAVE_FLAG="--save-plots"
            PLOT_DIR_FLAG="--plot-dir ${SPEED_DIR}"
            break
        fi
    done

    echo ""
    echo "============================================================"
    echo "Running angular_speed = pi * ${MULTIPLIER} = ${ANGULAR_SPEED} rad/s  (EE linear speed = ${EE_LINEAR_SPEED} m/s)"
    if [ -n "$SAVE_FLAG" ]; then
        echo "  (saving plots to ${SPEED_DIR})"
    fi
    echo "============================================================"

    OUTPUT=$(python3 "${SCRIPT_DIR}/run_approach_then_hybrid_mujoco.py" \
        --headless \
        --angular-speed "${ANGULAR_SPEED}" \
        ${SAVE_FLAG} \
        ${PLOT_DIR_FLAG} \
        2>&1)

    echo "$OUTPUT"

    # Extract metrics from output
    AVG_FORCE_ERROR=$(echo "$OUTPUT" | grep "AVG_FORCE_Z_ERROR:" | tail -1 | awk '{print $2}')
    AVG_POS_ERROR=$(echo "$OUTPUT" | grep "AVG_POSITION_ERROR:" | tail -1 | awk '{print $2}')
    [ -z "$AVG_FORCE_ERROR" ] && AVG_FORCE_ERROR="nan"
    [ -z "$AVG_POS_ERROR" ] && AVG_POS_ERROR="nan"

    echo "${MULTIPLIER},${ANGULAR_SPEED},${EE_LINEAR_SPEED},${AVG_FORCE_ERROR},${AVG_POS_ERROR}" >> "${RESULTS_CSV}"
    echo "  -> avg_force_z_error = ${AVG_FORCE_ERROR}  avg_position_error = ${AVG_POS_ERROR}"
done

echo ""
echo "============================================================"
echo "Sweep complete. Results saved to ${RESULTS_CSV}"
echo "Generating summary plot..."
echo "============================================================"

python3 "${PLOT_SCRIPT}" "${RESULTS_CSV}" "${OUTPUT_DIR}"
