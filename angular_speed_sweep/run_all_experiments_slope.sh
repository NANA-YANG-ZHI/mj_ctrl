#!/usr/bin/env bash
# Run the same 5 angular speed sweep experiments as run_all_experiments.sh
# but on a tilted slope surface.
#
# Set SLOPE_ANGLE below (degrees). Output goes to:
#   angular_speed_sweep/plots/slope<angle>_<method>/
# which is separate from the flat-surface results.

set -euo pipefail

# ---------------------------------------------------------------
# Configure slope angle here
# ---------------------------------------------------------------
SLOPE_ANGLE="30.0"

# ---------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP="${SCRIPT_DIR}/sweep_angular_speed.sh"

echo "========================================================"
echo "Running all 5 angular speed sweep experiments"
echo "Slope angle: ${SLOPE_ANGLE} degrees"
echo "========================================================"
echo ""

# ----------------------------------------------------------
# 1. feedforward
# ----------------------------------------------------------
echo "=== Experiment 1/5: feedforward ==="
SLOPE_ANGLE="${SLOPE_ANGLE}" \
SWEEP_NAME="slope${SLOPE_ANGLE}_feedforward" \
FORCE_CONTROL_METHOD="feedforward" \
USE_PI="false" \
KP_FORCE="" \
KD_FORCE="" \
KI_FORCE="" \
SKIP_SECONDS="0.0" \
bash "${SWEEP}"

echo ""

# ----------------------------------------------------------
# 2. feedforward + PI (kp=2.0, ki=5.0)
# ----------------------------------------------------------
echo "=== Experiment 2/5: feedforward + PI (kp=2.0, ki=5.0) ==="
SLOPE_ANGLE="${SLOPE_ANGLE}" \
SWEEP_NAME="slope${SLOPE_ANGLE}_feedforward_pi" \
FORCE_CONTROL_METHOD="feedforward" \
USE_PI="true" \
KP_FORCE="2.0" \
KD_FORCE="" \
KI_FORCE="5.0" \
SKIP_SECONDS="0.0" \
bash "${SWEEP}"

echo ""

# ----------------------------------------------------------
# 3. PD (kp=5.0, kd=0.5)
# ----------------------------------------------------------
echo "=== Experiment 3/5: PD (kp=5.0, kd=0.5) ==="
SLOPE_ANGLE="${SLOPE_ANGLE}" \
SWEEP_NAME="slope${SLOPE_ANGLE}_pd" \
FORCE_CONTROL_METHOD="pd" \
USE_PI="false" \
KP_FORCE="5.0" \
KD_FORCE="0.5" \
KI_FORCE="" \
SKIP_SECONDS="0.0" \
bash "${SWEEP}"

echo ""

# ----------------------------------------------------------
# 4. paper
# ----------------------------------------------------------
echo "=== Experiment 4/5: paper ==="
SLOPE_ANGLE="${SLOPE_ANGLE}" \
SWEEP_NAME="slope${SLOPE_ANGLE}_paper" \
FORCE_CONTROL_METHOD="paper" \
USE_PI="false" \
KP_FORCE="" \
KD_FORCE="" \
KI_FORCE="" \
SKIP_SECONDS="0.0" \
bash "${SWEEP}"

echo ""

# ----------------------------------------------------------
# 5. paper + PI (kp=2.0, ki=5.0)
# ----------------------------------------------------------
echo "=== Experiment 5/5: paper + PI (kp=2.0, ki=5.0) ==="
SLOPE_ANGLE="${SLOPE_ANGLE}" \
SWEEP_NAME="slope${SLOPE_ANGLE}_paper_pi" \
FORCE_CONTROL_METHOD="paper" \
USE_PI="true" \
KP_FORCE="2.0" \
KD_FORCE="" \
KI_FORCE="5.0" \
SKIP_SECONDS="0.0" \
bash "${SWEEP}"

echo ""
echo "========================================================"
echo "All 5 experiments complete."
echo "Results in: ${SCRIPT_DIR}/plots/"
echo "  slope${SLOPE_ANGLE}_feedforward/"
echo "  slope${SLOPE_ANGLE}_feedforward_pi/"
echo "  slope${SLOPE_ANGLE}_pd/"
echo "  slope${SLOPE_ANGLE}_paper/"
echo "  slope${SLOPE_ANGLE}_paper_pi/"
echo "Each folder contains sweep_results.csv and data/*.npz"
echo "========================================================"
