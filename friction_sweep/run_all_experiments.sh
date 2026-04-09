#!/usr/bin/env bash
# Run 2 friction sweep experiments for HFPD (paper method) and HFPD+PI.
# Each experiment saves time-series data (.npz) for all 10 friction values
# and aggregate metrics to sweep_results.csv under friction_sweep/plots/<name>/.
#
# Experiments:
#   1. paper (HFPD, no PI)
#   2. paper + PI  (kp=2.0, ki=5.0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP="${SCRIPT_DIR}/sweep_friction.sh"

echo "========================================================"
echo "Running friction sweep experiments (HFPD methods only)"
echo "========================================================"
echo ""

# ----------------------------------------------------------
# 1. paper (HFPD, no PI)
# ----------------------------------------------------------
echo "=== Experiment 1/2: paper (HFPD) ==="
SWEEP_NAME="paper" \
FORCE_CONTROL_METHOD="paper" \
USE_PI="false" \
KP_FORCE="" \
KD_FORCE="" \
KI_FORCE="" \
bash "${SWEEP}"

echo ""

# ----------------------------------------------------------
# 2. paper + PI (kp=2.0, ki=5.0)
# ----------------------------------------------------------
echo "=== Experiment 2/2: paper + PI (kp=2.0, ki=5.0) ==="
SWEEP_NAME="paper_pi" \
FORCE_CONTROL_METHOD="paper" \
USE_PI="true" \
KP_FORCE="2.0" \
KD_FORCE="" \
KI_FORCE="5.0" \
bash "${SWEEP}"

echo ""
echo "========================================================"
echo "Both experiments complete."
echo "Results in: ${SCRIPT_DIR}/plots/"
echo "  paper/   paper_pi/"
echo "Each folder contains sweep_results.csv and data/*.npz"
echo "========================================================"
