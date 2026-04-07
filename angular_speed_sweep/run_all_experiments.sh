#!/usr/bin/env bash
# Run 5 angular speed sweep experiments one after another.
# Each experiment saves time-series data (.npz) for all 50 speed points
# and aggregate metrics to sweep_results.csv under angular_speed_sweep/plots/<name>/.
#
# Experiments:
#   1. feedforward
#   2. feedforward + PI  (kp=2.0, ki=5.0)
#   3. pd                (kp=5.0, kd=0.5)
#   4. paper
#   5. paper + PI        (kp=2.0, ki=5.0)
#
# No samples are skipped at the start (SKIP_SECONDS=0.0).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP="${SCRIPT_DIR}/sweep_angular_speed.sh"

echo "========================================================"
echo "Running all 5 angular speed sweep experiments"
echo "========================================================"
echo ""

# # ----------------------------------------------------------
# # 1. feedforward
# # ----------------------------------------------------------
# echo "=== Experiment 1/5: feedforward ==="
# SWEEP_NAME="feedforward" \
# FORCE_CONTROL_METHOD="feedforward" \
# USE_PI="false" \
# KP_FORCE="" \
# KD_FORCE="" \
# KI_FORCE="" \
# SKIP_SECONDS="0.0" \
# bash "${SWEEP}"

# echo ""

# # ----------------------------------------------------------
# # 2. feedforward + PI (kp=2.0, ki=5.0)
# # ----------------------------------------------------------
# echo "=== Experiment 2/5: feedforward + PI (kp=2.0, ki=5.0) ==="
# SWEEP_NAME="feedforward_pi" \
# FORCE_CONTROL_METHOD="feedforward" \
# USE_PI="true" \
# KP_FORCE="2.0" \
# KD_FORCE="" \
# KI_FORCE="5.0" \
# SKIP_SECONDS="0.0" \
# bash "${SWEEP}"

# echo ""

# # ----------------------------------------------------------
# # 3. PD (kp=5.0, kd=0.5)
# # ----------------------------------------------------------
# echo "=== Experiment 3/5: PD (kp=5.0, kd=0.5) ==="
# SWEEP_NAME="pd" \
# FORCE_CONTROL_METHOD="pd" \
# USE_PI="false" \
# KP_FORCE="5.0" \
# KD_FORCE="0.5" \
# KI_FORCE="" \
# SKIP_SECONDS="0.0" \
# bash "${SWEEP}"

# echo ""

# ----------------------------------------------------------
# 4. paper
# ----------------------------------------------------------
echo "=== Experiment 4/5: paper ==="
SWEEP_NAME="paper" \
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
SWEEP_NAME="paper_pi" \
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
echo "  feedforward/   feedforward_pi/   pd/   paper/   paper_pi/"
echo "Each folder contains sweep_results.csv and data/*.npz"
echo "========================================================"
