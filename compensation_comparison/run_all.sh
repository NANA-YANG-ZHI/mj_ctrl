#!/usr/bin/env bash
# Run compensation ablation experiments and plot each.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── [COMMENTED OUT] Run 1: Flat surface, frictionless ────────────────────────
echo "=========================================================="
echo " Run 1: Frictionless (flat surface)"
echo "=========================================================="
python "$SCRIPT_DIR/run_experiments.py" \
    --robot fr3 \
    --headless \
    --circle-duration 10.0 \
    --slope-angle 0.0 \
    --angular-speed 3.2 \
    --data-dir "$SCRIPT_DIR/data/frictionless_angularv3.2"

# ── [COMMENTED OUT] Run 2: Flat surface, friction μ=0.7 ──────────────────────
echo ""
echo "=========================================================="
echo " Run 2: Surface friction μ=0.7 (flat surface)"
echo "=========================================================="
python "$SCRIPT_DIR/run_experiments.py" \
    --robot fr3_friction \
    --headless \
    --circle-duration 10.0 \
    --slope-angle 0.0 \
    --surface-friction 0.7 \
    --angular-speed 3.2 \
    --data-dir "$SCRIPT_DIR/data/friction_0.7_angularv3.2"

# ── Run 3: Slope 30°, frictionless ───────────────────────────────────────────
echo "=========================================================="
echo " Run 3: Slope 30° — frictionless"
echo "=========================================================="

python "$SCRIPT_DIR/run_experiments.py" \
    --robot fr3 \
    --headless \
    --circle-duration 10.0 \
    --slope-angle 30.0 \
    --angular-speed 3.2 \
    --data-dir "$SCRIPT_DIR/data/slope30_frictionless_angularv3.2"

echo ""

# ── Run 4: Slope 30°, friction μ=0.7 ─────────────────────────────────────────
echo "=========================================================="
echo " Run 4: Slope 30° — friction μ=0.7"
echo "=========================================================="

python "$SCRIPT_DIR/run_experiments.py" \
    --robot fr3_friction \
    --headless \
    --circle-duration 10.0 \
    --slope-angle 30.0 \
    --surface-friction 0.7 \
    --angular-speed 3.2 \
    --data-dir "$SCRIPT_DIR/data/slope30_friction_0.7_angularv3.2"

echo ""
echo "=========================================================="
echo " Done. Data saved to:"
echo "   $SCRIPT_DIR/data/slope30_frictionless/"
echo "   $SCRIPT_DIR/data/slope30_friction_0.7/"
echo "=========================================================="
