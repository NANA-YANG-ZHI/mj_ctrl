#!/usr/bin/env bash
# Run compensation ablation experiments for two conditions and plot each.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Run 1: Frictionless ───────────────────────────────────────────────
echo "=========================================================="
echo " Run 1: Frictionless"
echo "=========================================================="

python "$SCRIPT_DIR/run_experiments.py" \
    --robot fr3 \
    --headless \
    --circle-duration 10.0 \
    --data-dir "$SCRIPT_DIR/data/frictionless"

python "$SCRIPT_DIR/plot_comparison.py" \
    --data-dir "$SCRIPT_DIR/data/frictionless" \
    --plot-dir "$SCRIPT_DIR/plots/frictionless"

# ── Run 2: Surface friction μ=0.7 ─────────────────────────────────────
echo ""
echo "=========================================================="
echo " Run 2: Surface friction μ=0.7"
echo "=========================================================="

python "$SCRIPT_DIR/run_experiments.py" \
    --robot fr3_friction \
    --headless \
    --circle-duration 10.0 \
    --surface-friction 0.7 \
    --data-dir "$SCRIPT_DIR/data/friction_0.7"

python "$SCRIPT_DIR/plot_comparison.py" \
    --data-dir "$SCRIPT_DIR/data/friction_0.7" \
    --plot-dir "$SCRIPT_DIR/plots/friction_0.7"

echo ""
echo "=========================================================="
echo " Done. Plots saved to:"
echo "   $SCRIPT_DIR/plots/frictionless/compensation_comparison.png"
echo "   $SCRIPT_DIR/plots/friction_0.7/compensation_comparison.png"
echo "=========================================================="
