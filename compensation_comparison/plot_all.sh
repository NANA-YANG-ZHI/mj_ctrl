#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── [COMMENTED OUT] Flat surface, frictionless ───────────────────────────────
python "$SCRIPT_DIR/plot_comparison.py" \
    --data-dir "$SCRIPT_DIR/data/frictionless_angularv3.2" \
    --plot-dir "$SCRIPT_DIR/plots/frictionless_angularv3.2"

# ── [COMMENTED OUT] Flat surface, friction μ=0.7 ─────────────────────────────
python "$SCRIPT_DIR/plot_comparison.py" \
    --data-dir "$SCRIPT_DIR/data/friction_0.7_angularv3.2" \
    --plot-dir "$SCRIPT_DIR/plots/friction_0.7_angularv3.2"

# ── Slope 30°, frictionless ───────────────────────────────────────────────────
python "$SCRIPT_DIR/plot_comparison.py" \
    --data-dir "$SCRIPT_DIR/data/slope30_frictionless_angularv3.2" \
    --plot-dir "$SCRIPT_DIR/plots/slope30_frictionless_angularv3.2"

# ── Slope 30°, friction μ=0.7 ─────────────────────────────────────────────────
python "$SCRIPT_DIR/plot_comparison.py" \
    --data-dir "$SCRIPT_DIR/data/slope30_friction_0.7_angularv3.2" \
    --plot-dir "$SCRIPT_DIR/plots/slope30_friction_0.7_angularv3.2"

echo ""
echo "=========================================================="
echo " Done. Plots saved to:"
echo "   $SCRIPT_DIR/plots/slope30_frictionless/compensation_comparison.png"
echo "   $SCRIPT_DIR/plots/slope30_friction_0.7/compensation_comparison.png"
echo "=========================================================="
