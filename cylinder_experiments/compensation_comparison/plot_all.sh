#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/plot_comparison.py" \
    --data-dir "$SCRIPT_DIR/data/fr3_friction_cylinder" \
    --plot-dir "$SCRIPT_DIR/plots/fr3_friction_cylinder"

echo ""
echo "=========================================================="
echo " Done. Plot saved to:"
echo "   $SCRIPT_DIR/plots/fr3_friction_cylinder/compensation_comparison.png"
echo "=========================================================="
