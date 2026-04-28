#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================================="
echo " Cylinder — fr3_jointf_surff, omega=3.2 rad/s, trajectory 2"
echo "=========================================================="

python "$SCRIPT_DIR/run_experiments.py" \
    --robot fr3_jointf_surff \
    --angular-speed 3.2 \
    --trajectory 2 \
    --headless \
    --circle-duration 10.0 \
    --data-dir "$SCRIPT_DIR/data/fr3_jointf_surff_cylinder"

echo ""
echo "=========================================================="
echo " Done. Data saved to:"
echo "   $SCRIPT_DIR/data/fr3_jointf_surff_cylinder/"
echo "=========================================================="
