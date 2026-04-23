#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================================="
echo " Cylinder — fr3_friction, omega=3.2 rad/s, trajectory 1"
echo "=========================================================="

python "$SCRIPT_DIR/run_experiments.py" \
    --robot fr3_friction \
    --angular-speed 3.2 \
    --trajectory 1 \
    --headless \
    --circle-duration 10.0 \
    --data-dir "$SCRIPT_DIR/data/fr3_friction_cylinder"

echo ""
echo "=========================================================="
echo " Done. Data saved to:"
echo "   $SCRIPT_DIR/data/fr3_friction_cylinder/"
echo "=========================================================="
