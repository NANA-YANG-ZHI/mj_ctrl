#!/usr/bin/env bash
# Generate all flat_slope_experiments plots.
#
# Usage (from workspace root):
#   bash flat_slope_experiments/plot_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[1/9] E1 speed sweep (flat)..."
python E1_angular_speed_sweep/speed_sweep/plot.py

echo "[2/9] E1 speed sweep (slope30)..."
python E1_angular_speed_sweep/speed_sweep/plot.py --slope

echo "[3/9] E1 tracking (flat, mult 3.4 3.8 4.4)..."
python E1_angular_speed_sweep/tracking/plot.py --multiplier 3.4 3.8 4.4

echo "[4/9] E1 tracking (slope30, mult 3.4 3.8)..."
python E1_angular_speed_sweep/tracking/plot.py --multiplier 3.4 3.8 --slope

echo "[5/9] E2 combined friction sweep..."
python E2_friction_sweep/combined/plot.py

echo "[6/9] E2 surface comparison..."
python E2_friction_sweep/surface_comparison/plot.py

echo "[7/9] E2 joint friction comparison + timeseries..."
python E2_friction_sweep/joint_friction_comparison/plot.py

echo "[8/9] E2 friction tracking (mu 0.3 0.7)..."
python E2_friction_sweep/tracking/plot.py --friction 0.3 0.7

echo "[9/9] E3 compensation comparison..."
python E3_compensation_comparison/plot.py

echo ""
echo "Done. All plots generated."
