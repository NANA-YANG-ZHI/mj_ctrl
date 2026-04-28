#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROBOT="fr3_friction"
SPEEDS=("0.1" "0.5" "1.0")

for mult in "${SPEEDS[@]}"; do
    data_dir="$SCRIPT_DIR/data/${ROBOT}_cylinder/${mult}x"
    plot_dir="$SCRIPT_DIR/plots/${ROBOT}_cylinder/${mult}x"

    echo "Plotting ${ROBOT} at ${mult}π rad/s..."
    python "$SCRIPT_DIR/plot_comparison.py" \
        --data-dir "$data_dir" \
        --plot-dir "$plot_dir"

    echo " Done. Plot saved to: $plot_dir/compensation_comparison.png"
    echo ""
done

echo "=========================================================="
echo " All plots finished."
echo "=========================================================="
