#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROBOT="fr3_jointf_surff"
SPEEDS=("0.1" "0.5" "1.0")

for mult in "${SPEEDS[@]}"; do
    omega=$(python3 -c "import math; print(math.pi * ${mult})")
    data_dir="$SCRIPT_DIR/data/${ROBOT}_cylinder/${mult}x"

    echo "=========================================================="
    echo " Cylinder — ${ROBOT}, omega=${mult}π rad/s, trajectory 2"
    echo "=========================================================="

    python "$SCRIPT_DIR/run_experiments.py" \
        --robot "$ROBOT" \
        --angular-speed "$omega" \
        --trajectory 2 \
        --headless \
        --circle-duration 10.0 \
        --data-dir "$data_dir"

    echo ""
    echo " Done. Data saved to: $data_dir"
    echo ""
done

echo "=========================================================="
echo " All speeds finished."
echo "=========================================================="
