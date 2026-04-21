#!/usr/bin/env bash
# Run all cylinder surface experiments in parallel.
# Usage: bash cylinder_experiments/run_all.sh [--jobs N]
#        (must be executed from the workspace root)
set -euo pipefail

# Set locale to C to ensure printf uses '.' as decimal separator
export LC_NUMERIC=C

# Parse number of parallel jobs (default: 10)
JOBS=10
while [[ $# -gt 0 ]]; do
    case $1 in
        --jobs)
            JOBS=$2
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# ── Resolve workspace root regardless of invocation path ─────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# ── Experiment grid ───────────────────────────────────────────────────────────
ROBOTS=("fr3" "kuka")
SPEEDS=(0.314 0.628 0.8 1.0 1.2)

# Each entry: "method_safe:force_method:use_pi(0|1)"
METHODS_CFG=(
    "ff:feedforward:0"
    "ff_pi:feedforward:1"
    "pd:pd:0"
    "paper:paper:0"
    "paper_pi:paper:1"
)

EXP_DIR="cylinder_experiments"
DATA_BASE="$EXP_DIR/data"
PLOTS_BASE="$EXP_DIR/plots_individual"

mkdir -p "$DATA_BASE" "$PLOTS_BASE"

# ── Function to run a single experiment ────────────────────────────────────────
run_experiment() {
    local robot=$1
    local speed=$2
    local method=$3
    local force_method=$4
    local use_pi=$5
    local idx=$6
    local total=$7

    speed_str=$(printf "%.3f" "$speed")
    tag="${robot}_${method}_${speed_str}"
    data_dir="$DATA_BASE/$tag"
    plot_dir="$PLOTS_BASE/$tag"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [$idx/$total]  robot=$robot  method=$method  ω=${speed_str} rad/s"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    cmd=(python run_approach_then_hybrid_cylinder.py
        --robot         "$robot"
        --angular-speed "$speed"
        --force-control-method "$force_method"
        --headless
        --save-plots
        --plot-dir      "$plot_dir"
        --save-data
        --data-dir      "$data_dir"
    )
    [[ "$use_pi" == "1" ]] && cmd+=(--use-pi)

    "${cmd[@]}"
}

export -f run_experiment
export DATA_BASE PLOTS_BASE EXP_DIR

# ── Generate all experiment jobs ───────────────────────────────────────────────
total=$(( ${#ROBOTS[@]} * ${#SPEEDS[@]} * ${#METHODS_CFG[@]} ))
count=0

# Create a temporary file to store all jobs
JOBS_FILE=$(mktemp)
trap "rm -f $JOBS_FILE" EXIT

for robot in "${ROBOTS[@]}"; do
    for speed in "${SPEEDS[@]}"; do
        for cfg in "${METHODS_CFG[@]}"; do
            IFS=':' read -r method force_method use_pi <<< "$cfg"
            count=$(( count + 1 ))
            echo "$robot $speed $method $force_method $use_pi $count $total" >> "$JOBS_FILE"
        done
    done
done

echo "Running $total experiments with $JOBS parallel jobs..."
echo ""

# ── Run experiments in parallel ────────────────────────────────────────────────
cat "$JOBS_FILE" | xargs -P "$JOBS" -I {} bash -c 'run_experiment {}'

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  All $total runs complete. Generating comparison plots..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python "$EXP_DIR/plot_comparison.py"
echo "Done!"
