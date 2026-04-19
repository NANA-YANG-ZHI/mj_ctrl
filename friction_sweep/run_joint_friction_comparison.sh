#!/usr/bin/env bash
# Run joint friction comparison experiments for HFDC and HFDC+PI.
#
# 2x2x2 factorial design at ω = 2π rad/s:
#   Joint friction: yes (fr3_jointf / fr3_jointf_surff) vs. no (baseline reused)
#   Surface friction: no (condim=1) vs. yes μ=0.7 (condim=6)
#   Geometry: flat vs. slope 30°
#
# New runs saved to: friction_sweep/plots/joint_friction_comparison/<method>/<condition>/
# Baseline data reused from: friction_sweep/plots/surface_comparison/<method>/<condition>/
# Plot saved to: friction_sweep/plots/joint_friction_comparison/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/.."
PLOT_SCRIPT="${SCRIPT_DIR}/plot_joint_friction_comparison.py"

BASE_DIR="${SCRIPT_DIR}/plots/joint_friction_comparison"
BASELINE_DIR="${SCRIPT_DIR}/plots/surface_comparison"
mkdir -p "${BASE_DIR}"

# Shared simulation settings
ANGULAR_SPEED="${ANGULAR_SPEED:-6.283185307179586}"   # 2π rad/s
SKIP_SECONDS="${SKIP_SECONDS:-1.0}"
CIRCLE_DURATION="${CIRCLE_DURATION:-10.0}"

# ------------------------------------------------------------------
# Helper: run one simulation
#   $1 method_dir      (paper | paper_pi)
#   $2 cond_dir        (flat_jointf_no_surff | slope_jointf_no_surff | ...)
#   $3 robot           (fr3_jointf | fr3_jointf_surff)
#   $4 slope           (0.0 | 30.0)
#   $5 mu              (multiplier value, e.g. 0.0 or 0.7)
#   $6 friction_flag   (empty string or "--surface-friction 0.7")
#   $7 pi_flags        (empty string or "--use-pi --kp-force 2.0 --ki-force 5.0")
#   $8 label           (human-readable label for logging)
# ------------------------------------------------------------------
run_one() {
    local method_dir="$1" cond_dir="$2" robot="$3" slope="$4" mu="$5"
    local friction_flag="$6" pi_flags="$7" label="$8"

    local data_dir="${BASE_DIR}/${method_dir}/${cond_dir}"
    mkdir -p "${data_dir}"

    echo ""
    echo "──────────────────────────────────────────────────────"
    echo " ${label}"
    echo " → data: ${data_dir}"
    echo "──────────────────────────────────────────────────────"

    # shellcheck disable=SC2086
    python3 "${REPO_DIR}/run_approach_then_hybrid_mujoco.py" \
        --robot "${robot}" \
        --slope-angle "${slope}" \
        --headless \
        --angular-speed "${ANGULAR_SPEED}" \
        --force-control-method paper \
        --multiplier "${mu}" \
        --skip-seconds "${SKIP_SECONDS}" \
        --save-data \
        --data-dir "${data_dir}" \
        ${friction_flag} \
        ${pi_flags}
}

echo "=========================================================="
echo " Joint friction comparison experiments (HFDC and HFDC+PI)"
echo " ω = 2π rad/s | slope: 0° and 30°"
echo "=========================================================="

# ── HFDC (no PI) ──────────────────────────────────────────────────
echo ""
echo "=== Method 1/2: HFDC (no PI) ==="

run_one paper flat_jointf_no_surff \
    fr3_jointf 0.0 0.0 \
    "" "" \
    "HFDC | flat | joint friction | no surface friction"

run_one paper slope30_jointf_no_surff \
    fr3_jointf 30.0 0.0 \
    "" "" \
    "HFDC | slope 30° | joint friction | no surface friction"

run_one paper flat_jointf_surff_0.7 \
    fr3_jointf_surff 0.0 0.7 \
    "--surface-friction 0.7" "" \
    "HFDC | flat | joint + surface friction μ=0.7"

run_one paper slope30_jointf_surff_0.7 \
    fr3_jointf_surff 30.0 0.7 \
    "--surface-friction 0.7" "" \
    "HFDC | slope 30° | joint + surface friction μ=0.7"

# ── HFDC + PI ─────────────────────────────────────────────────────
echo ""
echo "=== Method 2/2: HFDC + PI (kp=2.0, ki=5.0) ==="

PI_FLAGS="--use-pi --kp-force 2.0 --ki-force 5.0"

run_one paper_pi flat_jointf_no_surff \
    fr3_jointf 0.0 0.0 \
    "" "${PI_FLAGS}" \
    "HFDC+PI | flat | joint friction | no surface friction"

run_one paper_pi slope30_jointf_no_surff \
    fr3_jointf 30.0 0.0 \
    "" "${PI_FLAGS}" \
    "HFDC+PI | slope 30° | joint friction | no surface friction"

run_one paper_pi flat_jointf_surff_0.7 \
    fr3_jointf_surff 0.0 0.7 \
    "--surface-friction 0.7" "${PI_FLAGS}" \
    "HFDC+PI | flat | joint + surface friction μ=0.7"

run_one paper_pi slope30_jointf_surff_0.7 \
    fr3_jointf_surff 30.0 0.7 \
    "--surface-friction 0.7" "${PI_FLAGS}" \
    "HFDC+PI | slope 30° | joint + surface friction μ=0.7"

echo ""
echo "=========================================================="
echo " All experiments complete. Generating comparison plot..."
echo "=========================================================="

python3 "${PLOT_SCRIPT}" \
    --base-dir "${BASE_DIR}" \
    --baseline-dir "${BASELINE_DIR}"

python3 "${SCRIPT_DIR}/plot_joint_friction_timeseries.py" \
    --base-dir "${BASE_DIR}"

echo ""
echo "Done. Plots saved to: ${BASE_DIR}/"
