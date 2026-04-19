"""Grouped bar chart comparing HFDC vs HFDC+PI across 6 joint-friction conditions.

Reads .npz files from:
  - joint_friction_comparison/<method>/<cond>/   (new joint-friction runs)
  - surface_comparison/<method>/<cond>/           (no-joint-friction baseline, reused)

Conditions (x-axis, left→right):
  flat         | no joint friction, no surface friction  (baseline)
  flat         | joint friction,    no surface friction
  flat         | joint + surface friction μ=0.7
  slope 30°    | no joint friction, no surface friction  (baseline)
  slope 30°    | joint friction,    no surface friction
  slope 30°    | joint + surface friction μ=0.7

Usage
-----
    python friction_sweep/plot_joint_friction_comparison.py
    python friction_sweep/plot_joint_friction_comparison.py \\
        --base-dir path/to/joint_friction_comparison \\
        --baseline-dir path/to/surface_comparison
"""

import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tueplots import bundles

plt.rcParams.update(bundles.icml2024(usetex=False))
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 8,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 8,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DT = 0.001   # simulation timestep (s)
SKIP_S = 1.0  # seconds to skip (exclude approach transient)

# Each entry: (source, method_subdir, cond_subdir, x_label)
# source: "baseline" → read from baseline_dir; "new" → read from base_dir
CONDITIONS = [
    # flat group
    ("baseline", "flat_frictionless",        "Flat\nno joint\nno surf."),
    ("baseline", "flat_friction_0.7",        "Flat\nno joint\nsurf. μ=0.7"),
    ("new",      "flat_jointf_no_surff",     "Flat\njoint\nno surf."),
    ("new",      "flat_jointf_surff_0.7",    "Flat\njoint\n+ surf. μ=0.7"),
    # slope group
    ("baseline", "slope30_frictionless",      "Slope 30°\nno joint\nno surf."),
    ("baseline", "slope30_friction_0.7",      "Slope 30°\nno joint\nsurf. μ=0.7"),
    ("new",      "slope30_jointf_no_surff",   "Slope 30°\njoint\nno surf."),
    ("new",      "slope30_jointf_surff_0.7",  "Slope 30°\njoint\n+ surf. μ=0.7"),
]

METHODS = [
    ("paper",    "HFDC",      "tab:blue"),
    ("paper_pi", "HFDC + PI", "tab:orange"),
]

METRICS = [
    dict(npz_key="force_error",    ylabel="Avg |Force Z Error| (N)",
         out="joint_friction_force_error.png"),
    dict(npz_key="position_error", ylabel="Avg Position Error (m)",
         out="joint_friction_position_error.png"),
]


def load_npz_mean_std(data_dir: str, npz_key: str):
    """Load all .npz in data_dir, skip first SKIP_S seconds, return (mean, std)."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not files:
        return float("nan"), float("nan")
    skip = int(SKIP_S / DT)
    arrays = []
    for fp in files:
        data = np.load(fp)
        if npz_key in data:
            arr = np.abs(data[npz_key][skip:])
            arr = arr[~np.isnan(arr)]
            if arr.size:
                arrays.append(arr)
    if not arrays:
        return float("nan"), float("nan")
    combined = np.concatenate(arrays)
    return float(np.mean(combined)), float(np.std(combined))


def make_bar_chart(cfg, base_dir, baseline_dir, out_dir):
    n_cond   = len(CONDITIONS)
    n_method = len(METHODS)
    width    = 0.32
    offsets  = np.linspace(-(n_method - 1) * width / 2,
                            (n_method - 1) * width / 2,
                            n_method)

    fig, ax = plt.subplots(figsize=(6.5, 2.25))
    x = np.arange(n_cond)

    # Vertical separator between flat and slope groups
    ax.axvline(x=3.5, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)

    for m_idx, (method_key, method_label, color) in enumerate(METHODS):
        means, stds = [], []
        for source, cond_key, _ in CONDITIONS:
            root = baseline_dir if source == "baseline" else base_dir
            data_dir = os.path.join(root, method_key, cond_key)
            mean, std = load_npz_mean_std(data_dir, cfg["npz_key"])
            means.append(mean)
            stds.append(std)

        means = np.array(means)
        stds  = np.array(stds)

        valid = ~np.isnan(means)
        bars = ax.bar(
            x[valid] + offsets[m_idx], means[valid],
            width=width * 0.9,
            yerr=stds[valid],
            capsize=3,
            color=color,
            alpha=0.85,
            label=method_label,
            error_kw=dict(elinewidth=0.8, capthick=0.8),
        )

        for bar, mean_val, std_val in zip(bars, means[valid], stds[valid]):
            top = bar.get_height() + (std_val if not np.isnan(std_val) else 0)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                top + ax.get_ylim()[1] * 0.005,
                f"{mean_val:.3f}",
                ha="center", va="bottom", fontsize=5.5,
            )

    # Group labels
    flat_center  = np.mean([0, 1, 2, 3])
    slope_center = np.mean([4, 5, 6, 7])
    y_group = ax.get_ylim()[1] * 1.02
    for cx, lbl in [(flat_center, "Flat"), (slope_center, "Slope 30°")]:
        ax.text(cx, y_group, lbl, ha="center", va="bottom",
                fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, _, label in CONDITIONS])
    ax.set_ylabel(cfg["ylabel"])
    ax.legend(loc="upper left", framealpha=0.85)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, cfg["out"])
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {os.path.relpath(out)}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot joint friction comparison: HFDC vs HFDC+PI across 6 conditions."
    )
    parser.add_argument(
        "--base-dir",
        default=os.path.join(SCRIPT_DIR, "plots", "joint_friction_comparison"),
        help="Directory with new joint-friction run data (paper/ and paper_pi/ subdirs).",
    )
    parser.add_argument(
        "--baseline-dir",
        default=os.path.join(SCRIPT_DIR, "plots", "surface_comparison"),
        help="Directory with baseline (no joint friction) data from run_surface_comparison.sh.",
    )
    args = parser.parse_args()

    out_dir = args.base_dir

    for cfg in METRICS:
        make_bar_chart(cfg, args.base_dir, args.baseline_dir, out_dir)

    print(f"\nDone. Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
