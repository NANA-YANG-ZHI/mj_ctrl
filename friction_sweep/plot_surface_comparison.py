"""Grouped bar chart comparing HFDC vs HFDC+PI across 4 surface conditions.

Reads .npz files produced by run_surface_comparison.sh and plots mean ± std
for force Z error and position error, grouped by surface condition.

Usage
-----
    python friction_sweep/plot_surface_comparison.py
    python friction_sweep/plot_surface_comparison.py --base-dir path/to/surface_comparison
"""

import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tueplots import bundles

plt.rcParams.update(bundles.icml2024(usetex=False))
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 8,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CONDITIONS = [
    ("flat_frictionless",   "Flat\nfrictionless"),
    ("flat_friction_0.7",   "Flat\nμ=0.7"),
    ("slope30_frictionless","Slope 30°\nfrictionless"),
    ("slope30_friction_0.7","Slope 30°\nμ=0.7"),
]

METHODS = [
    ("paper",    "HFDC",      "tab:blue"),
    ("paper_pi", "HFDC + PI", "tab:orange"),
]

METRICS = [
    dict(npz_key="force_error",    ylabel="Avg |Force Z Error| (N)",  out="surface_force_error.png"),
    dict(npz_key="position_error", ylabel="Avg Position Error (m)",    out="surface_position_error.png"),
]


def load_npz_mean_std(data_dir: str, npz_key: str):
    """Load all .npz in data_dir, concatenate npz_key arrays, return (mean, std)."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not files:
        return float("nan"), float("nan")
    arrays = []
    for fp in files:
        data = np.load(fp)
        if npz_key in data:
            arr = np.abs(data[npz_key])
            arr = arr[~np.isnan(arr)]
            if arr.size:
                arrays.append(arr)
    if not arrays:
        return float("nan"), float("nan")
    combined = np.concatenate(arrays)
    return float(np.mean(combined)), float(np.std(combined))


def make_bar_chart(cfg, base_dir, out_dir):
    n_cond   = len(CONDITIONS)
    n_method = len(METHODS)
    width    = 0.35
    offsets  = np.linspace(-(n_method - 1) * width / 2,
                            (n_method - 1) * width / 2,
                            n_method)

    fig, ax = plt.subplots()
    x = np.arange(n_cond)

    for m_idx, (method_key, method_label, color) in enumerate(METHODS):
        means, stds = [], []
        for cond_key, _ in CONDITIONS:
            data_dir = os.path.join(base_dir, method_key, cond_key)
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

        # value label on top of each bar
        for bar, mean in zip(bars, means[valid]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (stds[valid][list(means[valid]).index(mean)] if not np.isnan(stds[valid][list(means[valid]).index(mean)]) else 0) + ax.get_ylim()[1] * 0.01,
                f"{mean:.3f}",
                ha="center", va="bottom", fontsize=6,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in CONDITIONS])
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
        description="Plot surface comparison: HFDC vs HFDC+PI across 4 conditions."
    )
    parser.add_argument(
        "--base-dir",
        default=os.path.join(SCRIPT_DIR, "plots", "surface_comparison"),
        help="Base directory containing paper/ and paper_pi/ subdirectories.",
    )
    args = parser.parse_args()

    out_dir = args.base_dir

    for cfg in METRICS:
        make_bar_chart(cfg, args.base_dir, out_dir)

    print(f"\nDone. Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
