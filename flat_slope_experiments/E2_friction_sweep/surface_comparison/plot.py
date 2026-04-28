"""Grouped bar chart: HFDC vs HFDC+PI across 4 surface conditions.

Reads .npz from E2_friction_sweep/data/surface_comparison/<method>/<condition>/.

Outputs (saved to plots/):
  surface_force_error.png
  surface_position_error.png

Usage
-----
    python plot.py
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np

try:
    from tueplots import bundles
    plt.rcParams.update(bundles.icml2024(usetex=False))
except ImportError:
    pass

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 8,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT  = os.path.join(SCRIPT_DIR, "..", "data", "surface_comparison")
PLOTS_DIR  = os.path.join(SCRIPT_DIR, "plots")

CONDITIONS = [
    ("flat_frictionless",    "Flat\nfrictionless"),
    ("flat_friction_0.7",    "Flat\nμ=0.7"),
    ("slope30_frictionless", "Slope 30°\nfrictionless"),
    ("slope30_friction_0.7", "Slope 30°\nμ=0.7"),
]

METHODS = [
    ("paper",    "HFDC",      "tab:blue"),
    # ("paper_pi", "HFDC + PI", "tab:orange"),
]

METRICS = [
    dict(npz_key="force_error",    ylabel="Avg |Force Z Error| (N)",
         out="surface_force_error.png"),
    dict(npz_key="position_error", ylabel="Avg Position Error (m)",
         out="surface_position_error.png"),
]


def load_npz_mean_std(data_dir, npz_key):
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


def make_bar_chart(cfg):
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
            data_dir = os.path.join(DATA_ROOT, method_key, cond_key)
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
                top + ax.get_ylim()[1] * 0.01,
                f"{mean_val:.3f}",
                ha="center", va="bottom", fontsize=6,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in CONDITIONS])
    ax.set_ylabel(cfg["ylabel"])
    ax.legend(loc="upper left", framealpha=0.85)
    ax.set_ylim(bottom=0)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, cfg["out"])
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {cfg['out']}")


def main():
    for cfg in METRICS:
        make_bar_chart(cfg)


if __name__ == "__main__":
    main()
