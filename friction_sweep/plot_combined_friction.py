"""Combined friction sweep plot — mean ± std shaded band for HFDC and HFDC+PI.

Reads sweep_results.csv from both paper/ and paper_pi/ output directories and
produces two plots (force error, position error) with a solid mean line and a
±1 std shaded band per method, saved to friction_sweep/plots/combined/.

Usage
-----
    python friction_sweep/plot_combined_friction.py
    python friction_sweep/plot_combined_friction.py --slope-angle 30
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tueplots import bundles

plt.rcParams.update(bundles.icml2024(usetex=False))
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR  = os.path.join(SCRIPT_DIR, "plots")

STD_RATIO  = 1.0   # multiplier on std for the shaded band

METHOD_KEYS = [
    ("HFDC",      "paper",    "tab:blue",   "o"),
    ("HFDC + PI", "paper_pi", "tab:orange", "s"),
]

METRICS = [
    dict(
        col_mean="avg_force_z_error",
        col_var="var_force_z_error",
        ylabel=f"Avg |Force Z Error| ± {STD_RATIO:.0f}Std  (N)",
        out="force_combined.png",
    ),
    dict(
        col_mean="avg_position_error",
        col_var="var_position_error",
        ylabel=f"Avg Position Error ± {STD_RATIO:.0f}Std  (m)",
        out="position_combined.png",
    ),
]


def build_methods(slope_angle):
    methods = []
    for label, key, color, marker in METHOD_KEYS:
        dir_name = f"slope{slope_angle}_{key}" if slope_angle != 0.0 else key
        csv_path = os.path.join(PLOTS_DIR, dir_name, "sweep_results.csv")
        methods.append((label, csv_path, color, marker))
    return methods


def load_csv(csv_path):
    """Parse sweep_results.csv → dict of numpy arrays keyed by column name."""
    rows = {k: [] for k in (
        "friction_coeff",
        "avg_force_z_error", "var_force_z_error",
        "avg_position_error", "var_position_error",
    )}
    with open(csv_path) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.strip().split(",")
            if len(parts) < 5:
                continue
            try:
                rows["friction_coeff"].append(float(parts[0]))
                rows["avg_force_z_error"].append(float(parts[1]) if parts[1].strip() != "nan" else float("nan"))
                rows["var_force_z_error"].append(float(parts[2]) if parts[2].strip() != "nan" else float("nan"))
                rows["avg_position_error"].append(float(parts[3]) if parts[3].strip() != "nan" else float("nan"))
                rows["var_position_error"].append(float(parts[4]) if parts[4].strip() != "nan" else float("nan"))
            except (ValueError, IndexError):
                continue
    data = {k: np.array(v) for k, v in rows.items()}
    order = np.argsort(data["friction_coeff"])
    return {k: v[order] for k, v in data.items()}


def make_combined_plot(cfg, datasets, out_dir):
    """Solid mean line + shaded ±std band per method, x = friction coefficient."""
    fig, ax = plt.subplots()

    for name, data, color, marker in datasets:
        mu   = data["friction_coeff"]
        mean = data[cfg["col_mean"]]
        std  = np.sqrt(np.maximum(data[cfg["col_var"]], 0.0)) * STD_RATIO

        lo = np.maximum(mean - std, 0.0)
        hi = mean + std

        valid = ~np.isnan(mean)
        ax.fill_between(mu[valid], lo[valid], hi[valid],
                        alpha=0.20, color=color, linewidth=0)
        ax.plot(mu[valid], mean[valid],
                marker=marker, color=color, label=name,
                linewidth=1, markersize=2, markerfacecolor=color)

    ax.set_xlabel("Sliding Friction Coefficient μ")
    ax.set_ylabel(cfg["ylabel"])
    ax.set_xticks(np.round(np.arange(0.1, 1.05, 0.1), 1))
    ax.legend(loc="upper right", framealpha=0.85)
    ax.grid(True, alpha=0.3)

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, cfg["out"])
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {os.path.relpath(out)}")


def main():
    parser = argparse.ArgumentParser(
        description="Combined friction sweep plot: HFDC and HFDC+PI with mean±std bands."
    )
    parser.add_argument("--slope-angle", type=float, default=0.0,
                        help="Slope angle in degrees (default: 0 = flat surface). "
                             "Reads from slope<angle>_paper/ directories.")
    args = parser.parse_args()

    slope_angle = args.slope_angle
    slope_suffix = f"_slope{slope_angle:g}" if slope_angle != 0.0 else ""
    out_dir = os.path.join(PLOTS_DIR, f"combined{slope_suffix}")

    methods = build_methods(slope_angle)

    datasets = []
    for name, csv_path, color, marker in methods:
        if not os.path.isfile(csv_path):
            print(f"[SKIP] {name}: CSV not found ({csv_path})")
            continue
        datasets.append((name, load_csv(csv_path), color, marker))

    if not datasets:
        print("No data found. Run experiments first.")
        return

    for cfg in METRICS:
        make_combined_plot(cfg, datasets, out_dir)

    print(f"\nDone. Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
