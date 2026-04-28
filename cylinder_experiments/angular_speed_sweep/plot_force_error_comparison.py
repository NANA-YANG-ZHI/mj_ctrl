"""Compare force (and position) error across methods for the angular speed sweep.

Reads sweep_results.csv from sweep_speed_results/<method>/ for each method
and plots mean ± std vs EE linear speed, one subplot per robot configuration.

Usage
-----
# Default: compare paper vs baseline for all four robots
python cylinder_experiments/angular_speed_sweep/plot_force_error_comparison.py

# Specific methods
python cylinder_experiments/angular_speed_sweep/plot_force_error_comparison.py \
    --methods paper baseline

# Custom output directory
python cylinder_experiments/angular_speed_sweep/plot_force_error_comparison.py \
    --output-dir my_plots/
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

try:
    from tueplots import bundles
    plt.rcParams.update(bundles.icml2024(usetex=False))
except ImportError:
    pass

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "sweep_speed_results")
DEFAULT_OUT = os.path.join(SCRIPT_DIR, "sweep_speed_results", "comparison")

ROBOT_ORDER = ["fr3", "fr3_friction", "fr3_jointf", "fr3_jointf_surff"]
ROBOT_LABELS = {
    "fr3":              "FR3",
    "fr3_friction":     "FR3 + surface friction",
    "fr3_jointf":       "FR3 + joint friction",
    "fr3_jointf_surff": "FR3 + joint + surface friction",
}

METHOD_STYLES = {
    "paper":    ("tab:blue",   "o-",  "HFDC"),
    "baseline": ("tab:orange", "s--", "Baseline"),
    "ff":       ("tab:green",  "^-",  "Feedforward"),
    "ff_pi":    ("tab:purple", "D-",  "FF + PI"),
    "pd":       ("tab:red",    "v-",  "PD"),
    "paper_pi": ("tab:brown",  "P-",  "HFDC + PI"),
}
PLOT_DPI = 300


def load_csv(csv_path):
    """Return robot → list of (ee_speed, mean_force, std_force, mean_pos, std_pos)."""
    data = defaultdict(list)
    with open(csv_path) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            try:
                robot    = parts[0].strip()
                ee_speed = float(parts[2])
                mf       = float(parts[3])
                sf       = float(parts[4])
                mp       = float(parts[5])
                sp       = float(parts[6])
                data[robot].append((ee_speed, mf, sf, mp, sp))
            except ValueError:
                continue
    for robot in data:
        data[robot].sort(key=lambda x: x[0])
    return data


def plot_comparison(methods, metric, out_dir):
    """Plot one comparison figure for 'force' or 'position' metric.

    Parameters
    ----------
    methods : list of (method_name, csv_data_dict)
    metric  : "force" or "position"
    out_dir : output directory
    """
    all_robots = []
    for _, csv_data in methods:
        for r in ROBOT_ORDER:
            if r in csv_data and r not in all_robots:
                all_robots.append(r)

    ncols = 2
    nrows = (len(all_robots) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.25, nrows * 2.0),
                             sharex=False, squeeze=False)
    axes_flat = axes.flatten()

    ylabel = ("Mean |Force Error| ± std  (N)"
              if metric == "force" else
              "Mean Position Error ± std  (m)")
    title  = ("Force Error Comparison" if metric == "force"
              else "Position Error Comparison")

    for ax_i, robot in enumerate(all_robots):
        ax = axes_flat[ax_i]
        for method_name, csv_data in methods:
            if robot not in csv_data:
                continue
            rows   = csv_data[robot]
            v      = np.array([r[0] for r in rows])
            if metric == "force":
                mean = np.array([r[1] for r in rows])
                std  = np.array([r[2] for r in rows])
            else:
                mean = np.array([r[3] for r in rows])
                std  = np.array([r[4] for r in rows])

            color, ls, label = METHOD_STYLES.get(
                method_name, ("tab:gray", "x-", method_name)
            )
            ax.plot(v, mean, ls, lw=1.5, ms=4, color=color, label=label)
            ax.fill_between(v, mean - std, mean + std, alpha=0.15, color=color)

        ax.set_title(ROBOT_LABELS.get(robot, robot), fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("EE Linear Speed  (m/s)")
        ax.legend(loc="upper left")

    for j in range(len(all_robots), nrows * ncols):
        axes_flat[j].set_visible(False)

    fig.suptitle(title, fontweight="bold", fontsize=10)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    fname  = f"comparison_{metric}_error.png"
    fpath  = os.path.join(out_dir, fname)
    fig.savefig(fpath, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {fpath}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare force/position error across methods (angular speed sweep)."
    )
    parser.add_argument(
        "--methods", nargs="+", default=["paper", "baseline"],
        help="Method names matching sweep_speed_results/<name>/sweep_results.csv",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUT,
        help="Directory for output PNGs",
    )
    args = parser.parse_args()

    loaded = []
    for method in args.methods:
        csv_path = os.path.join(RESULTS_DIR, method, "sweep_results.csv")
        if not os.path.isfile(csv_path):
            print(f"[SKIP] CSV not found: {csv_path}")
            continue
        loaded.append((method, load_csv(csv_path)))
        print(f"[LOAD] {method}  ({csv_path})")

    if not loaded:
        print("[ERROR] No CSVs loaded.")
        return

    plot_comparison(loaded, "force",    args.output_dir)
    plot_comparison(loaded, "position", args.output_dir)


if __name__ == "__main__":
    main()
