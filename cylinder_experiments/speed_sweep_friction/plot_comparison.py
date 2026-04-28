"""Combined mean ± std comparison plots for speed_sweep_friction.

For each robot (fr3_friction, fr3_jointf_surff) generates:
  force_error_combined_<robot>.png    – avg |force Z error| ± std  (N)
  position_error_combined_<robot>.png – avg position error  ± std  (m)

Plot style mirrors angular_speed_sweep/plot_force_error_comparison.py exactly.

Usage:
    python cylinder_experiments/speed_sweep_friction/plot_comparison.py \\
        sweep_friction_results/results.csv  sweep_friction_results/plots
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

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

FIGSIZE = (3.25, 3)

_STYLE = {
    "plot_linewidth":     0.5,
    "plot_markersize":    1,
    "fill_alpha":         0.15,
    "hline_linewidth":    0.9,
    "hline_alpha":        0.55,
    "annot_fontsize":     7.5,
    "outlier_markersize": 4,
    "outlier_fontsize":   7,
    "bbox_lw":            0.7,
    "legend_framealpha":  0.85,
    "plot_dpi":           600,
}

STD_RATIO = 0.3

METHOD_KEYS = [
    ("Baseline",         "baseline",  "tab:gray",   "x"),
    ("Feedforward",      "ff",        "tab:blue",   "o"),
    # ("Feedforward + PI", "ff_pi",     "tab:purple", "s"),
    ("PD",               "pd",        "tab:green",  "^"),
    ("HFDC",             "paper",     "tab:orange", "D"),
    # ("HFDC + PI",        "paper_pi",  "tab:red",    "P"),
]

ROBOTS = [
    "fr3_friction",
    "fr3_jointf_surff",
]

COMBINED_METRICS = [
    dict(
        col_mean="mean_force_error",
        col_std="std_force_error",
        ylabel=f"Avg |Force Z Error| ± {STD_RATIO:.1f}Std  (N)",
        title="Force Z Error  (mean ± std)",
        out_prefix="force_error_combined",
        break_y=3.0,
        compress=10.0,
        top_max=20.0,
        yticks=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5, 8, 11, 14, 17, 20],
    ),
    dict(
        col_mean="mean_position_error",
        col_std="std_position_error",
        ylabel=f"Avg Position Error ± {STD_RATIO:.1f}Std  (m)",
        title="Position Error  (mean ± std)",
        out_prefix="position_error_combined",
        break_y=0.020,
        compress=25.0,
        top_max=0.50,
        yticks=[0, 0.004, 0.008, 0.012, 0.016, 0.020, 0.10, 0.20, 0.35, 0.50],
    ),
]

MAX_METRICS = [
    dict(
        col_mean="max_force_error",
        col_std=None,
        ylabel="Max |Force Z Error|  (N)",
        title="Force Z Error  (max)",
        out_prefix="force_error_max",
        break_y=3.0,
        compress=10.0,
        top_max=20.0,
        yticks=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5, 8, 11, 14, 17, 20],
    ),
    dict(
        col_mean="max_position_error",
        col_std=None,
        ylabel="Max Position Error  (m)",
        title="Position Error  (max)",
        out_prefix="position_error_max",
        break_y=0.020,
        compress=25.0,
        top_max=0.50,
        yticks=[0, 0.004, 0.008, 0.012, 0.016, 0.020, 0.10, 0.20, 0.35, 0.50],
    ),
]


def load_csv(results_csv, max_mult=None):
    """Return nested dict: all_data[robot][method] = dict of numpy arrays."""
    raw = {}
    with open(results_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            robot  = row["robot"]
            method = row["method"]
            raw.setdefault(robot, {}).setdefault(method, defaultdict(list))
            for col in ("omega_rad_s", "ee_linear_speed_m_s",
                        "mean_force_error", "max_force_error", "std_force_error",
                        "mean_position_error", "max_position_error", "std_position_error"):
                raw[robot][method][col].append(float(row[col]))

    all_data = {}
    for robot, methods in raw.items():
        all_data[robot] = {}
        for method, cols in methods.items():
            d = {k: np.array(v) for k, v in cols.items()}
            idx = np.argsort(d["ee_linear_speed_m_s"])
            d = {k: v[idx] for k, v in d.items()}
            if max_mult is not None:
                mask = d["omega_rad_s"] / np.pi <= max_mult + 1e-9
                d = {k: v[mask] for k, v in d.items()}
            all_data[robot][method] = d
    return all_data


def make_piecewise(break_y, compress):
    def forward(y):
        y = np.asarray(y, dtype=float)
        return np.where(y <= break_y, y, break_y + (y - break_y) / compress)

    def inverse(z):
        z = np.asarray(z, dtype=float)
        return np.where(z <= break_y, z, break_y + (z - break_y) * compress)

    return forward, inverse


def _fmt_val(v, col):
    if "force" in col:
        if abs(v) >= 1_000:
            return f"{v/1_000:.1f}k"
        return f"{v:.0f}"
    if "position" in col:
        if abs(v) >= 0.1:
            return f"{v:.2f}"
        return f"{v:.3f}"
    return f"{v:.3g}"


def _annotate_outliers(ax, datasets, col, top_max, plot_title=""):
    y_annot = top_max * 0.92
    groups = defaultdict(list)
    for name, data, color, marker in datasets:
        v    = data["ee_linear_speed_m_s"]
        vals = data[col]
        mask = vals > top_max
        for xi, yi in zip(v[mask], vals[mask]):
            groups[float(xi)].append((name, yi, color))

    if not groups:
        return

    for x_val, entries in groups.items():
        for _, _, color in entries:
            ax.plot(x_val, y_annot, marker="^", color=color,
                    markersize=_STYLE["outlier_markersize"], zorder=5, clip_on=False)

    print(f"\n[OUTLIERS] {plot_title} (clipped above {_fmt_val(top_max, col)})")
    for x_val in sorted(groups):
        for name, yi, _ in groups[x_val]:
            print(f"  v={x_val:.3f} m/s  {name}: {_fmt_val(yi, col)}")


def make_combined_plot(cfg, datasets, out_path):
    col_mean = cfg["col_mean"]
    col_std  = cfg["col_std"]
    break_y  = cfg["break_y"]
    compress = cfg["compress"]
    top_max  = cfg["top_max"]

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for name, data, color, marker in datasets:
        v    = data["ee_linear_speed_m_s"]
        mean = data[col_mean]

        if col_std is not None:
            std = data[col_std] * STD_RATIO
            lo  = np.maximum(mean - std, 0.0)
            hi  = np.minimum(mean + std, top_max)
            ax.fill_between(v, lo, hi, alpha=_STYLE["fill_alpha"], color=color, linewidth=0)

        mean_plot = np.where(mean > top_max, np.nan, mean)
        ax.plot(v, mean_plot, marker=marker, color=color, label=name,
                linewidth=_STYLE["plot_linewidth"], markersize=_STYLE["plot_markersize"],
                markerfacecolor=color)

    fwd, inv = make_piecewise(break_y, compress)
    ax.set_yscale("function", functions=(fwd, inv))
    ax.set_ylim(0.0, top_max)

    ax.set_yticks(cfg["yticks"])
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.ticklabel_format(style="plain", axis="y")

    ax.axhline(break_y, color="gray", linestyle="--",
               linewidth=_STYLE["hline_linewidth"], alpha=_STYLE["hline_alpha"])
    ax.annotate(
        f"  scale ÷{compress:.0f} above",
        xy=(0.0, break_y), xycoords=("axes fraction", "data"),
        fontsize=_STYLE["annot_fontsize"], color="gray", style="italic", va="bottom",
    )

    _annotate_outliers(ax, datasets, col_mean, top_max, cfg["title"])

    x_max = max(data["ee_linear_speed_m_s"].max() for _, data, _, _ in datasets)
    ax.set_xlim(0.0, x_max * 1.02)
    ax.set_xlabel("EE Linear Speed  v = r·ω  (m/s,  r = 0.1 m)")
    ax.set_ylabel(cfg["ylabel"])
    ax.legend(loc="upper left", framealpha=_STYLE["legend_framealpha"])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=_STYLE["plot_dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Combined mean±std comparison plots for speed_sweep_friction."
    )
    parser.add_argument("results_csv", help="Path to results.csv from collect_results.py")
    parser.add_argument("plots_dir",   help="Output directory for plots")
    parser.add_argument("--max-multiplier", type=float, default=1.0,
                        help="Only plot data with multiplier (ω/π) ≤ this value (default: 1.0). "
                             "Pass a large number (e.g. 999) to include all data.")
    args = parser.parse_args()

    if not os.path.isfile(args.results_csv):
        print(f"[ERROR] results CSV not found: {args.results_csv}")
        sys.exit(1)

    all_data = load_csv(args.results_csv, max_mult=args.max_multiplier)
    os.makedirs(args.plots_dir, exist_ok=True)

    for robot_key in ROBOTS:
        if robot_key not in all_data:
            print(f"[SKIP] No data for robot={robot_key}")
            continue

        datasets = []
        for label, method_key, color, marker in METHOD_KEYS:
            if method_key not in all_data[robot_key]:
                print(f"[SKIP] {robot_key}/{method_key}: no data")
                continue
            datasets.append((label, all_data[robot_key][method_key], color, marker))

        if not datasets:
            continue

        for cfg in COMBINED_METRICS + MAX_METRICS:
            out_fname = f"{cfg['out_prefix']}_{robot_key}.png"
            out_path  = os.path.join(args.plots_dir, out_fname)
            make_combined_plot(cfg, datasets, out_path)


if __name__ == "__main__":
    main()
