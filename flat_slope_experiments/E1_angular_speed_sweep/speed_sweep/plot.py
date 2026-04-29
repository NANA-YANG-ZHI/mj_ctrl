"""Compare control-method metrics vs EE linear speed — combined mean ± std plots.

Reads per-speed .npz files from E1_angular_speed_sweep/data/flat/ or slope30/
and produces two combined plots (force error, position error).

Outputs (saved to plots/):
  force_error_combined.png
  position_error_combined.png
  force_error_combined_slope30.png      (with --slope)
  position_error_combined_slope30.png   (with --slope)

Usage
-----
    python plot.py
    python plot.py --slope
"""

import argparse
import glob
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

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
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT  = os.path.join(SCRIPT_DIR, "..", "data")
PLOTS_DIR  = os.path.join(SCRIPT_DIR, "plots")

METHOD_KEYS = [
    ("Baseline",         "baseline",       "tab:gray",   "x"),
    ("Feedforward",      "feedforward",    "tab:blue",   "o"),
    ("Feedforward + PI", "feedforward_pi", "tab:purple", "s"),
    ("PD",               "pd",             "tab:green",  "^"),
    ("HFDC",             "paper",          "tab:orange", "D"),
]

STD_RATIO = 0.3

_STYLE = {
    "plot_linewidth":    0.5,
    "plot_markersize":   1,
    "fill_alpha":        0.15,
    "hline_linewidth":   0.9,
    "hline_alpha":       0.55,
    "annot_fontsize":    7.5,
    "outlier_markersize":4,
    "legend_framealpha": 0.85,
    "plot_dpi":          600,
}

FIGSIZE = (3.25, 3)

DT = 0.001
FORCE_SKIP_SAMPLES = int(1.0 / DT)


def build_combined_metrics(slope_angle):
    suffix = "" if slope_angle == 0.0 else f"_slope{slope_angle:g}"
    return [
        dict(
            col_mean="avg_force_z_error",
            col_var="var_force_z_error",
            ylabel=f"Avg |Force Z Error| ± {STD_RATIO:.1f}Std  (N)",
            out=f"force_error_combined{suffix}.png",
            break_y=3.0,
            compress=10.0,
            top_max=20.0,
            yticks=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5, 8, 11, 14, 17, 20],
        ),
        dict(
            col_mean="avg_position_error",
            col_var="var_position_error",
            ylabel=f"Avg Position Error ± {STD_RATIO:.1f}Std  (m)",
            out=f"position_error_combined{suffix}.png",
            break_y=0.020,
            compress=25.0,
            top_max=0.50,
            yticks=[0, 0.004, 0.008, 0.012, 0.016, 0.020, 0.10, 0.20, 0.35, 0.50],
        ),
    ]


def load_all(data_dir):
    npz_files = sorted(
        glob.glob(os.path.join(data_dir, "data_*.npz")),
        key=lambda p: float(
            re.match(r"data_([0-9]+(?:\.[0-9]+)?)", os.path.basename(p)).group(1)
        ),
    )
    rows = {k: [] for k in (
        "ee_linear_speed_m_s",
        "avg_force_z_error", "var_force_z_error",
        "avg_position_error", "var_position_error",
    )}
    for fpath in npz_files:
        d = np.load(fpath)
        rows["ee_linear_speed_m_s"].append(
            float(d["ee_linear_speed_m_s"]) if "ee_linear_speed_m_s" in d
            else float(d["angular_speed_rad_s"]) * 0.1
        )
        fe = d["force_error"]
        fe_ss = fe[FORCE_SKIP_SAMPLES:]
        if len(fe_ss) > 0:
            rows["avg_force_z_error"].append(np.mean(np.abs(fe_ss)))
            rows["var_force_z_error"].append(np.var(fe_ss))
        else:
            rows["avg_force_z_error"].append(float("nan"))
            rows["var_force_z_error"].append(float("nan"))
        pe = d["position_error"]
        if len(pe) > 0:
            rows["avg_position_error"].append(np.mean(pe))
            rows["var_position_error"].append(np.var(pe))
        else:
            rows["avg_position_error"].append(float("nan"))
            rows["var_position_error"].append(float("nan"))
    return {k: np.array(v) for k, v in rows.items()}


def make_piecewise(break_y, compress):
    def forward(y):
        y = np.asarray(y, dtype=float)
        return np.where(y <= break_y, y, break_y + (y - break_y) / compress)
    def inverse(z):
        z = np.asarray(z, dtype=float)
        return np.where(z <= break_y, z, break_y + (z - break_y) * compress)
    return forward, inverse


def _annotate_outliers(ax, datasets, col, top_max):
    y_annot = top_max * 0.92
    groups = defaultdict(list)
    for name, data, color, marker in datasets:
        v, vals = data["ee_linear_speed_m_s"], data[col]
        mask = vals > top_max
        for xi, yi in zip(v[mask], vals[mask]):
            groups[float(xi)].append((name, yi, color))
    for x_val, entries in groups.items():
        for _, _, color in entries:
            ax.plot(x_val, y_annot, marker="^", color=color,
                    markersize=_STYLE["outlier_markersize"], zorder=5, clip_on=False)


def make_combined_plot(cfg, datasets):
    col_mean = cfg["col_mean"]
    col_var  = cfg["col_var"]
    break_y  = cfg["break_y"]
    compress = cfg["compress"]
    top_max  = cfg["top_max"]

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for name, data, color, marker in datasets:
        v    = data["ee_linear_speed_m_s"]
        mean = data[col_mean]
        std  = np.sqrt(np.maximum(data[col_var], 0.0)) * STD_RATIO
        lo   = np.maximum(mean - std, 0.0)
        hi   = np.minimum(mean + std, top_max)
        mean_plot = np.where(mean > top_max, np.nan, mean)

        ax.fill_between(v, lo, hi, alpha=_STYLE["fill_alpha"], color=color, linewidth=0)
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
    _annotate_outliers(ax, datasets, col_mean, top_max)

    x_max = max(data["ee_linear_speed_m_s"].max() for _, data, _, _ in datasets)
    ax.set_xlim(0.0, x_max * 1.02)
    ax.set_xlabel("EE Linear Speed  v = r·ω  (m/s,  r = 0.1 m)")
    ax.set_ylabel(cfg["ylabel"])
    ax.legend(loc="upper left", framealpha=_STYLE["legend_framealpha"])

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, cfg["out"])
    fig.savefig(out, dpi=_STYLE["plot_dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {cfg['out']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slope", action="store_true",
                        help="Plot slope30 data (default: flat)")
    args = parser.parse_args()

    slope_angle = 30.0 if args.slope else 0.0
    surface = "slope30" if args.slope else "flat"
    print(f"[INFO] Surface: {surface}")

    datasets = []
    for label, key, color, marker in METHOD_KEYS:
        data_dir = os.path.join(DATA_ROOT, surface, key)
        if not os.path.isdir(data_dir):
            print(f"[SKIP] {label}: {data_dir}")
            continue
        datasets.append((label, load_all(data_dir), color, marker))

    if not datasets:
        print("No data found.")
        return

    for cfg in build_combined_metrics(slope_angle):
        make_combined_plot(cfg, datasets)


if __name__ == "__main__":
    main()
