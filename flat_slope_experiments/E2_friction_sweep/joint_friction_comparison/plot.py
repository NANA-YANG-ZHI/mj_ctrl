"""Bar chart + time-series for joint-friction comparison (HFDC vs HFDC+PI).

Bar charts across 8 conditions (flat/slope × joint/no-joint × surface/no-surface).
Time-series figures: force Z error and position error vs time.

Reads:
  ../data/joint_friction_comparison/<method>/<cond>/   (new joint-friction runs)
  ../data/surface_comparison/<method>/<cond>/           (no-joint-friction baseline)

Outputs (saved to plots/):
  joint_friction_force_error.png
  joint_friction_position_error.png
  timeseries_hfdc_flat.png
  timeseries_hfdc_pi_flat.png
  timeseries_hfdc_slope.png
  timeseries_hfdc_pi_slope.png

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
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 8,
    "legend.fontsize": 6,
})

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT    = os.path.join(SCRIPT_DIR, "..", "data")
BASE_DIR     = os.path.join(DATA_ROOT, "joint_friction_comparison")
BASELINE_DIR = os.path.join(DATA_ROOT, "surface_comparison")
PLOTS_DIR    = os.path.join(SCRIPT_DIR, "plots")

DT             = 0.001
SKIP_S         = 1.0
PLOT_DURATION_S = 2.0

# Each pair: (flat_source, flat_cond, slope_source, slope_cond, x_label)
CONDITION_PAIRS = [
    ("baseline", "flat_frictionless",     "baseline", "slope30_frictionless",     "No joint\nno surf."),
    ("baseline", "flat_friction_0.7",     "baseline", "slope30_friction_0.7",     "No joint\nsurf. μ=0.7"),
    ("new",      "flat_jointf_no_surff",  "new",      "slope30_jointf_no_surff",  "Joint\nno surf."),
    ("new",      "flat_jointf_surff_0.7", "new",      "slope30_jointf_surff_0.7", "Joint\n+ surf. μ=0.7"),
]

SURFACES = [
    ("Flat",      "tab:blue"),
    ("Slope 30°", "tab:orange"),
]

METHODS = [
    ("paper",    "HFDC"),
    # ("paper_pi", "HFDC + PI"),
]

BAR_METRICS = [
    dict(npz_key="force_error",    ylabel="Avg |Force Z Error| (N)",
         out="joint_friction_force_error.png"),
    dict(npz_key="position_error", ylabel="Avg Position Error (m)",
         out="joint_friction_position_error.png"),
]

FLAT_TRACES = [
    ("baseline", "flat_frictionless",     "No joint, no surf.",  "tab:blue",   "-"),
    ("new",      "flat_jointf_no_surff",  "Joint, no surf.",     "tab:orange", "-"),
    ("baseline", "flat_friction_0.7",     "No joint, surf.",     "tab:blue",   "--"),
    ("new",      "flat_jointf_surff_0.7", "Joint + surf.",       "tab:orange", "--"),
]

SLOPE_TRACES = [
    ("baseline", "slope30_frictionless",     "No joint, no surf.",  "tab:blue",   "-"),
    ("new",      "slope30_jointf_no_surff",  "Joint, no surf.",     "tab:orange", "-"),
    ("baseline", "slope30_friction_0.7",     "No joint, surf.",     "tab:blue",   "--"),
    ("new",      "slope30_jointf_surff_0.7", "Joint + surf.",       "tab:orange", "--"),
]

TIMESERIES_FIGS = [
    ("paper",    FLAT_TRACES,  "timeseries_hfdc_flat.png"),
    # ("paper_pi", FLAT_TRACES,  "timeseries_hfdc_pi_flat.png"),
    ("paper",    SLOPE_TRACES, "timeseries_hfdc_slope.png"),
    # ("paper_pi", SLOPE_TRACES, "timeseries_hfdc_pi_slope.png"),
]


def load_npz_mean_std(data_dir, npz_key):
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


def load_first_npz(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not files:
        return None
    return np.load(files[0])


def make_bar_chart(cfg):
    n_pairs  = len(CONDITION_PAIRS)
    n_series = len(METHODS) * len(SURFACES)
    width    = 0.35
    offsets  = np.linspace(-(n_series - 1) * width / 2,
                            (n_series - 1) * width / 2,
                            n_series)

    fig, ax = plt.subplots(figsize=(3.25, 2.25))
    x = np.arange(n_pairs)

    series_idx = 0
    for method_key, method_label in METHODS:
        for surf_label, color in SURFACES:
            means, stds = [], []
            for flat_src, flat_cond, slope_src, slope_cond, _ in CONDITION_PAIRS:
                if surf_label == "Flat":
                    src, cond = flat_src, flat_cond
                else:
                    src, cond = slope_src, slope_cond
                root     = BASELINE_DIR if src == "baseline" else BASE_DIR
                data_dir = os.path.join(root, method_key, cond)
                mean, std = load_npz_mean_std(data_dir, cfg["npz_key"])
                means.append(mean)
                stds.append(std)

            means = np.array(means)
            stds  = np.array(stds)
            valid = ~np.isnan(means)
            bar_label = f"{method_label} {surf_label}" if len(METHODS) > 1 else surf_label

            bars = ax.bar(
                x[valid] + offsets[series_idx], means[valid],
                width=width * 0.9,
                yerr=stds[valid],
                capsize=3,
                color=color,
                alpha=0.85,
                label=bar_label,
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
            series_idx += 1

    ax.set_xticks(x)
    ax.set_xticklabels([label for *_, label in CONDITION_PAIRS])
    ax.set_ylabel(cfg["ylabel"])
    ax.legend(loc="upper left", framealpha=0.85)
    ax.set_ylim(bottom=0)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, cfg["out"])
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {cfg['out']}")


def make_timeseries(method_dir, traces, filename):
    n_plot = int(PLOT_DURATION_S / DT)
    t = np.arange(n_plot) * DT

    fig, (ax_force, ax_pos) = plt.subplots(
        2, 1, sharex=True, figsize=(3.25, 3.012),
        gridspec_kw={"height_ratios": [1, 1]},
    )

    for source, cond_dir, label, color, ls in traces:
        root     = BASELINE_DIR if source == "baseline" else BASE_DIR
        data_dir = os.path.join(root, method_dir, cond_dir)
        data = load_first_npz(data_dir)
        if data is None:
            print(f"  [SKIP] no data in {data_dir}")
            continue

        fe = data["force_error"]
        pe = data["position_error"]
        n = min(len(fe), n_plot)
        ax_force.plot(t[:n], fe[:n], color=color, linestyle=ls,
                      linewidth=1.0, alpha=0.85, label=label)
        n = min(len(pe), n_plot)
        ax_pos.plot(t[:n], pe[:n], color=color, linestyle=ls,
                    linewidth=1.0, alpha=0.85, label=label)

    ax_force.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax_force.set_ylabel("Force Z Error (N)")

    ax_pos.set_xlabel("Time (s)")
    ax_pos.set_ylabel("Position Error (m)")
    ax_pos.legend(loc="upper right")

    fig.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, filename)
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {filename}")


def main():
    for cfg in BAR_METRICS:
        make_bar_chart(cfg)

    for method_dir, traces, filename in TIMESERIES_FIGS:
        make_timeseries(method_dir, traces, filename)


if __name__ == "__main__":
    main()
