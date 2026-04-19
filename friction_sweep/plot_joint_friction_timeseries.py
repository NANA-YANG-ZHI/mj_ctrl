"""Time-series plots for joint friction comparison experiments.

Four figures (HFDC flat / HFDC+PI flat / HFDC slope / HFDC+PI slope),
each with two subplots:
  top:    Force Z error vs time
  bottom: Position error magnitude vs time

Four traces per figure (color = joint friction, linestyle = surface friction):
  blue  solid  — no joint friction, no surface friction  (μ=0)
  orange solid  — joint friction,    no surface friction  (μ=0)
  blue  dashed  — no joint friction, surface friction     (μ=0.7)
  orange dashed  — joint friction,    surface friction     (μ=0.7)

Baseline data (no joint friction) read from --baseline-dir (surface_comparison/).
New data (with joint friction)  read from --base-dir     (joint_friction_comparison/).

Usage
-----
    python friction_sweep/plot_joint_friction_timeseries.py
    python friction_sweep/plot_joint_friction_timeseries.py \\
        --base-dir path/to/joint_friction_comparison \\
        --baseline-dir path/to/surface_comparison
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from tueplots import bundles

plt.rcParams.update(bundles.icml2024(usetex=False))
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 6,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DT = 0.001            # simulation timestep (s)
PLOT_DURATION_S = 2.0 # seconds to show
SKIP_S = 1.0          # seconds to skip for metric computation

# Each trace: (source, cond_dir, label, color, linestyle)
# source: "baseline" → baseline_dir, "new" → base_dir
FLAT_TRACES = [
    ("baseline", "flat_frictionless",       "No joint, no surf.",   "tab:blue",   "-"),
    ("new",      "flat_jointf_no_surff",    "Joint, no surf.",      "tab:orange", "-"),
    ("baseline", "flat_friction_0.7",       "No joint, surf.",      "tab:blue",   "--"),
    ("new",      "flat_jointf_surff_0.7",   "Joint + surf.",        "tab:orange", "--"),
]

SLOPE_TRACES = [
    ("baseline", "slope30_frictionless",      "No joint, no surf.",   "tab:blue",   "-"),
    ("new",      "slope30_jointf_no_surff",   "Joint, no surf.",      "tab:orange", "-"),
    ("baseline", "slope30_friction_0.7",      "No joint, surf.",      "tab:blue",   "--"),
    ("new",      "slope30_jointf_surff_0.7",  "Joint + surf.",        "tab:orange", "--"),
]

# (method_dir, geometry_label, traces, out_filename)
FIGURES = [
    ("paper",    "Flat surface — HFDC",      FLAT_TRACES,  "timeseries_hfdc_flat.png"),
    ("paper_pi", "Flat surface — HFDC+PI",   FLAT_TRACES,  "timeseries_hfdc_pi_flat.png"),
    ("paper",    "Slope 30° — HFDC",         SLOPE_TRACES, "timeseries_hfdc_slope.png"),
    ("paper_pi", "Slope 30° — HFDC+PI",      SLOPE_TRACES, "timeseries_hfdc_pi_slope.png"),
]


def load_first_npz(data_dir: str):
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not files:
        return None
    return np.load(files[0])


def compute_stats(arr: np.ndarray):
    """Return (mean, max) of |arr| skipping first SKIP_S seconds."""
    skip = int(SKIP_S / DT)
    arr = np.abs(arr[skip:])
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.max(arr))


def make_figure(method_dir, title, traces, base_dir, baseline_dir):
    """Build time-series figure and return (fig, stats_rows).

    stats_rows is a list of dicts with keys: label, force_mean, force_max,
    pos_mean, pos_max (all floats, positions in metres).
    """
    n_plot = int(PLOT_DURATION_S / DT)
    t = np.arange(n_plot) * DT

    fig, (ax_force, ax_pos) = plt.subplots(
        2, 1, 
        sharex=True, 
        figsize=(3.25, 3.012),
        gridspec_kw={"height_ratios": [1, 1]})
    stats_rows = []

    for source, cond_dir, label, color, ls in traces:
        root = baseline_dir if source == "baseline" else base_dir
        data_dir = os.path.join(root, method_dir, cond_dir)
        data = load_first_npz(data_dir)
        if data is None:
            print(f"[SKIP] no data in {data_dir}")
            continue

        fe = data["force_error"]
        pe = data["position_error"]

        # Compute stats (skip first SKIP_S seconds)
        fm, fx = compute_stats(fe)
        pm, px = compute_stats(pe)
        stats_rows.append(dict(label=label,
                               force_mean=fm, force_max=fx,
                               pos_mean=pm, pos_max=px))
        print(f"  [{label}]  force mean={fm:.4f} N  max={fx:.4f} N  "
              f"| pos mean={pm*1000:.3f} mm  max={px*1000:.3f} mm")

        n = min(len(fe), n_plot)
        ax_force.plot(t[:n], fe[:n], color=color, linestyle=ls,
                      linewidth=1.0, alpha=0.85, label=label)

        n = min(len(pe), n_plot)
        ax_pos.plot(t[:n], pe[:n], color=color, linestyle=ls,
                    linewidth=1.0, alpha=0.85, label=label)

    ax_force.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax_force.set_ylabel("Force Z Error (N)")
    ax_force.set_title(title)

    ax_pos.set_xlabel("Time (s)")
    ax_pos.set_ylabel("Position Error (m)")
    ax_pos.legend(loc="upper right")


    fig.tight_layout()
    return fig, stats_rows


def main():
    parser = argparse.ArgumentParser(
        description="Time-series plots for joint friction comparison (first 2 s)."
    )
    parser.add_argument(
        "--base-dir",
        default=os.path.join(SCRIPT_DIR, "plots", "joint_friction_comparison"),
        help="Directory with new joint-friction data (paper/ and paper_pi/ subdirs).",
    )
    parser.add_argument(
        "--baseline-dir",
        default=os.path.join(SCRIPT_DIR, "plots", "surface_comparison"),
        help="Directory with baseline (no joint friction) data.",
    )
    args = parser.parse_args()

    out_dir = args.base_dir
    os.makedirs(out_dir, exist_ok=True)

    all_stats = {}
    for method_dir, title, traces, filename in FIGURES:
        print(f"\n--- {title} ---")
        fig, stats_rows = make_figure(method_dir, title, traces, args.base_dir, args.baseline_dir)
        out = os.path.join(out_dir, filename)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] {os.path.relpath(out)}")
        all_stats[title] = stats_rows

    print(f"\nDone. Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
