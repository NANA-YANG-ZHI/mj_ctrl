"""Plot Y-Z trajectory and force error for speed_sweep_friction experiments.

Three modes
-----------
(default)         – 2×3 grid (all 6 methods) Y-Z trajectories + combined force error plot.
--method <key>    – single 2-panel plot: Y-Z trajectory (top) + force error vs time (bottom).
--all-separate    – one 2-panel plot per method.
--overlay         – all method trajectories overlaid on a single axes.

No burn-in skip is applied (unlike angular_speed_sweep).

Usage
-----
# Comparison grid for all methods at multiplier 1.0
python cylinder_experiments/speed_sweep_friction/plot_trajectory_and_force.py --multiplier 1.0

# Single method
python cylinder_experiments/speed_sweep_friction/plot_trajectory_and_force.py --multiplier 1.0 --method paper

# All methods, one file each
python cylinder_experiments/speed_sweep_friction/plot_trajectory_and_force.py --multiplier 1.0 --all-separate

# Different robot
python cylinder_experiments/speed_sweep_friction/plot_trajectory_and_force.py \\
    --multiplier 1.0 --robot fr3_jointf_surff

# Custom paths
python cylinder_experiments/speed_sweep_friction/plot_trajectory_and_force.py \\
    --data-root sweep_friction_results/data \\
    --output-dir sweep_friction_results/plots/traj \\
    --multiplier 2.0
"""

import argparse
import os

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
})

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(SCRIPT_DIR, "sweep_friction_results", "data")
DEFAULT_OUT  = os.path.join(SCRIPT_DIR, "sweep_friction_results", "plots", "traj")

DT              = 0.001
CYLINDER_RADIUS = 0.1  # m

METHOD_KEYS = [
    ("SUP",              "baseline",  "tab:gray",   "o"),
    ("FF",               "ff",        "tab:blue",   "o"),
    ("Feedforward + PI", "ff_pi",     "tab:purple", "s"),
    ("PD",               "pd",        "tab:green",  "^"),
    ("HFDC",             "paper",     "tab:orange", "D"),
    ("HFDC + PI",        "paper_pi",  "tab:red",    "P"),
]

DEFAULT_KEYS = {"baseline", "ff", "pd", "paper"}

PLOT_DPI = 300


def load_npz(data_root, robot, method, multiplier):
    base = os.path.join(data_root, robot, method)
    for suffix in (f"data_{multiplier:.1f}_all.npz", f"data_{multiplier:.1f}.npz"):
        path = os.path.join(base, suffix)
        if os.path.isfile(path):
            return np.load(path)
    raise FileNotFoundError(
        f"No NPZ for robot={robot}, method={method}, mult={multiplier:.1f} in {base}"
    )


def _mean_abs_force(d):
    return float(np.nanmean(np.abs(d["force_error"])))


def _panel_yz(ax, d, color):
    """Y-Z 2D trajectory: desired (dashed black) + actual (color)."""
    act = d["actual_positions"]
    des = d["desired_positions"]
    ax.plot(des[:, 1], des[:, 2], color="black", linestyle="--",
            linewidth=0.8, alpha=0.6, label="Desired")
    ax.plot(act[:, 1], act[:, 2], color=color, linewidth=0.8,
            alpha=0.85, label="Actual")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Y (m)")
    ax.set_ylabel("Z (m)")


def _panel_force(ax, d, color, label=None):
    """Force Z error vs time."""
    fe = d["force_error"]
    t  = np.arange(len(fe)) * DT
    avg = _mean_abs_force(d)
    lbl = f"{label}" if label else f""
    ax.plot(t, fe, color=color, linewidth=0.8, alpha=0.85, label=lbl)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force Z Error (N)")


# ── Mode 0: all trajectories overlaid on one axes ─────────────────────────────

def plot_overlay_yz(datasets, robot, multiplier, out_dir):
    fig, ax = plt.subplots(figsize=(3.25, 2.0))

    des = datasets[0][1]["desired_positions"]
    ax.plot(des[:, 1], des[:, 2], color="black", linestyle="--",
            linewidth=0.8, alpha=0.55, label="Desired")

    for label, d, color, _ in datasets:
        act = d["actual_positions"]
        ax.plot(act[:, 1], act[:, 2], color=color, linewidth=0.9,
                alpha=0.85, label=label)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Y (m)")
    ax.set_ylabel("Z (m)")
    ax.legend(loc="lower center", fontsize=7, framealpha=0.85)
    plt.tight_layout()

    out = os.path.join(out_dir, f"overlay_yz_{robot}_mult{multiplier:.1f}.png")
    fig.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {out}")


# ── Mode 1: 2×3 comparison grid ───────────────────────────────────────────────

def plot_comparison_grid(datasets, robot, multiplier, out_dir):
    ncols = 3
    nrows = (len(datasets) + ncols - 1) // ncols

    # Y-Z trajectory grid
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.75, nrows * 2.0), squeeze=False)
    axes_flat = axes.flatten()

    for i, (label, d, color, _) in enumerate(datasets):
        ax = axes_flat[i]
        _panel_yz(ax, d, color)
        if i == 0:
            ax.legend(loc="best", fontsize=7, framealpha=0.85)

    for j in range(len(datasets), nrows * ncols):
        axes_flat[j].set_visible(False)

    plt.tight_layout()

    out = os.path.join(out_dir, f"comparison_yz_{robot}_mult{multiplier:.1f}.png")
    fig.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {out}")

    # Combined force error time series
    fig2, ax2 = plt.subplots(figsize=(3.25, 2.0))
    for label, d, color, _ in datasets:
        _panel_force(ax2, d, color, label=label)
    ax2.set_xlim(0, 0.4)
    ax2.legend(loc="lower right", fontsize=7, framealpha=0.85)
    plt.tight_layout()

    out2 = os.path.join(out_dir, f"comparison_force_{robot}_mult{multiplier:.1f}.png")
    fig2.savefig(out2, dpi=PLOT_DPI)
    plt.close(fig2)
    print(f"[PLOT] {out2}")


# ── Mode 2 & 3: single method 2-panel ─────────────────────────────────────────

def plot_single_method(label, d, color, robot, multiplier, out_dir):
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(3.25, 4.0))

    _panel_yz(ax_top, d, color)
    ax_top.legend(loc="best", fontsize=7, framealpha=0.85)

    _panel_force(ax_bot, d, color)
    ax_bot.legend(loc="best", fontsize=7, framealpha=0.85)

    plt.tight_layout()

    safe = label.lower().replace(" + ", "_plus_").replace(" ", "_")
    out = os.path.join(out_dir, f"single_{safe}_{robot}_mult{multiplier:.1f}.png")
    fig.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot Y-Z trajectory and force error for speed_sweep_friction."
    )
    parser.add_argument("--data-root",    default=DEFAULT_DATA,
                        help="Path to the data root directory")
    parser.add_argument("--output-dir",   default=DEFAULT_OUT,
                        help="Directory for output plots")
    parser.add_argument("--robot",        default="fr3_friction",
                        choices=["fr3_friction", "fr3_jointf_surff"])
    parser.add_argument("--multiplier",   type=float, default=1.0,
                        help="Speed multiplier to plot (0.1 – 3.0)")
    parser.add_argument("--method",       default=None,
                        help="Single method key: baseline/ff/ff_pi/pd/paper/paper_pi")
    parser.add_argument("--all-separate", action="store_true",
                        help="Generate one 2-panel plot per method")
    parser.add_argument("--overlay", action="store_true",
                        help="Overlay all method trajectories on a single axes")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    method_filter = {args.method} if args.method else DEFAULT_KEYS
    datasets = []
    for label, key, color, marker in METHOD_KEYS:
        if key not in method_filter:
            continue
        try:
            d = load_npz(args.data_root, args.robot, key, args.multiplier)
            datasets.append((label, d, color, marker))
            omega  = float(d["angular_speed_rad_s"])
            ee_spd = float(d["ee_linear_speed_m_s"]) if "ee_linear_speed_m_s" in d \
                     else omega * CYLINDER_RADIUS
            print(f"[LOAD] {label:20s}  ω={omega:.3f} rad/s  v={ee_spd:.3f} m/s")
        except FileNotFoundError as e:
            print(f"[SKIP] {e}")

    if not datasets:
        print("[ERROR] No data loaded.")
        return

    if args.method:
        label, d, color, _ = datasets[0]
        plot_single_method(label, d, color, args.robot, args.multiplier, args.output_dir)
    elif args.all_separate:
        for label, d, color, _ in datasets:
            plot_single_method(label, d, color, args.robot, args.multiplier, args.output_dir)
    elif args.overlay:
        plot_overlay_yz(datasets, args.robot, args.multiplier, args.output_dir)
    else:
        plot_comparison_grid(datasets, args.robot, args.multiplier, args.output_dir)


if __name__ == "__main__":
    main()
