"""Position (X,Y,Z) and force tracking for all control methods at a given speed.

Reads NPZ files from E1_angular_speed_sweep/data/flat/ or slope30/.

Outputs (saved to plots/tracking_speed_<mult>[_slope30]/):
  position_tracking_xyz.png
  force_tracking.png

Usage
-----
    python plot.py --multiplier 3.4
    python plot.py --multiplier 3.4 3.8 4.4
    python plot.py --multiplier 3.4 --slope
"""

import argparse
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
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT  = os.path.join(SCRIPT_DIR, "..", "data")
PLOTS_DIR  = os.path.join(SCRIPT_DIR, "plots")

DT             = 0.001
SKIP_S         = 1.0
PLOT_DURATION_S = 2.0

METHOD_KEYS = [
    # ("Baseline",         "baseline",       "tab:gray",   "-",  "x"),
    ("Feedforward",      "feedforward",    "tab:blue",   "-",  "o"),
    ("Feedforward + PI", "feedforward_pi", "tab:purple", "-",  "s"),
    ("PD",               "pd",             "tab:green",  "-",  "^"),
    ("HFDC",             "paper",          "tab:orange", "-",  "D"),
]

AXES_LABELS = ["X", "Y", "Z"]


def load_npz(method_key, multiplier, slope):
    surface = "slope30" if slope else "flat"
    fname   = f"data_{multiplier}_all.npz" if slope else f"data_{multiplier}.npz"
    path = os.path.join(DATA_ROOT, surface, method_key, fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return np.load(path)


def plot_position_xyz(datasets, multiplier, out_dir):
    sk = int(SKIP_S / DT)

    for axis_idx, axis_label in enumerate(AXES_LABELS):
        print(f"\n  {axis_label} axis — avg / max |error| (m)")
        for name, data, color, ls, _ in datasets:
            pe = np.abs(data["actual_positions"][sk:, axis_idx]
                        - data["desired_positions"][sk:, axis_idx])
            print(f"    {name:<20s}  avg={np.mean(pe):.4f}  max={np.max(pe):.4f}")

    fig, axes = plt.subplots(
        3, 1, sharex=True, figsize=(3.25, 3.012),
        gridspec_kw={"height_ratios": [1, 1, 1]},
    )

    for axis_idx, ax in enumerate(axes):
        label = AXES_LABELS[axis_idx]
        desired_plotted = False
        for name, data, color, ls, _ in datasets:
            n = min(data["actual_positions"].shape[0], int(PLOT_DURATION_S / DT))
            t = np.arange(n) * DT
            ax.plot(t, data["actual_positions"][:n, axis_idx],
                    color=color, linestyle=ls, linewidth=0.8, label=name, alpha=0.85)
            if not desired_plotted:
                ax.plot(t, data["desired_positions"][:n, axis_idx],
                        color="black", linestyle="--", linewidth=0.8,
                        label="Desired", alpha=0.7)
                desired_plotted = True
        ax.set_ylabel(f"{label} (m)")
        if axis_idx == 0:
            ax.legend(loc="upper right", ncol=1)

    axes[-1].set_xlabel("Time (s)")

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "position_tracking_xyz.png")
    fig.savefig(out, dpi=600)
    plt.close(fig)
    print(f"[PLOT] {out}")


def plot_force_tracking(datasets, multiplier, out_dir):
    sk = int(SKIP_S / DT)
    fig, ax = plt.subplots(figsize=(3.25, 3))

    for name, data, color, ls, _ in datasets:
        fe = data["force_error"][:int(PLOT_DURATION_S / DT)]
        n  = len(fe)
        t  = np.arange(n) * DT
        avg = np.mean(np.abs(fe[sk:])) if n > sk else np.mean(np.abs(fe))
        max_err = np.max(np.abs(fe[sk:])) if n > sk else np.max(np.abs(fe))
        print(f"  {name:<20s}  avg={avg:.3f} N  max={max_err:.3f} N")
        ax.plot(t, fe, color=color, linestyle=ls, linewidth=0.8, alpha=0.80, label=name)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force Z Error (N)")
    ax.legend(loc="lower right")

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "force_tracking.png")
    fig.savefig(out, dpi=600)
    plt.close(fig)
    print(f"[PLOT] {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiplier", type=float, nargs="+", default=[3.4],
                        help="Speed multiplier(s) (ω/π) to plot")
    parser.add_argument("--slope", action="store_true",
                        help="Use slope30 data instead of flat")
    args = parser.parse_args()

    surface_label = "slope30" if args.slope else "flat"
    print(f"[INFO] Surface: {surface_label}")

    for mult in args.multiplier:
        slope_suffix = "_slope30" if args.slope else ""
        out_dir = os.path.join(PLOTS_DIR, f"tracking_speed_{mult}{slope_suffix}")

        print(f"\n[MULT] {mult}×")
        datasets = []
        for name, key, color, ls, marker in METHOD_KEYS:
            try:
                d = load_npz(key, mult, args.slope)
                datasets.append((name, d, color, ls, marker))
                print(f"  [LOAD] {name}")
            except FileNotFoundError as e:
                print(f"  [SKIP] {e}")

        if not datasets:
            print("  No data loaded, skipping.")
            continue

        plot_position_xyz(datasets, mult, out_dir)
        plot_force_tracking(datasets, mult, out_dir)


if __name__ == "__main__":
    main()
