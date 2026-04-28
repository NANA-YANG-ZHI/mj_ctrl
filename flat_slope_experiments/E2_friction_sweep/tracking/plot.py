"""Position (X,Y,Z) and force tracking at specific friction coefficients.

Reads:
  ../data/paper/data_<mu>.npz          HFDC with friction
  ../data/paper_pi/data_<mu>.npz       HFDC+PI with friction
  ../data/paper_wo_surface_friction/data_0.0.npz    HFDC no friction
  ../data/paper_pi_wo_surface_friction/data_0.0.npz HFDC+PI no friction

Outputs (saved to plots/tracking_friction_<mu>/):
  position_tracking_xyz.png
  force_tracking.png

Usage
-----
    python plot.py --friction 0.7
    python plot.py --friction 0.3 0.7
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

AXES_LABELS = ["X", "Y", "Z"]


def coeff_str(coeff):
    return f"{coeff:.1f}" if coeff == round(coeff, 1) else str(coeff)


def load_npz(method_key, coeff):
    path = os.path.join(DATA_ROOT, method_key, f"data_{coeff_str(coeff)}.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return np.load(path)


def build_datasets(coeff):
    entries = [
        ("HFDC (no friction)",              "paper_wo_surface_friction",    0.0,   "tab:red",  "-"),
        # ("HFDC+PI (no friction)",         "paper_pi_wo_surface_friction", 0.0,   "tab:red",  "--"),
        (f"HFDC (μ={coeff_str(coeff)})",    "paper",                        coeff, "tab:blue", "-"),
        # (f"HFDC+PI (μ={coeff_str(coeff)})", "paper_pi",                   coeff, "tab:blue", "--"),
    ]

    datasets = []
    for label, method_key, c, color, ls in entries:
        try:
            d = load_npz(method_key, c)
            datasets.append((label, d, color, ls))
            print(f"  [LOAD] {label}")
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
    return datasets


def plot_position_xyz(datasets, coeff, out_dir):
    sk = int(SKIP_S / DT)

    for axis_idx, axis_label in enumerate(AXES_LABELS):
        print(f"\n  {axis_label} axis — avg / max error (m)")
        for name, data, color, ls in datasets:
            pe = (data["actual_positions"][sk:, axis_idx]
                  - data["desired_positions"][sk:, axis_idx])
            print(f"    {name:<35s}  avg={np.mean(pe):.4f}  max={np.max(pe):.4f}")

    fig, axes = plt.subplots(
        3, 1, sharex=True, figsize=(3.25, 2.008),
        gridspec_kw={"height_ratios": [1, 1, 1]},
    )

    for axis_idx, ax in enumerate(axes):
        lbl = AXES_LABELS[axis_idx]
        for name, data, color, ls in datasets:
            n  = min(data["actual_positions"].shape[0], int(PLOT_DURATION_S / DT))
            t  = np.arange(n) * DT
            pe = (data["actual_positions"][:n, axis_idx]
                  - data["desired_positions"][:n, axis_idx])
            ax.plot(t, pe, color=color, linestyle=ls, linewidth=1.0,
                    label=name, alpha=0.85)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_ylabel(f"{lbl} err (m)")
        if axis_idx == 0:
            ax.legend(loc="upper right", ncol=2)

    axes[-1].set_xlabel("Time (s)")

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "position_tracking_xyz.png")
    fig.savefig(out, dpi=600)
    plt.close(fig)
    print(f"[PLOT] {out}")


def plot_force_tracking(datasets, coeff, out_dir):
    sk = int(SKIP_S / DT)
    fig, ax = plt.subplots(figsize=(3.25, 2.0))

    for name, data, color, ls in datasets:
        fe  = data["force_error"][:int(PLOT_DURATION_S / DT)]
        n   = len(fe)
        t   = np.arange(n) * DT
        avg = np.mean(np.abs(fe[sk:])) if n > sk else np.mean(np.abs(fe))
        mx  = np.max(np.abs(fe[sk:])) if n > sk else np.max(np.abs(fe))
        print(f"  {name:<35s}  avg={avg:.3f} N  max={mx:.3f} N")
        ax.plot(t, fe, color=color, linestyle=ls, linewidth=1.0, alpha=0.80, label=name)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force Z Error (N)")
    ax.legend(loc="upper right")

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "force_tracking.png")
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--friction", type=float, nargs="+", default=[0.7],
                        help="Friction coefficient(s) to plot")
    args = parser.parse_args()

    for coeff in args.friction:
        cs = coeff_str(coeff)
        out_dir = os.path.join(PLOTS_DIR, f"tracking_friction_{cs}")
        print(f"\n[μ={cs}]")

        datasets = build_datasets(coeff)
        if not datasets:
            print("  No data loaded, skipping.")
            continue

        plot_position_xyz(datasets, coeff, out_dir)
        plot_force_tracking(datasets, coeff, out_dir)


if __name__ == "__main__":
    main()
