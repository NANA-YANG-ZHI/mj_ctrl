"""Plot position (X, Y, Z) and force tracking comparing friction coefficients.

One set of plots is generated per friction coefficient supplied.  Each set
contains exactly 4 traces:
  - HFDC (no friction)     red, solid  — always included
  - HFDC+PI (no friction)  red, dashed — always included
  - HFDC   (μ=X)           tab:blue, solid
  - HFDC+PI(μ=X)           tab:blue, dashed

Outputs (one subdirectory per coefficient):
  plots/tracking_friction_<X>/position_tracking_x.png
  plots/tracking_friction_<X>/position_tracking_y.png
  plots/tracking_friction_<X>/position_tracking_z.png
  plots/tracking_friction_<X>/position_tracking_xyz.png
  plots/tracking_friction_<X>/force_tracking.png

Usage
-----
    # default (uses FRICTION_COEFFS list below)
    python friction_sweep/plot_tracking_results_friction.py

    # explicit coefficients via CLI — one output directory per value
    python friction_sweep/plot_tracking_results_friction.py 0.5 0.7
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tueplots import bundles

plt.rcParams.update(bundles.neurips2021(usetex=False))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR  = os.path.join(SCRIPT_DIR, "plots")

DT             = 0.001   # simulation timestep (s)
SKIP_S         = 0.0     # seconds to skip at start for steady-state metrics
FORCE_SKIP_S   = 1.0     # seconds to skip for force avg/max metrics
PLOT_DURATION_S = 2.0    # only plot this many seconds of data

# Default friction coefficients — overridden by command-line args
FRICTION_COEFFS = [0.5]

AXES_LABELS = ["X", "Y", "Z"]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_npz(method_dir: str, coeff: float) -> np.lib.npyio.NpzFile:
    path = os.path.join(PLOTS_DIR, method_dir, "data", f"data_{coeff}.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    return np.load(path)


def make_time_axis(n_samples: int) -> np.ndarray:
    return np.arange(n_samples) * DT


def coeff_str(coeff: float) -> str:
    return f"{coeff:.1f}" if coeff == round(coeff, 1) else str(coeff)


def build_datasets(coeff: float) -> list:
    """Return 4-entry list of (label, data, color, linestyle) for one coefficient."""
    entries = [
        ("HFDC (no friction)",           "paper_wo_surface_friction",    0.0,   "tab:red", "-"),
        ("HFDC+PI (no friction)",         "paper_pi_wo_surface_friction", 0.0,   "tab:red", "--"),
        (f"HFDC (μ={coeff_str(coeff)})",    "paper",                        coeff, "tab:blue", "-"),
        (f"HFDC+PI (μ={coeff_str(coeff)})", "paper_pi",                     coeff, "tab:blue", "--"),
    ]

    datasets = []
    for label, method_dir, c, color, ls in entries:
        try:
            d = load_npz(method_dir, c)
            datasets.append((label, d, color, ls))
            print(f"[LOAD] {label:35s} — {d['actual_positions'].shape[0]} timesteps")
        except FileNotFoundError as e:
            print(f"[SKIP] {e}")

    return datasets


# ── individual axis position plots ────────────────────────────────────────────

def plot_position_axis(datasets: list, axis_idx: int, coeff: float, out_dir: str) -> None:
    """One figure per axis (X / Y / Z): all datasets + desired reference."""
    lbl = AXES_LABELS[axis_idx]
    fig, ax = plt.subplots(figsize=(12, 4))

    desired_plotted = False
    for name, data, color, ls in datasets:
        n = min(data["actual_positions"].shape[0], int(PLOT_DURATION_S / DT))
        t = make_time_axis(n)
        ax.plot(t, data["actual_positions"][:n, axis_idx],
                color=color, linestyle=ls, linewidth=1.2, label=name, alpha=0.85)
        if not desired_plotted:
            ax.plot(t, data["desired_positions"][:n, axis_idx],
                    color="black", linestyle=":", linewidth=1.2,
                    label="Desired", alpha=0.7)
            desired_plotted = True

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{lbl} Position (m)")
    ax.set_title(f"Position Tracking — {lbl} Axis  |  μ={coeff_str(coeff)}")
    ax.legend(loc="best", fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = os.path.join(out_dir, f"position_tracking_{lbl.lower()}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {os.path.relpath(out)}")


# ── combined XYZ position plot ────────────────────────────────────────────────

def plot_position_xyz(datasets: list, coeff: float, out_dir: str) -> None:
    """3-row figure: one row per axis, all datasets overlaid, shared time axis."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"Position Tracking (X, Y, Z)  |  μ={coeff_str(coeff)}", fontsize=13
    )

    for axis_idx, ax in enumerate(axes):
        lbl = AXES_LABELS[axis_idx]
        desired_plotted = False
        for name, data, color, ls in datasets:
            n = min(data["actual_positions"].shape[0], int(PLOT_DURATION_S / DT))
            t = make_time_axis(n)
            ax.plot(t, data["actual_positions"][:n, axis_idx],
                    color=color, linestyle=ls, linewidth=1.2, label=name, alpha=0.85)
            if not desired_plotted:
                ax.plot(t, data["desired_positions"][:n, axis_idx],
                        color="black", linestyle=":", linewidth=1.2,
                        label="Desired", alpha=0.7)
                desired_plotted = True

        ax.set_ylabel(f"{lbl} (m)")
        ax.grid(True, alpha=0.3)
        if axis_idx == 0:
            ax.legend(loc="upper right", fontsize=8.5, framealpha=0.85, ncol=2)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    out = os.path.join(out_dir, "position_tracking_xyz.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {os.path.relpath(out)}")


# ── force tracking plot ───────────────────────────────────────────────────────

def plot_force_tracking(datasets: list, coeff: float, out_dir: str) -> None:
    """Force Z error over time for all datasets, with per-dataset avg/max."""
    sk       = int(SKIP_S / DT)
    force_sk = int(FORCE_SKIP_S / DT)

    fig, ax = plt.subplots()

    for name, data, color, ls in datasets:
        fe  = data["force_error"][:int(PLOT_DURATION_S / DT)]
        n   = len(fe)
        t   = make_time_axis(n)
        avg = np.mean(np.abs(fe[sk:]))       if n > sk       else np.mean(np.abs(fe))
        mx  = np.max(np.abs(fe[force_sk:]))  if n > force_sk else np.max(np.abs(fe))
        ax.plot(t, fe, color=color, linestyle=ls, linewidth=1.0, alpha=0.80,
                label=f"{name}  (avg|err|={avg:.3f} N, max|err|={mx:.3f} N)")

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force Z Error (N)")
    ax.set_title(f"Force Tracking — Force Z Error  |  μ={coeff_str(coeff)}")
    ax.legend(loc="upper right")

    out = os.path.join(out_dir, "force_tracking.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {os.path.relpath(out)}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # Parse friction coefficients from CLI or fall back to default
    if len(sys.argv) > 1:
        try:
            coeffs = [float(a) for a in sys.argv[1:]]
        except ValueError:
            print(f"Usage: {sys.argv[0]} [friction_coeff ...]  (e.g. 0.5 0.7)")
            sys.exit(1)
    else:
        coeffs = FRICTION_COEFFS

    print(f"Friction coefficients: {coeffs}\n")

    for coeff in coeffs:
        cs      = coeff_str(coeff)
        out_dir = os.path.join(PLOTS_DIR, f"tracking_friction_{cs}")
        os.makedirs(out_dir, exist_ok=True)

        print(f"── μ={cs} ──────────────────────────────────────")
        datasets = build_datasets(coeff)

        if not datasets:
            print(f"  No data loaded for μ={cs}, skipping.\n")
            continue

        print(f"  Generating plots -> {out_dir}")

        for axis_idx in range(3):
            plot_position_axis(datasets, axis_idx, coeff, out_dir)

        plot_position_xyz(datasets, coeff, out_dir)
        plot_force_tracking(datasets, coeff, out_dir)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
