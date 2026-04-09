"""Plot position (X, Y, Z) and force tracking comparing friction coefficients.

Always includes no-surface-friction baselines for both methods:
  - HFPD             (paper_wo_surface_friction/data/data_0.0.npz)
  - HFPD + PI        (paper_pi_wo_surface_friction/data/data_0.0.npz)

Plus, for each friction coefficient supplied, loads:
  - HFPD   (μ=X)    (paper/data/data_X.npz)
  - HFPD+PI(μ=X)    (paper_pi/data/data_X.npz)

Outputs (saved to plots/tracking_friction_<coeff1>_<coeff2>_…/):
  position_tracking_x.png    – X-axis: actual vs desired for all datasets
  position_tracking_y.png    – Y-axis
  position_tracking_z.png    – Z-axis
  position_tracking_xyz.png  – Combined 3-row figure (X, Y, Z)
  force_tracking.png          – Force Z error over time for all datasets

Usage
-----
    # default (uses FRICTION_COEFFS list below)
    python friction_sweep/plot_tracking_results_friction.py

    # explicit coefficients via CLI
    python friction_sweep/plot_tracking_results_friction.py 0.3 0.5
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR  = os.path.join(SCRIPT_DIR, "plots")

DT           = 0.001   # simulation timestep (s)
SKIP_S       = 0.0     # seconds to skip at start for steady-state metrics
FORCE_SKIP_S = 1.0     # seconds to skip for force avg/max metrics

# Default friction coefficients — overridden by command-line args
FRICTION_COEFFS = [0.5]

AXES_LABELS = ["X", "Y", "Z"]

# Colors cycled across friction coefficient values (tab10 palette)
_COEFF_COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red",
    "tab:purple", "tab:brown", "tab:pink", "tab:cyan",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_npz(method_dir: str, coeff: float) -> np.lib.npyio.NpzFile:
    path = os.path.join(PLOTS_DIR, method_dir, "data", f"data_{coeff}.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    return np.load(path)


def make_time_axis(n_samples: int) -> np.ndarray:
    return np.arange(n_samples) * DT


def build_datasets(coeffs: list) -> list:
    """Return list of (label, data, color, linestyle) for all datasets.

    No-friction baselines are always prepended; then one pair of entries per
    friction coefficient in *coeffs*.
    """
    datasets = []

    # ── no-friction baselines ─────────────────────────────────────────────
    baselines = [
        ("HFPD (no friction)",    "paper_wo_surface_friction",    "dimgray", "-"),
        ("HFPD+PI (no friction)", "paper_pi_wo_surface_friction", "dimgray", "--"),
    ]
    for label, method_dir, color, ls in baselines:
        try:
            d = load_npz(method_dir, 0.0)
            datasets.append((label, d, color, ls))
            print(f"[LOAD] {label:30s} — {d['actual_positions'].shape[0]} timesteps")
        except FileNotFoundError as e:
            print(f"[SKIP] {e}")

    # ── per-coefficient pairs ─────────────────────────────────────────────
    for i, coeff in enumerate(coeffs):
        color = _COEFF_COLORS[i % len(_COEFF_COLORS)]
        coeff_str = f"{coeff:.1f}" if coeff == round(coeff, 1) else str(coeff)
        pairs = [
            (f"HFPD (μ={coeff_str})",    "paper",    color, "-"),
            (f"HFPD+PI (μ={coeff_str})", "paper_pi", color, "--"),
        ]
        for label, method_dir, col, ls in pairs:
            try:
                d = load_npz(method_dir, coeff)
                datasets.append((label, d, col, ls))
                print(f"[LOAD] {label:30s} — {d['actual_positions'].shape[0]} timesteps")
            except FileNotFoundError as e:
                print(f"[SKIP] {e}")

    return datasets


# ── individual axis position plots ────────────────────────────────────────────

def plot_position_axis(datasets: list, axis_idx: int, out_dir: str) -> None:
    """One figure per axis (X / Y / Z): all datasets + desired reference."""
    label = AXES_LABELS[axis_idx]
    fig, ax = plt.subplots(figsize=(12, 4))

    desired_plotted = False
    for name, data, color, ls in datasets:
        n = data["actual_positions"].shape[0]
        t = make_time_axis(n)
        ax.plot(t, data["actual_positions"][:, axis_idx],
                color=color, linestyle=ls, linewidth=1.2, label=name, alpha=0.85)
        if not desired_plotted:
            ax.plot(t, data["desired_positions"][:, axis_idx],
                    color="black", linestyle=":", linewidth=1.2,
                    label="Desired", alpha=0.7)
            desired_plotted = True

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{label} Position (m)")
    ax.set_title(f"Position Tracking — {label} Axis  |  Friction Comparison")
    ax.legend(loc="best", fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = os.path.join(out_dir, f"position_tracking_{label.lower()}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {os.path.relpath(out)}")


# ── combined XYZ position plot ────────────────────────────────────────────────

def plot_position_xyz(datasets: list, out_dir: str) -> None:
    """3-row figure: one row per axis, all datasets overlaid, shared time axis."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Position Tracking (X, Y, Z)  |  Friction Comparison", fontsize=13)

    for axis_idx, ax in enumerate(axes):
        label = AXES_LABELS[axis_idx]
        desired_plotted = False
        for name, data, color, ls in datasets:
            n = data["actual_positions"].shape[0]
            t = make_time_axis(n)
            ax.plot(t, data["actual_positions"][:, axis_idx],
                    color=color, linestyle=ls, linewidth=1.2, label=name, alpha=0.85)
            if not desired_plotted:
                ax.plot(t, data["desired_positions"][:, axis_idx],
                        color="black", linestyle=":", linewidth=1.2,
                        label="Desired", alpha=0.7)
                desired_plotted = True

        ax.set_ylabel(f"{label} (m)")
        ax.grid(True, alpha=0.3)
        if axis_idx == 0:
            ax.legend(loc="upper right", fontsize=8.5, framealpha=0.85, ncol=3)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    out = os.path.join(out_dir, "position_tracking_xyz.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {os.path.relpath(out)}")


# ── force tracking plot ───────────────────────────────────────────────────────

def plot_force_tracking(datasets: list, out_dir: str) -> None:
    """Force Z error over time for all datasets, with per-dataset avg/max."""
    sk      = int(SKIP_S / DT)
    force_sk = int(FORCE_SKIP_S / DT)

    fig, ax = plt.subplots(figsize=(14, 5))

    for name, data, color, ls in datasets:
        fe = data["force_error"]
        n  = len(fe)
        t  = make_time_axis(n)
        avg = np.mean(np.abs(fe[sk:])) if n > sk else np.mean(np.abs(fe))
        mx  = np.max(np.abs(fe[force_sk:])) if n > force_sk else np.max(np.abs(fe))
        ax.plot(t, fe, color=color, linestyle=ls, linewidth=1.0, alpha=0.80,
                label=f"{name}  (avg|err|={avg:.3f} N, max|err|={mx:.3f} N)")

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force Z Error (N)")
    ax.set_title("Force Tracking — Force Z Error  |  Friction Comparison")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

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
            print(f"Usage: {sys.argv[0]} [friction_coeff ...]  (e.g. 0.3 0.5)")
            sys.exit(1)
    else:
        coeffs = FRICTION_COEFFS

    coeff_tag = "_".join(str(c) for c in coeffs)
    out_dir   = os.path.join(PLOTS_DIR, f"tracking_friction_{coeff_tag}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Friction coefficients: {coeffs}")
    datasets = build_datasets(coeffs)

    if not datasets:
        print("No data loaded. Exiting.")
        return

    print(f"\nGenerating plots -> {out_dir}\n")

    for axis_idx in range(3):
        plot_position_axis(datasets, axis_idx, out_dir)

    plot_position_xyz(datasets, out_dir)
    plot_force_tracking(datasets, out_dir)

    print(f"\nDone. All plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
