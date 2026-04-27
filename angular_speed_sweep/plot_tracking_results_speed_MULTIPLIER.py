"""Plot position (X, Y, Z) and force tracking for all 5 control methods at a given speed multiplier.

Experiments:
  1. Feedforward
  2. Feedforward + PI  (kp=2.0, ki=5.0)
  3. PD                (kp=5.0, kd=0.5)
  4. Paper
  5. Paper + PI        (kp=2.0, ki=5.0)

Outputs (saved to plots/tracking_speed_<multiplier>[_slope<angle>]/):
  position_tracking_xyz.png – Combined 3-row figure (X, Y, Z)
  force_tracking.png         – Force error over time for all 5 methods

Usage
-----
    python angular_speed_sweep/plot_tracking_results_speed_MULTIPLIER.py --multiplier 3.4
    python angular_speed_sweep/plot_tracking_results_speed_MULTIPLIER.py --multiplier 3.4 --slope-angle 30
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from tueplots import bundles

plt.rcParams.update(bundles.icml2024(usetex=False))
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR   = os.path.join(SCRIPT_DIR, "plots")

DT          = 0.001          # simulation timestep (s)
SKIP_S      = 1.0            # seconds to skip at the start for steady-state metrics

METHOD_KEYS = [
    ("Baseline",          "baseline",       "tab:gray",   "-",  "x"),
    ("Feedforward",       "feedforward",    "tab:blue",   "-",  "o"),
    ("Feedforward + PI",  "feedforward_pi", "tab:purple", "-",  "s"),
    ("PD",                "pd",             "tab:green",  "-",  "^"),
    ("HFDC",              "paper",          "tab:orange", "-",  "D"),
    ("HFDC + PI",         "paper_pi",       "tab:red",    "-",  "P"),
]

AXES_LABELS = ["X", "Y", "Z"]
PLOT_DURATION_S = 2.0  # only plot this many seconds of data


def build_methods(slope_angle):
    """Return (label, dir_name, color, ls, marker) for the given slope angle."""
    methods = []
    for label, key, color, ls, marker in METHOD_KEYS:
        dir_name = f"slope{slope_angle}_{key}" if slope_angle != 0.0 else key
        methods.append((label, dir_name, color, ls, marker))
    return methods


# ── helpers ───────────────────────────────────────────────────────────────────

def load_npz(method_dir: str, multiplier: float, slope_angle: float = 0.0) -> np.lib.npyio.NpzFile:
    path = os.path.join(PLOTS_DIR, method_dir, "data", f"data_{multiplier}.npz")
    if slope_angle != 0.0:
        path = os.path.join(PLOTS_DIR, method_dir, "data", f"data_{multiplier}_all.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    return np.load(path)


def make_time_axis(n_samples: int) -> np.ndarray:
    return np.arange(n_samples) * DT


# ── individual axis position plots ────────────────────────────────────────────

def plot_position_axis(datasets: list, axis_idx: int, multiplier: float,
                       surface_label: str, out_dir: str) -> None:
    """One figure per axis (X / Y / Z): all methods + desired reference."""
    label = AXES_LABELS[axis_idx]
    fig, ax = plt.subplots(figsize=(12, 4))

    desired_plotted = False
    for name, data, color, ls, _ in datasets:
        n = min(data["actual_positions"].shape[0], int(PLOT_DURATION_S / DT))
        t = make_time_axis(n)
        ax.plot(t, data["actual_positions"][:n, axis_idx],
                color=color, linestyle=ls, linewidth=1.2, label=name, alpha=0.85)
        if not desired_plotted:
            ax.plot(t, data["desired_positions"][:n, axis_idx],
                    color="black", linestyle="--", linewidth=1.2,
                    label="Desired", alpha=0.7)
            desired_plotted = True

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{label} Position (m)")
    ax.set_title(
        f"Position Tracking — {label} Axis  |  Speed Multiplier {multiplier}×  [{surface_label}]"
    )
    ax.legend(loc="best", fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = os.path.join(out_dir, f"position_tracking_{label.lower()}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {os.path.relpath(out)}")


# ── combined XYZ position plot ────────────────────────────────────────────────

def plot_position_xyz(datasets: list, multiplier: float,
                      surface_label: str, out_dir: str) -> None:
    """3-row figure: one row per axis, all methods overlaid, shared time axis."""
    # Print per-axis position error summary to terminal
    col_w = 20
    sk = int(SKIP_S / DT)
    for axis_idx, axis_label in enumerate(AXES_LABELS):
        print(f"\n  {axis_label} axis — avg / max |error| (m)")
        print(f"  {'Method':<{col_w}}  {'Avg':>10}  {'Max':>10}")
        print("  " + "-" * (col_w + 24))
        for name, data, color, ls, _ in datasets:
            pe = np.abs(data["actual_positions"][sk:, axis_idx]
                        - data["desired_positions"][sk:, axis_idx])
            print(f"  {name:<{col_w}}  {np.mean(pe):>10.4f}  {np.max(pe):>10.4f}")
    print()

    fig, axes = plt.subplots(
        3, 1, sharex=True, figsize=(3.25, 3.012),
        gridspec_kw={"height_ratios": [1, 1, 1]},
    )

    for axis_idx, ax in enumerate(axes):
        label = AXES_LABELS[axis_idx]
        desired_plotted = False
        for name, data, color, ls, _ in datasets:
            n = min(data["actual_positions"].shape[0], int(PLOT_DURATION_S / DT))
            t = make_time_axis(n)
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

    out = os.path.join(out_dir, "position_tracking_xyz.png")
    fig.savefig(out, dpi=600)
    plt.close(fig)
    print(f"[PLOT] {os.path.relpath(out)}")


# ── force tracking plot ───────────────────────────────────────────────────────

def plot_force_tracking(datasets: list, multiplier: float,
                        surface_label: str, out_dir: str) -> None:
    """Force error (Z) over time for all 5 methods, with per-method avg |error|."""
    fig, ax = plt.subplots()

    for name, data, color, ls, _ in datasets:
        fe = data["force_error"][:int(PLOT_DURATION_S / DT)]
        n  = len(fe)
        t  = make_time_axis(n)
        sk = int(SKIP_S / DT)
        avg = np.mean(np.abs(fe[sk:])) if n > sk else np.mean(np.abs(fe))
        max_err = np.max(np.abs(fe[sk:])) if n > sk else np.max(np.abs(fe))
        print(f"{name:20s} — avg |error| = {avg:.3f} N, max |error| = {max_err:.3f} N")

        ax.plot(t, fe, color=color, linestyle=ls, linewidth=1.0, alpha=0.80,
                label=f"{name}")

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.9, alpha=0.6)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force Z Error (N)")
    ax.legend(loc="lower right")

    out = os.path.join(out_dir, "force_tracking.png")
    fig.savefig(out, dpi=600)
    plt.close(fig)
    print(f"[PLOT] {os.path.relpath(out)}")


# ── per-method detail plot (position + force in one figure) ──────────────────

def plot_per_method_detail(datasets: list, multiplier: float,
                           surface_label: str, out_dir: str) -> None:
    """4-row figure per method: X, Y, Z position + force error on one page."""
    for name, data, color, ls, marker in datasets:
        n  = min(data["actual_positions"].shape[0], int(PLOT_DURATION_S / DT))
        t  = make_time_axis(n)
        sk = int(SKIP_S / DT)

        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        fig.suptitle(
            f"{name}  |  Speed Multiplier {multiplier}×  [{surface_label}]", fontsize=13
        )

        # X, Y, Z position rows
        for axis_idx in range(3):
            ax = axes[axis_idx]
            label = AXES_LABELS[axis_idx]
            ax.plot(t, data["actual_positions"][:n, axis_idx],
                    color=color, linewidth=1.2, label="Actual")
            ax.plot(t, data["desired_positions"][:n, axis_idx],
                    color="black", linestyle="--", linewidth=1.2, label="Desired")
            ax.set_ylabel(f"{label} (m)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)

        # Force error row
        fe  = data["force_error"][:n]
        avg = np.mean(np.abs(fe[sk:])) if n > sk else np.mean(np.abs(fe))
        axes[3].plot(t, fe, color=color, linewidth=1.0,
                     label=f"Force Z error  (avg|err|={avg:.3f} N)")
        axes[3].axhline(0, color="gray", linestyle="--", linewidth=0.9)
        axes[3].axvline(SKIP_S, color="gray", linestyle=":", linewidth=0.9, alpha=0.7)
        axes[3].set_ylabel("Force Z Error (N)")
        axes[3].grid(True, alpha=0.3)
        axes[3].legend(loc="upper right", fontsize=8)

        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()

        safe_name = name.lower().replace(" + ", "_").replace(" ", "_")
        out = os.path.join(out_dir, f"detail_{safe_name}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] {os.path.relpath(out)}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot position and force tracking for all control methods."
    )
    parser.add_argument(
        "--multiplier",
        type=float,
        default=3.4,
        help="Angular speed multiplier (omega/pi) to plot (default: 3.4)",
    )
    parser.add_argument(
        "--slope-angle",
        type=float,
        default=0.0,
        help="Slope angle in degrees (default: 0 = flat surface). "
             "Reads from slope<angle>_<method>/data directories.",
    )
    args = parser.parse_args()

    multiplier  = args.multiplier
    slope_angle = args.slope_angle
    surface_label = f"{slope_angle:g}° slope" if slope_angle != 0.0 else "flat surface"

    slope_suffix = f"_slope{slope_angle:g}" if slope_angle != 0.0 else ""
    out_dir = os.path.join(PLOTS_DIR, f"tracking_speed_{multiplier}{slope_suffix}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Multiplier: {multiplier}×  |  Surface: {surface_label}")

    methods = build_methods(slope_angle)

    # Load data for every method
    datasets = []
    for name, method_dir, color, ls, marker in methods:
        try:
            d = load_npz(method_dir, multiplier, slope_angle)
            datasets.append((name, d, color, ls, marker))
            ee_spd = float(d["ee_linear_speed_m_s"]) if "ee_linear_speed_m_s" in d \
                     else float(d["angular_speed_rad_s"]) * 0.1
            print(f"[LOAD] {name:20s} — {d['actual_positions'].shape[0]} timesteps, "
                  f"angular_speed={float(d['angular_speed_rad_s']):.2f} rad/s, "
                  f"ee_speed={ee_spd:.3f} m/s")
        except FileNotFoundError as e:
            print(f"[SKIP] {e}")

    if not datasets:
        print("No data loaded. Exiting.")
        return

    print(f"\nGenerating plots → {out_dir}\n")

    plot_position_xyz(datasets, multiplier, surface_label, out_dir)
    plot_force_tracking(datasets, multiplier, surface_label, out_dir)

    print(f"\nDone. All plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
