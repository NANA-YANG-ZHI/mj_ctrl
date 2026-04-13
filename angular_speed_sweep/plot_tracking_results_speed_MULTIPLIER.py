"""Plot position (X, Y, Z) and force tracking for all 5 control methods at speed multiplier 3.6.

Experiments:
  1. Feedforward
  2. Feedforward + PI  (kp=2.0, ki=5.0)
  3. PD                (kp=5.0, kd=0.5)
  4. Paper
  5. Paper + PI        (kp=2.0, ki=5.0)

Outputs (saved to plots/tracking_speed_3.6/):
  position_tracking_x.png   – X-axis: actual vs desired for all 5 methods
  position_tracking_y.png   – Y-axis: actual vs desired for all 5 methods
  position_tracking_z.png   – Z-axis: actual vs desired for all 5 methods
  position_tracking_xyz.png – Combined 3-row figure (X, Y, Z)
  force_tracking.png         – Force error over time for all 5 methods

Usage
-----
    python angular_speed_sweep/plot_tracking_results_speed_3_6.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tueplots import bundles

plt.rcParams.update(bundles.neurips2021(usetex=False))

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR   = os.path.join(SCRIPT_DIR, "plots")

MULTIPLIER  = 3.4
DT          = 0.001          # simulation timestep (s)
SKIP_S      = 0.0            # seconds to skip at the start for steady-state metrics
OUT_DIR     = os.path.join(PLOTS_DIR, f"tracking_speed_{MULTIPLIER}")

METHODS = [
    ("Feedforward",       "feedforward",    "tab:blue",   "-",  "o"),
    ("Feedforward + PI",  "feedforward_pi", "tab:purple", "-",  "s"),
    ("PD",                "pd",             "tab:green",  "-",  "^"),
    ("HFPD",             "paper",          "tab:orange",    "-",  "D"),
    ("HFPD + PI",        "paper_pi",       "tab:red", "-",  "P"),
]

AXES_LABELS = ["X", "Y", "Z"]
PLOT_DURATION_S = 2.0  # only plot this many seconds of data


# ── helpers ───────────────────────────────────────────────────────────────────

def load_npz(method_dir: str, multiplier: float) -> np.lib.npyio.NpzFile:
    path = os.path.join(PLOTS_DIR, method_dir, "data", f"data_{multiplier}.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    return np.load(path)


def make_time_axis(n_samples: int) -> np.ndarray:
    return np.arange(n_samples) * DT


def skip_samples(n_samples: int) -> int:
    return int(SKIP_S / DT)


# ── individual axis position plots ────────────────────────────────────────────

def plot_position_axis(datasets: list, axis_idx: int) -> None:
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
        f"Position Tracking — {label} Axis  |  Speed Multiplier {MULTIPLIER}×"
    )
    ax.legend(loc="best", fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = os.path.join(OUT_DIR, f"position_tracking_{label.lower()}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {os.path.relpath(out)}")


# ── combined XYZ position plot ────────────────────────────────────────────────

def plot_position_xyz(datasets: list) -> None:
    """3-row figure: one row per axis, all methods overlaid, shared time axis."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"Position Tracking (X, Y, Z)  |  Speed Multiplier {MULTIPLIER}×",
        fontsize=13,
    )

    for axis_idx, ax in enumerate(axes):
        label = AXES_LABELS[axis_idx]
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

        ax.set_ylabel(f"{label} (m)")
        ax.grid(True, alpha=0.3)
        if axis_idx == 0:
            ax.legend(loc="upper right", fontsize=8.5, framealpha=0.85, ncol=3)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    out = os.path.join(OUT_DIR, "position_tracking_xyz.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {os.path.relpath(out)}")


# ── force tracking plot ───────────────────────────────────────────────────────

def plot_force_tracking(datasets: list) -> None:
    """Force error (Z) over time for all 5 methods, with per-method avg |error|."""
    fig, ax = plt.subplots()

    for name, data, color, ls, _ in datasets:
        fe = data["force_error"][:int(PLOT_DURATION_S / DT)]
        n  = len(fe)
        t  = make_time_axis(n)
        sk = int(SKIP_S / DT)
        avg = np.mean(np.abs(fe[sk:])) if n > sk else np.mean(np.abs(fe))
        sk = int(1.0 / DT)
        max = np.max(np.abs(fe[sk:])) if n > sk else np.max(np.abs(fe))

        ax.plot(t, fe, color=color, linestyle=ls, linewidth=1.0, alpha=0.80,
                label=f"{name}  (avg|err|={avg:.3f} N) (max|err|={max:.3f} N)")

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.9, alpha=0.6)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force Z Error (N)")
    ax.set_title(
        f"Force Tracking — Force Z Error  |  Speed Multiplier {MULTIPLIER}×"
    )
    ax.legend(loc="upper right")

    out = os.path.join(OUT_DIR, "force_tracking.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {os.path.relpath(out)}")


# ── per-method detail plot (position + force in one figure) ──────────────────

def plot_per_method_detail(datasets: list) -> None:
    """4-row figure per method: X, Y, Z position + force error on one page."""
    for name, data, color, ls, marker in datasets:
        n  = min(data["actual_positions"].shape[0], int(PLOT_DURATION_S / DT))
        t  = make_time_axis(n)
        sk = int(SKIP_S / DT)

        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        fig.suptitle(
            f"{name}  |  Speed Multiplier {MULTIPLIER}×", fontsize=13
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
        out = os.path.join(OUT_DIR, f"detail_{safe_name}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] {os.path.relpath(out)}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load data for every method
    datasets = []
    for name, method_dir, color, ls, marker in METHODS:
        try:
            d = load_npz(method_dir, MULTIPLIER)
            datasets.append((name, d, color, ls, marker))
            print(f"[LOAD] {name:20s} — {d['actual_positions'].shape[0]} timesteps, "
                  f"angular_speed={float(d['angular_speed_rad_s']):.2f} rad/s, "
                  f"ee_speed={float(d['ee_linear_speed_m_s']):.3f} m/s")
        except FileNotFoundError as e:
            print(f"[SKIP] {e}")

    if not datasets:
        print("No data loaded. Exiting.")
        return

    print(f"\nGenerating plots → {OUT_DIR}\n")

    # Per-axis position comparison
    for axis_idx in range(3):
        plot_position_axis(datasets, axis_idx)

    # Combined XYZ position comparison
    plot_position_xyz(datasets)

    # Force tracking comparison
    plot_force_tracking(datasets)

    # Per-method detail (position + force)
    plot_per_method_detail(datasets)

    print(f"\nDone. All plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
