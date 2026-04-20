"""Plot force error vs time and actual EE velocity vs time for all 5 methods.

Two-row figure:
  Row 1: Force Z error (N) vs time
  Row 2: Actual EE velocity magnitude (m/s) vs time

Usage
-----
    python angular_speed_sweep/plot_force_and_velocity.py --multiplier 3.4
    python angular_speed_sweep/plot_force_and_velocity.py --multiplier 3.4 --slope-angle 30
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR  = os.path.join(SCRIPT_DIR, "plots")

DT             = 0.001   # simulation timestep (s)
PLOT_DURATION_S = 2.0    # seconds of data to show

METHOD_KEYS = [
    ("Feedforward",      "feedforward",    "tab:blue",   "-"),
    ("Feedforward + PI", "feedforward_pi", "tab:purple", "-"),
    ("PD",               "pd",             "tab:green",  "-"),
    ("HFDC",             "paper",          "tab:orange", "-"),
    ("HFDC + PI",        "paper_pi",       "tab:red",    "-"),
]


def build_methods(slope_angle):
    methods = []
    for label, key, color, ls in METHOD_KEYS:
        dir_name = f"slope{slope_angle}_{key}" if slope_angle != 0.0 else key
        methods.append((label, dir_name, color, ls))
    return methods


def load_npz(method_dir, multiplier, slope_angle=0.0):
    if slope_angle != 0.0:
        fname = f"data_{multiplier}_all.npz"
    else:
        fname = f"data_{multiplier}.npz"
    path = os.path.join(PLOTS_DIR, method_dir, "data", fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return np.load(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiplier",   type=float, default=3.4)
    parser.add_argument("--slope-angle",  type=float, default=0.0)
    args = parser.parse_args()

    multiplier   = args.multiplier
    slope_angle  = args.slope_angle
    surface_label = f"{slope_angle:g}° slope" if slope_angle != 0.0 else "flat surface"

    slope_suffix = f"_slope{slope_angle:g}" if slope_angle != 0.0 else ""
    out_dir = os.path.join(PLOTS_DIR, f"tracking_speed_{multiplier}{slope_suffix}")
    os.makedirs(out_dir, exist_ok=True)

    methods = build_methods(slope_angle)

    datasets = []
    for label, dir_name, color, ls in methods:
        try:
            d = load_npz(dir_name, multiplier, slope_angle)
            datasets.append((label, d, color, ls))
            print(f"[LOAD] {label:20s} — {d['actual_positions'].shape[0]} timesteps")
        except FileNotFoundError as e:
            print(f"[SKIP] {e}")

    if not datasets:
        print("No data loaded. Exiting.")
        return

    n_plot = int(PLOT_DURATION_S / DT)
    t = np.arange(n_plot) * DT

    fig, (ax_force, ax_vel) = plt.subplots(
        2, 1, sharex=True, figsize=(3.25, 4.0),
    )

    for label, data, color, ls in datasets:
        # ── Force error ──────────────────────────────────────────────
        fe = data["force_error"][:n_plot]
        ax_force.plot(t[:len(fe)], fe, color=color, linestyle=ls,
                      linewidth=0.8, label=label, alpha=0.85)

        # ── Actual EE velocity magnitude ─────────────────────────────
        vel = data["actual_velocitys"][:n_plot]   # (N, 3)
        vel_mag = np.linalg.norm(vel, axis=1)
        ax_vel.plot(t[:len(vel_mag)], vel_mag, color=color, linestyle=ls,
                    linewidth=0.8, label=label, alpha=0.85)

    ax_force.axhline(0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
    ax_force.set_ylabel("Force Z Error (N)")
    ax_force.legend(loc="upper right", fontsize=7, framealpha=0.85)

    ax_vel.set_ylabel("EE Velocity (m/s)")
    ax_vel.set_xlabel("Time (s)")

    fig.suptitle(
        f"Speed {multiplier}×  [{surface_label}]", fontsize=9
    )

    out = os.path.join(out_dir, f"force_and_velocity_{multiplier}.png")
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
