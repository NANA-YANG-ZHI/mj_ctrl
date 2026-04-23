"""
Plot force error and position trajectories for each omega.

Produces two separate figures per omega:
  1. force_omega_X.X.png  — |Force Error| vs time, 4 conditions
  2. traj_omega_X.X.png   — Y-Z plane trajectories (actual + desired), 4 conditions

Usage:
    python plot_timeseries_speed_sweep.py
        [--data-root sweep_speed_results/paper/data]
        [--output-dir sweep_speed_results/paper/timeseries]
        [--omegas 0.3 0.6 0.8 1.2 1.6]
        [--dt 0.001]
        [--skip-seconds 0.0]
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
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

ROBOT_ORDER = ["fr3", "fr3_friction", "fr3_jointf", "fr3_jointf_surff"]
ROBOT_LABELS = {
    "fr3":              "FR3",
    "fr3_friction":     "FR3 + surface friction",
    "fr3_jointf":       "FR3 + joint friction",
    "fr3_jointf_surff": "FR3 + joint + surface friction",
}
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]


def load_run(data_root, robot, omega, skip_n):
    fname = f"data_{omega:.1f}_all.npz"
    fpath = os.path.join(data_root, robot, fname)
    if not os.path.exists(fpath):
        return None
    d = np.load(fpath)
    return {
        "force_err":       np.abs(d["force_error"][skip_n:]),
        "actual_pos":      d["actual_positions"][skip_n:],    # (N, 3)
        "desired_pos":     d["desired_positions"][skip_n:],   # (N, 3)
    }


def plot_force(omega, runs, output_dir, dt, skip_n):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    fig.suptitle(f"Force Error — ω = {omega:.1f} rad/s  (v = {0.1*omega:.2f} m/s)",
                 fontweight="bold")

    for robot, color in zip(ROBOT_ORDER, COLORS):
        if robot not in runs:
            continue
        fe = runs[robot]["force_err"]
        t  = np.arange(len(fe)) * dt + skip_n * dt
        ax.plot(t, fe, lw=0.8, color=color, label=ROBOT_LABELS[robot])

    ax.set_ylabel("|Force Error| (N)")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=7)

    plt.tight_layout()
    out = os.path.join(output_dir, f"force_omega_{omega:.1f}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[FORCE] → {out}")


def plot_trajectory(omega, runs, output_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle(f"Y-Z Trajectory — ω = {omega:.1f} rad/s  (v = {0.1*omega:.2f} m/s)",
                 fontweight="bold")

    desired_plotted = False
    for robot, color in zip(ROBOT_ORDER, COLORS):
        if robot not in runs:
            continue
        actual  = runs[robot]["actual_pos"]   # (N, 3): col0=x, col1=y, col2=z
        desired = runs[robot]["desired_pos"]

        # Plot desired once (same for all conditions)
        if not desired_plotted:
            ax.plot(desired[:, 1], desired[:, 2],
                    lw=1.5, ls="--", color="black", label="Desired", zorder=5)
            desired_plotted = True

        ax.plot(actual[:, 1], actual[:, 2],
                lw=0.8, color=color, label=ROBOT_LABELS[robot])

    ax.set_xlabel("Y (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7)

    plt.tight_layout()
    out = os.path.join(output_dir, f"traj_omega_{omega:.1f}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[TRAJ]  → {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",
                        default="sweep_speed_results/paper/data")
    parser.add_argument("--output-dir",
                        default="sweep_speed_results/paper/timeseries")
    parser.add_argument("--omegas", type=float, nargs="+",
                        default=[0.3, 0.6, 0.8, 1.2, 1.6])
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--skip-seconds", type=float, default=0.0)
    args = parser.parse_args()

    skip_n = int(args.skip_seconds / args.dt)
    os.makedirs(args.output_dir, exist_ok=True)

    for omega in args.omegas:
        runs = {}
        for robot in ROBOT_ORDER:
            result = load_run(args.data_root, robot, omega, skip_n)
            if result is None:
                print(f"[SKIP] {robot} omega={omega:.1f} not found")
            else:
                runs[robot] = result

        if not runs:
            print(f"[SKIP] No data found for omega={omega:.1f}")
            continue

        plot_force(omega, runs, args.output_dir, args.dt, skip_n)
        plot_trajectory(omega, runs, args.output_dir)


if __name__ == "__main__":
    main()
