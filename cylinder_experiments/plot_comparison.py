"""
Comparison plots for cylinder surface experiments.

For each (robot, angular_speed) pair, produces two figures:
  - force_error.png    : force error over time for all methods
  - position_error.png : position error (mm) over time for all methods

Run from the workspace root:
    python cylinder_experiments/plot_comparison.py
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Grid definition (must match run_all.sh) ───────────────────────────────────
ROBOTS  = ["fr3", "kuka"]
SPEEDS  = [0.314, 0.628, 0.8, 1.0, 1.2]
METHODS = ["ff", "ff_pi", "pd", "paper", "paper_pi"]

METHOD_LABELS = {
    "ff":       "FF",
    "ff_pi":    "FF + PI",
    "pd":       "PD",
    "paper":    "Paper",
    "paper_pi": "Paper + PI",
}
METHOD_COLORS = {
    "ff":       "steelblue",
    "ff_pi":    "deepskyblue",
    "pd":       "seagreen",
    "paper":    "darkorange",
    "paper_pi": "crimson",
}
METHOD_LS = {
    "ff":       "-",
    "ff_pi":    "--",
    "pd":       "-",
    "paper":    "-",
    "paper_pi": "--",
}

DATA_BASE = "cylinder_experiments/data"
PLOT_OUT  = "cylinder_experiments/plots_comparison"


def load(data_dir: str) -> dict | None:
    path = os.path.join(data_dir, "data.npz")
    if not os.path.isfile(path):
        print(f"  [WARN] missing {path}")
        return None
    return dict(np.load(path))


def make_force_figure(robot: str, speed_str: str, datasets: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        f"Force Error — {robot.upper()}  |  ω = {speed_str} rad/s",
        fontsize=13, fontweight="bold",
    )

    for method in METHODS:
        d = datasets.get(method)
        if d is None:
            continue
        t         = d["t"]
        force_err = d["force_err"]
        mean_abs  = np.mean(np.abs(force_err))
        label = f"{METHOD_LABELS[method]}  (mean |e| = {mean_abs:.3f} N)"
        ax.plot(t, force_err,
                color=METHOD_COLORS[method],
                ls=METHOD_LS[method],
                lw=1.6, label=label)

    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force error (N)")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.25)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    return fig


def make_position_figure(robot: str, speed_str: str, datasets: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        f"Position Error — {robot.upper()}  |  ω = {speed_str} rad/s",
        fontsize=13, fontweight="bold",
    )

    for method in METHODS:
        d = datasets.get(method)
        if d is None:
            continue
        t       = d["t"]
        pos_err = d["pos_err"] * 1e3  # m → mm
        mean_v  = np.mean(pos_err)
        label = f"{METHOD_LABELS[method]}  (mean = {mean_v:.3f} mm)"
        ax.plot(t, pos_err,
                color=METHOD_COLORS[method],
                ls=METHOD_LS[method],
                lw=1.6, label=label)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position error (mm)")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.25)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    return fig


def main() -> None:
    os.makedirs(PLOT_OUT, exist_ok=True)
    missing_any = False

    for robot in ROBOTS:
        for speed in SPEEDS:
            speed_str = f"{speed:.3f}"
            print(f"\n[{robot}  ω={speed_str}]")

            datasets: dict[str, dict | None] = {}
            for method in METHODS:
                tag      = f"{robot}_{method}_{speed_str}"
                data_dir = os.path.join(DATA_BASE, tag)
                datasets[method] = load(data_dir)
                if datasets[method] is None:
                    missing_any = True

            if all(v is None for v in datasets.values()):
                print("  No data — skipping.")
                continue

            out_dir = os.path.join(PLOT_OUT, robot, speed_str)
            os.makedirs(out_dir, exist_ok=True)

            fig_f = make_force_figure(robot, speed_str, datasets)
            path_f = os.path.join(out_dir, "force_error.png")
            fig_f.savefig(path_f, dpi=150)
            plt.close(fig_f)
            print(f"  [SAVED] {path_f}")

            fig_p = make_position_figure(robot, speed_str, datasets)
            path_p = os.path.join(out_dir, "position_error.png")
            fig_p.savefig(path_p, dpi=150)
            plt.close(fig_p)
            print(f"  [SAVED] {path_p}")

    if missing_any:
        print("\n[WARN] Some data files were missing — run run_all.sh first.")


if __name__ == "__main__":
    main()
