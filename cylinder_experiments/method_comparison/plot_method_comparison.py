"""
Method comparison plots for the cylinder surface experiment.

For each speed, produces one figure with two subplots (force error, position
error).  Each subplot is a grouped bar chart: robot groups on the x-axis,
one bar per force-control method.  Error bars show ± std.

Usage:
    python plot_method_comparison.py <results.csv> <output_dir>
"""
import os
import sys
from collections import defaultdict

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

METHOD_ORDER  = ["ff", "ff_pi", "pd", "paper", "paper_pi"]
METHOD_LABELS = {
    "ff":       "Feedforward",
    "ff_pi":    "Feedforward + PI",
    "pd":       "PD",
    "paper":    "Paper",
    "paper_pi": "Paper + PI",
}
METHOD_COLORS = {
    "ff":       "tab:blue",
    "ff_pi":    "tab:cyan",
    "pd":       "tab:orange",
    "paper":    "tab:green",
    "paper_pi": "tab:red",
}

ROBOT_ORDER  = ["fr3", "fr3_friction", "fr3_jointf", "fr3_jointf_surff"]
ROBOT_LABELS = {
    "fr3":              "FR3",
    "fr3_friction":     "FR3\n+surf. fric.",
    "fr3_jointf":       "FR3\n+joint fric.",
    "fr3_jointf_surff": "FR3\n+both fric.",
}


def load_csv(path):
    # (speed, robot, method) → (mean_force, std_force, mean_pos, std_pos)
    data = {}
    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            try:
                robot  = parts[0].strip()
                method = parts[1].strip()
                omega  = float(parts[2])
                data[(omega, robot, method)] = (
                    float(parts[4]),  # mean_force
                    float(parts[5]),  # std_force
                    float(parts[6]),  # mean_pos
                    float(parts[7]),  # std_pos
                )
            except ValueError:
                continue
    return data


def plot_speed(speed, data, robots, methods, output_dir):
    """One figure per speed: grouped bar chart comparing methods across robots."""
    n_robots  = len(robots)
    n_methods = len(methods)
    bar_w     = 0.8 / n_methods
    x         = np.arange(n_robots)

    fig, (ax_f, ax_p) = plt.subplots(2, 1, figsize=(max(8, n_robots * 2), 6), sharex=True)
    ee_speed = speed * 0.1
    fig.suptitle(f"Method Comparison  —  ω = {speed} rad/s  (v = {ee_speed:.2f} m/s)",
                 fontweight="bold")

    for mi, method in enumerate(methods):
        offset = (mi - (n_methods - 1) / 2) * bar_w
        mf_vals = [data.get((speed, r, method), (np.nan,))[0] for r in robots]
        sf_vals = [data.get((speed, r, method), (np.nan, np.nan))[1] for r in robots]
        mp_vals = [data.get((speed, r, method), (np.nan, np.nan, np.nan))[2] for r in robots]
        sp_vals = [data.get((speed, r, method), (np.nan, np.nan, np.nan, np.nan))[3] for r in robots]

        kw = dict(width=bar_w, color=METHOD_COLORS[method],
                  label=METHOD_LABELS[method], capsize=3, error_kw={"linewidth": 0.8})

        ax_f.bar(x + offset, mf_vals, yerr=sf_vals, **kw)
        ax_p.bar(x + offset, mp_vals, yerr=sp_vals, **kw)

    robot_tick_labels = [ROBOT_LABELS.get(r, r) for r in robots]
    ax_f.set_ylabel("Mean |Normal Force Error| ± std  (N)")
    ax_f.set_xticks(x)
    ax_f.set_xticklabels(robot_tick_labels)
    ax_f.grid(True, axis="y", alpha=0.3)
    ax_f.legend(loc="upper right", ncol=n_methods)

    ax_p.set_ylabel("Mean Position Error ± std  (m)")
    ax_p.set_xticks(x)
    ax_p.set_xticklabels(robot_tick_labels)
    ax_p.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"speed_{str(speed).replace('.', 'p')}_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[PLOT] → {out}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python plot_method_comparison.py <results.csv> <output_dir>")
        sys.exit(1)

    csv_path   = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        sys.exit(1)

    data = load_csv(csv_path)

    speeds  = sorted({k[0] for k in data})
    robots  = [r for r in ROBOT_ORDER  if any(k[1] == r for k in data)] + \
              [r for r in {k[1] for k in data} if r not in ROBOT_ORDER]
    methods = [m for m in METHOD_ORDER  if any(k[2] == m for k in data)] + \
              [m for m in {k[2] for k in data} if m not in METHOD_ORDER]

    for speed in speeds:
        plot_speed(speed, data, robots, methods, output_dir)


if __name__ == "__main__":
    main()
