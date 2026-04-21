"""
Plot cylinder angular-speed sweep: mean ± std per robot condition.

Reads the CSV produced by collect_cylinder_results.py and plots two subplots:
  - Mean |normal force error| ± std  vs  EE linear speed
  - Mean position error       ± std  vs  EE linear speed

One shaded line per robot.

Usage:
    python plot_cylinder_speed_sweep.py <sweep_results.csv> <output_dir>
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

ROBOT_LABELS = {
    "fr3":              "FR3",
    "fr3_friction":     "FR3 + surface friction",
    "fr3_jointf":       "FR3 + joint friction",
    "fr3_jointf_surff": "FR3 + joint + surface friction",
}
ROBOT_ORDER = list(ROBOT_LABELS)
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]


def load_csv(path):
    # robot → list of (omega, ee_speed, mean_force, std_force, mean_pos, std_pos)
    data = defaultdict(list)
    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            try:
                robot    = parts[0].strip()
                omega    = float(parts[1])
                ee_speed = float(parts[2])
                mf       = float(parts[3])
                sf       = float(parts[4])
                mp       = float(parts[5])
                sp       = float(parts[6])
                data[robot].append((ee_speed, mf, sf, mp, sp))
            except ValueError:
                continue
    # Sort each robot's rows by ee_speed
    for robot in data:
        data[robot].sort(key=lambda x: x[0])
    return data


def main():
    if len(sys.argv) < 3:
        print("Usage: python plot_cylinder_speed_sweep.py <sweep_results.csv> <output_dir>")
        sys.exit(1)

    csv_path   = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        sys.exit(1)

    raw = load_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    ordered = [r for r in ROBOT_ORDER if r in raw] + [r for r in raw if r not in ROBOT_ORDER]

    fig, (ax_force, ax_pos) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    fig.suptitle("Cylinder Surface: Angular Speed Sweep", fontweight="bold")

    for robot, color in zip(ordered, COLORS):
        rows     = raw[robot]
        v        = np.array([r[0] for r in rows])
        mf       = np.array([r[1] for r in rows])
        sf       = np.array([r[2] for r in rows])
        mp       = np.array([r[3] for r in rows])
        sp       = np.array([r[4] for r in rows])
        label    = ROBOT_LABELS.get(robot, robot)

        for ax, mean, std in [(ax_force, mf, sf), (ax_pos, mp, sp)]:
            ax.plot(v, mean, "o-", lw=1.5, ms=4, color=color, label=label)
            ax.fill_between(v, mean - std, mean + std, alpha=0.18, color=color)

    ax_force.set_ylabel("Mean |Normal Force Error| ± std  (N)")
    ax_force.grid(True, alpha=0.3)
    ax_force.legend(loc="upper left")

    ax_pos.set_ylabel("Mean Position Error ± std  (m)")
    ax_pos.set_xlabel("EE Linear Speed  v = 0.1 · ω  (m/s)")
    ax_pos.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "cylinder_speed_sweep.png")
    fig.savefig(out_path, dpi=150)
    print(f"[PLOT] → {out_path}")


if __name__ == "__main__":
    main()
