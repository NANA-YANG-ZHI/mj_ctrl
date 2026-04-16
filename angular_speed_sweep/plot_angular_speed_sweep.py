"""Plot EE linear speed vs force/position error metrics from sweep CSV.

X-axis: EE linear speed  v = r × ω  (m/s,  r = 0.1 m)
Four subplots: avg force error | var force error | avg position error | var position error
"""
import sys
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
    # "legend.fontsize": 14,
})


def _parse_col(val: str) -> float:
    return float(val) if val.strip() != "nan" else float("nan")


def main():
    if len(sys.argv) < 3:
        print("Usage: python plot_angular_speed_sweep.py <results_csv> <output_dir>")
        sys.exit(1)

    csv_path = sys.argv[1]
    output_dir = sys.argv[2]
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        sys.exit(1)

    ee_linear_speeds = []
    avg_force_errors = []
    var_force_errors = []
    avg_pos_errors = []
    var_pos_errors = []

    with open(csv_path) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # skip header
            parts = line.strip().split(",")
            # columns: multiplier, angular_speed, ee_linear_speed,
            #          avg_force_z_error, var_force_z_error,
            #          avg_position_error, var_position_error
            if len(parts) < 5:
                continue
            try:
                ee_linear_speeds.append(float(parts[2]))
                avg_force_errors.append(_parse_col(parts[3]))
                var_force_errors.append(_parse_col(parts[4]) if len(parts) > 4 else float("nan"))
                avg_pos_errors.append(_parse_col(parts[5]) if len(parts) > 5 else float("nan"))
                var_pos_errors.append(_parse_col(parts[6]) if len(parts) > 6 else float("nan"))
            except (ValueError, IndexError):
                continue

    v = np.array(ee_linear_speeds)
    avg_fe = np.array(avg_force_errors)
    var_fe = np.array(var_force_errors)
    avg_pe = np.array(avg_pos_errors)
    var_pe = np.array(var_pos_errors)

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    fig.suptitle("EE Linear Speed Sweep — Controller Performance", fontsize=14)
    xlabel = "EE Linear Speed  v = r·ω  (m/s,  r = 0.1 m)"

    specs = [
        (avg_fe, "Avg |Force Z Error| (N)",  "tab:blue"),
        (var_fe, "Var Force Z Error (N²)",    "tab:cyan"),
        (avg_pe, "Avg Position Error (m)",    "tab:orange"),
        (var_pe, "Var Position Error (m²)",   "tab:red"),
    ]
    markers = ["o", "o", "s", "s"]

    for ax, (data, ylabel, color), marker in zip(axes, specs, markers):
        valid = ~np.isnan(data)
        ax.plot(v[valid], data[valid], f"{marker}-", linewidth=1.5, markersize=5, color=color)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(xlabel)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "angular_speed_sweep.png")
    fig.savefig(out_path, dpi=150)
    print(f"[PLOT] Sweep plot saved to {out_path}")


if __name__ == "__main__":
    main()
