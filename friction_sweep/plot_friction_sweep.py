"""Plot surface friction coefficient vs force/position error metrics from sweep CSV.

X-axis: sliding friction coefficient μ
Four subplots: avg |force Z error| | var force Z error | avg position error | var position error
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use('science')


def _parse_col(val: str) -> float:
    return float(val) if val.strip() != "nan" else float("nan")


def main():
    if len(sys.argv) < 3:
        print("Usage: python plot_friction_sweep.py <results_csv> <output_dir>")
        sys.exit(1)

    csv_path = sys.argv[1]
    output_dir = sys.argv[2]
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        sys.exit(1)

    friction_coeffs = []
    avg_force_errors = []
    var_force_errors = []
    avg_pos_errors = []
    var_pos_errors = []

    with open(csv_path) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # skip header
            parts = line.strip().split(",")
            # columns: friction_coeff, avg_force_z_error, var_force_z_error,
            #          avg_position_error, var_position_error
            if len(parts) < 2:
                continue
            try:
                friction_coeffs.append(float(parts[0]))
                avg_force_errors.append(_parse_col(parts[1]))
                var_force_errors.append(_parse_col(parts[2]) if len(parts) > 2 else float("nan"))
                avg_pos_errors.append(_parse_col(parts[3]) if len(parts) > 3 else float("nan"))
                var_pos_errors.append(_parse_col(parts[4]) if len(parts) > 4 else float("nan"))
            except (ValueError, IndexError):
                continue

    mu = np.array(friction_coeffs)
    avg_fe = np.array(avg_force_errors)
    var_fe = np.array(var_force_errors)
    avg_pe = np.array(avg_pos_errors)
    var_pe = np.array(var_pos_errors)

    # Sort by friction coefficient in case parallel workers finished out of order
    order = np.argsort(mu)
    mu, avg_fe, var_fe, avg_pe, var_pe = (
        mu[order], avg_fe[order], var_fe[order], avg_pe[order], var_pe[order]
    )

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
    fig.suptitle("Surface Friction Sweep — HFPD Controller Performance", fontsize=14)

    specs = [
        (avg_fe, "Avg |Force Z Error| (N)",  "tab:blue"),
        (var_fe, "Var Force Z Error (N²)",    "tab:cyan"),
        (avg_pe, "Avg Position Error (m)",    "tab:orange"),
        (var_pe, "Var Position Error (m²)",   "tab:red"),
    ]
    markers = ["o", "o", "s", "s"]

    for ax, (data, ylabel, color), marker in zip(axes, specs, markers):
        valid = ~np.isnan(data)
        ax.plot(mu[valid], data[valid], f"{marker}-", linewidth=1.5, markersize=6, color=color)
        ax.set_ylabel(ylabel)
        ax.set_xticks(np.round(np.arange(0.1, 1.05, 0.1), 1))
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Sliding Friction Coefficient μ")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "friction_sweep.png")
    fig.savefig(out_path, dpi=150)
    print(f"[PLOT] Sweep plot saved to {out_path}")


if __name__ == "__main__":
    main()
