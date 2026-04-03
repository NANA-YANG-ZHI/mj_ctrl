"""Plot EE linear speed vs avg_force_z_error and avg_position_error from sweep CSV.

X-axis: EE linear speed  v = r × ω  (m/s,  r = 0.1 m)
Two subplots: force Z error (top) | position error (bottom)
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt


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
    force_errors = []
    pos_errors = []

    with open(csv_path) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # skip header
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            try:
                # parts[0]=multiplier, parts[1]=angular_speed, parts[2]=ee_linear_speed
                ee_v = float(parts[2])
                f_err = float(parts[3]) if parts[3].strip() != "nan" else float("nan")
                p_err = float(parts[4]) if len(parts) > 4 and parts[4].strip() != "nan" else float("nan")
                ee_linear_speeds.append(ee_v)
                force_errors.append(f_err)
                pos_errors.append(p_err)
            except (ValueError, IndexError):
                continue

    ee_linear_speeds = np.array(ee_linear_speeds)
    force_errors = np.array(force_errors)
    pos_errors = np.array(pos_errors)

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("EE Linear Speed Sweep — Controller Performance", fontsize=14)

    valid_f = ~np.isnan(force_errors)
    valid_p = ~np.isnan(pos_errors)

    # --- Force error subplot ---
    axes[0].plot(ee_linear_speeds[valid_f], force_errors[valid_f], "o-", linewidth=1.5, markersize=5, color="tab:blue")
    axes[0].set_ylabel("Avg |Force Z Error| (N)")
    axes[0].grid(True, alpha=0.3)

    # --- Position error subplot ---
    axes[1].plot(ee_linear_speeds[valid_p], pos_errors[valid_p], "s-", linewidth=1.5, markersize=5, color="tab:orange")
    axes[1].set_ylabel("Avg Position Error (m)")
    axes[1].set_xlabel("EE Linear Speed  v = r·ω  (m/s,  r = 0.1 m)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "angular_speed_sweep.png")
    fig.savefig(out_path, dpi=150)
    print(f"[PLOT] Sweep plot saved to {out_path}")


if __name__ == "__main__":
    main()
