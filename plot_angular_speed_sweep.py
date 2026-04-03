"""Plot angular_speed vs avg_force_z_error from sweep_angular_speed.sh results CSV."""
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

    multipliers = []
    angular_speeds = []
    errors = []

    with open(csv_path) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # skip header
            parts = line.strip().split(",")
            if len(parts) != 3:
                continue
            mult, speed, err = parts
            try:
                multipliers.append(float(mult))
                angular_speeds.append(float(speed))
                errors.append(float(err) if err != "nan" else float("nan"))
            except ValueError:
                continue

    multipliers = np.array(multipliers)
    angular_speeds = np.array(angular_speeds)
    errors = np.array(errors)

    plot_dir = output_dir
    os.makedirs(plot_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    valid = ~np.isnan(errors)
    ax.plot(multipliers[valid], errors[valid], "o-", linewidth=1.5, markersize=5)
    ax.set_xlabel("Angular Speed Multiplier (× π rad/s)")
    ax.set_ylabel("Avg |Force Z Error| (N)")
    ax.set_title("Angular Speed vs Average Force Z Error")
    ax.grid(True, alpha=0.3)

    # Add secondary x-axis showing actual rad/s values
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    tick_mults = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    ax2.set_xticks(tick_mults)
    ax2.set_xticklabels([f"{m * np.pi:.2f}" for m in tick_mults], fontsize=8)
    ax2.set_xlabel("Angular Speed (rad/s)")

    plt.tight_layout()
    out_path = os.path.join(plot_dir, "angular_speed_sweep.png")
    fig.savefig(out_path, dpi=150)
    print(f"[PLOT] Sweep plot saved to {out_path}")


if __name__ == "__main__":
    main()
