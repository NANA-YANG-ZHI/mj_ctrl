"""Plot Kp x Ki heatmap of steady-state force error from PI gain sweep CSV.

Usage:
    python plot_pi_gain_sweep.py <results_csv> <output_dir>
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt


def _parse_col(val: str) -> float:
    return float(val) if val.strip() != "nan" else float("nan")


def main():
    if len(sys.argv) < 3:
        print("Usage: python plot_pi_gain_sweep.py <results_csv> <output_dir>")
        sys.exit(1)

    csv_path = sys.argv[1]
    output_dir = sys.argv[2]
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        sys.exit(1)

    kp_vals, ki_vals = [], []
    avg_force_errors, var_force_errors = [], []
    avg_pos_errors, var_pos_errors = [], []

    with open(csv_path) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # skip header
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            try:
                kp_vals.append(float(parts[0]))
                ki_vals.append(float(parts[1]))
                avg_force_errors.append(_parse_col(parts[2]))
                var_force_errors.append(_parse_col(parts[3]) if len(parts) > 3 else float("nan"))
                avg_pos_errors.append(_parse_col(parts[4]) if len(parts) > 4 else float("nan"))
                var_pos_errors.append(_parse_col(parts[5]) if len(parts) > 5 else float("nan"))
            except (ValueError, IndexError):
                continue

    kp_unique = sorted(set(kp_vals))
    ki_unique = sorted(set(ki_vals))
    nkp = len(kp_unique)
    nki = len(ki_unique)

    kp_idx = {v: i for i, v in enumerate(kp_unique)}
    ki_idx = {v: i for i, v in enumerate(ki_unique)}

    def make_grid(values):
        grid = np.full((nkp, nki), float("nan"))
        for kp, ki, v in zip(kp_vals, ki_vals, values):
            grid[kp_idx[kp], ki_idx[ki]] = v
        return grid

    grid_avg_fe  = make_grid(avg_force_errors)
    grid_var_fe  = make_grid(var_force_errors)
    grid_avg_pe  = make_grid(avg_pos_errors)
    grid_var_pe  = make_grid(var_pos_errors)

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("PI Gain Sweep — Steady-State Performance", fontsize=14)

    specs = [
        (axes[0, 0], grid_avg_fe, "Avg |Force Z Error| (N)",  "YlOrRd"),
        (axes[0, 1], grid_var_fe, "Var Force Z Error (N²)",   "YlOrRd"),
        (axes[1, 0], grid_avg_pe, "Avg Position Error (m)",   "Blues"),
        (axes[1, 1], grid_var_pe, "Var Position Error (m²)",  "Blues"),
    ]

    KP = np.array(kp_unique)
    KI = np.array(ki_unique)

    for ax, grid, title, cmap in specs:
        im = ax.pcolormesh(KI, KP, grid, cmap=cmap, shading="auto")
        fig.colorbar(im, ax=ax)

        # mark the minimum with a star
        if not np.all(np.isnan(grid)):
            min_idx = np.unravel_index(np.nanargmin(grid), grid.shape)
            best_kp = KP[min_idx[0]]
            best_ki = KI[min_idx[1]]
            ax.plot(best_ki, best_kp, "w*", markersize=14,
                    label=f"min: Kp={best_kp}, Ki={best_ki}")
            ax.legend(fontsize=8, loc="upper right")

        ax.set_xlabel("Ki")
        ax.set_ylabel("Kp")
        ax.set_title(title)
        ax.set_xticks(KI)
        ax.set_yticks(KP)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "pi_gain_heatmap.png")
    fig.savefig(out_path, dpi=150)
    print(f"[PLOT] Heatmap saved to {out_path}")

    # Print best (Kp, Ki) by avg force error
    if not np.all(np.isnan(grid_avg_fe)):
        min_idx = np.unravel_index(np.nanargmin(grid_avg_fe), grid_avg_fe.shape)
        print(f"[RESULT] Best Kp={KP[min_idx[0]]}, Ki={KI[min_idx[1]]}  "
              f"→ avg_force_error={grid_avg_fe[min_idx]:.6f} N")


if __name__ == "__main__":
    main()
