"""
Compensation Term Ablation — Cylinder Surface Plot
===================================================
Reads .npz files produced by run_experiments.py and generates a signed
force-error comparison plot (normal force projected onto cylinder surface).

Usage:
    python cylinder_experiments/compensation_comparison/plot_comparison.py \\
        --data-dir cylinder_experiments/compensation_comparison/data/fr3_friction_cylinder \\
        --plot-dir cylinder_experiments/compensation_comparison/plots/fr3_friction_cylinder
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

try:
    from tueplots import bundles
    plt.rcParams.update(bundles.icml2024(usetex=False))
except ImportError:
    pass

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

CONFIGS = [
    {
        "label":     "No contact force compensation",
        "suffix":    "_no_contact",
        "color":     "tab:orange",
        "linestyle": "-",
        "linewidth": 1.2,
        "zorder":    2,
    },
    {
        "label":     "No velocity term",
        "suffix":    "_no_vel",
        "color":     "tab:green",
        "linestyle": "-",
        "linewidth": 1.2,
        "zorder":    2,
    },
    {
        "label":     "No control force compensation",
        "suffix":    "_no_ctrl",
        "color":     "tab:red",
        "linestyle": "-",
        "linewidth": 1.2,
        "zorder":    2,
    },
    {
        "label":     "All compensations ON",
        "suffix":    "_all",
        "color":     "tab:blue",
        "linestyle": "--",
        "linewidth": 1.8,
        "zorder":    3,
    },
]

DT = 0.001


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot cylinder force error comparison from saved .npz data"
    )
    parser.add_argument("--data-dir", required=True,
                        help="Directory containing .npz data files")
    parser.add_argument("--plot-dir", required=True,
                        help="Directory to save the output plot")
    parser.add_argument("--skip-seconds", type=float, default=1.0,
                        help="Seconds to skip when computing avg/max |error|")
    parser.add_argument("--plot-duration", type=float, default=None,
                        help="Seconds of data to plot (default: all)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.plot_dir, exist_ok=True)

    skip        = int(args.skip_seconds / DT)
    max_samples = int(args.plot_duration / DT) if args.plot_duration else None

    print(f"\n{'Config':<35s}  {'Avg |err| (N)':>14}  {'Max |err| (N)':>14}")
    print("-" * 67)

    fig, ax = plt.subplots()

    for cfg in CONFIGS:
        fpath = os.path.join(args.data_dir, f"data_0.0{cfg['suffix']}.npz")
        if not os.path.exists(fpath):
            print(f"[WARN] Missing {fpath} — skipping {cfg['label']}")
            continue

        d = np.load(fpath)
        force_error = d["force_error"]
        if max_samples is not None:
            force_error = force_error[:max_samples]
        t = np.arange(len(force_error)) * DT
        n = len(force_error)
        fe_sk   = force_error[skip:] if n > skip else force_error
        avg_abs = np.mean(np.abs(fe_sk))
        max_abs = np.max(np.abs(fe_sk))
        print(f"{cfg['label']:<35s}  {avg_abs:>14.3f}  {max_abs:>14.3f}")

        ax.plot(t, force_error,
                color=cfg["color"], linestyle=cfg["linestyle"],
                linewidth=cfg["linewidth"], zorder=cfg["zorder"],
                label=cfg["label"])

    print()
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normal Force Error (N)")
    ax.set_title("Cylinder Surface — Compensation Ablation")
    ax.legend(loc="upper right")

    out_path = os.path.join(args.plot_dir, "compensation_comparison.png")
    fig.savefig(out_path, dpi=300)
    print(f"[PLOT] Saved to: {out_path}")


if __name__ == "__main__":
    main()
