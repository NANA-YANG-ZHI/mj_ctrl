"""Compensation term ablation — force error time-series per condition.

Auto-discovers condition subdirectories under data/ and plots one figure per
condition, saved to plots/<condition>/compensation_comparison.png.

Usage
-----
    python plot.py
"""

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
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT  = os.path.join(SCRIPT_DIR, "data")
PLOTS_DIR  = os.path.join(SCRIPT_DIR, "plots")

CONFIGS = [
    dict(label="No contact force compensation", suffix="_no_contact",
         color="tab:orange", linestyle="-",  linewidth=1.2, zorder=2),
    dict(label="No velocity term",              suffix="_no_vel",
         color="tab:green",  linestyle="-",  linewidth=1.2, zorder=2),
    dict(label="No control force compensation", suffix="_no_ctrl",
         color="tab:red",    linestyle="-",  linewidth=1.2, zorder=2),
    dict(label="All compensations ON",          suffix="_all",
         color="tab:blue",   linestyle="--", linewidth=1.8, zorder=3),
]

PLOT_DURATION_S = 2.0
SKIP_S          = 1.0
DT              = 0.001


def plot_condition(condition, data_dir, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    skip        = int(SKIP_S / DT)
    max_samples = int(PLOT_DURATION_S / DT)

    fig, ax = plt.subplots()
    has_data = False

    for cfg in CONFIGS:
        fpath = os.path.join(data_dir, f"data_0.0{cfg['suffix']}.npz")
        if not os.path.exists(fpath):
            print(f"  [SKIP] {cfg['label']}: {os.path.basename(fpath)} not found")
            continue

        d           = np.load(fpath)
        force_error = d["force_error"][:max_samples]
        t           = np.arange(len(force_error)) * DT
        n           = len(force_error)
        fe_sk       = force_error[skip:] if n > skip else force_error
        avg_abs     = np.mean(np.abs(fe_sk))
        max_abs     = np.max(np.abs(fe_sk))
        print(f"  {cfg['label']:<35s}  avg={avg_abs:.3f} N  max={max_abs:.3f} N")

        ax.plot(t, force_error,
                color=cfg["color"], linestyle=cfg["linestyle"],
                linewidth=cfg["linewidth"], zorder=cfg["zorder"],
                label=cfg["label"])
        has_data = True

    if not has_data:
        plt.close(fig)
        return

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force Z Error (N)")
    ax.legend(loc="upper right")

    out = os.path.join(plot_dir, "compensation_comparison.png")
    fig.savefig(out, dpi=600)
    plt.close(fig)
    print(f"  [PLOT] {out}")


def main():
    if not os.path.isdir(DATA_ROOT):
        print(f"[ERROR] data dir not found: {DATA_ROOT}")
        return

    conditions = sorted(
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d))
    )

    if not conditions:
        print("No condition subdirectories found in data/.")
        return

    for condition in conditions:
        print(f"\n[CONDITION] {condition}")
        data_dir = os.path.join(DATA_ROOT, condition)
        plot_dir = os.path.join(PLOTS_DIR, condition)
        plot_condition(condition, data_dir, plot_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
