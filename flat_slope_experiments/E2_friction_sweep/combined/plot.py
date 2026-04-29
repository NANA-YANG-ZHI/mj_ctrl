"""Combined friction sweep — mean ± std for HFDC and HFDC+PI vs friction coefficient.

Reads per-friction .npz files from E2_friction_sweep/data/paper/ and data/paper_pi/.

Outputs (saved to plots/):
  force_combined.png
  position_combined.png

Usage
-----
    python plot.py
"""

import glob
import os
import re

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
DATA_ROOT  = os.path.join(SCRIPT_DIR, "..", "data")
PLOTS_DIR  = os.path.join(SCRIPT_DIR, "plots")

DT               = 0.001
FORCE_SKIP_SAMPLES = int(1.0 / DT)

STD_RATIO = 1.0

METHOD_KEYS = [
    ("Baseline",  "baseline", "tab:gray",   "x"),
    ("HFDC",      "paper",    "tab:blue",   "o"),
    # ("HFDC + PI", "paper_pi", "tab:orange", "s"),
]

METRICS = [
    dict(
        col_mean="avg_force_z_error",
        col_var="var_force_z_error",
        ylabel=f"Avg |Force Z Error| ± {STD_RATIO:.0f}Std  (N)",
        out="force_combined.png",
    ),
    dict(
        col_mean="avg_position_error",
        col_var="var_position_error",
        ylabel=f"Avg Position Error ± {STD_RATIO:.0f}Std  (m)",
        out="position_combined.png",
    ),
]


def load_all(data_dir):
    npz_files = sorted(
        glob.glob(os.path.join(data_dir, "data_*.npz")),
        key=lambda p: float(
            re.match(r"data_([0-9]+(?:\.[0-9]+)?)", os.path.basename(p)).group(1)
        ),
    )
    rows = {k: [] for k in (
        "friction_coeff",
        "avg_force_z_error", "var_force_z_error",
        "avg_position_error", "var_position_error",
    )}
    for fpath in npz_files:
        m = re.match(r"data_([0-9]+(?:\.[0-9]+)?)", os.path.basename(fpath))
        rows["friction_coeff"].append(float(m.group(1)))

        d = np.load(fpath)
        fe = d["force_error"]
        fe_ss = fe[FORCE_SKIP_SAMPLES:]
        if len(fe_ss) > 0:
            rows["avg_force_z_error"].append(np.mean(np.abs(fe_ss)))
            rows["var_force_z_error"].append(np.var(fe_ss))
        else:
            rows["avg_force_z_error"].append(float("nan"))
            rows["var_force_z_error"].append(float("nan"))

        pe = d["position_error"]
        if len(pe) > 0:
            rows["avg_position_error"].append(np.mean(pe))
            rows["var_position_error"].append(np.var(pe))
        else:
            rows["avg_position_error"].append(float("nan"))
            rows["var_position_error"].append(float("nan"))

    return {k: np.array(v) for k, v in rows.items()}


def make_combined_plot(cfg, datasets):
    fig, ax = plt.subplots()

    for name, data, color, marker in datasets:
        mu   = data["friction_coeff"]
        mean = data[cfg["col_mean"]]
        std  = np.sqrt(np.maximum(data[cfg["col_var"]], 0.0)) * STD_RATIO

        lo = np.maximum(mean - std, 0.0)
        hi = mean + std

        valid = ~np.isnan(mean)
        ax.fill_between(mu[valid], lo[valid], hi[valid],
                        alpha=0.20, color=color, linewidth=0)
        ax.plot(mu[valid], mean[valid],
                marker=marker, color=color, label=name,
                linewidth=1, markersize=2, markerfacecolor=color)

    ax.set_xlabel("Sliding Friction Coefficient μ")
    ax.set_ylabel(cfg["ylabel"])
    ax.set_xticks(np.round(np.arange(0.1, 1.05, 0.1), 1))
    ax.legend(loc="upper right", framealpha=0.85)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, cfg["out"])
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {cfg['out']}")


def main():
    datasets = []
    for label, key, color, marker in METHOD_KEYS:
        data_dir = os.path.join(DATA_ROOT, key)
        if not os.path.isdir(data_dir):
            print(f"[SKIP] {label}: {data_dir}")
            continue
        datasets.append((label, load_all(data_dir), color, marker))

    if not datasets:
        print("No data found.")
        return

    for cfg in METRICS:
        make_combined_plot(cfg, datasets)


if __name__ == "__main__":
    main()
