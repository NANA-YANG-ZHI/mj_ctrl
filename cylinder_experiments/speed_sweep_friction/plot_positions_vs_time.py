"""Plot actual vs desired X/Y/Z positions over time for a single method.

Usage
-----
python cylinder_experiments/speed_sweep_friction/plot_positions_vs_time.py \
    --multiplier 0.1 --robot fr3_friction --method paper

Method keys
-----------
  baseline   Baseline
  ff         Feedforward
  ff_pi      Feedforward + PI
  pd         PD
  paper      HFDC
  paper_pi   HFDC + PI
"""

import argparse
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
})

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(SCRIPT_DIR, "sweep_friction_results", "data")
DEFAULT_OUT  = os.path.join(SCRIPT_DIR, "sweep_friction_results", "plots", "traj")

DT       = 0.001
PLOT_DPI = 300

METHOD_KEYS = [
    ("Baseline",         "baseline",  "tab:gray",   ),
    ("Feedforward",      "ff",        "tab:blue",   ),
    ("Feedforward + PI", "ff_pi",     "tab:purple", ),
    ("PD",               "pd",        "tab:green",  ),
    ("HFDC",             "paper",     "tab:orange", ),
    ("HFDC + PI",        "paper_pi",  "tab:red",    ),
]


def load_npz(data_root, robot, method, multiplier):
    base = os.path.join(data_root, robot, method)
    for suffix in (f"data_{multiplier:.1f}_all.npz", f"data_{multiplier:.1f}.npz"):
        path = os.path.join(base, suffix)
        if os.path.isfile(path):
            return np.load(path)
    raise FileNotFoundError(
        f"No NPZ for robot={robot}, method={method}, mult={multiplier:.1f} in {base}"
    )


def plot_positions_vs_time(data_root, robot, method_key, multiplier, out_dir):
    """Plot actual and desired X/Y/Z positions over time for one method.

    Parameters
    ----------
    data_root   : path to the data root (contains <robot>/<method>/ dirs)
    robot       : "fr3_friction" or "fr3_jointf_surff"
    method_key  : baseline / ff / ff_pi / pd / paper / paper_pi
    multiplier  : speed multiplier, e.g. 0.1, 0.5, 1.0
    out_dir     : directory where the PNG is saved
    """
    label, color = next(
        ((lbl, col) for lbl, key, col in METHOD_KEYS if key == method_key),
        (method_key, "tab:blue"),
    )

    d   = load_npz(data_root, robot, method_key, multiplier)
    act = d["actual_positions"]   # (N, 3)  X Y Z
    des = d["desired_positions"]  # (N, 3)
    t   = np.arange(len(act)) * DT

    pos_err  = np.linalg.norm(act - des, axis=1) * 1e3  # mm
    err_mean = np.nanmean(pos_err)
    err_var  = np.nanvar(pos_err)

    fig, axes = plt.subplots(3, 1, figsize=(6.0, 5.5), sharex=True)
    for i, (ax, ylabel) in enumerate(zip(axes, ["X (m)", "Y (m)", "Z (m)"])):
        ax.plot(t, des[:, i], color="black", linestyle="--",
                linewidth=0.8, alpha=0.6, label="Desired")
        ax.plot(t, act[:, i], color=color, linewidth=0.8,
                alpha=0.85, label="Actual")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.85)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"Position vs Time  |  {label}  |  robot={robot}  mult={multiplier:.1f}×\n"
        f"pos err: mean={err_mean:.2f} mm,  var={err_var:.2f} mm²",
        fontsize=9,
    )
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    safe = label.lower().replace(" + ", "_plus_").replace(" ", "_")
    out  = os.path.join(out_dir, f"pos_time_{safe}_{robot}_mult{multiplier:.1f}.png")
    fig.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {out}")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Plot actual vs desired X/Y/Z positions over time."
    )
    parser.add_argument("--data-root",  default=DEFAULT_DATA)
    parser.add_argument("--output-dir", default=DEFAULT_OUT)
    parser.add_argument("--robot",      default="fr3_friction",
                        choices=["fr3_friction", "fr3_jointf_surff"])
    parser.add_argument("--multiplier", type=float, default=0.1,
                        help="Speed multiplier, e.g. 0.1, 0.5, 1.0")
    parser.add_argument("--method",     required=True,
                        help="baseline / ff / ff_pi / pd / paper / paper_pi")
    args = parser.parse_args()

    plot_positions_vs_time(
        args.data_root, args.robot, args.method, args.multiplier, args.output_dir
    )


if __name__ == "__main__":
    main()
