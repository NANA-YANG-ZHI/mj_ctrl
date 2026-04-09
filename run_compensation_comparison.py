"""
Compensation Term Ablation Comparison
======================================
Runs run_approach_then_hybrid_mujoco.py four times, each with a different
compensation term disabled, saves the force error as .npz, then plots all
4 time series on a single figure for comparison.

Usage:
    python run_compensation_comparison.py --headless --circle-duration 10.0
    python run_compensation_comparison.py --headless --data-dir /tmp/comp_data --plot-dir plots/comp
"""
import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np


CONFIGS = [
    {
        "label": "All compensations ON",
        "flags": [],
        "suffix": "_all",
        "color": "tab:blue",
    },
    {
        "label": "No contact force compensation",
        "flags": ["--no-contact-force-compensation"],
        "suffix": "_no_contact",
        "color": "tab:orange",
    },
    {
        "label": "No velocity term",
        "flags": ["--no-velocity-term"],
        "suffix": "_no_vel",
        "color": "tab:green",
    },
    {
        "label": "No control force compensation",
        "flags": ["--no-control-force-compensation"],
        "suffix": "_no_ctrl",
        "color": "tab:red",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run 4 compensation-term configurations and compare force error time series"
    )
    parser.add_argument("--robot", type=str, default="fr3",
                        choices=["fr3", "kuka", "panda", "fr3_friction"])
    parser.add_argument("--approach-duration", type=float, default=20.0)
    parser.add_argument("--circle-duration", type=float, default=10.0)
    parser.add_argument("--angular-speed", type=float, default=float("nan"),
                        help="Angular speed in rad/s (default: pi*2 from the run script)")
    parser.add_argument("--force-control-method", type=str, default="paper",
                        choices=["paper", "pd", "feedforward"])
    parser.add_argument("--skip-seconds", type=float, default=1.0,
                        help="Seconds to skip at the start when computing avg |error| in legend")
    parser.add_argument("--headless", action="store_true",
                        help="Run all sub-simulations without the MuJoCo viewer")
    parser.add_argument("--plot-dir", type=str,
                        default="plots/compensation_comparison",
                        help="Directory to save the comparison plot")
    parser.add_argument("--data-dir", type=str, default="",
                        help="Directory for per-run .npz files. Uses a temp dir if empty.")
    return parser.parse_args()


def build_run_cmd(args, extra_flags, data_dir):
    """Build the subprocess command for one configuration run."""
    runner = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "run_approach_then_hybrid_mujoco.py")
    cmd = [
        sys.executable, runner,
        "--robot", args.robot,
        "--approach-duration", str(args.approach_duration),
        "--circle-duration", str(args.circle_duration),
        "--force-control-method", args.force_control_method,
        "--skip-seconds", str(args.skip_seconds),
        "--save-data",
        "--data-dir", data_dir,
        "--multiplier", "0.0",
    ]
    if not math.isnan(args.angular_speed):
        cmd += ["--angular-speed", str(args.angular_speed)]
    if args.headless:
        cmd.append("--headless")
    cmd.extend(extra_flags)
    return cmd


def run_all_configs(args, data_dir):
    for cfg in CONFIGS:
        print(f"\n{'='*60}")
        print(f"Running: {cfg['label']}")
        print(f"{'='*60}")
        cmd = build_run_cmd(args, cfg["flags"], data_dir)
        subprocess.run(cmd, check=True)
        print(f"[DONE] {cfg['label']}")


def make_comparison_plot(args, data_dir):
    dt = 0.001  # matches ControllerConfig.dt

    fig, ax = plt.subplots(figsize=(12, 5))

    for cfg in CONFIGS:
        fname = f"data_0.0{cfg['suffix']}.npz"
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print(f"[WARN] Missing {fpath} — skipping {cfg['label']}")
            continue

        d = np.load(fpath)
        force_error = d["force_error"]
        t = np.arange(len(force_error)) * dt
        skip = int(args.skip_seconds / dt)
        avg_abs = np.mean(np.abs(force_error[skip:])) if len(force_error) > skip else float("nan")
        label = f"{cfg['label']}  (avg |err| = {avg_abs:.3f} N)"
        ax.plot(t, force_error, linewidth=1.2, color=cfg["color"], label=label)

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force Error Z (N)")
    ax.set_title("Force Tracking Error: Contribution of Each Compensation Term")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(args.plot_dir, exist_ok=True)
    out_path = os.path.join(args.plot_dir, "compensation_comparison_surface_friction.png")
    fig.savefig(out_path, dpi=150)
    print(f"\n[PLOT] Comparison plot saved to: {out_path}")
    plt.show()


def main():
    args = parse_args()

    if args.data_dir:
        data_dir = os.path.abspath(args.data_dir)
        os.makedirs(data_dir, exist_ok=True)
        cleanup = False
    else:
        data_dir = tempfile.mkdtemp(prefix="mj_ctrl_comp_")
        cleanup = True
        print(f"[INFO] Using temp data dir: {data_dir}")

    try:
        run_all_configs(args, data_dir)
        make_comparison_plot(args, data_dir)
    finally:
        if cleanup:
            shutil.rmtree(data_dir, ignore_errors=True)
            print(f"[INFO] Temp data dir removed: {data_dir}")


if __name__ == "__main__":
    main()
