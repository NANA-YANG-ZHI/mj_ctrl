"""
Compensation Term Ablation — Data Collection
============================================
Runs run_approach_then_hybrid_mujoco.py four times, each time with one
compensation term disabled, and saves .npz data files to --data-dir.

Usage:
    python compensation_comparison/run_experiments.py \\
        --headless --data-dir compensation_comparison/data/frictionless

    python compensation_comparison/run_experiments.py \\
        --headless --robot fr3_friction --surface-friction 0.7 \\
        --data-dir compensation_comparison/data/friction_0.7
"""
import argparse
import math
import os
import subprocess
import sys

CONFIGS = [
    {"label": "No contact force compensation", "flags": ["--no-contact-force-compensation"]},
    {"label": "No velocity term",              "flags": ["--no-velocity-term"]},
    {"label": "No control force compensation", "flags": ["--no-control-force-compensation"]},
    {"label": "All compensations ON",          "flags": []},
]

RUNNER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "run_approach_then_hybrid_mujoco.py"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run 4 compensation ablation configs and save .npz data"
    )
    parser.add_argument("--robot", default="fr3",
                        choices=["fr3", "kuka", "panda", "fr3_friction"])
    parser.add_argument("--approach-duration", type=float, default=20.0)
    parser.add_argument("--circle-duration",   type=float, default=10.0)
    parser.add_argument("--angular-speed",     type=float, default=float("nan"),
                        help="Angular speed in rad/s (default: 2π from the run script)")
    parser.add_argument("--force-control-method", default="paper",
                        choices=["paper", "pd", "feedforward"])
    parser.add_argument("--skip-seconds", type=float, default=1.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--surface-friction", type=float, default=None)
    parser.add_argument("--slope-angle", type=float, default=0.0,
                        help="Slope angle in degrees (default: 0.0 = flat surface)")
    parser.add_argument("--data-dir", required=True,
                        help="Directory to save .npz data files")
    return parser.parse_args()


def build_cmd(args, extra_flags):
    cmd = [
        sys.executable, RUNNER,
        "--robot",                args.robot,
        "--approach-duration",    str(args.approach_duration),
        "--circle-duration",      str(args.circle_duration),
        "--force-control-method", args.force_control_method,
        "--skip-seconds",         str(args.skip_seconds),
        "--save-data",
        "--data-dir",             os.path.abspath(args.data_dir),
        "--multiplier",           "0.0",
    ]
    if not math.isnan(args.angular_speed):
        cmd += ["--angular-speed", str(args.angular_speed)]
    if args.surface_friction is not None:
        cmd += ["--surface-friction", str(args.surface_friction)]
    cmd += ["--slope-angle", str(args.slope_angle)]
    if args.headless:
        cmd.append("--headless")
    cmd.extend(extra_flags)
    return cmd


def main():
    args = parse_args()
    os.makedirs(args.data_dir, exist_ok=True)

    for cfg in CONFIGS:
        print(f"\n{'='*60}")
        print(f"Running: {cfg['label']}")
        print(f"{'='*60}")
        subprocess.run(build_cmd(args, cfg["flags"]), check=True)
        print(f"[DONE] {cfg['label']}")

    print(f"\n[INFO] All data saved to: {os.path.abspath(args.data_dir)}")


if __name__ == "__main__":
    main()
