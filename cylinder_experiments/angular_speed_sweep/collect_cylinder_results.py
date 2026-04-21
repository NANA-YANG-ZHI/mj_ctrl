"""
Collect NPZ experiment files into a summary CSV for the cylinder speed sweep.

Scans  <data_root>/<robot>/data_*.npz  for every robot sub-directory,
trims the burn-in period, computes mean and std of force / position error,
and writes one row per (robot, omega) to the output CSV.

Usage:
    python collect_cylinder_results.py <data_root>
        [--output sweep_results.csv]
        [--skip-seconds 1.0]
        [--dt 0.001]
"""
import argparse
import glob
import math
import os
import sys

import numpy as np


ROBOT_ORDER = ["fr3", "fr3_friction", "fr3_jointf", "fr3_jointf_surff"]


def collect(data_root: str, skip_seconds: float, dt: float):
    skip_n = int(skip_seconds / dt)
    rows = []
    missing = []

    robot_dirs = sorted(os.listdir(data_root))
    # Put known robots first in canonical order, then any unknown ones
    robot_dirs = [r for r in ROBOT_ORDER if r in robot_dirs] + \
                 [r for r in robot_dirs if r not in ROBOT_ORDER]

    for robot in robot_dirs:
        robot_path = os.path.join(data_root, robot)
        if not os.path.isdir(robot_path):
            continue

        npz_files = sorted(glob.glob(os.path.join(robot_path, "data_*.npz")))
        if not npz_files:
            print(f"[WARN] No NPZ files found in {robot_path}")
            continue

        for fpath in npz_files:
            try:
                d = np.load(fpath)
            except Exception as e:
                print(f"[WARN] Failed to load {fpath}: {e}")
                missing.append(fpath)
                continue

            omega    = float(d["angular_speed_rad_s"])
            ee_speed = float(d["ee_linear_speed_m_s"])

            force_err = d["force_error"][skip_n:]
            pos_err   = d["position_error"][skip_n:]

            if force_err.size == 0 or pos_err.size == 0:
                print(f"[WARN] {fpath}: empty after burn-in trim, skipping")
                missing.append(fpath)
                continue

            mean_force = float(np.nanmean(np.abs(force_err)))
            std_force  = float(np.nanstd(force_err))
            mean_pos   = float(np.nanmean(pos_err))
            std_pos    = float(np.nanstd(pos_err))

            rows.append((robot, omega, ee_speed, mean_force, std_force, mean_pos, std_pos))

    rows.sort(key=lambda r: (ROBOT_ORDER.index(r[0]) if r[0] in ROBOT_ORDER else 99, r[1]))
    return rows, missing


def main():
    parser = argparse.ArgumentParser(description="Collect cylinder NPZ results → CSV")
    parser.add_argument("data_root", help="Directory containing <robot>/*.npz sub-folders")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: <data_root>/../sweep_results.csv)")
    parser.add_argument("--skip-seconds", type=float, default=1.0,
                        help="Burn-in seconds to drop from each run (default: 1.0)")
    parser.add_argument("--dt", type=float, default=0.001,
                        help="Simulation timestep in seconds (default: 0.001)")
    args = parser.parse_args()

    if not os.path.isdir(args.data_root):
        print(f"[ERROR] data_root not found: {args.data_root}")
        sys.exit(1)

    output = args.output or os.path.join(args.data_root, "..", "sweep_results.csv")
    output = os.path.normpath(output)

    print(f"Scanning {args.data_root}  (skip={args.skip_seconds}s = {int(args.skip_seconds/args.dt)} samples)")

    rows, missing = collect(args.data_root, args.skip_seconds, args.dt)

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        f.write("robot,omega_rad_s,ee_linear_speed_m_s,"
                "mean_force_error,std_force_error,"
                "mean_position_error,std_position_error\n")
        for robot, omega, ee_speed, mf, sf, mp, sp in rows:
            f.write(f"{robot},{omega:.4f},{ee_speed:.4f},"
                    f"{mf:.6f},{sf:.6f},{mp:.6f},{sp:.6f}\n")

    print(f"Wrote {len(rows)} rows → {output}")
    if missing:
        print(f"[WARN] {len(missing)} file(s) skipped: {missing}")


if __name__ == "__main__":
    main()
