"""
Collect NPZ experiment files into a summary CSV for the speed_sweep_friction experiment.

Scans  <data_root>/<robot>/<method>/data_*.npz  for every combination,
trims burn-in, computes mean and std of force / position error.

Handles both:
  - run_approach_then_hybrid_mujoco.py output: data_<mult>_all.npz (has ee_linear_speed_m_s)
  - run_baseline_mujoco.py output:             data_<mult>.npz     (no ee_linear_speed_m_s)

Usage:
    python collect_results.py <data_root>
        [--output results.csv]
        [--skip-seconds 0.2]
        [--dt 0.001]
"""
import argparse
import glob
import os
import sys

import numpy as np

CYLINDER_RADIUS = 0.1  # m

ROBOT_ORDER  = ["fr3_friction", "fr3_jointf_surff"]
METHOD_ORDER = ["baseline", "ff", "ff_pi", "pd", "paper", "paper_pi"]


def _robot_key(r):
    return ROBOT_ORDER.index(r) if r in ROBOT_ORDER else 99


def _method_key(m):
    return METHOD_ORDER.index(m) if m in METHOD_ORDER else 99


def collect(data_root: str, skip_n: int, max_mult: float = None):
    rows = []
    missing = []

    for robot in sorted(os.listdir(data_root), key=_robot_key):
        robot_path = os.path.join(data_root, robot)
        if not os.path.isdir(robot_path):
            continue

        for method in sorted(os.listdir(robot_path), key=_method_key):
            method_path = os.path.join(robot_path, method)
            if not os.path.isdir(method_path):
                continue

            for fpath in sorted(glob.glob(os.path.join(method_path, "data_*.npz"))):
                try:
                    d = np.load(fpath)
                except Exception as e:
                    print(f"[WARN] Cannot load {fpath}: {e}")
                    missing.append(fpath)
                    continue

                omega = float(d["angular_speed_rad_s"])
                if max_mult is not None and omega / np.pi > max_mult + 1e-9:
                    continue

                if "ee_linear_speed_m_s" in d:
                    ee_speed = float(d["ee_linear_speed_m_s"])
                else:
                    ee_speed = CYLINDER_RADIUS * omega

                force_err = d["force_error"][skip_n:]
                pos_err   = d["position_error"][skip_n:]

                if force_err.size == 0 or pos_err.size == 0:
                    print(f"[WARN] {fpath}: empty after burn-in, skipping")
                    missing.append(fpath)
                    continue

                rows.append((
                    robot, method, omega, ee_speed,
                    float(np.nanmean(np.abs(force_err))),
                    float(np.nanmax(np.abs(force_err))),
                    float(np.nanstd(force_err)),
                    float(np.nanmean(pos_err)),
                    float(np.nanmax(pos_err)),
                    float(np.nanstd(pos_err)),
                ))

    rows.sort(key=lambda r: (_robot_key(r[0]), _method_key(r[1]), r[2]))
    return rows, missing


def main():
    parser = argparse.ArgumentParser(
        description="Collect speed-sweep-friction NPZ files → CSV"
    )
    parser.add_argument("data_root")
    parser.add_argument("--output", default=None)
    parser.add_argument("--skip-seconds", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--max-multiplier", type=float, default=None,
                        help="Only include files with multiplier (ω/π) ≤ this value")
    args = parser.parse_args()

    if not os.path.isdir(args.data_root):
        print(f"[ERROR] data_root not found: {args.data_root}")
        sys.exit(1)

    skip_n = int(args.skip_seconds / args.dt)
    output = args.output or os.path.normpath(
        os.path.join(args.data_root, "..", "results.csv")
    )

    print(f"Scanning {args.data_root}  (skip={args.skip_seconds}s = {skip_n} samples, "
          f"max_mult={args.max_multiplier if args.max_multiplier is not None else 'all'})")

    rows, missing = collect(args.data_root, skip_n, max_mult=args.max_multiplier)

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        f.write("robot,method,omega_rad_s,ee_linear_speed_m_s,"
                "mean_force_error,max_force_error,std_force_error,"
                "mean_position_error,max_position_error,std_position_error\n")
        for row in rows:
            robot, method, omega, ee_speed, mf, xf, sf, mp, xp, sp = row
            f.write(f"{robot},{method},{omega:.4f},{ee_speed:.4f},"
                    f"{mf:.6f},{xf:.6f},{sf:.6f},{mp:.6f},{xp:.6f},{sp:.6f}\n")

    print(f"Wrote {len(rows)} rows → {output}")
    if missing:
        print(f"[WARN] {len(missing)} file(s) skipped.")


if __name__ == "__main__":
    main()
