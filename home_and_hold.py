"""
home_and_hold.py — Move robot arm to home configuration and hold it.

Two phases:
  1. APPROACH  : Track a cosine S-curve trajectory from the current pose to the
                 home configuration using joint-space PD + gravity compensation.
  2. HOLD      : Pure gravity compensation once the arm is within tolerance.

Usage:
  python home_and_hold.py [--robot {fr3,kuka,panda}] [--T 5.0] [--tol 0.02]
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
import pinocchio as pino

from mujoco_robot_interface import MujocoRobotInterface, Torques
from src import get_robot_config
from src.controller_config import ControllerConfig

# ── Joint-space PD gains (Nm/rad, Nm·s/rad) ──────────────────────────────────
KP = 200.0   # proportional gain (uniform across joints)
KD = 20.0    # derivative gain   (uniform across joints)


def cosine_s_curve(q0: np.ndarray, qf: np.ndarray, t: float, T: float) -> np.ndarray:
    """Smooth interpolation: q(t) = q0 + 0.5*(1 - cos(π*t/T)) * (qf - q0)."""
    if t >= T:
        return qf.copy()
    s = 0.5 * (1.0 - np.cos(np.pi * t / T))
    return q0 + s * (qf - q0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Move robot arm to home and hold.")
    parser.add_argument("--robot", choices=["fr3", "kuka", "panda"], default="fr3",
                        help="Robot type (default: fr3)")
    parser.add_argument("--T", type=float, default=5.0,
                        help="Approach duration in seconds (default: 5.0)")
    parser.add_argument("--tol", type=float, default=0.02,
                        help="Joint-space tolerance to switch to hold mode (rad, default: 0.02)")
    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────────
    robot_cfg  = get_robot_config(args.robot)
    common_cfg = ControllerConfig()

    pino_model = pino.buildModelFromMJCF(robot_cfg.pinocchio_xml_path)
    pino_data  = pino_model.createData()

    mj_iface = MujocoRobotInterface(
        common_cfg,
        joint_names=robot_cfg.joint_names,
        xml_path=robot_cfg.mujoco_scene_xml_path,
    )

    q_home = robot_cfg.q0.copy()

    # Read start configuration from simulation
    robot_state, _ = mj_iface.readOnce()
    q_start = np.array(robot_state.q)

    print(f"Robot  : {args.robot.upper()}")
    print(f"Home q : {np.round(q_home, 4).tolist()}")
    print(f"Start q: {np.round(q_start, 4).tolist()}")
    print(f"Moving to home over {args.T}s, then holding — close the viewer to exit.")

    t = 0.0
    phase = "APPROACH"

    # ── Viewer loop ───────────────────────────────────────────────────────────
    with mujoco.viewer.launch_passive(
        mj_iface.model, mj_iface.data,
        show_left_ui=False, show_right_ui=False,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(mj_iface.model, viewer.cam)

        while viewer.is_running():
            step_start = time.time()

            robot_state, dt = mj_iface.readOnce()
            q  = np.array(robot_state.q)
            dq = np.array(robot_state.dq)

            # Gravity compensation torques
            tau_grav = pino.computeGeneralizedGravity(pino_model, pino_data, q)

            if phase == "APPROACH":
                t += dt
                q_ref = cosine_s_curve(q_start, q_home, t, args.T)

                # PD on reference trajectory + gravity feed-forward
                tau = tau_grav + KP * (q_ref - q) - KD * dq

                error = np.linalg.norm(q_home - q)
                if t >= args.T and error < args.tol:
                    phase = "HOLD"
                    print(f"Reached home (error={error:.4f} rad). Holding.")

            else:  # HOLD
                tau = tau_grav

            mj_iface.writeOnce(Torques(tau.tolist()))
            viewer.sync()

            elapsed = time.time() - step_start
            remaining = common_cfg.dt - elapsed
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
