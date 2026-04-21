# ------------------------------------------------------------------------------
# Approach + Hybrid Force-Impedance Control on a Curved (Cylinder) Surface
#
# Phase 1: Move end-effector to the top of the cylinder (CartesianSpacePDController)
# Phase 2: Sweep along the cylinder surface with hybrid force/motion control.
#
# Key difference from the flat-slope version: the surface normal changes as the
# EE moves around the cylinder, so S_f, S_v, and target_rot are recomputed each
# control step from the current EE position.
#
# Cylinder geometry (from kuka_iiwa_14/table_cylinder.xml):
#   center: [0.5, 0, 0.45]   (body pos)
#   axis:   [1, 0, 0]         (body euler="0 pi/2 0" → local Z becomes world X)
#   radius: 0.1 m             (size="0.1 0.1" → first value is radius)
# ------------------------------------------------------------------------------
import argparse
import mujoco
import mujoco.viewer
import numpy as np
import time
import pinocchio as pino
from scipy.spatial.transform import Rotation

from src import (
    ControllerConfig,
    CartesianSpacePDController,
    CartesianSpacePDControlConfig,
    ControlPhase,
    HybridControllerConfig,
    get_robot_config,
)
from utils_libfranka import (
    compute_ee_pose_error,
    task_space_inertiaM,
    null_space_tau,
    dynamically_consistent_inv,
    feedforward_PD,
    PI_term,
)
from mujoco_robot_interface import MujocoRobotInterface, Torques

# ──────────────────────────────────────────────────────────────────────────────
# Cylinder geometry constants
# ──────────────────────────────────────────────────────────────────────────────
CYLINDER_CENTER = np.array([0.5, 0.0, 0.45])
CYLINDER_AXIS   = np.array([1.0, 0.0, 0.0])   # horizontal, along world X
CYLINDER_RADIUS = 0.1                           # metres


def cylinder_surface_normal(ee_pos: np.ndarray) -> np.ndarray:
    """Outward surface normal at *ee_pos*, pointing away from the cylinder axis."""
    radial = ee_pos - CYLINDER_CENTER
    radial -= np.dot(radial, CYLINDER_AXIS) * CYLINDER_AXIS  # remove axial component
    norm = np.linalg.norm(radial)
    return radial / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])


def cylinder_ee_rotation(normal: np.ndarray) -> np.ndarray:
    """
    Build target EE rotation matrix so that:
      x_ee = cylinder axis  [1, 0, 0]
      z_ee = -normal         (pressing into surface)
      y_ee = z_ee × x_ee    (right-hand frame)
    """
    x_ee = CYLINDER_AXIS.copy()
    z_ee = -normal
    y_ee = np.cross(z_ee, x_ee)
    return np.column_stack([x_ee, y_ee, z_ee])


def cylinder_selection_matrices(normal: np.ndarray):
    """
    Build selection matrices S_f (6×1, force space) and S_v (6×5, motion space)
    from the outward surface normal.

    Force direction : outward normal n
    Motion dirs     : [cylinder_axis, tangent, rx, ry, rz]
    where tangent = n × cylinder_axis  (circumferential direction)
    """
    t1 = CYLINDER_AXIS                         # axial tangent
    t2 = np.cross(normal, t1)                  # circumferential tangent
    t2 /= np.linalg.norm(t2) if np.linalg.norm(t2) > 1e-8 else 1.0

    S_f = np.zeros((6, 1))
    S_f[:3, 0] = normal

    S_v = np.zeros((6, 5))
    S_v[:3, 0] = t1   # axial
    S_v[:3, 1] = t2   # circumferential
    S_v[3, 2]  = 1    # rx
    S_v[4, 3]  = 1    # ry
    S_v[5, 4]  = 1    # rz

    return S_f, S_v


def cylinder_trajectory(elapsed: float, omega: float, theta_start: float = 0.0):
    """
    Arc trajectory on the cylinder surface.

    The EE sweeps around the cylinder in the Y-Z plane (perpendicular to the
    cylinder axis) at constant angular speed *omega*.

    Returns
    -------
    target_pos : (3,)
    x_dot      : (3,)  desired linear velocity in world frame
    x_ddot     : (3,)  desired linear acceleration in world frame
    theta      : float  current angle (radians)
    """
    theta = theta_start + omega * elapsed
    sin_t, cos_t = np.sin(theta), np.cos(theta)

    normal  = np.array([0.0,  sin_t,  cos_t])
    tangent = np.array([0.0,  cos_t, -sin_t])  # d(normal)/dtheta

    target_pos = CYLINDER_CENTER + CYLINDER_RADIUS * normal
    x_dot      = CYLINDER_RADIUS * omega * tangent
    x_ddot     = -CYLINDER_RADIUS * omega**2 * normal   # centripetal

    return target_pos, x_dot, x_ddot, theta


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────

def _plot_position_tracking(t, ee_pos, tgt_pos, pos_err, save_dir=None):
    """
    Figure 1 – Position tracking.

    Row 1-3 : X / Y / Z  (EE vs target, time-series)
    Row 4   : 3-D cylinder-surface trajectory (EE vs target as arc)
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(13, 11))
    fig.suptitle("Position Tracking on Cylinder Surface", fontsize=13, fontweight="bold")
    gs  = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.35)

    ax_x  = fig.add_subplot(gs[0, 0])
    ax_y  = fig.add_subplot(gs[1, 0])
    ax_z  = fig.add_subplot(gs[2, 0])
    ax_e  = fig.add_subplot(gs[3, 0])
    ax_3d = fig.add_subplot(gs[:, 1], projection="3d")

    labels = ["X (m)", "Y (m)", "Z (m)"]
    for ax, col, lbl in zip([ax_x, ax_y, ax_z], range(3), labels):
        ax.plot(t, ee_pos[:, col],  "b-",  lw=1.5, label="EE")
        ax.plot(t, tgt_pos[:, col], "r--", lw=1.5, label="Target")
        ax.set_ylabel(lbl)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    ax_e.plot(t, pos_err * 1e3, "m-", lw=1.5, label="||EE − target||")
    ax_e.set_ylabel("Position error (mm)")
    ax_e.set_xlabel("Time (s)")
    ax_e.grid(True, alpha=0.3)
    ax_e.legend(fontsize=8)
    avg_err = np.mean(pos_err) * 1e3
    ax_e.set_title(f"Mean error: {avg_err:.2f} mm", fontsize=9)

    # 3-D view: draw cylinder wireframe + EE and target trajectories
    theta_wire = np.linspace(0, 2 * np.pi, 60)
    # Two end-cap rings at x = center ± half-length
    for x_end in [CYLINDER_CENTER[0] - CYLINDER_RADIUS,
                  CYLINDER_CENTER[0] + CYLINDER_RADIUS]:
        ax_3d.plot(
            x_end * np.ones_like(theta_wire),
            CYLINDER_CENTER[1] + CYLINDER_RADIUS * np.sin(theta_wire),
            CYLINDER_CENTER[2] + CYLINDER_RADIUS * np.cos(theta_wire),
            "k-", alpha=0.15, lw=0.8,
        )

    # Lateral cylinder lines
    for th in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        ax_3d.plot(
            [CYLINDER_CENTER[0] - CYLINDER_RADIUS,
             CYLINDER_CENTER[0] + CYLINDER_RADIUS],
            [CYLINDER_CENTER[1] + CYLINDER_RADIUS * np.sin(th)] * 2,
            [CYLINDER_CENTER[2] + CYLINDER_RADIUS * np.cos(th)] * 2,
            "k-", alpha=0.15, lw=0.8,
        )

    ax_3d.plot(tgt_pos[:, 0], tgt_pos[:, 1], tgt_pos[:, 2],
               "r--", lw=1.5, label="Target arc")
    ax_3d.plot(ee_pos[:, 0],  ee_pos[:, 1],  ee_pos[:, 2],
               "b-",  lw=1.5, label="EE path")
    ax_3d.set_xlabel("X (m)"); ax_3d.set_ylabel("Y (m)"); ax_3d.set_zlabel("Z (m)")
    ax_3d.legend(fontsize=8)
    ax_3d.set_title("3-D trajectory", fontsize=9)

    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        path = f"{save_dir}/position_tracking.png"
        fig.savefig(path, dpi=150)
        print(f"[PLOT] Position tracking → {path}")


def _plot_contact_force(t, cf, normals, f_proj, f_desired, force_err, save_dir=None):
    """
    Figure 2 – Contact force.

    Row 1 : Normal-direction force projection vs desired   (main metric)
    Row 2 : Force error  (f_proj − desired)
    Row 3 : World-frame force components  Fx, Fy, Fz
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    fig.suptitle("Contact Force – Hybrid Phase on Cylinder", fontsize=13, fontweight="bold")

    # ── Row 1: normal force vs desired ─────────────────────────────────────
    axes[0].plot(t, f_proj,   "b-",  lw=1.5, label="Normal force (meas.)")
    axes[0].axhline(f_desired[0], color="r", lw=1.5, ls="--", label=f"Desired ({f_desired[0]:.1f} N)")
    axes[0].set_ylabel("Force along n̂ (N)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)
    avg_f = np.mean(f_proj)
    axes[0].set_title(f"Mean measured: {avg_f:.2f} N", fontsize=9)

    # ── Row 2: force error ──────────────────────────────────────────────────
    axes[1].plot(t, force_err, "m-", lw=1.3, label="Error = meas − desired")
    axes[1].axhline(0, color="k", lw=0.8, ls="--")
    axes[1].fill_between(t, force_err, alpha=0.15, color="m")
    axes[1].set_ylabel("Force error (N)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)
    avg_abs = np.mean(np.abs(force_err))
    axes[1].set_title(f"Mean |error|: {avg_abs:.3f} N", fontsize=9)

    # ── Row 3: raw world-frame components ────────────────────────────────────
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for i, (lbl, col) in enumerate(zip(["Fx", "Fy", "Fz"], colors)):
        axes[2].plot(t, cf[:, i], color=col, lw=1.2, label=lbl)
    axes[2].set_ylabel("World-frame force (N)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=9, ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        path = f"{save_dir}/contact_force.png"
        fig.savefig(path, dpi=150)
        print(f"[PLOT] Contact force  → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Approach + Hybrid Force Control on Cylinder Surface"
    )
    parser.add_argument("--robot", type=str, default="fr3",
                        choices=["fr3", "kuka", "fr3_friction", "fr3_jointf", "fr3_jointf_surff"],
                        help="Robot to use (default: kuka)")
    parser.add_argument("--approach-duration", type=float, default=20.0)
    parser.add_argument("--trajectory",       type=int,   default=1, choices=[1, 2],
                        help="1: θ 0°→75°  |  2: θ −75°→75°")
    parser.add_argument("--angular-speed",    type=float, default=np.pi / 4,
                        help="Angular speed in rad/s (default pi/4 ≈ 45 deg/s)")
    parser.add_argument("--force-desired",    type=float, default=-10.0,
                        help="Desired contact force (negative = pressing in)")
    parser.add_argument("--headless",         action="store_true")
    parser.add_argument("--save-plots",       action="store_true")
    parser.add_argument("--plot-dir",         type=str,
                        default="plots/run_approach_then_hybrid_cylinder")
    args = parser.parse_args()

    if args.trajectory == 1:
        theta_start    = 0.0
        theta_end      = np.radians(60.0)
    else:
        theta_start    = np.radians(-60.0)
        theta_end      = np.radians(60.0)
    sweep_duration = (theta_end - theta_start) / args.angular_speed

    print(f"[CONFIG] Trajectory {args.trajectory}: θ {np.degrees(theta_start):.1f}° → {np.degrees(theta_end):.1f}°  ({sweep_duration:.2f}s at ω={args.angular_speed:.4f} rad/s)")

    _cylinder_config_map = {
        "fr3":              "fr3_cylinder",
        "kuka":             "kuka_cylinder",
        "fr3_friction":     "fr3_friction_cylinder",
        "fr3_jointf":       "fr3_jointf_cylinder",
        "fr3_jointf_surff": "fr3_jointf_surff_cylinder",
    }
    robot_cfg = get_robot_config(_cylinder_config_map[args.robot])

    print(f"\n[CONFIG] Robot  : {robot_cfg.name}")
    print(f"[CONFIG] Scene  : {robot_cfg.mujoco_scene_xml_path}")
    print(f"[CONFIG] Pinocchio: {robot_cfg.pinocchio_xml_path}")

    # ── shared config (dt / gravity / etc.) ──────────────────────────────────
    common_config = ControllerConfig()
    common_config.dt                 = 0.001
    common_config.gravity_compensation = True
    common_config.circle_center      = CYLINDER_CENTER
    common_config.circle_radius      = CYLINDER_RADIUS
    common_config.size_z             = 0.002  # radial standoff above surface for approach
    common_config.euler              = np.array([0.0, 0.0, 0.0])  # not used for cylinder
    common_config.force_control_method = "paper"
    common_config.use_pi             = True
    common_config.circle_duration    = sweep_duration
    common_config.angular_speed      = args.angular_speed

    approach_config = CartesianSpacePDControlConfig()
    hybrid_config   = HybridControllerConfig()
    hybrid_config.F_desired_contact = np.array([args.force_desired])

    # ── initial joint configuration ───────────────────────────────────────────
    q0 = robot_cfg.q0.copy()

    # ── Pinocchio model ────────────────────────────────────────────────────────
    pino_model = pino.buildModelFromMJCF(robot_cfg.pinocchio_xml_path)
    pino_data  = pino_model.createData()
    pino_frame_id = pino_model.getFrameId(robot_cfg.ee_frame_name)

    # ── Approach target: slightly above cylinder surface at theta_start ────────
    # The approach controller targets a point 1 cm radially outward from the
    # cylinder surface.  The hybrid force controller then presses inward.
    theta0     = theta_start
    approach_normal     = np.array([0.0, np.sin(theta0), np.cos(theta0)])
    approach_target_pos = (
        CYLINDER_CENTER
        + (CYLINDER_RADIUS + common_config.size_z) * approach_normal
    )
    approach_target_rot = cylinder_ee_rotation(approach_normal)

    print(f"\n[APPROACH] Target pos : {approach_target_pos}")
    print(f"[APPROACH] Target rot :\n{approach_target_rot}")

    if not args.headless:
        input("Press Enter to start...")

    # ── MuJoCo interface (no slope XML injection, use cylinder scene directly) ─
    mujoco_interface = MujocoRobotInterface(
        common_config,
        joint_names=robot_cfg.joint_names,
        xml_path=robot_cfg.mujoco_scene_xml_path,
        add_surface=False,
        contact_geom_name="cylinder_geom",
        cylinder_center=CYLINDER_CENTER,
        cylinder_axis=CYLINDER_AXIS,
    )

    # ── Controllers ────────────────────────────────────────────────────────────
    approach_controller = CartesianSpacePDController(
        approach_config,
        common_config,
        n_joints=robot_cfg.n_joints,
        ee_frame_name=robot_cfg.ee_frame_name,
    )

    # ── Logging ────────────────────────────────────────────────────────────────
    log_contact_forces  = []
    log_desired_forces  = []
    log_ee_positions    = []
    log_target_positions = []
    log_normals         = []

    # ── Control loop ───────────────────────────────────────────────────────────
    hybrid_sim_time = 0.0
    integral_force_error = np.zeros(1)

    def run_control_loop(viewer=None):
        nonlocal hybrid_sim_time, integral_force_error

        mujoco_interface.reset_to_keyframe()
        mujoco.mj_step(mujoco_interface.model, mujoco_interface.data)
        if viewer is not None:
            mujoco.mjv_defaultFreeCamera(mujoco_interface.model, viewer.cam)

        robot_state, _ = mujoco_interface.readOnce()
        O_T_EE = np.array(robot_state.O_T_EE).reshape(4, 4).T
        start_pos = O_T_EE[:3, 3]

        # ── Phase 1 setup ────────────────────────────────────────────────────
        control_phase = ControlPhase.APPROACHING
        approach_controller.starting(
            start_pos, approach_target_pos, approach_target_rot,
            q0, pino_model, pino_data
        )
        print("\n" + "=" * 58)
        print("PHASE 1: APPROACHING CYLINDER")
        print("=" * 58)

        prev_tau = np.zeros(robot_cfg.n_joints)

        while True:
            if viewer is not None and not viewer.is_running():
                break

            step_start = time.time()
            robot_state, dt = mujoco_interface.readOnce()

            O_T_EE      = np.array(robot_state.O_T_EE).reshape(4, 4).T
            current_pos = O_T_EE[:3, 3]
            current_mat = O_T_EE[:3, :3]
            q  = np.array(robot_state.q)
            dq = np.array(robot_state.dq)

            # ── Phase state machine ──────────────────────────────────────────
            if control_phase == ControlPhase.APPROACHING:
                tau = approach_controller.update(dt, robot_state)

                if approach_controller.time_elapsed >= args.approach_duration:
                    print(f"\n[WARN] Approach timed out at {approach_controller.time_elapsed:.2f}s")
                    control_phase = ControlPhase.STOPPED

                elif approach_controller.is_target_reached(robot_state):
                    print("\n" + "=" * 58)
                    print(f"CONTACT REACHED at t={approach_controller.time_elapsed:.2f}s")
                    print("PHASE 2: HYBRID CONTROL ON CYLINDER SURFACE")
                    print("=" * 58 + "\n")
                    hybrid_sim_time = 0.0
                    integral_force_error = np.zeros(1)
                    control_phase = ControlPhase.CIRCLE_DRAWING

            elif control_phase == ControlPhase.CIRCLE_DRAWING:
                elapsed = hybrid_sim_time

                # ── Trajectory ───────────────────────────────────────────────
                target_pos, x_dot_des, x_ddot_des, theta = cylinder_trajectory(
                    elapsed, args.angular_speed, theta_start
                )

                if theta >= theta_end:
                    print(f"\nSweep finished: θ={np.degrees(theta):.1f}° at t={elapsed:.2f}s")
                    control_phase = ControlPhase.STOPPED
                    continue

                # ── Dynamic surface geometry ─────────────────────────────────
                normal  = cylinder_surface_normal(current_pos)
                S_f, S_v = cylinder_selection_matrices(normal)
                target_rot = cylinder_ee_rotation(normal)

                # ── Kinematics / Dynamics ────────────────────────────────────
                pino.forwardKinematics(pino_model, pino_data, q, dq)
                pino.computeJointJacobians(pino_model, pino_data)
                pino.updateFramePlacements(pino_model, pino_data)
                jac = pino.getFrameJacobian(
                    pino_model, pino_data, pino_frame_id, pino.LOCAL_WORLD_ALIGNED
                )
                M_inv = pino.computeMinverse(pino_model, pino_data, q)

                J_phi   = S_f.T @ jac           # (1 × n_joints)
                J_motion = S_v.T @ jac           # (5 × n_joints)
                jac_1   = np.vstack([J_phi, J_motion])

                Mx_constraint = task_space_inertiaM(M_inv, J_phi)
                Mx_motion     = task_space_inertiaM(M_inv, J_motion)

                # ── External force ───────────────────────────────────────────
                F_ext       = np.array(robot_state.O_F_ext_hat_K)  # world frame
                F_ext_phi   = F_ext @ S_f   # (1,) pressing → negative
                F_ext_x     = F_ext @ S_v   # (5,)

                # ── Null-space torque ─────────────────────────────────────────
                jac_1_inv  = dynamically_consistent_inv(jac_1, M_inv)
                N2         = np.eye(robot_cfg.n_joints) - jac_1.T @ jac_1_inv.T
                tau_null   = null_space_tau(q, dq, q0,
                                            hybrid_config.Kp_null,
                                            hybrid_config.Kd_null)
                tau_ctrl_v = N2 @ tau_null

                # ── Motion-space control ──────────────────────────────────────
                twist = compute_ee_pose_error(
                    target_pos, current_pos, target_rot, current_mat.flatten()
                )
                x_ddot_sel   = np.concatenate([x_ddot_des, [0, 0, 0]]) @ S_v
                x_tilde      = twist @ S_v
                site_vel     = jac @ dq
                x_dot_tilde  = (np.concatenate([x_dot_des, [0, 0, 0]]) - site_vel) @ S_v

                a_motion     = feedforward_PD(
                    x_acc_desired=x_ddot_sel,
                    x_delta=x_tilde,
                    x_dot_delta=x_dot_tilde,
                    Kp=hybrid_config.Kp @ S_v,
                    Kd=hybrid_config.Kd @ S_v,
                )
                tau_ctrl_x   = J_motion.T @ (Mx_motion @ a_motion)

                # ── Constraint-space (force) control — Bruno paper method ─────
                C       = pino.computeCoriolisMatrix(pino_model, pino_data, q, dq)
                J_dot   = pino.getFrameJacobianTimeVariation(
                    pino_model, pino_data, pino_frame_id, pino.LOCAL_WORLD_ALIGNED
                )
                J_phi_dot = S_f.T @ J_dot

                F_ext_x_trans = F_ext_x.copy()
                F_ext_x_trans[-3:] = 0          # ignore torque components

                ctrl_comp    = -Mx_constraint @ J_phi @ M_inv @ (tau_ctrl_x + tau_ctrl_v)
                contact_comp =  Mx_constraint @ J_phi @ M_inv @ (J_motion.T @ F_ext_x_trans)
                vel_term     =  Mx_constraint @ (J_phi @ M_inv @ C - J_phi_dot) @ dq

                F_ctrl = (
                    hybrid_config.F_desired_contact
                    + ctrl_comp
                    + contact_comp
                    + vel_term
                )

                # PI correction on top
                pi, integral_force_error = PI_term(
                    F_ext_phi,
                    hybrid_config.F_desired_contact,
                    common_config.dt,
                    integral_force_error,
                    kp=hybrid_config.Kp_force,
                    ki=hybrid_config.Ki_force,
                )
                F_ctrl = F_ctrl + pi

                tau_ctrl_phi = J_phi.T @ F_ctrl
                tau = tau_ctrl_phi + tau_ctrl_x + tau_ctrl_v

                # Gravity compensation
                if common_config.gravity_compensation:
                    tau += pino.computeGeneralizedGravity(pino_model, pino_data, q)

                # Torque rate limiting
                delta = np.clip(
                    tau - prev_tau,
                    -hybrid_config.max_delta_tau,
                     hybrid_config.max_delta_tau
                )
                tau = prev_tau + delta
                prev_tau = tau.copy()

                hybrid_sim_time += common_config.dt

                # ── Logging ──────────────────────────────────────────────────
                log_contact_forces.append(F_ext[:3].copy())
                log_desired_forces.append(hybrid_config.F_desired_contact.copy())
                log_ee_positions.append(current_pos.copy())
                log_target_positions.append(target_pos.copy())
                log_normals.append(normal.copy())

            else:  # STOPPED
                tau = pino.computeGeneralizedGravity(pino_model, pino_data, q)
                cmd = Torques(tau.tolist())
                cmd.motion_finished = True
                mujoco_interface.writeOnce(cmd)
                break

            cmd = Torques(tau.tolist())
            mujoco_interface.writeOnce(cmd)

            if viewer is not None:
                viewer.sync()

            elapsed_step = time.time() - step_start
            sleep_time = common_config.dt - elapsed_step
            if sleep_time > 0:
                time.sleep(sleep_time)

    try:
        if args.headless:
            run_control_loop()
        else:
            with mujoco.viewer.launch_passive(
                mujoco_interface.model, mujoco_interface.data,
                show_left_ui=False, show_right_ui=False,
            ) as viewer:
                run_control_loop(viewer)
    except Exception as exc:
        import traceback
        print(f"\nError: {exc}")
        traceback.print_exc()
        return

    # ── Metrics & Plots ────────────────────────────────────────────────────────
    if not log_contact_forces:
        print("\n[DONE] No hybrid-phase data collected.")
        return

    cf  = np.array(log_contact_forces)   # (N, 3) world-frame force
    df  = np.array(log_desired_forces)   # (N, 1)
    ep  = np.array(log_ee_positions)     # (N, 3)
    tp  = np.array(log_target_positions) # (N, 3)
    nor = np.array(log_normals)          # (N, 3)
    dt  = common_config.dt
    t   = np.arange(len(cf)) * dt

    # Force projected onto outward surface normal (negative when pressing)
    f_normal_proj   = np.einsum('ij,ij->i', cf, nor)   # dot product per row
    f_desired_scalar = df[:, 0]
    force_err = f_normal_proj - f_desired_scalar

    pos_err = np.linalg.norm(ep - tp, axis=1)

    print(f"\nAVG_FORCE_ERROR   : {np.mean(np.abs(force_err)):.6f} N")
    print(f"VAR_FORCE_ERROR   : {np.var(force_err):.6f}")
    print(f"AVG_POSITION_ERROR: {np.mean(pos_err):.6f} m")
    print(f"VAR_POSITION_ERROR: {np.var(pos_err):.6f}")

    _plot_position_tracking(t, ep, tp, pos_err,
                            args.plot_dir if args.save_plots else None)
    _plot_contact_force(t, cf, nor, f_normal_proj, f_desired_scalar, force_err,
                        args.plot_dir if args.save_plots else None)

    import matplotlib.pyplot as plt
    plt.show()

    print("\n[DONE] Cylinder hybrid control finished.")


if __name__ == "__main__":
    main()
