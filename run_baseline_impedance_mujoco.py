# ------------------------------------------------------------------------------
# Approach + Impedance Force-Control Script
#
# Phase 1: Move end-effector to target surface (CartesianSpacePDController)
# Phase 2: Impedance + feedforward force (ImpedanceController)
#   twist      = [Kpos/dt * dx,  Kori/dt * d_ori]
#   y          = jac_inv @ Md_inv @ (Kp * twist - Kd * (jac @ dq))
#   tau_motion = M @ y  +  tau_null
#   tau_f      = jac.T @ (force_mag * n)  [+ PI]
#   tau        = tau_motion + tau_f + qfrc_bias
#
# Supports slope (flat/angled) and cylinder surface (--cylinder).
# Supports fr3, kuka, panda, fr3_friction, fr3_jointf, fr3_jointf_surff.
# ------------------------------------------------------------------------------
import argparse
import mujoco
import mujoco.viewer
import numpy as np
import os
import re
import time
import pinocchio as pino
from scipy.spatial.transform import Rotation

from src import (
    ControllerConfig,
    CartesianSpacePDController,
    CartesianSpacePDControlConfig,
    ControlPhase,
    get_robot_config,
)
from src.impedance_controller import ImpedanceController, ImpedanceControllerConfig
from src.cylinder_helper import (
    CYLINDER_CENTER, CYLINDER_AXIS, CYLINDER_RADIUS,
    CYLINDER_CONFIG_MAP,
    get_cylinder_approach_target,
    plot_cylinder_position_tracking,
    plot_cylinder_contact_force,
)
from src.trajectory import CylinderTrajectory
from src.hybrid_controller import generate_circle_trajectory
from utils_plot import plot_ee_positions, plot_joint_torques, plot_force_error_z
from utils_libfranka import euler_to_rot_matrix, generate_start_position
from mujoco_robot_interface import MujocoRobotInterface, Torques


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Approach then Impedance Force-Control"
    )
    parser.add_argument(
        "--robot", type=str, default="fr3",
        choices=["fr3", "kuka", "panda", "fr3_friction", "fr3_jointf", "fr3_jointf_surff"],
        help="Robot type (default: fr3)"
    )
    parser.add_argument(
        "--approach-duration", type=float, default=20.0,
        help="Max approach phase duration in seconds (default: 20.0)"
    )
    parser.add_argument(
        "--circle-duration", type=float, default=10.0,
        help="Duration of circle/sweep motion in seconds (default: 10.0)"
    )
    parser.add_argument(
        "--angular-speed", type=float, default=np.pi * 2,
        help="Angular speed for circle/sweep in rad/s (default: pi*2)"
    )
    parser.add_argument(
        "--force-desired", type=float, default=-8.0, dest="force_desired",
        help="Desired contact force magnitude in N, negative = pressing (default: -8.0)"
    )
    parser.add_argument(
        "--use-pi", action="store_true",
        help="Add PI force correction on top of feedforward"
    )
    parser.add_argument(
        "--kp-force", type=float, default=None,
        help="Proportional gain for PI force control (overrides config default)"
    )
    parser.add_argument(
        "--ki-force", type=float, default=None,
        help="Integral gain for PI force control (overrides config default)"
    )
    parser.add_argument(
        "--slope-angle", type=float, default=30.0,
        help="Slope angle in degrees around X axis (default: 30.0)"
    )
    parser.add_argument(
        "--cylinder", action="store_true",
        help="Use cylinder surface instead of flat slope"
    )
    parser.add_argument(
        "--trajectory", type=int, default=1, choices=[1, 2],
        help="Cylinder sweep: 1 → θ 0°→75°, 2 → θ −75°→75° (only with --cylinder)"
    )
    parser.add_argument(
        "--skip-seconds", type=float, default=1.0,
        help="Seconds to skip at start when computing metrics (default: 1.0)"
    )
    parser.add_argument(
        "--surface-friction", type=float, default=None, dest="surface_friction",
        help="Override sliding friction coefficient (requires --robot fr3_friction)"
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without MuJoCo viewer"
    )
    parser.add_argument(
        "--save-plots", action="store_true",
        help="Save plots after simulation"
    )
    parser.add_argument(
        "--plot-dir", type=str, default="plots/run_baseline_impedance_mujoco",
        help="Directory to save plots"
    )
    parser.add_argument(
        "--save-data", action="store_true",
        help="Save time-series data to .npz file"
    )
    parser.add_argument(
        "--data-dir", type=str, default="",
        help="Directory to save .npz data file (used with --save-data)"
    )
    parser.add_argument(
        "--multiplier", type=float, default=0.0,
        help="Speed multiplier label used in saved filename"
    )
    args = parser.parse_args()

    # ============================================================
    # 1. Robot Configuration
    # ============================================================
    if args.cylinder:
        robot_cfg = get_robot_config(CYLINDER_CONFIG_MAP[args.robot])
        theta_start = 0.0 if args.trajectory == 1 else np.radians(-75.0)
        theta_end   = np.radians(75.0)
        sweep_duration = (theta_end - theta_start) / args.angular_speed
        print(f"[CONFIG] Cylinder sweep: θ {np.degrees(theta_start):.1f}° → "
              f"{np.degrees(theta_end):.1f}°  ({sweep_duration:.2f}s)")
    else:
        robot_cfg = get_robot_config(args.robot)

    if args.surface_friction is not None:
        mu = args.surface_friction
        base_xml = robot_cfg.pinocchio_xml_path
        with open(base_xml, 'r') as _f:
            robot_xml_str = _f.read()
        assets_dir = os.path.abspath(os.path.join(os.path.dirname(base_xml), 'assets'))
        robot_xml_str = robot_xml_str.replace('meshdir="assets"', f'meshdir="{assets_dir}"')
        robot_xml_str = re.sub(
            r'(friction=")[0-9.]+( [0-9.]+ [0-9.]+")',
            rf'\g<1>{mu:.4f}\2',
            robot_xml_str
        )
        tmp_dir = f"/tmp/mj_ctrl_friction_{mu:.4f}"
        os.makedirs(tmp_dir, exist_ok=True)
        robot_tmp_path = os.path.join(tmp_dir, "robot.xml")
        with open(robot_tmp_path, 'w') as _f:
            _f.write(robot_xml_str)
        scene_tmp_content = (
            '<mujoco model="fr3 scene">\n'
            '  <include file="robot.xml"/>\n\n'
            '  <statistic center="0.2 0 0.4" extent=".8"/>\n\n'
            '  <visual>\n'
            '    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>\n'
            '    <rgba haze="0.15 0.25 0.35 1"/>\n'
            '    <global azimuth="120" elevation="-20"/>\n'
            '  </visual>\n\n'
            '  <asset>\n'
            '    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>\n'
            '    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"\n'
            '      markrgb="0.8 0.8 0.8" width="300" height="300"/>\n'
            '    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>\n'
            '  </asset>\n\n'
            '  <worldbody>\n'
            '    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>\n'
            '    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>\n'
            '  </worldbody>\n'
            '</mujoco>\n'
        )
        scene_tmp_path = os.path.join(tmp_dir, "scene.xml")
        with open(scene_tmp_path, 'w') as _f:
            _f.write(scene_tmp_content)
        robot_cfg.mujoco_scene_xml_path = scene_tmp_path
        print(f"[CONFIG] Surface friction override: mu={mu:.4f}")

    print(f"\n[CONFIG] Robot       : {robot_cfg.name}")
    print(f"[CONFIG] Pinocchio   : {robot_cfg.pinocchio_xml_path}")
    print(f"[CONFIG] MuJoCo scene: {robot_cfg.mujoco_scene_xml_path}")

    # ============================================================
    # 2. Configurations
    # ============================================================
    common_config = ControllerConfig(circle_duration=args.circle_duration)
    common_config.size_z             = 0.01
    common_config.gravity_compensation = True
    common_config.angular_speed      = args.angular_speed
    common_config.use_pi             = args.use_pi
    common_config.euler              = np.array([np.deg2rad(args.slope_angle), 0.0, 0.0])
    if args.cylinder:
        common_config.circle_center  = CYLINDER_CENTER
        common_config.circle_radius  = CYLINDER_RADIUS
        common_config.size_z         = 0.002
        common_config.circle_duration = sweep_duration

    approach_config   = CartesianSpacePDControlConfig()
    impedance_config  = ImpedanceControllerConfig(force_mag=args.force_desired)
    if args.kp_force is not None:
        impedance_config.Kp_force = args.kp_force
    if args.ki_force is not None:
        impedance_config.Ki_force = args.ki_force

    q0 = robot_cfg.q0.copy() if args.cylinder else \
        np.array([0.0225, 0.7064, -0.0243, -2.3135, -0.0095, 3.0422, -0.2441])

    # ============================================================
    # 3. Pinocchio model  (approach phase only)
    # ============================================================
    pino_model = pino.buildModelFromMJCF(robot_cfg.pinocchio_xml_path)
    pino_data  = pino_model.createData()

    try:
        print("\n" + "=" * 60)
        print("APPROACH + IMPEDANCE FORCE-CONTROL")
        print("=" * 60)
        print(f"Robot        : {robot_cfg.name.upper()}")
        print(f"Surface      : {'cylinder' if args.cylinder else f'slope {args.slope_angle}°'}")
        print(f"Force desired: {args.force_desired} N")
        print(f"PI enabled   : {args.use_pi}")
        print("=" * 60)
        if not args.headless:
            input("Press Enter to continue...")

        # ============================================================
        # 4. Controllers
        # ============================================================
        approach_controller = CartesianSpacePDController(
            approach_config, common_config,
            n_joints=robot_cfg.n_joints,
            ee_frame_name=robot_cfg.ee_frame_name,
        )

        impedance_controller = ImpedanceController(
            impedance_config, common_config,
            n_joints=robot_cfg.n_joints,
            site_name="attachment_site",
            trajectory=CylinderTrajectory(theta_start, theta_end, args.angular_speed)
                       if args.cylinder else None,
        )

        # ============================================================
        # 5. Approach target
        # ============================================================
        if args.cylinder:
            target_pos, _cylinder_approach_rot = get_cylinder_approach_target(
                theta_start, common_config.size_z
            )
        else:
            R_slope = euler_to_rot_matrix(common_config.euler)
            target_pos = generate_start_position(
                common_config.circle_radius,
                common_config.circle_center,
                common_config.size_z,
                R_slope,
            )

        # ============================================================
        # 6. MuJoCo interface
        # ============================================================
        print("\nStarting torque control...")
        if args.cylinder:
            mujoco_interface = MujocoRobotInterface(
                common_config,
                joint_names=robot_cfg.joint_names,
                xml_path=robot_cfg.mujoco_scene_xml_path,
                add_surface=False,
                contact_geom_name="cylinder_geom",
                cylinder_center=CYLINDER_CENTER,
                cylinder_axis=CYLINDER_AXIS,
            )
        else:
            mujoco_interface = MujocoRobotInterface(
                common_config,
                joint_names=robot_cfg.joint_names,
                xml_path=robot_cfg.mujoco_scene_xml_path,
            )

        # ============================================================
        # 7. Control loop
        # ============================================================
        hybrid_sim_time = 0.0

        def run_control_loop(sync_viewer=None):
            nonlocal hybrid_sim_time

            mujoco_interface.reset_to_keyframe()
            mujoco.mj_step(mujoco_interface.model, mujoco_interface.data)
            if sync_viewer is not None:
                mujoco.mjv_defaultFreeCamera(mujoco_interface.model, sync_viewer.cam)

            robot_state, duration = mujoco_interface.readOnce()
            O_T_EE = np.array(robot_state.O_T_EE).reshape(4, 4).T
            start_pos = O_T_EE[:3, 3]

            # Approach target orientation
            if args.cylinder:
                target_rot = _cylinder_approach_rot
            else:
                rot_slope   = Rotation.from_euler('xyz', common_config.euler)
                rot_default = Rotation.from_quat(np.roll(robot_cfg.target_quat, -1))
                target_rot  = (rot_slope * rot_default).as_matrix()

            control_phase = ControlPhase.APPROACHING
            approach_controller.starting(start_pos, target_pos, target_rot, q0, pino_model, pino_data)

            print("\n" + "=" * 60)
            print("PHASE 1: APPROACHING TARGET POSITION")
            print("=" * 60)

            while True:
                if sync_viewer is not None and not sync_viewer.is_running():
                    break

                step_start = time.time()
                robot_state, duration = mujoco_interface.readOnce()

                O_T_EE      = np.array(robot_state.O_T_EE).reshape(4, 4).T
                current_pos = O_T_EE[:3, 3]
                current_mat = O_T_EE[:3, :3]

                # ── State machine ─────────────────────────────────────────────
                if control_phase == ControlPhase.APPROACHING:
                    tau = approach_controller.update(duration, robot_state)

                    if approach_controller.time_elapsed >= args.approach_duration:
                        print(f"\n[WARN] Approach timed out at "
                              f"{approach_controller.time_elapsed:.2f}s")
                        control_phase = ControlPhase.STOPPED

                    elif approach_controller.is_target_reached(robot_state):
                        print("\n" + "=" * 60)
                        print(f"CONTACT REACHED at t={approach_controller.time_elapsed:.2f}s!")
                        print("PHASE 2: IMPEDANCE FORCE-CONTROL")
                        print("=" * 60 + "\n")

                        O_T_EE = np.array(robot_state.O_T_EE).reshape(4, 4).T
                        surface_target_rot = O_T_EE[:3, :3]
                        surface_q0 = np.array(robot_state.q)
                        impedance_controller.starting(
                            hybrid_sim_time,
                            surface_target_rot,
                            surface_q0,
                            mujoco_interface.model,
                            mujoco_interface.data,
                            mujoco_interface.site_id,
                            mujoco_interface.dof_ids,
                        )
                        impedance_controller.target_pos = O_T_EE[:3, 3].copy()
                        control_phase = ControlPhase.CIRCLE_DRAWING

                elif control_phase == ControlPhase.CIRCLE_DRAWING:
                    elapsed = hybrid_sim_time

                    # For slope: drive circle trajectory externally
                    if not args.cylinder:
                        if elapsed < common_config.circle_duration:
                            tp, xd, xdd = generate_circle_trajectory(
                                elapsed,
                                common_config.circle_center,
                                common_config.circle_radius,
                                common_config.angular_speed,
                                R_slope,
                                common_config.size_z,
                            )
                            impedance_controller.target_pos        = tp
                            impedance_controller.x_dot_desired[:]  = xd
                            impedance_controller.x_ddot_desired[:] = xdd
                        else:
                            impedance_controller.x_dot_desired[:]  = 0.0
                            impedance_controller.x_ddot_desired[:] = 0.0
                            impedance_controller.is_drawing = False

                    tau = impedance_controller.update(hybrid_sim_time, robot_state)
                    hybrid_sim_time += common_config.dt

                    if impedance_controller.is_finished():
                        print("\n" + "=" * 60)
                        print(f"IMPEDANCE CONTROL FINISHED at t={hybrid_sim_time:.2f}s!")
                        print("=" * 60)
                        control_phase = ControlPhase.STOPPED

                else:  # STOPPED
                    tau = pino.computeGeneralizedGravity(
                        pino_model, pino_data, np.array(robot_state.q)
                    )
                    cmd = Torques(tau.tolist())
                    cmd.motion_finished = True
                    mujoco_interface.writeOnce(cmd)
                    break

                cmd = Torques(tau.tolist())
                mujoco_interface.writeOnce(cmd)
                if sync_viewer is not None:
                    sync_viewer.sync()

                time_left = common_config.dt - (time.time() - step_start)
                if time_left > 0:
                    time.sleep(time_left)

        if args.headless:
            run_control_loop(sync_viewer=None)
        else:
            with mujoco.viewer.launch_passive(
                mujoco_interface.model, mujoco_interface.data,
                show_left_ui=False, show_right_ui=False,
            ) as viewer:
                run_control_loop(sync_viewer=viewer)

        # ============================================================
        # 8. Metrics
        # ============================================================
        skip_samples = int(args.skip_seconds / common_config.dt)

        contact_forces   = np.array(impedance_controller.contact_forces) \
            if impedance_controller.contact_forces else np.empty((0, 3))
        desired_forces   = np.array(impedance_controller.desired_forces) \
            if impedance_controller.desired_forces else np.empty((0, 1))
        normals_arr      = np.array(impedance_controller.normals) \
            if impedance_controller.normals else np.empty((0, 3))
        ee_positions     = np.array(impedance_controller.ee_positions) \
            if impedance_controller.ee_positions else np.empty((0, 3))
        target_positions = np.array(impedance_controller.target_positions) \
            if impedance_controller.target_positions else np.empty((0, 3))

        if contact_forces.size > 0 and normals_arr.size > 0:
            cf_ss  = contact_forces[skip_samples:]
            df_ss  = desired_forces[skip_samples:]
            nor_ss = normals_arr[skip_samples:]
            if cf_ss.shape[0] > 0:
                force_proj = np.einsum('ij,ij->i', cf_ss, nor_ss)
                force_err  = force_proj - df_ss[:, 0]
                print(f"AVG_FORCE_ERROR   : {np.mean(np.abs(force_err)):.6f}")
                print(f"VAR_FORCE_ERROR   : {np.var(force_err):.6f}")
            else:
                print("AVG_FORCE_ERROR   : nan")
                print("VAR_FORCE_ERROR   : nan")
        else:
            print("AVG_FORCE_ERROR   : nan")
            print("VAR_FORCE_ERROR   : nan")

        if ee_positions.size > 0 and target_positions.size > 0 \
                and ee_positions.shape == target_positions.shape:
            ep_ss = ee_positions[skip_samples:]
            tp_ss = target_positions[skip_samples:]
            if ep_ss.shape[0] > 0:
                pos_err = np.linalg.norm(ep_ss - tp_ss, axis=1)
                print(f"AVG_POSITION_ERROR: {np.mean(pos_err):.6f}")
                print(f"VAR_POSITION_ERROR: {np.var(pos_err):.6f}")
            else:
                print("AVG_POSITION_ERROR: nan")
                print("VAR_POSITION_ERROR: nan")
        else:
            print("AVG_POSITION_ERROR: nan")
            print("VAR_POSITION_ERROR: nan")

        # ============================================================
        # 9. Save data
        # ============================================================
        if args.save_data and args.data_dir:
            os.makedirs(args.data_dir, exist_ok=True)

            actual_vel  = np.gradient(ee_positions,     common_config.dt, axis=0) \
                if ee_positions.size > 0 else np.empty((0, 3))
            desired_vel = np.gradient(target_positions, common_config.dt, axis=0) \
                if target_positions.size > 0 else np.empty((0, 3))

            if contact_forces.size > 0 and normals_arr.size > 0:
                force_error_full = (
                    np.einsum('ij,ij->i', contact_forces, normals_arr)
                    - desired_forces[:, 0]
                )
            else:
                force_error_full = np.empty(0)

            if ee_positions.size > 0 and target_positions.size > 0 \
                    and ee_positions.shape == target_positions.shape:
                pos_error_full = np.linalg.norm(ee_positions - target_positions, axis=1)
            else:
                pos_error_full = np.empty(0)

            fname = f"data_{args.multiplier:.1f}.npz"
            fpath = os.path.join(args.data_dir, fname)
            np.savez(
                fpath,
                force_error=force_error_full,
                position_error=pos_error_full,
                actual_positions=ee_positions,
                desired_positions=target_positions,
                actual_velocitys=actual_vel,
                desired_velocitys=desired_vel,
                multiplier=np.array(args.multiplier),
                angular_speed_rad_s=np.array(args.angular_speed),
            )
            print(f"DATA_SAVED: {fpath}")

        # ============================================================
        # 10. Plots
        # ============================================================
        if args.save_plots:
            print("\n[MAIN] Generating plots...")
            plot_dir = args.plot_dir
            t = np.arange(len(contact_forces)) * common_config.dt

            if args.cylinder and contact_forces.size > 0:
                import matplotlib.pyplot as plt
                force_proj_full = np.einsum('ij,ij->i', contact_forces, normals_arr)
                force_err_full  = force_proj_full - desired_forces[:, 0]
                pos_err_full    = np.linalg.norm(ee_positions - target_positions, axis=1) \
                    if ee_positions.shape == target_positions.shape else np.zeros(len(t))
                plot_cylinder_position_tracking(t, ee_positions, target_positions,
                                                pos_err_full, save_dir=plot_dir)
                plot_cylinder_contact_force(t, contact_forces, normals_arr,
                                            force_proj_full, desired_forces[:, 0],
                                            force_err_full, save_dir=plot_dir)
            else:
                plot_joint_torques(impedance_controller, "joint_torques",
                                   common_config.dt, plot_dir=plot_dir)
                plot_ee_positions(impedance_controller, common_config.dt, plot_dir=plot_dir)
                plot_force_error_z(impedance_controller, common_config.dt,
                                   robot_cfg.name, plot_dir=plot_dir)

            if not args.headless:
                import matplotlib.pyplot as plt
                plt.show()

        print("\n[MAIN] Impedance control finished")
        print(f"Approach time : {approach_controller.time_elapsed:.2f}s")
        print(f"Surface time  : {hybrid_sim_time:.2f}s")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return -1


if __name__ == "__main__":
    main()
