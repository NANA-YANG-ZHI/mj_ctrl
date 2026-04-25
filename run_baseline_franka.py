# ------------------------------------------------------------------------------
# Baseline Force-Impedance Control Script (Real Robot via libfranka)
#
# Baseline full-space control (BaselineController):
#   tau = J^T*Mx*(x_ddot + Kp*dx + Kd*dxdot) + J^T*(force_mag*n) [+PI] + g
#
# Assumes the robot is already in contact with the surface before running.
# Supports slope (flat/angled) and cylinder surface (--cylinder).
# ------------------------------------------------------------------------------
import argparse
import gc
import numpy as np
import os
import pinocchio as pino
from pylibfranka import Robot, Torques

from src import (
    ControllerConfig,
    ControlPhase,
    get_robot_config,
)
from src.baseline_controller import BaselineController, BaselineControllerConfig
from src.cylinder_helper import (
    CYLINDER_CENTER, CYLINDER_RADIUS,
    plot_cylinder_position_tracking,
    plot_cylinder_contact_force,
)
from src.trajectory import CylinderTrajectory
from src.hybrid_controller import generate_circle_trajectory
from utils_plot import plot_ee_positions, plot_joint_torques, plot_force_error_z
from utils_libfranka import euler_to_rot_matrix


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline Force-Impedance Control (real robot)"
    )
    parser.add_argument("--ip", type=str, default="localhost", help="Robot IP address")
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
        "--save-plots", action="store_true",
        help="Save plots after run"
    )
    parser.add_argument(
        "--plot-dir", type=str, default="plots/run_baseline_franka",
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
    # 1. Robot Configuration (always fr3 on real hardware)
    # ============================================================
    robot_cfg = get_robot_config("fr3")

    if args.cylinder:
        theta_start = 0.0 if args.trajectory == 1 else np.radians(-75.0)
        theta_end   = np.radians(75.0)
        sweep_duration = (theta_end - theta_start) / args.angular_speed
        print(f"[CONFIG] Cylinder sweep: θ {np.degrees(theta_start):.1f}° → "
              f"{np.degrees(theta_end):.1f}°  ({sweep_duration:.2f}s)")

    print(f"\n[CONFIG] Robot     : {robot_cfg.name}")
    print(f"[CONFIG] Pinocchio : {robot_cfg.pinocchio_xml_path}")

    # ============================================================
    # 2. Configurations
    # ============================================================
    common_config = ControllerConfig(circle_duration=args.circle_duration)
    common_config.size_z               = 0.01
    common_config.gravity_compensation = True
    common_config.angular_speed        = args.angular_speed
    common_config.use_pi               = args.use_pi
    common_config.euler                = np.array([np.deg2rad(args.slope_angle), 0.0, 0.0])
    if args.cylinder:
        common_config.circle_center  = CYLINDER_CENTER
        common_config.circle_radius  = CYLINDER_RADIUS
        common_config.size_z         = 0.002
        common_config.circle_duration = sweep_duration

    baseline_config = BaselineControllerConfig(force_mag=args.force_desired)
    if args.kp_force is not None:
        baseline_config.Kp_force = args.kp_force
    if args.ki_force is not None:
        baseline_config.Ki_force = args.ki_force

    q0 = np.array([0.0225, 0.7064, -0.0243, -2.3135, -0.0095, 3.0422, -0.2441])

    if not args.cylinder:
        R_slope = euler_to_rot_matrix(common_config.euler)

    # ============================================================
    # 3. Pinocchio model
    # ============================================================
    pino_model = pino.buildModelFromMJCF(robot_cfg.pinocchio_xml_path)
    pino_data  = pino_model.createData()

    robot = None
    try:
        print("\n" + "=" * 60)
        print("BASELINE FORCE-IMPEDANCE CONTROL")
        print("=" * 60)
        print(f"Robot        : {robot_cfg.name.upper()}")
        print(f"Surface      : {'cylinder' if args.cylinder else f'slope {args.slope_angle}°'}")
        print(f"Force desired: {args.force_desired} N")
        print(f"PI enabled   : {args.use_pi}")
        print("=" * 60)

        print(f"\nConnecting to robot at {args.ip}...")
        robot = Robot(args.ip)

        robot.set_collision_behavior(
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        )

        print("\nWARNING: This will move the robot!")
        print("Make sure:")
        print("  1. The workspace is clear")
        print("  2. Emergency stop is accessible")
        print("  3. Robot is already in contact with the surface")
        input("Press Enter to continue...")

        # ============================================================
        # 4. Controller
        # ============================================================
        baseline_controller = BaselineController(
            baseline_config, common_config,
            n_joints=robot_cfg.n_joints,
            ee_frame_name=robot_cfg.ee_frame_name,
            trajectory=CylinderTrajectory(theta_start, theta_end, args.angular_speed)
                       if args.cylinder else None,
        )

        # ============================================================
        # 5. Start torque control
        # ============================================================
        print("\nStarting torque control...")
        active_control = robot.start_torque_control()
        robot_state, duration = active_control.readOnce()
        O_T_EE = np.array(robot_state.O_T_EE).reshape(4, 4).T

        baseline_controller.starting(
            0.0, O_T_EE[:3, :3], np.array(robot_state.q),
            pino_model, pino_data,
        )
        baseline_controller.target_pos = O_T_EE[:3, 3].copy()

        # ============================================================
        # 6. Pinocchio warmup (trigger lazy init before real-time loop)
        # ============================================================
        _warmup_q  = np.array(q0)
        _warmup_dq = np.zeros(robot_cfg.n_joints)
        pino.forwardKinematics(pino_model, pino_data, _warmup_q, _warmup_dq)
        pino.computeJointJacobians(pino_model, pino_data)
        pino.updateFramePlacements(pino_model, pino_data)
        _warmup_frame_id = pino_model.getFrameId(robot_cfg.ee_frame_name)
        pino.getFrameJacobian(pino_model, pino_data, _warmup_frame_id, pino.LOCAL_WORLD_ALIGNED)
        pino.computeMinverse(pino_model, pino_data, _warmup_q)
        pino.crba(pino_model, pino_data, _warmup_q)
        pino.computeGeneralizedGravity(pino_model, pino_data, _warmup_q)
        pino.computeCoriolisMatrix(pino_model, pino_data, _warmup_q, _warmup_dq)
        pino.getFrameJacobianTimeVariation(pino_model, pino_data, _warmup_frame_id, pino.LOCAL_WORLD_ALIGNED)
        del _warmup_q, _warmup_dq, _warmup_frame_id

        gc.collect()
        gc.disable()

        print("\n" + "=" * 60)
        print("BASELINE FORCE-IMPEDANCE CONTROL RUNNING")
        print("=" * 60)

        hybrid_sim_time = 0.0
        control_phase = ControlPhase.CIRCLE_DRAWING

        # ============================================================
        # 7. Control loop
        # ============================================================
        try:
            while True:
                robot_state, duration = active_control.readOnce()

                if control_phase == ControlPhase.CIRCLE_DRAWING:
                    if not args.cylinder:
                        if hybrid_sim_time < common_config.circle_duration:
                            tp, xd, xdd = generate_circle_trajectory(
                                hybrid_sim_time,
                                common_config.circle_center,
                                common_config.circle_radius,
                                common_config.angular_speed,
                                R_slope,
                                common_config.size_z,
                            )
                            baseline_controller.target_pos        = tp
                            baseline_controller.x_dot_desired[:]  = xd
                            baseline_controller.x_ddot_desired[:] = xdd
                        else:
                            baseline_controller.x_dot_desired[:]  = 0.0
                            baseline_controller.x_ddot_desired[:] = 0.0
                            baseline_controller.is_drawing = False

                    tau = baseline_controller.update(hybrid_sim_time, robot_state)
                    hybrid_sim_time += duration.to_sec()

                    if baseline_controller.is_finished():
                        print("\n" + "=" * 60)
                        print(f"BASELINE CONTROL FINISHED at t={hybrid_sim_time:.2f}s!")
                        print("=" * 60)
                        control_phase = ControlPhase.STOPPED

                else:  # STOPPED
                    tau = pino.computeGeneralizedGravity(
                        pino_model, pino_data, np.array(robot_state.q)
                    )
                    cmd = Torques(tau.tolist())
                    cmd.motion_finished = True
                    active_control.writeOnce(cmd)
                    break

                cmd = Torques(tau.tolist())
                active_control.writeOnce(cmd)

        except KeyboardInterrupt:
            gc.enable()
            print("\nControl interrupted by user")
            cmd = Torques([0.0] * robot_cfg.n_joints)
            cmd.motion_finished = True
            active_control.writeOnce(cmd)

        gc.enable()
        gc.collect()

        # ============================================================
        # 8. Metrics
        # ============================================================
        skip_samples = int(args.skip_seconds / common_config.dt)

        contact_forces   = np.array(baseline_controller.contact_forces) \
            if baseline_controller.contact_forces else np.empty((0, 3))
        desired_forces   = np.array(baseline_controller.desired_forces) \
            if baseline_controller.desired_forces else np.empty((0, 1))
        normals_arr      = np.array(baseline_controller.normals) \
            if baseline_controller.normals else np.empty((0, 3))
        ee_positions     = np.array(baseline_controller.ee_positions) \
            if baseline_controller.ee_positions else np.empty((0, 3))
        target_positions = np.array(baseline_controller.target_positions) \
            if baseline_controller.target_positions else np.empty((0, 3))

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

            actual_vel  = np.gradient(ee_positions,   common_config.dt, axis=0) \
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
                plot_joint_torques(baseline_controller, "joint_torques",
                                   common_config.dt, plot_dir=plot_dir)
                plot_ee_positions(baseline_controller, common_config.dt, plot_dir=plot_dir)
                plot_force_error_z(baseline_controller, common_config.dt,
                                   robot_cfg.name, plot_dir=plot_dir)

            import matplotlib.pyplot as plt
            plt.show()

        print("\n[MAIN] Baseline control finished")
        print(f"Baseline time : {hybrid_sim_time:.2f}s")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return -1
    finally:
        if robot is not None:
            robot.stop()


if __name__ == "__main__":
    main()
