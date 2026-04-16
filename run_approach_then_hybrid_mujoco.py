# ------------------------------------------------------------------------------
# Combined Approach + Hybrid Force-Impedance Control Script
# Phase 1: Move end-effector to target surface position (CartesianSpacePDController)
# Phase 2: Perform circle drawing with hybrid force/motion control (HybridController)
# The goal of the approach phase is the start point of the hybrid control phase.
# Supports FR3, KUKA, and Panda robots via command-line argument
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
    HybridController,
    HybridControllerConfig,
    get_robot_config
)
from utils_plot import plot_ee_positions, plot_joint_torques, plot_control_torques, plot_hybrid_results, plot_force_error_z
from utils_libfranka import euler_to_rot_matrix, generate_start_position
from mujoco_robot_interface import MujocoRobotInterface, Torques


def main() -> None:
    """Main function: approach to surface, then hybrid force-impedance control."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Approach then Hybrid Force-Impedance Control"
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="fr3",
        choices=["fr3", "kuka", "panda", "fr3_friction"],
        help="Robot type: fr3, kuka, or panda (default: fr3)"
    )
    parser.add_argument(
        "--approach-duration",
        type=float,
        default=20.0,
        help="Maximum duration for approach phase in seconds (default: 20.0)"
    )
    parser.add_argument(
        "--circle-duration",
        type=float,
        default=10.0,
        help="Duration of circle drawing in seconds (default: 10.0)"
    )
    parser.add_argument(
        "--angular-speed",
        type=float,
        default=np.pi * 2,
        help="Angular speed for circle drawing in rad/s (default: pi*2)"
    )
    parser.add_argument(
        "--force-control-method",
        type=str,
        default="paper",
        choices=["paper", "pd", "feedforward"],
        help="Force control method: paper (default), pd, or feedforward"
    )
    parser.add_argument(
        "--use-pi",
        action="store_true",
        help="Add PI force correction on top of the selected force control method"
    )
    parser.add_argument(
        "--kp-force",
        type=float,
        default=None,
        help="Proportional gain for PI/PD force control (overrides HybridControllerConfig default)"
    )
    parser.add_argument(
        "--ki-force",
        type=float,
        default=None,
        help="Integral PI gain (overrides HybridControllerConfig default)"
    )
    parser.add_argument(
        "--kd-force",
        type=float,
        default=None,
        help="Derivative PD gain (overrides HybridControllerConfig default)"
    )
    parser.add_argument(
        "--skip-seconds",
        type=float,
        default=1.0,
        help="Seconds of data to skip at start when computing metrics (default: 1.0)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without MuJoCo viewer"
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots after simulation (default: False)"
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="plots/run_approach_then_hybrid_mujoco",
        help="Directory to save plots (default: plots/run_approach_then_hybrid_mujoco)"
    )
    parser.add_argument(
        "--save-data",
        action="store_true",
        help="Save time-series data to a .npz file (default: False)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="",
        help="Directory to save .npz data file (used with --save-data)"
    )
    parser.add_argument(
        "--multiplier",
        type=float,
        default=0.0,
        help="Angular speed multiplier (omega/pi); used as part of the saved data filename"
    )
    parser.add_argument(
        "--surface-friction",
        type=float,
        default=None,
        dest="surface_friction",
        help="Override the sliding friction coefficient (first value in friction='mu ...'). "
             "Requires --robot fr3_friction. Generates a temp XML with the specified value."
    )
    parser.add_argument(
        "--no-control-force-compensation",
        dest="use_control_force_compensation",
        action="store_false",
        default=True,
        help="Disable the control force compensation term in F_ctrl_constraint (paper method only)"
    )
    parser.add_argument(
        "--no-contact-force-compensation",
        dest="use_contact_force_compensation",
        action="store_false",
        default=True,
        help="Disable the contact force compensation term in F_ctrl_constraint (paper method only)"
    )
    parser.add_argument(
        "--no-velocity-term",
        dest="use_velocity_term",
        action="store_false",
        default=True,
        help="Disable the velocity term in F_ctrl_constraint (paper method only)"
    )
    parser.add_argument(
        "--slope-angle",
        type=float,
        default=30.0,
        help="Slope angle in degrees around the X axis (default: 30.0)"
    )
    args = parser.parse_args()

    # ============================================================
    # 1. Get Robot Configuration
    # ============================================================
    robot_cfg = get_robot_config(args.robot)

    # If --surface-friction is given, generate a temp MuJoCo XML with the overridden value.
    # Pinocchio only needs kinematics, so its XML stays unchanged.
    if args.surface_friction is not None:
        mu = args.surface_friction
        base_xml = robot_cfg.pinocchio_xml_path
        with open(base_xml, 'r') as _f:
            robot_xml_str = _f.read()
        # Make meshdir absolute so the temp XML can live anywhere
        assets_dir = os.path.abspath(os.path.join(os.path.dirname(base_xml), 'assets'))
        robot_xml_str = robot_xml_str.replace('meshdir="assets"', f'meshdir="{assets_dir}"')
        # Replace first friction coefficient: friction="<old> 0.02 0.01"
        robot_xml_str = re.sub(
            r'(friction=")[0-9.]+( [0-9.]+ [0-9.]+")',
            rf'\g<1>{mu:.4f}\2',
            robot_xml_str
        )
        # Unique temp dir per friction value — safe for parallel runs
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

    print(f"\n[CONFIG] Using robot: {robot_cfg.name}")
    print(f"[CONFIG] Pinocchio XML: {robot_cfg.pinocchio_xml_path}")
    print(f"[CONFIG] MuJoCo XML: {robot_cfg.mujoco_scene_xml_path}")

    # ============================================================
    # 2. Create Configurations
    # ============================================================
    common_config = ControllerConfig(circle_duration=args.circle_duration)
    common_config.size_z = 0.01
    common_config.gravity_compensation = True
    common_config.angular_speed = args.angular_speed
    common_config.force_control_method = args.force_control_method
    common_config.use_pi = args.use_pi
    common_config.euler = np.array([np.deg2rad(args.slope_angle), 0.0, 0.0])

    approach_config = CartesianSpacePDControlConfig()
    hybrid_config = HybridControllerConfig()
    if args.kp_force is not None:
        hybrid_config.Kp_force = args.kp_force
    if args.ki_force is not None:
        hybrid_config.Ki_force = args.ki_force
    if args.kd_force is not None:
        hybrid_config.Kd_force = args.kd_force
    hybrid_config.use_control_force_compensation = args.use_control_force_compensation
    hybrid_config.use_contact_force_compensation = args.use_contact_force_compensation
    hybrid_config.use_velocity_term = args.use_velocity_term

    # Initial joint configuration (before approach)
    q0 = np.array([0.0225, 0.7064, -0.0243, -2.3135, -0.0095, 3.0422, -0.2441])

    # ============================================================
    # 3. Load Pinocchio Model
    # ============================================================
    pino_model = pino.buildModelFromMJCF(robot_cfg.pinocchio_xml_path)
    pino_data = pino_model.createData()

    try:
        print("\n" + "=" * 60)
        print("APPROACH + HYBRID FORCE-IMPEDANCE CONTROL")
        print("=" * 60)
        print(f"Robot: {robot_cfg.name.upper()}")
        print(f"Force control method: {args.force_control_method}")
        print("This will:")
        print("  1. Approach the target surface position")
        print("  2. Perform circle drawing with hybrid force control")
        print("\nMake sure:")
        print("  1. The workspace is clear")
        print("  2. Emergency stop is accessible")
        print("=" * 60)
        if not args.headless:
            input("Press Enter to continue...")

        # ============================================================
        # 4. Create Controllers
        # ============================================================
        approach_controller = CartesianSpacePDController(
            approach_config,
            common_config,
            n_joints=robot_cfg.n_joints,
            ee_frame_name=robot_cfg.ee_frame_name
        )
        hybrid_controller = HybridController(
            hybrid_config,
            common_config,
            n_joints=robot_cfg.n_joints,
            ee_frame_name=robot_cfg.ee_frame_name
        )

        # ============================================================
        # 5. Setup Approach Target
        # The approach goal position is also the hybrid control start point.
        # ============================================================
        R_slope = euler_to_rot_matrix(common_config.euler)
        target_pos = generate_start_position(
            common_config.circle_radius,
            common_config.circle_center,
            common_config.size_z,
            R_slope
        )

        # ============================================================
        # 6. Create MuJoCo Interface
        # ============================================================
        print("\nStarting torque control...")
        mujoco_interface = MujocoRobotInterface(
            common_config,
            joint_names=robot_cfg.joint_names,
            xml_path=robot_cfg.mujoco_scene_xml_path
        )

        # ============================================================
        # 7. Run Combined Control Loop
        # ============================================================
        def run_control_loop(sync_viewer=None):
            """Run the approach + hybrid control loop. Pass a viewer to sync, or None for headless."""
            nonlocal hybrid_sim_time

            # Reset simulation to initial keyframe
            mujoco_interface.reset_to_keyframe()
            mujoco.mj_step(mujoco_interface.model, mujoco_interface.data)
            if sync_viewer is not None:
                mujoco.mjv_defaultFreeCamera(mujoco_interface.model, sync_viewer.cam)

            # Read initial robot state
            robot_state, duration = mujoco_interface.readOnce()
            O_T_EE = np.array(robot_state.O_T_EE).reshape(4, 4).T
            start_pos = O_T_EE[:3, 3]

            # Compute slope-aware target orientation: slope rotation * default EE orientation
            rot_slope = Rotation.from_euler('xyz', common_config.euler)
            rot_default = Rotation.from_quat(np.roll(robot_cfg.target_quat, -1))
            target_rot = (rot_slope * rot_default).as_matrix()

            # Initialize approach controller
            control_phase = ControlPhase.APPROACHING
            approach_controller.starting(start_pos, target_pos, target_rot, q0, pino_model, pino_data)

            print("\n" + "=" * 60)
            print("PHASE 1: APPROACHING TARGET POSITION")
            print("=" * 60)

            while True:
                if sync_viewer is not None and not sync_viewer.is_running():
                    break

                step_start = time.time()

                # Read robot state
                robot_state, duration = mujoco_interface.readOnce()

                # ============================================================
                # State Machine
                # ============================================================
                if control_phase == ControlPhase.APPROACHING:
                    tau = approach_controller.update(duration, robot_state)

                    # Check approach timeout
                    if approach_controller.time_elapsed >= args.approach_duration:
                        print(f"\n[WARN] Approach timed out at {approach_controller.time_elapsed:.2f}s")
                        control_phase = ControlPhase.STOPPED

                    # Check if approach target reached
                    elif approach_controller.is_target_reached(robot_state):
                        print("\n" + "=" * 60)
                        print(f"TARGET REACHED at t={approach_controller.time_elapsed:.2f}s!")
                        print("PHASE 2: HYBRID FORCE-IMPEDANCE CONTROL")
                        print("=" * 60 + "\n")

                        # Use current robot state as the start of hybrid control
                        O_T_EE = np.array(robot_state.O_T_EE).reshape(4, 4).T
                        hybrid_target_rot = O_T_EE[:3, :3]
                        hybrid_q0 = np.array(robot_state.q)

                        hybrid_controller.starting(
                            hybrid_sim_time, hybrid_target_rot, hybrid_q0, pino_model, pino_data
                        )
                        control_phase = ControlPhase.CIRCLE_DRAWING

                elif control_phase == ControlPhase.CIRCLE_DRAWING:
                    tau = hybrid_controller.update(hybrid_sim_time, robot_state)
                    hybrid_sim_time += common_config.dt

                    if hybrid_controller.is_finished():
                        print("\n" + "=" * 60)
                        print(f"HYBRID CONTROL FINISHED at t={hybrid_sim_time:.2f}s!")
                        print("=" * 60)
                        control_phase = ControlPhase.STOPPED

                else:  # STOPPED
                    tau = pino.computeGeneralizedGravity(
                        pino_model, pino_data, np.array(robot_state.q)
                    )
                    torque_cmd = Torques(tau.tolist())
                    torque_cmd.motion_finished = True
                    mujoco_interface.writeOnce(torque_cmd)
                    break

                # Apply torques
                torque_cmd = Torques(tau.tolist())
                mujoco_interface.writeOnce(torque_cmd)

                if sync_viewer is not None:
                    sync_viewer.sync()

                # Maintain real-time rate
                time_until_next_step = common_config.dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

        hybrid_sim_time = 0.0

        if args.headless:
            run_control_loop(sync_viewer=None)
        else:
            with mujoco.viewer.launch_passive(
                mujoco_interface.model, mujoco_interface.data,
                show_left_ui=False, show_right_ui=False
            ) as viewer:
                run_control_loop(sync_viewer=viewer)

        # ============================================================
        # 8. Report metrics (always printed for sweep scripts)
        # ============================================================
        skip_samples = int(args.skip_seconds / common_config.dt)
        contact_forces = np.array(hybrid_controller.contact_forces) if hybrid_controller.contact_forces else np.empty((0, 3))
        desired_forces = np.array(hybrid_controller.desired_forces) if hybrid_controller.desired_forces else np.empty((0, 1))
        if contact_forces.size > 0 and contact_forces.ndim == 2 and contact_forces.shape[1] >= 3 and desired_forces.size > 0:
            cf_ss = contact_forces[skip_samples:]
            df_ss = desired_forces[skip_samples:]
            if cf_ss.shape[0] > 0 and df_ss.shape[0] > 0:
                error_z = cf_ss[:, 2] - df_ss[:, 0]
                avg_abs_force_error = np.mean(np.abs(error_z))
                var_force_error = np.var(error_z)
            else:
                avg_abs_force_error = float('nan')
                var_force_error = float('nan')
            print(f"AVG_FORCE_Z_ERROR: {avg_abs_force_error:.6f}")
            print(f"VAR_FORCE_Z_ERROR: {var_force_error:.6f}")
        else:
            print(f"AVG_FORCE_Z_ERROR: nan")
            print(f"VAR_FORCE_Z_ERROR: nan")

        ee_positions = np.array(hybrid_controller.ee_positions) if hybrid_controller.ee_positions else np.empty((0, 3))
        target_positions = np.array(hybrid_controller.target_positions) if hybrid_controller.target_positions else np.empty((0, 3))
        if ee_positions.size > 0 and target_positions.size > 0 and ee_positions.shape == target_positions.shape:
            ep_ss = ee_positions[skip_samples:]
            tp_ss = target_positions[skip_samples:]
            if ep_ss.shape[0] > 0:
                pos_error = np.linalg.norm(ep_ss - tp_ss, axis=1)
                avg_abs_pos_error = np.mean(pos_error)
                var_pos_error = np.var(pos_error)
            else:
                avg_abs_pos_error = float('nan')
                var_pos_error = float('nan')
            print(f"AVG_POSITION_ERROR: {avg_abs_pos_error:.6f}")
            print(f"VAR_POSITION_ERROR: {var_pos_error:.6f}")
        else:
            print(f"AVG_POSITION_ERROR: nan")
            print(f"VAR_POSITION_ERROR: nan")

        # ============================================================
        # 9. Save time-series data (only if --save-data is set)
        # ============================================================
        if args.save_data and args.data_dir:
            import os as _os
            _os.makedirs(args.data_dir, exist_ok=True)

            # Velocities via numerical differentiation (consistent with utils_plot.py)
            if ee_positions.size > 0:
                actual_velocitys = np.gradient(ee_positions, common_config.dt, axis=0)
            else:
                actual_velocitys = np.empty((0, 3))

            if target_positions.size > 0:
                desired_velocitys = np.gradient(target_positions, common_config.dt, axis=0)
            else:
                desired_velocitys = np.empty((0, 3))

            # Force error time series (full, not skip-trimmed)
            if contact_forces.size > 0 and desired_forces.size > 0:
                force_error = contact_forces[:, 2] - desired_forces[:, 0]
            else:
                force_error = np.empty(0)

            # Position error time series (full, not skip-trimmed)
            if ee_positions.size > 0 and target_positions.size > 0 and ee_positions.shape == target_positions.shape:
                position_error = np.linalg.norm(ee_positions - target_positions, axis=1)
            else:
                position_error = np.empty(0)

            ee_linear_speed = 0.1 * args.angular_speed  # circle radius = 0.1 m
            disabled_parts = []
            if not args.use_control_force_compensation:
                disabled_parts.append("no_ctrl")
            if not args.use_contact_force_compensation:
                disabled_parts.append("no_contact")
            if not args.use_velocity_term:
                disabled_parts.append("no_vel")
            comp_suffix = ("_" + "_".join(disabled_parts)) if disabled_parts else ""
            fname = f"data_{args.multiplier:.1f}{comp_suffix}.npz"
            fpath = _os.path.join(args.data_dir, fname)
            np.savez(
                fpath,
                force_error=force_error,
                position_error=position_error,
                actual_positions=ee_positions,
                desired_positions=target_positions,
                actual_velocitys=actual_velocitys,
                desired_velocitys=desired_velocitys,
                multiplier=np.array(args.multiplier),
                angular_speed_rad_s=np.array(args.angular_speed),
                ee_linear_speed_m_s=np.array(ee_linear_speed),
            )
            print(f"DATA_SAVED: {fpath}")

        # ============================================================
        # 10. Plot Results (only if --save-plots is set)
        # ============================================================
        if args.save_plots:
            print("\n[MAIN] Simulation complete. Generating plots...")
            plot_dir = args.plot_dir
            # plot_joint_torques(approach_controller, common_config.dt, plot_dir="mj_ctrl/plots/sim/approach")
            # plot_ee_positions(approach_controller, common_config.dt, plot_dir="mj_ctrl/plots/sim/approach")
            plot_joint_torques(hybrid_controller, "joint_torques", common_config.dt, plot_dir=plot_dir)
            plot_joint_torques(hybrid_controller, "joint_g_torques", common_config.dt, plot_dir=plot_dir)
            plot_ee_positions(hybrid_controller, common_config.dt, plot_dir=plot_dir)
            plot_control_torques(hybrid_controller, common_config.dt, plot_dir=plot_dir)
            plot_hybrid_results(hybrid_controller, common_config.dt, robot_cfg.name, plot_dir=plot_dir)
            plot_force_error_z(hybrid_controller, common_config.dt, robot_cfg.name, plot_dir=plot_dir)

        print("\n[MAIN] Combined control finished")
        print(f"Approach time: {approach_controller.time_elapsed:.2f}s")
        print(f"Hybrid time:   {hybrid_sim_time:.2f}s")

    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        return -1


if __name__ == "__main__":
    main()
