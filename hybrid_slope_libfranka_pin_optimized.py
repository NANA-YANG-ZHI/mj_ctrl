# ------------------------------------------------------------------------------
# Hybrid Force-Impedance Control for Fast End-Effector Motions
# OPTIMIZED VERSION - Removed print/logging from real-time loop
# Added profiling instrumentation to identify bottlenecks
# ------------------------------------------------------------------------------
import argparse
import numpy as np
import time
import pinocchio as pino
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from utils_libfranka import *
from franka_bindings import Robot, Torques
from scipy.spatial.transform import Rotation

from profile_control_loop import ControlLoopProfiler

# Disable logging in real-time loop - use file logging only after control ends
import logging
logging.basicConfig(
    filename="robot.log",
    level=logging.WARNING,  # Changed from INFO to WARNING
    filemode="w"
)

def generate_circle_trajectory(elapsed_time: float,
                               circle_center: np.ndarray,
                               circle_radius: float,
                               angular_speed: float,
                               R_slope: np.ndarray,
                               size_z: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate desired position, velocity, and acceleration for circle trajectory."""
    angle = angular_speed * elapsed_time % (2 * np.pi)
    target_pos_local = np.zeros(3)
    x_dot_desired_local = np.zeros(3)
    x_ddot_desired_local = np.zeros(3)

    target_pos_local[0] = circle_radius * np.cos(angle)
    target_pos_local[1] = circle_radius * np.sin(angle)
    target_pos_local[2] = size_z

    x_dot_desired_local[0] = -circle_radius * angular_speed * np.sin(angle)
    x_dot_desired_local[1] =  circle_radius * angular_speed * np.cos(angle)
    x_dot_desired_local[2] = 0.0

    x_ddot_desired_local[0] = -circle_radius * angular_speed**2 * np.cos(angle)
    x_ddot_desired_local[1] = -circle_radius * angular_speed**2 * np.sin(angle)
    x_ddot_desired_local[2] = 0.0

    return circle_center + (R_slope @ target_pos_local), R_slope @ x_dot_desired_local, R_slope @ x_ddot_desired_local


class ControlPhase(Enum):
    APPROACHING = 1
    CIRCLE_DRAWING = 2
    STOPPED = 3


@dataclass
class ControllerConfig:
    dt: float = 0.001
    gravity_compensation: bool = True
    circle_center: np.ndarray = None
    circle_radius: float = 0.1
    circle_duration: float = 10.0
    angular_speed: float = np.pi * 2
    position_tolerance: float = 0.05
    euler: np.ndarray = None
    size_z: float = 0.00
    use_table: bool = False

    def __post_init__(self):
        if self.circle_center is None:
            self.circle_center = np.array([0.5, 0.0, 0.3])
        if self.euler is None:
            self.euler = np.array([np.deg2rad(0), 0, 0])


@dataclass
class CartesianSpacePDControlConfig:
    Kpos: float = 0.95
    Kp: np.ndarray = None
    Kd: np.ndarray = None
    Kp_null: np.ndarray = None
    Kd_null: np.ndarray = None
    impedance_pos: np.ndarray = None
    impedance_ori: np.ndarray = None

    def __post_init__(self):
        if self.impedance_pos is None:
            self.impedance_pos = np.asarray([50.0, 50.0, 50.0]) * 0.2
        if self.impedance_ori is None:
            self.impedance_ori = np.asarray([25.0, 25.0, 25.0]) * 0.2
        if self.Kp is None:
            self.Kp = np.concatenate([self.impedance_pos, self.impedance_ori], axis=0)
        if self.Kd is None:
            damping_ratio = 1.0
            damping_pos = damping_ratio * 2 * np.sqrt(self.impedance_pos)
            damping_ori = damping_ratio * 2 * np.sqrt(self.impedance_ori)
            self.Kd = np.concatenate([damping_pos, damping_ori], axis=0)
        if self.Kp_null is None:
            self.Kp_null = np.asarray([75.0, 75.0, 50.0, 50.0, 40.0, 25.0, 25.0]) * 0.2
        if self.Kd_null is None:
            damping_ratio = 1.0
            self.Kd_null = damping_ratio * 2 * np.sqrt(self.Kp_null)


@dataclass
class HybridControllerConfig:
    damping_ratio: float = 1.0
    impedance_pos: np.ndarray = None
    impedance_ori: np.ndarray = None
    Kp_null: np.ndarray = None
    Kd_null: np.ndarray = None
    k_normal: float = 5000.0
    Kp_force: float = 0.4
    Kd_force: float = 0.002
    Ki_force: float = 0.4
    F_desired_contact: np.ndarray = None

    def __post_init__(self):
        if self.impedance_pos is None:
            self.impedance_pos = np.asarray([50.0, 50.0, 50.0]) * 0.2
        if self.impedance_ori is None:
            self.impedance_ori = np.asarray([25.0, 25.0, 25.0]) * 0.2
        if self.Kp_null is None:
            self.Kp_null = np.asarray([75.0, 75.0, 50.0, 50.0, 40.0, 25.0, 25.0]) * 0.2
            self.Kd_null = self.damping_ratio * 2 * np.sqrt(self.Kp_null)
        if self.F_desired_contact is None:
            self.F_desired_contact = np.array([-10.0])


class CartesianSpacePDController:
    """Controller for moving end-effector to desired position."""

    def __init__(self, config: CartesianSpacePDControlConfig, common_config: ControllerConfig):
        self.config = config
        self.common_config = common_config
        self.pino_model: Optional[pino.Model] = None
        self.pino_data: Optional[pino.Data] = None
        self.target_pos: Optional[np.ndarray] = None
        self.target_quat: Optional[np.ndarray] = None
        self.q0: Optional[np.ndarray] = None
        self.tau: np.ndarray = np.zeros(7)

        # Preallocate arrays to avoid memory allocation in loop
        self._twist = np.zeros(6)
        self._q = np.zeros(7)
        self._dq = np.zeros(7)

        # Data logging - use preallocated arrays instead of lists
        self.max_log_size = 50000
        self.ee_positions = np.zeros((self.max_log_size, 3))
        self.target_positions = np.zeros((self.max_log_size, 3))
        self.log_idx = 0

    def starting(self, target_pos: np.ndarray, target_quat: np.ndarray, q0: np.ndarray,
                 pino_model: pino.Model, pino_data: pino.Data) -> None:
        self.pino_model = pino_model
        self.pino_data = pino_data
        self.target_pos = target_pos.copy()
        self.target_quat = target_quat.copy()
        self.q0 = q0.copy()
        self.log_idx = 0
        self.tau[:] = 0.0
        print(f"[APPROACH START] Target position: {self.target_pos}")
        print(f"[APPROACH START] Target quaternion: {self.target_quat}")

    def update(self, robot_state, profiler: Optional[ControlLoopProfiler] = None) -> np.ndarray:
        """Compute control torques - OPTIMIZED: no prints, preallocated arrays."""

        # Get current state - reuse preallocated arrays
        with profiler.measure("state_copy") if profiler else nullcontext():
            self._q[:] = robot_state.q
            self._dq[:] = robot_state.dq

        # 1. Compute End-Effector Pose Error
        with profiler.measure("pose_transform") if profiler else nullcontext():
            O_T_EE = np.array(robot_state.O_T_EE).reshape(4, 4).T
            current_pos = O_T_EE[:3, 3]
            current_mat = O_T_EE[:3, :3]

        with profiler.measure("pose_error") if profiler else nullcontext():
            twist = compute_ee_pose_error(
                self.target_pos, current_pos,
                self.target_quat, current_mat.flatten(),
                Kpos=self.config.Kpos
            )

        # 2. Compute Jacobian
        with profiler.measure("pinocchio_fk") if profiler else nullcontext():
            pino.forwardKinematics(self.pino_model, self.pino_data, self._q, self._dq)
            pino.computeJointJacobians(self.pino_model, self.pino_data)
            pino.updateFramePlacements(self.pino_model, self.pino_data)

        with profiler.measure("jacobian") if profiler else nullcontext():
            pino_frame_id = self.pino_model.getFrameId("attachment")
            jac = pino.getFrameJacobian(self.pino_model, self.pino_data, pino_frame_id, pino.LOCAL_WORLD_ALIGNED)

        # 3. Compute Task-Space Inertia Matrix
        with profiler.measure("mass_matrix") if profiler else nullcontext():
            M_inv = pino.computeMinverse(self.pino_model, self.pino_data, self._q)
            Mx = task_space_inertiaM(M_inv, jac)

        # 4. Compute Task-Space Control
        with profiler.measure("control_law") if profiler else nullcontext():
            self.tau[:] = jac.T @ Mx @ (
                    self.config.Kp * twist - self.config.Kd * (jac @ self._dq)
            )

        # 5. Add Nullspace Control
        with profiler.measure("nullspace") if profiler else nullcontext():
            Jbar = M_inv @ jac.T @ Mx
            ddq = null_space_tau(self._q, self._dq, self.q0, self.config.Kp_null, self.config.Kd_null)
            self.tau += (np.eye(7) - jac.T @ Jbar.T) @ ddq

        # 6. Add Gravity Compensation
        with profiler.measure("gravity") if profiler else nullcontext():
            if self.common_config.gravity_compensation:
                self.tau += pino.computeGeneralizedGravity(self.pino_model, self.pino_data, self._q)

        # 7. Log Data (preallocated, no list append)
        if self.log_idx < self.max_log_size:
            self.ee_positions[self.log_idx] = current_pos
            self.target_positions[self.log_idx] = self.target_pos
            self.log_idx += 1

        return self.tau

    def is_target_reached(self, robot_state) -> bool:
        O_T_EE = np.array(robot_state.O_T_EE).reshape(4, 4).T
        current_pos = O_T_EE[:3, 3]
        distance = np.linalg.norm(current_pos - self.target_pos)
        return distance < self.common_config.position_tolerance


class HybridController:
    """Controller for drawing circles with force control - OPTIMIZED."""

    def __init__(self, config: HybridControllerConfig, common_config: ControllerConfig):
        self.config = config
        self.common_config = common_config

        damping_pos = self.config.damping_ratio * 2 * np.sqrt(self.config.impedance_pos)
        damping_ori = self.config.damping_ratio * 2 * np.sqrt(self.config.impedance_ori)
        self.Kp = np.concatenate([self.config.impedance_pos, self.config.impedance_ori])
        self.Kd = np.concatenate([damping_pos, damping_ori])

        self.S_fc = np.zeros((6, 1))
        self.S_fc[2, 0] = 1

        self.S_vc = np.zeros((6, 5))
        self.S_vc[0, 0] = 1
        self.S_vc[1, 1] = 1
        self.S_vc[3, 2] = 1
        self.S_vc[4, 3] = 1
        self.S_vc[5, 4] = 1

        self.R_slope = euler_to_rot_matrix(self.common_config.euler)
        rot_slope = Rotation.from_euler('xyz', self.common_config.euler)
        self.quat_slope = np.roll(rot_slope.as_quat(), 1)

        self.R = np.zeros((6, 6))
        self.R[0:3, 0:3] = self.R_slope
        self.R[3:6, 3:6] = self.R_slope
        self.S_f = self.R @ self.S_fc
        self.S_v = self.R @ self.S_vc

        self.target_pos: Optional[np.ndarray] = None
        self.target_quat: Optional[np.ndarray] = None
        self.x_dot_desired: Optional[np.ndarray] = np.zeros(3)
        self.x_ddot_desired: Optional[np.ndarray] = np.zeros(3)
        self.q0: Optional[np.ndarray] = None

        self.start_time: float = 0.0
        self.is_drawing: bool = False
        self.tau = np.zeros(7)

        # Preallocate workspace arrays
        self._q = np.zeros(7)
        self._dq = np.zeros(7)

        # Preallocated logging arrays
        self.max_log_size = 50000
        self.contact_forces = np.zeros((self.max_log_size, 3))
        self.desired_forces = np.zeros((self.max_log_size, 1))
        self.ee_positions = np.zeros((self.max_log_size, 3))
        self.target_positions = np.zeros((self.max_log_size, 3))
        self.control_force_compensation_arr = np.zeros((self.max_log_size, 1))
        self.contact_force_compensation_arr = np.zeros((self.max_log_size, 1))
        self.velocity_term_arr = np.zeros((self.max_log_size, 1))
        self.F_ctrl_constraint_arr = np.zeros((self.max_log_size, 1))
        self.log_idx = 0

    def starting(self, current_time: float, target_pos: np.ndarray, target_quat: np.ndarray,
                 q0: np.ndarray, pino_model: pino.Model, pino_data: pino.Data) -> None:
        self.q0 = q0.copy()
        self.pino_model = pino_model
        self.pino_data = pino_data
        self.start_time = current_time
        self.is_drawing = True
        self.target_pos = target_pos.copy()
        self.target_quat = target_quat.copy()
        self.log_idx = 0
        self.tau[:] = 0.0

        print(f"[CIRCLE START] Circle drawing started at t={current_time:.2f}s")
        print(f"[CIRCLE START] Center: {self.common_config.circle_center}")
        print(f"[CIRCLE START] Radius: {self.common_config.circle_radius}")
        print(f"[CIRCLE START] Force control: F_desired={self.config.F_desired_contact}")

    def update(self, current_time: float, robot_state, profiler: Optional[ControlLoopProfiler] = None) -> np.ndarray:
        """Compute control torques - OPTIMIZED: no prints/logging in loop."""

        # 1. Update Trajectory
        with profiler.measure("trajectory") if profiler else nullcontext():
            elapsed = current_time - self.start_time
            if elapsed < self.common_config.circle_duration:
                self.target_pos, self.x_dot_desired, self.x_ddot_desired = \
                    generate_circle_trajectory(
                        elapsed,
                        self.common_config.circle_center,
                        self.common_config.circle_radius,
                        self.common_config.angular_speed,
                        self.R_slope,
                        self.common_config.size_z
                    )
            else:
                self.x_dot_desired[:] = 0.0
                self.x_ddot_desired[:] = 0.0
                self.is_drawing = False

        # Get current state
        with profiler.measure("state_copy") if profiler else nullcontext():
            self._q[:] = robot_state.q
            self._dq[:] = robot_state.dq
            O_T_EE = np.array(robot_state.O_T_EE).reshape(4, 4).T
            current_pos = O_T_EE[:3, 3]
            current_mat = O_T_EE[:3, :3]

        # 2. Compute Jacobian and Dynamics
        with profiler.measure("pinocchio_fk") if profiler else nullcontext():
            pino.forwardKinematics(self.pino_model, self.pino_data, self._q, self._dq)
            pino.computeJointJacobians(self.pino_model, self.pino_data)
            pino.updateFramePlacements(self.pino_model, self.pino_data)

        with profiler.measure("jacobian") if profiler else nullcontext():
            pino_frame_id = self.pino_model.getFrameId("attachment")
            jac = pino.getFrameJacobian(self.pino_model, self.pino_data, pino_frame_id, pino.LOCAL_WORLD_ALIGNED)

        with profiler.measure("dynamics") if profiler else nullcontext():
            M = pino.crba(self.pino_model, self.pino_data, self._q)
            M_inv = pino.computeMinverse(self.pino_model, self.pino_data, self._q)

        with profiler.measure("selection_matrices") if profiler else nullcontext():
            J_phi = self.S_f.T @ jac
            J_motion = self.S_v.T @ jac
            jac_1 = np.vstack([J_phi, J_motion])
            Mx_constraint = task_space_inertiaM(M_inv, J_phi)
            Mx_motion = task_space_inertiaM(M_inv, J_motion)

        # 4. Get Contact Information
        F_ext_world = np.zeros(6)
        current_force_local = F_ext_world
        F_ext_phi = current_force_local @ self.S_fc
        F_ext_x = current_force_local @ self.S_vc

        # Null Space torque
        with profiler.measure("nullspace") if profiler else nullcontext():
            jac_1_inv = dynamically_consistent_inv(jac_1, M_inv)
            N2 = np.eye(7) - jac_1.T @ jac_1_inv.T
            tau_ctrl_v = null_space_tau(self._q, self._dq, self.q0, self.config.Kp_null, self.config.Kd_null)
            tau_ctrl_v = N2 @ tau_ctrl_v

        # Motion Space
        with profiler.measure("pose_error") if profiler else nullcontext():
            twist = compute_ee_pose_error(
                self.target_pos, current_pos,
                self.target_quat, current_mat.flatten()
            )

        with profiler.measure("motion_control") if profiler else nullcontext():
            x_ddot_desired_sel = np.concatenate([self.x_ddot_desired, [0, 0, 0]]) @ self.S_v
            x_tilde = twist @ self.S_v
            site_vel = jac @ self._dq
            x_dot_tilde = (np.concatenate([self.x_dot_desired, [0, 0, 0]]) - site_vel) @ self.S_v
            a_motion = feedforward_PD(
                x_acc_desired=x_ddot_desired_sel, x_delta=x_tilde,
                x_dot_delta=x_dot_tilde,
                Kp=self.Kp @ self.S_v, Kd=self.Kd @ self.S_v
            )
            F_ctrl_x = Mx_motion @ a_motion
            tau_ctrl_x = J_motion.T @ F_ctrl_x

        # Constraint space
        with profiler.measure("constraint_control") if profiler else nullcontext():
            C = pino.computeCoriolisMatrix(self.pino_model, self.pino_data, self._q, self._dq)
            J_dot = pino.getFrameJacobianTimeVariation(self.pino_model, self.pino_data, pino_frame_id, pino.LOCAL_WORLD_ALIGNED)
            J_phi_dot = self.S_f.T @ J_dot

            F_ext_x_new = F_ext_x.copy()
            F_ext_x_new[-3:] = 0
            control_force_compensation = -Mx_constraint @ J_phi @ M_inv @ (tau_ctrl_x + tau_ctrl_v)
            contact_force_compensation = Mx_constraint @ J_phi @ M_inv @ (J_motion.T @ F_ext_x_new)
            velocity_term = Mx_constraint @ (J_phi @ M_inv @ C - J_phi_dot) @ self._dq
            F_ctrl_constraint = (
                self.config.F_desired_contact +
                control_force_compensation +
                contact_force_compensation + velocity_term
            )

        # Sum up torques
        with profiler.measure("torque_sum") if profiler else nullcontext():
            self.tau[:] = J_phi.T @ F_ctrl_constraint + tau_ctrl_x + tau_ctrl_v

        # 6. Add Gravity Compensation
        with profiler.measure("gravity") if profiler else nullcontext():
            if self.common_config.gravity_compensation:
                self.tau += pino.computeGeneralizedGravity(self.pino_model, self.pino_data, self._q)

        # 7. Log Data (preallocated arrays, no list append)
        if self.log_idx < self.max_log_size:
            self.contact_forces[self.log_idx] = current_force_local[:3]
            self.desired_forces[self.log_idx] = -self.config.F_desired_contact
            self.ee_positions[self.log_idx] = current_pos
            self.target_positions[self.log_idx] = self.target_pos
            self.control_force_compensation_arr[self.log_idx] = control_force_compensation
            self.contact_force_compensation_arr[self.log_idx] = contact_force_compensation
            self.velocity_term_arr[self.log_idx] = velocity_term
            self.F_ctrl_constraint_arr[self.log_idx] = F_ctrl_constraint
            self.log_idx += 1

        return self.tau

    def is_finished(self) -> bool:
        return not self.is_drawing


# Null context manager for when profiler is None
class nullcontext:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


def main() -> None:
    """Main function with two-phase control - OPTIMIZED with profiling."""

    parser = argparse.ArgumentParser(description="Hybrid force/impedance control for Franka Panda")
    parser.add_argument("--ip", type=str, default="localhost", help="Robot IP address")
    parser.add_argument("--approach-only", action="store_true", help="Only run approach phase")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    args = parser.parse_args()

    # Create profiler
    profiler = ControlLoopProfiler() if args.profile else None

    # 1. Create Configurations
    common_config = ControllerConfig()
    approach_config = CartesianSpacePDControlConfig()
    circle_config = HybridControllerConfig()
    q0 = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])

    # 2. Load Model
    pino_model = pino.buildModelFromMJCF("mj_ctrl/franka_fr3/fr3.xml")
    pino_data = pino_model.createData()

    try:
        print(f"Connecting to robot at {args.ip}...")
        robot = Robot(args.ip)

        robot.set_collision_behavior(
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        )

        print("\n" + "=" * 60)
        print("WARNING: This will move the robot!")
        print("Make sure:")
        print("  1. The workspace is clear")
        print("  2. Emergency stop is accessible")
        print("  3. You understand the trajectory")
        print("=" * 60)
        input("Press Enter to continue...")

        # 3. Create Controllers
        approach_controller = CartesianSpacePDController(approach_config, common_config)
        circle_controller = HybridController(circle_config, common_config)

        # 4. Setup Initial Targets
        R_slope = euler_to_rot_matrix(common_config.euler)
        target_pos = generate_start_position(
            common_config.circle_radius,
            common_config.circle_center,
            common_config.size_z,
            R_slope
        )

        target_quat = np.array([0., 1., 0., 0.])
        rot_slope = Rotation.from_euler('xyz', common_config.euler)
        rot_target = Rotation.from_quat(np.roll(target_quat, -1))
        target_quat = np.roll((rot_slope * rot_target).as_quat(), 1)

        print("\nStarting torque control...")
        active_control = robot.start_torque_control()

        # 5. Start Approach Phase
        control_phase = ControlPhase.APPROACHING
        approach_controller.starting(target_pos, target_quat, q0, pino_model, pino_data)

        print("\n" + "=" * 60)
        print("PHASE 1: APPROACHING TARGET POSITION")
        print("=" * 60)
        print(f"Target Quat: {target_quat}")

        # 6. Run Control Loop
        sim_time = 0.0
        transition_time = 0.0
        iteration_count = 0

        try:
            while True:
                if profiler:
                    profiler.start_iteration()

                # Read robot state
                with profiler.measure("readOnce") if profiler else nullcontext():
                    robot_state, duration = active_control.readOnce()

                # State Machine: Switch Controllers
                if control_phase == ControlPhase.APPROACHING:
                    with profiler.measure("approach_update") if profiler else nullcontext():
                        tau = approach_controller.update(robot_state, profiler)

                    if approach_controller.is_target_reached(robot_state):
                        print("\n" + "=" * 60)
                        print(f"TARGET REACHED at t={sim_time:.2f}s!")
                        print("=" * 60 + "\n")

                        if args.approach_only:
                            print("Approach-only mode: stopping here.")
                            control_phase = ControlPhase.STOPPED
                        else:
                            print("PHASE 2: CIRCLE DRAWING")
                            print("=" * 60)
                            control_phase = ControlPhase.CIRCLE_DRAWING
                            transition_time = sim_time
                            circle_controller.starting(sim_time, target_pos, target_quat, q0, pino_model, pino_data)

                elif control_phase == ControlPhase.CIRCLE_DRAWING:
                    with profiler.measure("circle_update") if profiler else nullcontext():
                        tau = circle_controller.update(sim_time, robot_state, profiler)

                    if circle_controller.is_finished():
                        print("\n" + "=" * 60)
                        print(f"CIRCLE DRAWING FINISHED at t={sim_time:.2f}s!")
                        print("=" * 60 + "\n")
                        control_phase = ControlPhase.STOPPED

                else:  # STOPPED
                    tau = pino.computeGeneralizedGravity(pino_model, pino_data, np.array(robot_state.q))
                    torque_cmd = Torques(tau.tolist())
                    torque_cmd.motion_finished = True
                    active_control.writeOnce(torque_cmd)
                    break

                # Apply Control
                with profiler.measure("writeOnce") if profiler else nullcontext():
                    torque_cmd = Torques(tau.tolist())
                    active_control.writeOnce(torque_cmd)

                if profiler:
                    profiler.end_iteration()

                # Update time
                sim_time += duration.to_sec()
                iteration_count += 1

                # Print profiling stats periodically (outside real-time critical path)
                if profiler and iteration_count % 5000 == 0:
                    profiler.print_stats()

            # Final profiling report
            if profiler:
                profiler.print_stats()

            print("\n[MAIN] Simulation complete.")
            np.savez(
                "force_details.npz",
                control_force_compensation_arr=circle_controller.control_force_compensation_arr[:circle_controller.log_idx],
                contact_force_compensation_arr=circle_controller.contact_force_compensation_arr[:circle_controller.log_idx],
                velocity_term_arr=circle_controller.velocity_term_arr[:circle_controller.log_idx],
                F_ctrl_constraint_arr=circle_controller.F_ctrl_constraint_arr[:circle_controller.log_idx]
            )

        except KeyboardInterrupt:
            print("\nControl interrupted by user")
            torque_cmd = Torques([0.0] * 7)
            torque_cmd.motion_finished = True
            active_control.writeOnce(torque_cmd)

            if profiler:
                profiler.print_stats()

            np.savez(
                "force_details.npz",
                control_force_compensation_arr=circle_controller.control_force_compensation_arr[:circle_controller.log_idx],
                contact_force_compensation_arr=circle_controller.contact_force_compensation_arr[:circle_controller.log_idx],
                velocity_term_arr=circle_controller.velocity_term_arr[:circle_controller.log_idx],
                F_ctrl_constraint_arr=circle_controller.F_ctrl_constraint_arr[:circle_controller.log_idx]
            )

        print("\n[MAIN] Control finished")
        print(f"Total time: {sim_time:.2f}s")

    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        if robot is not None:
            robot.stop()
        return -1
    finally:
        robot.stop()

    return 0


if __name__ == "__main__":
    main()
