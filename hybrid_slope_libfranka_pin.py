# ------------------------------------------------------------------------------
# Hybrid Force-Impedance Control for Fast End-Effector Motions
# Separated into Approach Controller and Circle Drawing Controller
# 1. Use pinocchio to load model dynamics and calculate jac, M and g
# 2. Using libfranka for robot states and send control signal
# ------------------------------------------------------------------------------
import argparse
import numpy as np
import time
import os
import pinocchio as pino
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from utils_libfranka import *
import matplotlib.pyplot as plt
# from geom_visualizer import visualize_normal_arrow, reset_scene
from franka_bindings import Robot, Torques
from scipy.spatial.transform import Rotation
import logging
import time

logging.basicConfig(
    filename="robot.log",
    level=logging.INFO,
    filemode="w"
)


class TrajectoryPlanner:
    """
    Quintic polynomial trajectory planner for smooth motion.

    Plans trajectories with continuous position, velocity, and acceleration.
    Re-plans every specified interval (default 1 second) to handle disturbances.
    """

    def __init__(self, planning_horizon: float = 1.0):
        """
        Initialize trajectory planner.

        Args:
            planning_horizon: Time duration for each trajectory segment (seconds)
        """
        self.planning_horizon = planning_horizon

        # Trajectory coefficients for each dimension (x, y, z)
        # Quintic: p(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        self.coeffs_pos = None  # Shape: (3, 6)
        self.coeffs_quat = None  # For orientation (simplified SLERP)

        # Trajectory state
        self.start_time = 0.0
        self.trajectory_duration = 0.0
        self.is_planned = False

        # Start and end states
        self.start_pos = None
        self.start_vel = None
        self.start_acc = None
        self.end_pos = None
        self.end_vel = None
        self.end_acc = None

        # Orientation (using SLERP)
        self.start_quat = None
        self.end_quat = None

    def plan(self,
             current_time: float,
             start_pos: np.ndarray,
             start_vel: np.ndarray,
             end_pos: np.ndarray,
             end_vel: np.ndarray = None,
             start_acc: np.ndarray = None,
             end_acc: np.ndarray = None,
             duration: float = None,
             start_quat: np.ndarray = None,
             end_quat: np.ndarray = None) -> None:
        """
        Plan a quintic polynomial trajectory from start to end.

        Args:
            current_time: Current time (seconds)
            start_pos: Starting position (3,)
            start_vel: Starting velocity (3,)
            end_pos: End position (3,)
            end_vel: End velocity (3,), defaults to zero
            start_acc: Starting acceleration (3,), defaults to zero
            end_acc: End acceleration (3,), defaults to zero
            duration: Trajectory duration, defaults to planning_horizon
            start_quat: Starting quaternion (w, x, y, z)
            end_quat: Ending quaternion (w, x, y, z)
        """
        if end_vel is None:
            end_vel = np.zeros(3)
        if start_acc is None:
            start_acc = np.zeros(3)
        if end_acc is None:
            end_acc = np.zeros(3)
        if duration is None:
            duration = self.planning_horizon

        self.start_time = current_time
        self.trajectory_duration = duration

        self.start_pos = start_pos.copy()
        self.start_vel = start_vel.copy()
        self.start_acc = start_acc.copy()
        self.end_pos = end_pos.copy()
        self.end_vel = end_vel.copy()
        self.end_acc = end_acc.copy()

        # Store orientation for SLERP
        if start_quat is not None:
            self.start_quat = start_quat.copy()
        if end_quat is not None:
            self.end_quat = end_quat.copy()

        # Compute quintic polynomial coefficients for each dimension
        # p(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        # v(t) = a1 + 2*a2*t + 3*a3*t^2 + 4*a4*t^3 + 5*a5*t^4
        # a(t) = 2*a2 + 6*a3*t + 12*a4*t^2 + 20*a5*t^3

        T = duration
        self.coeffs_pos = np.zeros((3, 6))

        for i in range(3):
            p0, v0, a0 = start_pos[i], start_vel[i], start_acc[i]
            pf, vf, af = end_pos[i], end_vel[i], end_acc[i]

            # Boundary conditions give us the coefficients
            # a0 = p0
            # a1 = v0
            # a2 = a0/2
            # Solve for a3, a4, a5 using end conditions

            self.coeffs_pos[i, 0] = p0
            self.coeffs_pos[i, 1] = v0
            self.coeffs_pos[i, 2] = a0 / 2.0

            # Solve the 3x3 system for a3, a4, a5
            T2 = T * T
            T3 = T2 * T
            T4 = T3 * T
            T5 = T4 * T

            # From position equation at T: pf = a0 + a1*T + a2*T^2 + a3*T^3 + a4*T^4 + a5*T^5
            # From velocity equation at T: vf = a1 + 2*a2*T + 3*a3*T^2 + 4*a4*T^3 + 5*a5*T^4
            # From acceleration equation at T: af = 2*a2 + 6*a3*T + 12*a4*T^2 + 20*a5*T^3

            # Rearrange to solve for a3, a4, a5
            b1 = pf - p0 - v0*T - (a0/2.0)*T2
            b2 = vf - v0 - a0*T
            b3 = af - a0

            # Matrix equation: A * [a3, a4, a5]^T = [b1, b2, b3]^T
            A = np.array([
                [T3, T4, T5],
                [3*T2, 4*T3, 5*T4],
                [6*T, 12*T2, 20*T3]
            ])

            b = np.array([b1, b2, b3])
            coeffs = np.linalg.solve(A, b)

            self.coeffs_pos[i, 3] = coeffs[0]
            self.coeffs_pos[i, 4] = coeffs[1]
            self.coeffs_pos[i, 5] = coeffs[2]

        self.is_planned = True

    def evaluate(self, current_time: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate the trajectory at the given time.

        Args:
            current_time: Current time (seconds)

        Returns:
            Tuple of (position, velocity, acceleration)
        """
        if not self.is_planned:
            raise RuntimeError("Trajectory not planned. Call plan() first.")

        # Compute normalized time
        t = current_time - self.start_time

        # Clamp to trajectory duration
        if t < 0:
            t = 0
        elif t > self.trajectory_duration:
            t = self.trajectory_duration

        # Evaluate quintic polynomial
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t

        pos = np.zeros(3)
        vel = np.zeros(3)
        acc = np.zeros(3)

        for i in range(3):
            a = self.coeffs_pos[i]
            pos[i] = a[0] + a[1]*t + a[2]*t2 + a[3]*t3 + a[4]*t4 + a[5]*t5
            vel[i] = a[1] + 2*a[2]*t + 3*a[3]*t2 + 4*a[4]*t3 + 5*a[5]*t4
            acc[i] = 2*a[2] + 6*a[3]*t + 12*a[4]*t2 + 20*a[5]*t3

        return pos, vel, acc

    def evaluate_orientation(self, current_time: float) -> np.ndarray:
        """
        Evaluate orientation using SLERP.

        Args:
            current_time: Current time (seconds)

        Returns:
            Interpolated quaternion (w, x, y, z)
        """
        if self.start_quat is None or self.end_quat is None:
            return None

        t = current_time - self.start_time

        # Clamp and normalize
        if t < 0:
            t = 0
        elif t > self.trajectory_duration:
            t = self.trajectory_duration

        s = t / self.trajectory_duration if self.trajectory_duration > 0 else 1.0

        # Convert from (w, x, y, z) to scipy format (x, y, z, w)
        start_scipy = np.roll(self.start_quat, -1)
        end_scipy = np.roll(self.end_quat, -1)

        # SLERP using scipy
        from scipy.spatial.transform import Slerp, Rotation
        key_rots = Rotation.from_quat([start_scipy, end_scipy])
        slerp = Slerp([0, 1], key_rots)
        interp_rot = slerp(s)

        # Convert back to (w, x, y, z)
        return np.roll(interp_rot.as_quat(), 1)

    def needs_replan(self, current_time: float, replan_threshold: float = 0.1) -> bool:
        """
        Check if trajectory needs to be re-planned.

        Args:
            current_time: Current time (seconds)
            replan_threshold: Time before end to trigger replan (seconds)

        Returns:
            True if replan is needed
        """
        if not self.is_planned:
            return True

        elapsed = current_time - self.start_time
        return elapsed >= (self.trajectory_duration - replan_threshold)

    def get_remaining_time(self, current_time: float) -> float:
        """Get remaining time in current trajectory segment."""
        if not self.is_planned:
            return 0.0
        elapsed = current_time - self.start_time
        return max(0.0, self.trajectory_duration - elapsed)


def generate_circle_trajectory(elapsed_time: float,
                               circle_center: np.ndarray,
                               circle_radius: float,
                               angular_speed: float,
                               R_slope: np.ndarray,
                               size_z: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate desired position, velocity, and acceleration for circle trajectory.

    Args:
        elapsed: Elapsed time since start of circle drawing
        center: Center of the circle (3D position)
        radius: Radius of the circle
        angular_speed: Angular speed (rad/s)
    """
    angle = angular_speed * elapsed_time % (2 * np.pi)
    target_pos_local = np.zeros(3)
    x_dot_desired_local = np.zeros(3)
    x_ddot_desired_local = np.zeros(3)
    # x 
    target_pos_local[0] = circle_radius * np.cos(angle)
    target_pos_local[1] = circle_radius * np.sin(angle)
    target_pos_local[2] = size_z # Keep Z at table height
    # x_dot
    x_dot_desired_local[0] = -circle_radius * angular_speed * np.sin(angle)
    x_dot_desired_local[1] =  circle_radius * angular_speed * np.cos(angle)
    x_dot_desired_local[2] = 0.0
    # x_ddot
    x_ddot_desired_local[0] = -circle_radius * angular_speed**2 * np.cos(angle)
    x_ddot_desired_local[1] = -circle_radius * angular_speed**2 * np.sin(angle)
    x_ddot_desired_local[2] = 0.0

    # target_pos[:] = circle_center + (R_slope @ target_pos_local)
    # x_dot_desired[:] = R_slope @ x_dot_desired_local
    # x_ddot_desired[:] = R_slope @ x_ddot_desired_local

    return circle_center + (R_slope @ target_pos_local), R_slope @ x_dot_desired_local, R_slope @ x_ddot_desired_local

class ControlPhase(Enum):
    """Control phase state machine."""
    APPROACHING = 1
    CIRCLE_DRAWING = 2
    STOPPED = 3


@dataclass
class ControllerConfig:
    """Configuration parameters shared across all controllers."""
    # Simulation parameters
    dt: float = 0.001 # only for result plotting
    gravity_compensation: bool = True

    # Circle drawing parameters
    circle_center: np.ndarray = None
    circle_radius: float = 0.1
    circle_duration: float = 10.0
    angular_speed: float = np.pi * 2

    # Contact detection thresholds
    position_tolerance: float = 0.05  # 1cm tolerance for reaching target

    # Constraint geometry
    euler: np.ndarray = None
    size_z: float = 0.00
    use_table: bool = False

    def __post_init__(self):
        """Set default values for array parameters."""
        if self.circle_center is None:
            self.circle_center = np.array([0.5, 0.0, 0.3])
        if self.euler is None:
            self.euler = np.array([np.deg2rad(0), 0, 0])


@dataclass
class CartesianSpacePDControlConfig:
    """
    Configuration for Operational Space PD control.

    Control law:
        tau = J^T M_x (Kp * twist - Kd * J * qvel) + N^T tau_null + g(q)

    where twist is computed from pose error with gain Kpos.
    """
    Kpos: float = 0.95  # Position error gain
    Kp: np.ndarray = None  # Task space proportional gain
    Kd: np.ndarray = None  # Task space derivative gain
    Kp_null: np.ndarray = None
    Kd_null: np.ndarray = None
    impedance_pos: np.ndarray = None
    impedance_ori: np.ndarray = None

    # Trajectory planning parameters
    planning_horizon: float = 1.0  # Re-plan trajectory every this many seconds
    use_trajectory_planner: bool = True  # Enable/disable trajectory planning

    def __post_init__(self):
        if self.impedance_pos is None:
            self.impedance_pos = np.asarray([50.0, 50.0, 50.0]) * 0.2
        if self.impedance_ori is None:
            self.impedance_ori = np.asarray([25.0, 25.0, 25.0]) * 0.2
        if self.Kp is None:
            self.Kp = np.concatenate([self.impedance_pos, self.impedance_ori], axis=0)
        if  self.Kd is None:
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
    """Configuration for circle drawing controller."""
    # Impedance control gains
    damping_ratio: float = 1.0
    impedance_pos: np.ndarray = None
    impedance_ori: np.ndarray = None
    Kp_null: np.ndarray = None
    Kd_null: np.ndarray = None

    # Material stiffness
    k_normal: float = 5000.0

    # Force control gains
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
    """
    Controller for moving end-effector to desired position.

    Uses task-space impedance control with nullspace joint control.
    Includes trajectory planning that re-plans every planning_horizon seconds.
    Transitions to circle drawing when target is reached.
    """

    def __init__(self, config: CartesianSpacePDControlConfig, common_config: ControllerConfig):
        """
        Initialize approach controller.

        Args:
            config: Approach-specific configuration
            common_config: Shared configuration parameters
        """
        self.config = config
        self.common_config = common_config

        self.pino_model: Optional[pino.Model] = None
        self.pino_data: Optional[pino.Data] = None

        # Target pose (final goal)
        self.target_pos: Optional[np.ndarray] = None
        self.target_quat: Optional[np.ndarray] = None
        self.q0: Optional[np.ndarray] = None  # Home configuration

        # Trajectory planner
        self.trajectory_planner: Optional[TrajectoryPlanner] = None
        if self.config.use_trajectory_planner:
            self.trajectory_planner = TrajectoryPlanner(
                planning_horizon=self.config.planning_horizon
            )

        # Current desired state from trajectory
        self.desired_pos: Optional[np.ndarray] = None
        self.desired_vel: Optional[np.ndarray] = None
        self.desired_acc: Optional[np.ndarray] = None
        self.desired_quat: Optional[np.ndarray] = None

        # Timing
        self.start_time: float = 0.0
        self.current_time: float = 0.0
        self.last_replan_time: float = 0.0

        # Control output
        self.tau: np.ndarray = np.zeros(7)

        # Data logging
        self.ee_positions: list = []
        self.target_positions: list = []
        self.desired_positions: list = []  # Trajectory-planned desired positions
        self.joint_torques: list = []

    def starting(self, target_pos: np.ndarray, target_quat: np.ndarray, q0: np.ndarray, pino_model: pino.Model, pino_data: pino.Data) -> None:
        """
        Reset controller state.

        Args:
            target_pos: Target end-effector position
            target_quat: Target end-effector quaternion
        """
        self.pino_model = pino_model
        self.pino_data = pino_data
        self.target_pos = target_pos.copy()
        self.target_quat = target_quat.copy()
        self.q0 = q0.copy()

        # Reset timing
        self.start_time = 0.0
        self.current_time = 0.0
        self.last_replan_time = -float('inf')  # Force initial planning

        # Clear logging
        self.ee_positions = []
        self.target_positions = []
        self.desired_positions = []
        self.joint_torques = []

        # Zero control
        self.tau[:] = 0.0

        # Reset trajectory planner
        if self.trajectory_planner is not None:
            self.trajectory_planner.is_planned = False

        print(f"[APPROACH START] Target position: {self.target_pos}")
        print(f"[APPROACH START] Target quaternion: {self.target_quat}")
        print(f"[APPROACH START] Trajectory planning enabled: {self.config.use_trajectory_planner}")
        print(f"[APPROACH START] Planning horizon: {self.config.planning_horizon}s")

    def update(self, robot_state, current_time: float = 0.0) -> np.ndarray:
        """
        Compute control torques for approaching target.

        Args:
            robot_state: Current robot state
            current_time: Current time in seconds (used for trajectory planning)

        Returns:
            Control torques
        """
        self.current_time = current_time

        # Get current state
        q = np.array(robot_state.q)
        dq = np.array(robot_state.dq)

        # ============================================================
        # 0. Get End-Effector Pose
        # ============================================================
        O_T_EE = np.array(robot_state.O_T_EE).reshape(4, 4).T
        current_pos = O_T_EE[:3, 3]
        current_mat = O_T_EE[:3, :3]

        # ============================================================
        # 1. Compute Jacobian (needed for velocity)
        # ============================================================
        pino.forwardKinematics(self.pino_model, self.pino_data, q, dq)
        pino.computeJointJacobians(self.pino_model, self.pino_data)
        pino.updateFramePlacements(self.pino_model, self.pino_data)
        pino_frame_id = self.pino_model.getFrameId("attachment")
        jac = pino.getFrameJacobian(self.pino_model, self.pino_data, pino_frame_id, pino.LOCAL_WORLD_ALIGNED)

        # Current end-effector velocity
        current_vel = (jac @ dq)[:3]  # Only position velocity

        # ============================================================
        # 2. Trajectory Planning (re-plan every planning_horizon seconds)
        # ============================================================
        if self.config.use_trajectory_planner and self.trajectory_planner is not None:
            # Check if we need to (re-)plan
            time_since_last_replan = current_time - self.last_replan_time
            needs_replan = (
                not self.trajectory_planner.is_planned or
                time_since_last_replan >= self.config.planning_horizon
            )

            if needs_replan:
                # Get current orientation as quaternion
                current_rot = Rotation.from_matrix(current_mat)
                current_quat = np.roll(current_rot.as_quat(), 1)  # Convert to (w, x, y, z)

                # Plan trajectory from current state to target
                self.trajectory_planner.plan(
                    current_time=current_time,
                    start_pos=current_pos,
                    start_vel=current_vel,
                    end_pos=self.target_pos,
                    end_vel=np.zeros(3),  # Zero velocity at target
                    start_acc=np.zeros(3),  # Assume zero acceleration
                    end_acc=np.zeros(3),
                    duration=self.config.planning_horizon,
                    start_quat=current_quat,
                    end_quat=self.target_quat
                )
                self.last_replan_time = current_time
                logging.info("[TRAJECTORY] Re-planned at t=%.3f, distance to target: %.4f",
                           current_time, np.linalg.norm(self.target_pos - current_pos))

            # Evaluate trajectory at current time
            self.desired_pos, self.desired_vel, self.desired_acc = \
                self.trajectory_planner.evaluate(current_time)
            self.desired_quat = self.trajectory_planner.evaluate_orientation(current_time)
            if self.desired_quat is None:
                self.desired_quat = self.target_quat

        else:
            # No trajectory planning - go directly to target
            self.desired_pos = self.target_pos
            self.desired_vel = np.zeros(3)
            self.desired_acc = np.zeros(3)
            self.desired_quat = self.target_quat

        # ============================================================
        # 3. Compute End-Effector Pose Error (using desired from trajectory)
        # ============================================================
        twist = compute_ee_pose_error(
            self.desired_pos,
            current_pos,
            self.desired_quat,
            current_mat.flatten(),
            Kpos=self.config.Kpos
        )

        # ============================================================
        # 4. Compute Task-Space Inertia Matrix
        # ============================================================
        M_inv = pino.computeMinverse(self.pino_model, self.pino_data, q)
        Mx = task_space_inertiaM(M_inv, jac)

        # ============================================================
        # 5. Compute Task-Space Control with Feedforward
        # ============================================================
        # Velocity error (only for position, not orientation)
        vel_error = np.concatenate([self.desired_vel - current_vel, np.zeros(3)])

        # Feedforward acceleration (only for position)
        x_ddot_ff = np.concatenate([self.desired_acc, np.zeros(3)])

        # Control law: tau = J^T * Mx * (x_ddot_ff + Kp * twist + Kd * vel_error)
        self.tau[:] = jac.T @ Mx @ (
            x_ddot_ff + self.config.Kp * twist + self.config.Kd * vel_error
        )
        logging.info("position control: %s", np.round(self.tau, 4))

        # ============================================================
        # 6. Add Nullspace Control
        # ============================================================
        Jbar = M_inv @ jac.T @ Mx
        ddq = null_space_tau(
            q,
            dq,
            self.q0,
            self.config.Kp_null,
            self.config.Kd_null
        )
        self.tau += (np.eye(7) - jac.T @ Jbar.T) @ ddq

        # ============================================================
        # 7. Add Gravity Compensation
        # ============================================================
        if self.common_config.gravity_compensation:
            g_ctrl = pino.computeGeneralizedGravity(self.pino_model, self.pino_data, q)
            self.tau += g_ctrl

        # ============================================================
        # 8. Log Data
        # ============================================================
        self.ee_positions.append(current_pos.copy())
        self.target_positions.append(self.target_pos.copy())
        self.desired_positions.append(self.desired_pos.copy())
        self.joint_torques.append(self.tau.copy())

        return self.tau

    def is_target_reached(self, robot_state) -> bool:
        """
        Check if end-effector has reached target position.

        Returns:
            True if within tolerance
        """
        O_T_EE = np.array(robot_state.O_T_EE).reshape(4, 4).T
        current_pos = O_T_EE[:3, 3]
        distance = np.linalg.norm(current_pos - self.target_pos)
        return distance < self.common_config.position_tolerance


class HybridController:
    """
    Controller for drawing circles with force control.

    Uses hybrid force/motion control:
    - Force control in normal direction
    - Motion control in tangential directions
    """

    def __init__(self, config: HybridControllerConfig, common_config: ControllerConfig):
        """
        Initialize circle drawing controller.

        Args:
            config: Circle drawing configuration
            common_config: Shared configuration
        """
        self.config = config
        self.common_config = common_config

        # Control matrices
        damping_pos = self.config.damping_ratio * 2 * np.sqrt(self.config.impedance_pos)
        damping_ori = self.config.damping_ratio * 2 * np.sqrt(self.config.impedance_ori)
        self.Kp = np.concatenate([self.config.impedance_pos, self.config.impedance_ori])
        self.Kd = np.concatenate([damping_pos, damping_ori])

        # Selection matrices
        self.S_fc = np.zeros((6, 1))
        self.S_fc[2, 0] = 1  # Normal force (z)

        self.S_vc = np.zeros((6, 5))
        self.S_vc[0, 0] = 1  # x tangential
        self.S_vc[1, 1] = 1  # y tangential
        self.S_vc[3, 2] = 1  # rx rotation
        self.S_vc[4, 3] = 1  # ry rotation
        self.S_vc[5, 4] = 1  # rz rotation

        # Constraint geometry

        self.R_slope = euler_to_rot_matrix(self.common_config.euler)
        rot_slope = Rotation.from_euler('xyz', self.common_config.euler)
        self.quat_slope = np.roll(rot_slope.as_quat(),1)

        self.R = np.zeros((6, 6))
        self.R[0:3, 0:3] = self.R_slope
        self.R[3:6, 3:6] = self.R_slope
        self.S_f = self.R @ self.S_fc
        self.S_v = self.R @ self.S_vc

        


        # Trajectory state
        self.target_pos: Optional[np.ndarray] = None
        self.target_quat: Optional[np.ndarray] = None
        self.x_dot_desired: Optional[np.ndarray] = np.zeros(3)
        self.x_ddot_desired: Optional[np.ndarray] = np.zeros(3)
        self.q0: Optional[np.ndarray] = None

        # Circle drawing state
        self.start_time: float = 0.0
        self.is_drawing: bool = False

        # Preallocated workspace
        self.tau = np.zeros(7)

        # Data logging
        self.contact_forces: list = []
        self.desired_forces: list = []
        self.ee_positions: list = []
        self.target_positions: list = []
        self.control_force_compensation_arr: list = []
        self.contact_force_compensation_arr: list = []
        self.velocity_term_arr: list = []
        self.F_ctrl_constraint_arr: list = []
        self.joint_torques: list = []

    def starting(self, current_time: float, target_pos: np.ndarray, target_quat: np.ndarray, q0: np.ndarray, pino_model: pino.Model, pino_data: pino.Data) -> None:
        """
        Reset controller state when starting circle drawing.

        Args:
            current_time: Current simulation time
            target_pos: Starting position for circle
            target_quat: Target orientation
        """
        self.q0 = q0.copy()
        self.pino_model = pino_model
        self.pino_data = pino_data

        self.start_time = current_time
        self.is_drawing = True

        self.target_pos = target_pos.copy()
        self.target_quat = target_quat.copy()

        # Clear logging
        self.contact_forces = []
        self.desired_forces = []
        self.ee_positions = []
        self.target_positions = []
        self.control_force_compensation_arr = []
        self.contact_force_compensation_arr = []
        self.velocity_term_arr = []
        self.F_ctrl_constraint_arr = []
        self.joint_torques = []

        # Zero control
        self.tau[:] = 0.0

        print(f"[CIRCLE START] Circle drawing started at t={current_time:.2f}s")
        print(f"[CIRCLE START] Center: {self.common_config.circle_center}")
        print(f"[CIRCLE START] Radius: {self.common_config.circle_radius}")
        print(f"[CIRCLE START] Force control: F_desired={self.config.F_desired_contact}")
        print(f"[CIRCLE START] target quat = {self.target_quat}")

    def update(self, current_time: float, robot_state) -> np.ndarray:
        """
        Compute control torques for circle drawing.

        Args:
            current_time: Current simulation time

        Returns:
            Control torques
        """
        # ============================================================
        # 1. Update Trajectory
        # ============================================================
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
            # Stop after duration
            self.x_dot_desired[:] = 0.0
            self.x_ddot_desired[:] = 0.0
            self.is_drawing = False

        # Get current state
        q = np.array(robot_state.q)
        dq = np.array(robot_state.dq)
        # Get end-effector pose
        O_T_EE = np.array(robot_state.O_T_EE).reshape(4, 4).T
        current_pos = O_T_EE[:3, 3]
        current_mat = O_T_EE[:3, :3]
        # ============================================================
        # 2. Compute Jacobian and Dynamics
        # ============================================================
        pino.forwardKinematics(self.pino_model, self.pino_data, q, dq)
        pino.computeJointJacobians(self.pino_model, self.pino_data)
        pino.updateFramePlacements(self.pino_model, self.pino_data)
        pino_frame_id = self.pino_model.getFrameId("attachment")
        jac = pino.getFrameJacobian(self.pino_model, self.pino_data, pino_frame_id, pino.LOCAL_WORLD_ALIGNED)
        M = pino.crba(self.pino_model, self.pino_data, q)
        # M_inv = np.linalg.inv(M)
        M_inv = pino.computeMinverse(self.pino_model, self.pino_data, q)

        J_phi = self.S_f.T @ jac
        J_motion = self.S_v.T @ jac
        jac_1 = np.vstack([J_phi, J_motion])

        Mx_constraint = task_space_inertiaM(M_inv, J_phi)
        Mx_motion = task_space_inertiaM(M_inv, J_motion)

        # ============================================================
        # 4. Get Contact Information
        # ============================================================
        # if self.common_config.use_table:
        #     current_force_world, current_force_local, contact_pos = check_world_ee_contact_force(self.data, self.model)
        # else:
        #     current_force_world, current_force_local, contact_pos = check_world_ee_contact_force(self.data, self.model, obj_name='slope_geom')
        # F_ext_world = np.array(robot_state.O_F_ext_hat_K)
        F_ext_world = np.zeros(6)
        # TODO
        current_force_local = F_ext_world
        F_ext_phi = current_force_local @ self.S_fc
        F_ext_x = current_force_local @ self.S_vc
        F_ext_v = None

        # ============================================================
        # Null Space torque
        # ============================================================
        jac_1_inv = dynamically_consistent_inv(jac_1, M_inv)
        N2 = np.eye(7) - jac_1.T @ jac_1_inv.T
        tau_ctrl_v = null_space_tau(q, dq, self.q0, self.config.Kp_null, self.config.Kd_null)
        tau_ctrl_v = N2 @ tau_ctrl_v

        #---------------------------------------------------
        # Motion Space
        #----------------------------------------------------
        # Compute the motion-space inertia matrix for x-y plane
        twist = compute_ee_pose_error(
                    self.target_pos, 
                    current_pos,
                    self.target_quat,
                    current_mat.flatten()
                    )
        # logging.info("current pos: %s", current_pos)
        # logging.info("current mat: %s", current_mat)
        # logging.info("target pos: %s", self.target_pos)
        # logging.info("target quat: %s", self.target_quat)

        
        x_ddot_desired_sel = np.concatenate([self.x_ddot_desired, [0,0,0]]) @ self.S_v
        x_tilde = twist @ self.S_v
        site_vel = jac @ dq #[vx, vy, vz, wx, wy, wz]
        x_dot_tilde = (np.concatenate([self.x_dot_desired, [0,0,0]]) - site_vel) @ self.S_v
        a_motion = feedforward_PD(
            x_acc_desired=x_ddot_desired_sel,x_delta=x_tilde,
            x_dot_delta=x_dot_tilde,
            Kp=self.Kp @ self.S_v, Kd=self.Kd @ self.S_v
            )
        F_ctrl_x = Mx_motion @ a_motion
        tau_ctrl_x = J_motion.T @ F_ctrl_x

        #------------------------------------------------------
        # Constraint space
        #------------------------------------------------------
        # TODO
        C = pino.computeCoriolisMatrix(self.pino_model, self.pino_data, q, dq)
        J_dot = pino.getFrameJacobianTimeVariation(self.pino_model, self.pino_data, pino_frame_id, pino.LOCAL_WORLD_ALIGNED)
        J_phi_dot = self.S_f.T @ J_dot

        F_ext_x_new = F_ext_x.copy()
        F_ext_x_new[-3:] = 0
        control_force_compensation = 1 * (- Mx_constraint @ J_phi @ M_inv @ (tau_ctrl_x + tau_ctrl_v))
        contact_force_compensation = 1 * (Mx_constraint @ J_phi @ M_inv @ (J_motion.T @ F_ext_x_new))
        verlociy_term = 1 * Mx_constraint @ (J_phi @ M_inv @ C - J_phi_dot) @ dq
        F_ctrl_constraint = (
            self.config.F_desired_contact +
            control_force_compensation +
            contact_force_compensation + verlociy_term
        )

        #------------------------------------------------------
        # Sum up torques
        #------------------------------------------------------
        self.tau[:] = J_phi.T @ F_ctrl_constraint + tau_ctrl_x + tau_ctrl_v
        # self.tau[:] = tau_ctrl_x + tau_ctrl_v

        # Store for logging
        self._last_control_compensation = control_force_compensation
        self._last_contact_compensation = contact_force_compensation
        self._last_velocity_term = verlociy_term
        self._last_F_ctrl_constraint = F_ctrl_constraint

        # ============================================================
        # 6. Add Gravity Compensation
        # ============================================================
        # Use Pinocchio to compute gravity
        if self.common_config.gravity_compensation:
            self.tau += pino.computeGeneralizedGravity(self.pino_model, self.pino_data, q)
        # ============================================================
        # 7. Log Data
        # ============================================================
        self._log_data(current_force_local, current_pos)

        return self.tau

    def _log_data(self, F_ext_local: np.ndarray, current_pos: np.ndarray) -> None:
        """Log data for plotting."""
        self.contact_forces.append(F_ext_local[:3].copy())
        self.desired_forces.append(-self.config.F_desired_contact.copy())
        self.ee_positions.append(current_pos.copy())
        self.target_positions.append(self.target_pos.copy())
        self.joint_torques.append(self.tau.copy())

        if hasattr(self, '_last_control_compensation'):
            self.control_force_compensation_arr.append(self._last_control_compensation.copy())
            self.contact_force_compensation_arr.append(self._last_contact_compensation.copy())
            self.velocity_term_arr.append(self._last_velocity_term.copy())
            self.F_ctrl_constraint_arr.append(self._last_F_ctrl_constraint.copy())
        else:
            self.control_force_compensation_arr.append(np.zeros(1))
            self.contact_force_compensation_arr.append(np.zeros(1))
            self.velocity_term_arr.append(np.zeros(1))
            self.F_ctrl_constraint_arr.append(np.zeros(1))

    def is_finished(self) -> bool:
        """Check if circle drawing is finished."""
        return not self.is_drawing


def plot_joint_torques(
        approach_controller: CartesianSpacePDController,
        circle_controller: HybridController,
        dt: float,
        transition_time: float
) -> None:
    """Plot joint torques from both controllers for each joint."""
    # Ensure plots directory exists
    os.makedirs("plots", exist_ok=True)

    # Combine torques from both controllers
    approach_torques = np.array(approach_controller.joint_torques) if approach_controller.joint_torques else np.empty((0, 7))
    circle_torques = np.array(circle_controller.joint_torques) if circle_controller.joint_torques else np.empty((0, 7))

    all_torques = np.vstack([approach_torques, circle_torques]) if approach_torques.size > 0 and circle_torques.size > 0 else (
        approach_torques if approach_torques.size > 0 else circle_torques
    )

    if all_torques.size == 0:
        print("[PLOT] No torque data to plot")
        return

    time_steps = np.arange(len(all_torques)) * dt

    # Create figure with 7 subplots (one per joint)
    fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
    fig.suptitle('Joint Torques Over Time', fontsize=14)

    for i in range(7):
        axes[i].plot(time_steps, all_torques[:, i], 'b-', linewidth=1.5)
        if transition_time > 0:
            axes[i].axvline(transition_time, color='g', linestyle='--', alpha=0.7, label='Transition')
        axes[i].set_ylabel(f'Joint {i+1} (Nm)')
        axes[i].grid(True, alpha=0.3)
        if i == 0 and transition_time > 0:
            axes[i].legend(loc='upper right')

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    fig.savefig("plots/joint_torques.png", dpi=150)
    print("[PLOT] Joint torques saved to plots/joint_torques.png")
    plt.show()


# def plot_results(
#         approach_controller: CartesianSpacePDController,
#         circle_controller: HybridController,
#         dt: float,
#         transition_time: float
# ) -> None:
#     """Plot results from both controllers."""

#     # Combine data from both controllers
#     all_ee_pos = approach_controller.ee_positions + circle_controller.ee_positions
#     all_target_pos = approach_controller.target_positions + circle_controller.target_positions

#     # ============================================================
#     # Plot Position Tracking
#     # ============================================================
#     ee_positions = np.array(all_ee_pos)
#     target_positions = np.array(all_target_pos)
#     time_steps = np.arange(len(ee_positions)) * dt

#     fig, axes = plt.subplots(3, 1, figsize=(10, 8))
#     axes_labels = ['X', 'Y', 'Z']

#     for i in range(3):
#         axes[i].plot(time_steps, ee_positions[:, i], 'b-', linewidth=2, label='End-Effector')
#         axes[i].plot(time_steps, target_positions[:, i], 'r--', linewidth=2, label='Target')
#         axes[i].axvline(transition_time, color='g', linestyle=':', label='Transition')
#         axes[i].set_ylabel(f'{axes_labels[i]} Position (m)')
#         axes[i].legend()
#         axes[i].grid(True, alpha=0.3)
#         axes[i].set_title(f'{axes_labels[i]} Position Tracking')

#     axes[2].set_xlabel('Time (s)')
#     plt.tight_layout()
#     fig.savefig("plots/combined_position_tracking.png")

#     # ============================================================
#     # Plot Contact Forces (Circle Drawing Phase Only)
#     # ============================================================
#     contact_forces = np.array(circle_controller.contact_forces)
#     desired_forces = np.array(circle_controller.desired_forces)

#     if len(contact_forces) > 0:
#         if contact_forces.ndim == 1:
#             contact_forces = contact_forces[:, None]
#             desired_forces = desired_forces[:, None]

#         timesteps, n_dim = contact_forces.shape
#         t = np.arange(timesteps) * dt + transition_time

#         plt.figure(figsize=(8, 3 * n_dim))
#         for i in range(n_dim):
#             plt.subplot(n_dim, 1, i + 1)
#             plt.plot(t, contact_forces[:, i], label="Contact force")
#             plt.plot(t, desired_forces[:, 0], label="Desired force")
#             plt.ylabel(f"Dim {i + 1}")
#             plt.xlabel("Time [s]")
#             plt.legend()
#             plt.grid(True)
#         plt.tight_layout()
#         plt.savefig("plots/contact_forces.png")

#     plt.show()
#     print("[PLOT] Results saved to plots/ directory")


def main() -> None:
    """Main function with two-phase control."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Hybrid force/impedance control for Franka Panda")
    parser.add_argument("--ip", type=str, default="localhost", help="Robot IP address")
    parser.add_argument("--approach-only", action="store_true", help="Only run approach phase (no circle drawing)")
    args = parser.parse_args()

    # ============================================================
    # 1. Create Configurations
    # ============================================================
    common_config = ControllerConfig()
    approach_config = CartesianSpacePDControlConfig()
    circle_config = HybridControllerConfig()
    q0 = np.array([0,0,0,-1.57079,0,1.57079,-0.7853])
    # q0 = [0.02366284, 0.94320843, -0.01978183, -1.85594285, 0.04376186, 2.78281701, 0.6891366]

    # ============================================================
    # 2. Load Model
    # ============================================================
    pino_model = pino.buildModelFromMJCF("mj_ctrl/franka_fr3/fr3.xml")
    pino_data = pino_model.createData()
    try:
        # Connect to robot
        print(f"Connecting to robot at {args.ip}...")
        robot = Robot(args.ip)

        # Set collision behavior
        robot.set_collision_behavior(
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        )

        # Safety warning
        print("\n" + "="*60)
        print("WARNING: This will move the robot!")
        print("Make sure:")
        print("  1. The workspace is clear")
        print("  2. Emergency stop is accessible")
        print("  3. You understand the trajectory")
        print("="*60)
        input("Press Enter to continue...")

        # ============================================================
        # 3. Create Controllers
        # ============================================================
        approach_controller = CartesianSpacePDController(approach_config, common_config)
        circle_controller = HybridController(circle_config, common_config)


        # ============================================================
        # 4. Setup Initial Targets
        # ============================================================

        # Generate target position
        R_slope = euler_to_rot_matrix(common_config.euler)
        target_pos = generate_start_position(
            common_config.circle_radius,
            common_config.circle_center,
            common_config.size_z,
            R_slope
        )

        # Generate target orientation
        # q = (w, x, y, z)
        target_quat = np.array([0., 1., 0., 0.])
        # quat_slope = np.zeros(4)
        # mujoco.mju_euler2Quat(quat_slope, common_config.euler, 'XYZ')
        # mujoco.mju_mulQuat(target_quat, quat_slope, target_quat)
        rot_slope = Rotation.from_euler('xyz', common_config.euler)
        rot_target = Rotation.from_quat(np.roll(target_quat, -1))
        target_quat = np.roll((rot_slope * rot_target).as_quat(), 1)

        # Start torque control
        print("\nStarting torque control...")
        active_control = robot.start_torque_control()
        # this function doesn't work, get rid of it
        # model = robot.load_model()

        # ============================================================
        # 5. Start Approach Phase
        # ============================================================
        control_phase = ControlPhase.APPROACHING
        approach_controller.starting(target_pos, target_quat, q0, pino_model, pino_data)

        print("\n" + "=" * 60)
        print("PHASE 1: APPROACHING TARGET POSITION")
        print("=" * 60)
        print(f"Target Quat: {target_quat}")

        # ============================================================
        # 6. Run Control Loop
        # ============================================================
        sim_time = 0.0
        transition_time = 0.0
        try:
            while True:
                loop_start = time.perf_counter()
                step_start = time.time()
                # Read robot state
                robot_state, duration = active_control.readOnce()
                try:
                    logging.info("Last commanded torques from controller: %s", np.round(robot_state.tau_J_d, 4).tolist())
                except (AttributeError, TypeError):
                    print("  Last commanded torques from controller: <not available>")

                # ============================================================
                # State Machine: Switch Controllers
                # ============================================================
                if control_phase == ControlPhase.APPROACHING:
                    # Use approach controller
                    tau = approach_controller.update(robot_state, sim_time)
                    # Check if target reached
                    if approach_controller.is_target_reached(robot_state):
                        print("\n" + "=" * 60)
                        print(f"TARGET REACHED at t={sim_time:.2f}s!")
                        print("PHASE 2: CIRCLE DRAWING")
                        print("=" * 60 + "\n")

                        if args.approach_only:
                            print("Approach-only mode: stopping here.")
                            control_phase = ControlPhase.STOPPED
                        else:
                            print("PHASE 2: CIRCLE DRAWING")
                            print("="*60)
                            control_phase = ControlPhase.CIRCLE_DRAWING
                            transition_time = sim_time
                            circle_controller.starting(sim_time, target_pos, target_quat, q0, pino_model, pino_data)

                elif control_phase == ControlPhase.CIRCLE_DRAWING:
                    # Use circle drawing controller
                    tau = circle_controller.update(sim_time, robot_state)

                    # Check if finished
                    if circle_controller.is_finished():
                        print("\n" + "=" * 60)
                        print(f"CIRCLE DRAWING FINISHED at t={sim_time:.2f}s!")
                        print("=" * 60 + "\n")
                        control_phase = ControlPhase.STOPPED

                else:  # STOPPED
                    tau = pino.computeGeneralizedGravity(pino_model, pino_data, np.array(robot_state.q))
                    # tau = np.array(model.gravity(robot_state))

                    # Signal motion finished and exit
                    torque_cmd = Torques(tau.tolist())
                    torque_cmd.motion_finished = True
                    active_control.writeOnce(torque_cmd)
                    break

                # ============================================================
                # Apply Control and Step Simulation
                # ============================================================
                logging.info("tau: %s", np.round(tau, 4))
                torque_cmd = Torques(tau.tolist())
                active_control.writeOnce(torque_cmd)
                loop_end = time.perf_counter()
                logging.info("time: %s", (loop_end - loop_start) * 1e6)

                # Update time
                sim_time += duration.to_sec()

                # # # Maintain control rate (1kHz)
                # elapsed = time.time() - step_start
                # if elapsed < common_config.dt:
                #     time.sleep(common_config.dt - elapsed)

                # sim_time += common_config.dt
            # ============================================================
            # 7. Plot Results
            # ============================================================
            print("\n[MAIN] Simulation complete. Generating plots...")
            # plot_results(approach_controller, circle_controller, common_config.dt, transition_time)
            np.savez(
                "force_details.npz",
                control_force_compensation_arr=circle_controller.control_force_compensation_arr,
                contact_force_compensation_arr=circle_controller.contact_force_compensation_arr,
                velocity_term_arr=circle_controller.velocity_term_arr,
                F_ctrl_constraint_arr=circle_controller.F_ctrl_constraint_arr,
                approach_joint_torques=approach_controller.joint_torques,
                circle_joint_torques=circle_controller.joint_torques
                )
            plot_joint_torques(approach_controller, circle_controller, common_config.dt, transition_time)
        except KeyboardInterrupt:
            print("\nControl interrupted by user")
            # Send zero torques
            torque_cmd = Torques([0.0] * 7)
            torque_cmd.motion_finished = True
            active_control.writeOnce(torque_cmd)
            print("\n[MAIN] Save force detail data into npz file...")
            np.savez(
                "force_details.npz",
                control_force_compensation_arr=circle_controller.control_force_compensation_arr,
                contact_force_compensation_arr=circle_controller.contact_force_compensation_arr,
                velocity_term_arr=circle_controller.velocity_term_arr,
                F_ctrl_constraint_arr=circle_controller.F_ctrl_constraint_arr,
                approach_joint_torques=approach_controller.joint_torques,
                circle_joint_torques=circle_controller.joint_torques
                )
            plot_joint_torques(approach_controller, circle_controller, common_config.dt, transition_time)

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