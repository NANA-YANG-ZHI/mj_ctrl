# ------------------------------------------------------------------------------
# Hybrid Force-Impedance Control for Fast End-Effector Motions
# Adapted for libfranka-python - Real Robot Control
# 1. Use libfranka-python for robot communication and control
# 2. Use pinocchio to load model dynamics and calculate jac, M and g
# 3. Real-time control loop at 1kHz
# ------------------------------------------------------------------------------

import numpy as np
import time
import pinocchio as pino
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import argparse

# Import libfranka-python
try:
    import libfranka
except ImportError:
    print("ERROR: libfranka-python not installed!")
    print("Please install with: pip install libfranka-python")
    exit(1)

from utils import *
import matplotlib.pyplot as plt


def generate_circle_trajectory(elapsed_time: float,
                               circle_center: np.ndarray,
                               circle_radius: float,
                               angular_speed: float,
                               R_slope: np.ndarray,
                               size_z: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate desired position, velocity, and acceleration for circle trajectory.

    Args:
        elapsed_time: Elapsed time since start of circle drawing
        circle_center: Center of the circle (3D position)
        circle_radius: Radius of the circle
        angular_speed: Angular speed (rad/s)
        R_slope: Rotation matrix for the slope
        size_z: Z offset for the circle
    """
    angle = angular_speed * elapsed_time % (2 * np.pi)
    target_pos_local = np.zeros(3)
    x_dot_desired_local = np.zeros(3)
    x_ddot_desired_local = np.zeros(3)
    # x
    target_pos_local[0] = circle_radius * np.cos(angle)
    target_pos_local[1] = circle_radius * np.sin(angle)
    target_pos_local[2] = size_z  # Keep Z at table height
    # x_dot
    x_dot_desired_local[0] = -circle_radius * angular_speed * np.sin(angle)
    x_dot_desired_local[1] =  circle_radius * angular_speed * np.cos(angle)
    x_dot_desired_local[2] = 0.0
    # x_ddot
    x_ddot_desired_local[0] = -circle_radius * angular_speed**2 * np.cos(angle)
    x_ddot_desired_local[1] = -circle_radius * angular_speed**2 * np.sin(angle)
    x_ddot_desired_local[2] = 0.0

    return circle_center + (R_slope @ target_pos_local), R_slope @ x_dot_desired_local, R_slope @ x_ddot_desired_local


class ControlPhase(Enum):
    """Control phase state machine."""
    APPROACHING = 1
    CIRCLE_DRAWING = 2
    STOPPED = 3


@dataclass
class ControllerConfig:
    """Configuration parameters shared across all controllers."""
    # Control parameters
    dt: float = 0.001  # 1kHz control loop
    gravity_compensation: bool = True

    # Circle drawing parameters
    circle_center: np.ndarray = None
    circle_radius: float = 0.1
    circle_duration: float = 10.0
    angular_speed: float = np.pi

    # Contact detection thresholds
    position_tolerance: float = 0.01  # 1cm tolerance for reaching target

    # Constraint geometry
    euler: np.ndarray = None
    size_z: float = 0.01
    use_table: bool = False

    def __post_init__(self):
        """Set default values for array parameters."""
        if self.circle_center is None:
            self.circle_center = np.array([0.5, 0.0, 0.45])
        if self.euler is None:
            self.euler = np.array([np.deg2rad(-10), 0, 0])


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

    def __post_init__(self):
        if self.impedance_pos is None:
            self.impedance_pos = np.asarray([500.0, 500.0, 500.0])
        if self.impedance_ori is None:
            self.impedance_ori = np.asarray([250.0, 250.0, 250.0])
        if self.Kp is None:
            self.Kp = np.concatenate([self.impedance_pos, self.impedance_ori], axis=0)
        if self.Kd is None:
            damping_ratio = 1.0
            damping_pos = damping_ratio * 2 * np.sqrt(self.impedance_pos)
            damping_ori = damping_ratio * 2 * np.sqrt(self.impedance_ori)
            self.Kd = np.concatenate([damping_pos, damping_ori], axis=0)
        if self.Kp_null is None:
            self.Kp_null = np.asarray([75.0, 75.0, 50.0, 50.0, 40.0, 25.0, 25.0])
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
            self.impedance_pos = np.asarray([500.0, 500.0, 500.0]) * 2
        if self.impedance_ori is None:
            self.impedance_ori = np.asarray([250.0, 250.0, 250.0]) * 2
        if self.Kp_null is None:
            self.Kp_null = np.asarray([75.0, 75.0, 50.0, 50.0, 40.0, 25.0, 25.0])
            self.Kd_null = self.damping_ratio * 2 * np.sqrt(self.Kp_null)
        if self.F_desired_contact is None:
            self.F_desired_contact = np.array([-10.0])


class RobotState:
    """Container for robot state from libfranka."""
    def __init__(self):
        self.q = np.zeros(7)  # Joint positions
        self.dq = np.zeros(7)  # Joint velocities
        self.tau_J = np.zeros(7)  # Joint torques
        self.O_T_EE = np.zeros(16)  # End-effector pose (4x4 matrix flattened)
        self.O_F_ext_hat_K = np.zeros(6)  # External forces/torques


class CartesianSpacePDController:
    """
    Controller for moving end-effector to desired position.

    Uses task-space impedance control with nullspace joint control.
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

        # Robot structure (set in init)
        self.pino_model: Optional[pino.Model] = None
        self.pino_data: Optional[pino.Data] = None
        self.pino_frame_id: int = -1
        self.n_joints: int = 7

        # Target pose
        self.target_pos: Optional[np.ndarray] = None
        self.target_quat: Optional[np.ndarray] = None
        self.q0: Optional[np.ndarray] = None  # Home configuration

        # Control output
        self.tau: np.ndarray = np.zeros(7)

        # Data logging
        self.ee_positions: list = []
        self.target_positions: list = []

    def init(
            self,
            pino_model: pino.Model,
            pino_data: pino.Data,
            q0: np.ndarray
    ) -> bool:
        """
        Initialize controller with robot model.

        Args:
            pino_model: Pinocchio model
            pino_data: Pinocchio data
            q0: Home joint configuration

        Returns:
            True if successful
        """
        try:
            self.pino_model = pino_model
            self.pino_data = pino_data
            self.n_joints = pino_model.nq
            self.pino_frame_id = pino_model.getFrameId("panda_hand")

            # Preallocate workspace
            self.jac = np.zeros((6, self.n_joints))
            self.M_inv = np.zeros((self.n_joints, self.n_joints))
            self.Mx = np.zeros((6, 6))
            self.tau = np.zeros(self.n_joints)

            # Get home configuration
            self.q0 = q0.copy()

            print(f"[APPROACH INIT] Controller initialized")
            print(f"  - Kpos={self.config.Kpos}, Kp={self.config.Kp}, Kd={self.config.Kd}")

            return True

        except Exception as e:
            print(f"[APPROACH INIT] Failed: {e}")
            return False

    def starting(self, target_pos: np.ndarray, target_quat: np.ndarray) -> None:
        """
        Reset controller state.

        Args:
            target_pos: Target end-effector position
            target_quat: Target end-effector quaternion
        """
        self.target_pos = target_pos.copy()
        self.target_quat = target_quat.copy()

        # Clear logging
        self.ee_positions = []
        self.target_positions = []

        # Zero control
        self.tau[:] = 0.0

        print(f"[APPROACH START] Target position: {self.target_pos}")
        print(f"[APPROACH START] Target quaternion: {self.target_quat}")

    def update(self, robot_state: RobotState) -> np.ndarray:
        """
        Compute control torques for approaching target.

        Args:
            robot_state: Current robot state from libfranka

        Returns:
            Control torques
        """
        # ============================================================
        # 1. Get current end-effector pose from robot state
        # ============================================================
        # O_T_EE is a 4x4 transformation matrix in column-major format
        ee_transform = robot_state.O_T_EE.reshape(4, 4, order='F')
        current_pos = ee_transform[:3, 3]
        current_rot = ee_transform[:3, :3]

        # ============================================================
        # 2. Compute End-Effector Pose Error
        # ============================================================
        twist = self._compute_pose_error(
            self.target_pos,
            current_pos,
            self.target_quat,
            current_rot
        )

        # ============================================================
        # 3. Compute Jacobian using Pinocchio
        # ============================================================
        pino.forwardKinematics(self.pino_model, self.pino_data, robot_state.q, robot_state.dq)
        pino.computeJointJacobians(self.pino_model, self.pino_data)
        pino.updateFramePlacements(self.pino_model, self.pino_data)
        self.jac[:] = pino.getFrameJacobian(self.pino_model, self.pino_data, self.pino_frame_id, pino.LOCAL_WORLD_ALIGNED)

        # ============================================================
        # 4. Compute Task-Space Inertia Matrix
        # ============================================================
        self.M_inv = pino.computeMinverse(self.pino_model, self.pino_data, robot_state.q)
        self.Mx = task_space_inertiaM(self.M_inv, self.jac)

        # ============================================================
        # 5. Compute Task-Space Control
        # ============================================================
        self.tau[:] = self.jac.T @ self.Mx @ (
                self.config.Kp * twist - self.config.Kd * (self.jac @ robot_state.dq)
        )

        # ============================================================
        # 6. Add Nullspace Control
        # ============================================================
        Jbar = self.M_inv @ self.jac.T @ self.Mx
        ddq = null_space_tau(robot_state.q, robot_state.dq, self.q0, self.config.Kp_null, self.config.Kd_null)
        self.tau += (np.eye(self.n_joints) - self.jac.T @ Jbar.T) @ ddq

        # ============================================================
        # 7. Add Gravity Compensation
        # ============================================================
        if self.common_config.gravity_compensation:
            g = pino.computeGeneralizedGravity(self.pino_model, self.pino_data, robot_state.q)
            self.tau += g

        # ============================================================
        # 8. Log Data
        # ============================================================
        self.ee_positions.append(current_pos.copy())
        self.target_positions.append(self.target_pos.copy())

        return self.tau

    def _compute_pose_error(self, target_pos, current_pos, target_quat, current_rot):
        """Compute 6D pose error as a twist."""
        twist = np.zeros(6)

        # Position error
        dx = target_pos - current_pos
        twist[:3] = self.config.Kpos * dx

        # Orientation error
        # Convert rotation matrix to quaternion
        current_quat = self._rot_to_quat(current_rot)

        # Quaternion error
        error_quat = self._quat_multiply(target_quat, self._quat_conjugate(current_quat))

        # Convert to angular velocity
        twist[3:] = self.config.Kpos * 2.0 * error_quat[1:]

        return twist

    @staticmethod
    def _rot_to_quat(R):
        """Convert rotation matrix to quaternion [w, x, y, z]."""
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        return np.array([w, x, y, z])

    @staticmethod
    def _quat_conjugate(q):
        """Return conjugate of quaternion [w, x, y, z]."""
        return np.array([q[0], -q[1], -q[2], -q[3]])

    @staticmethod
    def _quat_multiply(q1, q2):
        """Multiply two quaternions [w, x, y, z]."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def is_target_reached(self, robot_state: RobotState) -> bool:
        """
        Check if end-effector has reached target position.

        Args:
            robot_state: Current robot state

        Returns:
            True if within tolerance
        """
        ee_transform = robot_state.O_T_EE.reshape(4, 4, order='F')
        current_pos = ee_transform[:3, 3]
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

        # Robot structure (set in init)
        self.pino_model: Optional[pino.Model] = None
        self.pino_data: Optional[pino.Data] = None
        self.pino_frame_id: int = -1
        self.n_joints: int = 7

        # Control matrices
        self.Kp: Optional[np.ndarray] = None
        self.Kd: Optional[np.ndarray] = None
        self.K_material: Optional[np.ndarray] = None
        self.Compliance_matrix: Optional[np.ndarray] = None

        # Selection matrices
        self.S_fc: Optional[np.ndarray] = None
        self.S_vc: Optional[np.ndarray] = None
        self.S_v: Optional[np.ndarray] = None
        self.S_f: Optional[np.ndarray] = None

        # Constraint geometry
        self.R_slope: Optional[np.ndarray] = None
        self.quat_slope: Optional[np.ndarray] = None
        self.R: Optional[np.ndarray] = None

        # Trajectory state
        self.target_pos: Optional[np.ndarray] = None
        self.target_quat: Optional[np.ndarray] = None
        self.x_dot_desired: Optional[np.ndarray] = None
        self.x_ddot_desired: Optional[np.ndarray] = None
        self.q0: Optional[np.ndarray] = None

        # Circle drawing state
        self.start_time: float = 0.0
        self.is_drawing: bool = False

        # Preallocated workspace
        self.tau: Optional[np.ndarray] = None

        # Data logging
        self.contact_forces: list = []
        self.desired_forces: list = []
        self.ee_positions: list = []
        self.target_positions: list = []
        self.control_force_compensation_arr: list = []
        self.contact_force_compensation_arr: list = []
        self.velocity_term_arr: list = []
        self.F_ctrl_constraint_arr: list = []

    def init(
            self,
            pino_model: pino.Model,
            pino_data: pino.Data,
            q0: np.ndarray
    ) -> bool:
        """
        Initialize controller with robot model.

        Args:
            pino_model: Pinocchio model
            pino_data: Pinocchio data
            q0: Home joint configuration

        Returns:
            True if successful
        """
        try:
            self.pino_model = pino_model
            self.pino_data = pino_data
            self.pino_frame_id = pino_model.getFrameId("panda_hand")
            self.n_joints = pino_model.nq

            # ============================================================
            # Setup Control Gains
            # ============================================================
            damping_pos = self.config.damping_ratio * 2 * np.sqrt(self.config.impedance_pos)
            damping_ori = self.config.damping_ratio * 2 * np.sqrt(self.config.impedance_ori)
            self.Kp = np.concatenate([self.config.impedance_pos, self.config.impedance_ori])
            self.Kd = np.concatenate([damping_pos, damping_ori])

            # ============================================================
            # Setup Material Stiffness
            # ============================================================
            k_n = self.config.k_normal
            self.K_material = np.diag([
                k_n * 0.1, k_n * 0.1, k_n * 0.1,  # xyz
                k_n * 0.01, k_n * 0.01, k_n * 0.01  # rotations
            ])
            self.Compliance_matrix = np.linalg.inv(self.K_material)

            # ============================================================
            # Setup Constraint Geometry
            # ============================================================
            self.R_slope = euler_to_rot_matrix(self.common_config.euler)

            # ============================================================
            # Setup Selection Matrices
            # ============================================================
            self.S_fc = np.zeros((6, 1))
            self.S_fc[2, 0] = 1  # Normal force (z)

            self.S_vc = np.zeros((6, 5))
            self.S_vc[0, 0] = 1  # x tangential
            self.S_vc[1, 1] = 1  # y tangential
            self.S_vc[3, 2] = 1  # rx rotation
            self.S_vc[4, 3] = 1  # ry rotation
            self.S_vc[5, 4] = 1  # rz rotation
            self.R = np.zeros((6, 6))
            self.R[0:3, 0:3] = self.R_slope
            self.R[3:6, 3:6] = self.R_slope
            self.S_f = self.R @ self.S_fc
            self.S_v = self.R @ self.S_vc

            # ============================================================
            # Preallocate Workspace
            # ============================================================
            self.tau = np.zeros(self.n_joints)

            # Trajectory variables
            self.x_dot_desired = np.zeros(3)
            self.x_ddot_desired = np.zeros(3)

            # Get home configuration
            self.q0 = q0.copy()

            print(f"[CIRCLE INIT] Controller initialized")
            print(f"  - Force control: F_desired={self.config.F_desired_contact}")
            print(f"  - Material stiffness: k_normal={self.config.k_normal}")

            return True

        except Exception as e:
            print(f"[CIRCLE INIT] Failed: {e}")
            return False

    def starting(self, current_time: float, target_pos: np.ndarray, target_quat: np.ndarray) -> None:
        """
        Reset controller state when starting circle drawing.

        Args:
            current_time: Current time
            target_pos: Starting position for circle
            target_quat: Target orientation
        """
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

        # Zero control
        self.tau[:] = 0.0

        print(f"[CIRCLE START] Circle drawing started at t={current_time:.2f}s")
        print(f"[CIRCLE START] Center: {self.common_config.circle_center}")
        print(f"[CIRCLE START] Radius: {self.common_config.circle_radius}")

    def update(self, current_time: float, robot_state: RobotState) -> np.ndarray:
        """
        Compute control torques for circle drawing.

        Args:
            current_time: Current time
            robot_state: Current robot state from libfranka

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

        # ============================================================
        # 2. Compute Jacobian and Dynamics
        # ============================================================
        pino.forwardKinematics(self.pino_model, self.pino_data, robot_state.q, robot_state.dq)
        pino.computeJointJacobians(self.pino_model, self.pino_data)
        pino.updateFramePlacements(self.pino_model, self.pino_data)
        jac = pino.getFrameJacobian(self.pino_model, self.pino_data, self.pino_frame_id, pino.LOCAL_WORLD_ALIGNED)
        M = pino.crba(self.pino_model, self.pino_data, robot_state.q)
        M_inv = pino.computeMinverse(self.pino_model, self.pino_data, robot_state.q)

        J_phi = self.S_f.T @ jac
        J_motion = self.S_v.T @ jac
        jac_1 = np.vstack([J_phi, J_motion])

        Mx_constraint = task_space_inertiaM(M_inv, J_phi)
        Mx_motion = task_space_inertiaM(M_inv, J_motion)

        # ============================================================
        # 3. Get Current EE Pose
        # ============================================================
        ee_transform = robot_state.O_T_EE.reshape(4, 4, order='F')
        current_pos = ee_transform[:3, 3]
        current_rot = ee_transform[:3, :3]

        # ============================================================
        # 4. Get Contact Information (from external forces)
        # ============================================================
        # O_F_ext_hat_K contains estimated external wrench [fx, fy, fz, tx, ty, tz]
        F_ext_world = robot_state.O_F_ext_hat_K

        # Transform to constraint frame
        F_ext_constraint = np.zeros(6)
        F_ext_constraint[:3] = self.R_slope.T @ F_ext_world[:3]
        F_ext_constraint[3:] = self.R_slope.T @ F_ext_world[3:]

        F_ext_phi = F_ext_constraint @ self.S_fc
        F_ext_x = F_ext_constraint @ self.S_vc

        # ============================================================
        # 5. Null Space torque
        # ============================================================
        jac_1_inv = dynamically_consistent_inv(jac_1, M_inv)
        N2 = np.eye(self.n_joints) - jac_1.T @ jac_1_inv.T
        tau_ctrl_v = null_space_tau(robot_state.q, robot_state.dq, self.q0, self.config.Kp_null, self.config.Kd_null)
        # null space projection
        tau_ctrl_v = N2 @ tau_ctrl_v

        # ============================================================
        # 6. Motion Space Control
        # ============================================================
        # Compute pose error
        twist = np.zeros(6)
        dx = self.target_pos - current_pos
        twist[:3] = dx
        # Orientation error (simplified)
        twist[3:] = 0.0

        x_ddot_desired_sel = np.concatenate([self.x_ddot_desired, [0, 0, 0]]) @ self.S_v
        x_tilde = twist @ self.S_v
        site_vel = jac @ robot_state.dq
        x_dot_tilde = (np.concatenate([self.x_dot_desired, [0, 0, 0]]) - site_vel) @ self.S_v
        a_motion = feedforward_PD(
            x_acc_desired=x_ddot_desired_sel, x_delta=x_tilde,
            x_dot_delta=x_dot_tilde,
            Kp=self.Kp @ self.S_v, Kd=self.Kd @ self.S_v
        )
        F_ctrl_x = Mx_motion @ a_motion
        tau_ctrl_x = J_motion.T @ F_ctrl_x

        # ============================================================
        # 7. Constraint space (Force Control)
        # ============================================================
        C = pino.computeCoriolisMatrix(self.pino_model, self.pino_data, robot_state.q, robot_state.dq)
        J_dot = pino.getFrameJacobianTimeVariation(self.pino_model, self.pino_data, self.pino_frame_id, pino.LOCAL_WORLD_ALIGNED)
        J_phi_dot = self.S_f.T @ J_dot

        F_ext_x_new = F_ext_x.copy()
        F_ext_x_new[-3:] = 0
        control_force_compensation = 1 * (- Mx_constraint @ J_phi @ M_inv @ (tau_ctrl_x + tau_ctrl_v))
        contact_force_compensation = 1 * (Mx_constraint @ J_phi @ M_inv @ (J_motion.T @ F_ext_x_new))
        velocity_term = 1 * Mx_constraint @ (J_phi @ M_inv @ C - J_phi_dot) @ robot_state.dq
        F_ctrl_constraint = (
            self.config.F_desired_contact +
            control_force_compensation +
            contact_force_compensation + velocity_term
        )

        # ============================================================
        # 8. Sum up torques
        # ============================================================
        self.tau[:] = J_phi.T @ F_ctrl_constraint + tau_ctrl_x + tau_ctrl_v

        # Store for logging
        self._last_control_compensation = control_force_compensation
        self._last_contact_compensation = contact_force_compensation
        self._last_velocity_term = velocity_term
        self._last_F_ctrl_constraint = F_ctrl_constraint

        # ============================================================
        # 9. Add Gravity Compensation
        # ============================================================
        if self.common_config.gravity_compensation:
            g = pino.computeGeneralizedGravity(self.pino_model, self.pino_data, robot_state.q)
            self.tau += g

        # ============================================================
        # 10. Log Data
        # ============================================================
        self._log_data(F_ext_constraint, current_pos)

        return self.tau

    def _log_data(self, F_ext_local: np.ndarray, current_pos: np.ndarray) -> None:
        """Log data for plotting."""
        self.contact_forces.append(F_ext_local[:3].copy())
        self.desired_forces.append(-self.config.F_desired_contact.copy())
        self.ee_positions.append(current_pos.copy())
        self.target_positions.append(self.target_pos.copy())

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


def plot_results(
        approach_controller: CartesianSpacePDController,
        circle_controller: HybridController,
        dt: float,
        transition_time: float
) -> None:
    """Plot results from both controllers."""

    # Combine data from both controllers
    all_ee_pos = approach_controller.ee_positions + circle_controller.ee_positions
    all_target_pos = approach_controller.target_positions + circle_controller.target_positions

    # ============================================================
    # Plot Position Tracking
    # ============================================================
    ee_positions = np.array(all_ee_pos)
    target_positions = np.array(all_target_pos)
    time_steps = np.arange(len(ee_positions)) * dt

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    axes_labels = ['X', 'Y', 'Z']

    for i in range(3):
        axes[i].plot(time_steps, ee_positions[:, i], 'b-', linewidth=2, label='End-Effector')
        axes[i].plot(time_steps, target_positions[:, i], 'r--', linewidth=2, label='Target')
        axes[i].axvline(transition_time, color='g', linestyle=':', label='Transition')
        axes[i].set_ylabel(f'{axes_labels[i]} Position (m)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_title(f'{axes_labels[i]} Position Tracking')

    axes[2].set_xlabel('Time (s)')
    plt.tight_layout()
    fig.savefig("plots/combined_position_tracking.png")

    # ============================================================
    # Plot Contact Forces (Circle Drawing Phase Only)
    # ============================================================
    contact_forces = np.array(circle_controller.contact_forces)
    desired_forces = np.array(circle_controller.desired_forces)

    if len(contact_forces) > 0:
        if contact_forces.ndim == 1:
            contact_forces = contact_forces[:, None]
            desired_forces = desired_forces[:, None]

        timesteps, n_dim = contact_forces.shape
        t = np.arange(timesteps) * dt + transition_time

        plt.figure(figsize=(8, 3 * n_dim))
        for i in range(n_dim):
            plt.subplot(n_dim, 1, i + 1)
            plt.plot(t, contact_forces[:, i], label="Contact force")
            plt.plot(t, desired_forces[:, 0], label="Desired force")
            plt.ylabel(f"Dim {i + 1}")
            plt.xlabel("Time [s]")
            plt.legend()
            plt.grid(True)
        plt.tight_layout()
        plt.savefig("plots/contact_forces.png")

    plt.show()
    print("[PLOT] Results saved to plots/ directory")


def main() -> None:
    """Main function with two-phase control using libfranka."""

    # ============================================================
    # 1. Parse Arguments
    # ============================================================
    parser = argparse.ArgumentParser(description='Hybrid force-impedance control with libfranka')
    parser.add_argument('--robot-ip', type=str, required=True,
                        help='IP address of the Franka robot')
    args = parser.parse_args()

    # ============================================================
    # 2. Create Configurations
    # ============================================================
    common_config = ControllerConfig()
    approach_config = CartesianSpacePDControlConfig()
    circle_config = HybridControllerConfig()

    # ============================================================
    # 3. Load Pinocchio Model
    # ============================================================
    print("[MAIN] Loading Pinocchio model...")
    pino_model = pino.buildModelFromMJCF("franka_emika_panda/panda_nohand.xml")
    pino_data = pino_model.createData()

    # ============================================================
    # 4. Connect to Robot
    # ============================================================
    print(f"[MAIN] Connecting to robot at {args.robot_ip}...")
    try:
        robot = libfranka.Robot(args.robot_ip)
        print("[MAIN] Successfully connected to robot!")

        # Set collision behavior (more permissive for contact tasks)
        robot.setCollisionBehavior(
            [20.0] * 7, [20.0] * 7, [20.0] * 7, [20.0] * 7,
            [20.0] * 6, [20.0] * 6, [20.0] * 6, [20.0] * 6
        )
        print("[MAIN] Collision behavior set")

    except Exception as e:
        print(f"[MAIN] Failed to connect to robot: {e}")
        return

    # ============================================================
    # 5. Create Controllers
    # ============================================================
    approach_controller = CartesianSpacePDController(approach_config, common_config)
    circle_controller = HybridController(circle_config, common_config)

    # Get home configuration (neutral pose)
    q0 = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])

    # Initialize both controllers
    if not approach_controller.init(pino_model, pino_data, q0):
        print("Approach controller init failed!")
        return

    if not circle_controller.init(pino_model, pino_data, q0):
        print("Circle controller init failed!")
        return

    # ============================================================
    # 6. Setup Initial Targets
    # ============================================================
    R_slope = euler_to_rot_matrix(common_config.euler)
    target_pos = generate_start_position(
        common_config.circle_radius,
        common_config.circle_center,
        common_config.size_z,
        R_slope
    )

    # Generate target orientation (quaternion [w, x, y, z])
    target_quat = np.array([1., 0., 0., 0.])  # Identity quaternion
    # TODO: Apply slope rotation to target_quat if needed

    # ============================================================
    # 7. Start Approach Phase
    # ============================================================
    control_phase = ControlPhase.APPROACHING
    approach_controller.starting(target_pos, target_quat)

    print("\n" + "=" * 60)
    print("PHASE 1: APPROACHING TARGET POSITION")
    print("=" * 60)

    # ============================================================
    # 8. Run Control Loop
    # ============================================================
    robot_state = RobotState()
    start_time = time.time()
    sim_time = 0.0
    transition_time = 0.0

    print("[MAIN] Starting control loop at 1kHz...")
    print("[MAIN] Press Ctrl+C to stop")

    try:
        while True:
            loop_start = time.time()

            # ============================================================
            # Get Robot State
            # ============================================================
            state = robot.readOnce()
            robot_state.q = np.array(state.q)
            robot_state.dq = np.array(state.dq)
            robot_state.tau_J = np.array(state.tau_J)
            robot_state.O_T_EE = np.array(state.O_T_EE)
            robot_state.O_F_ext_hat_K = np.array(state.O_F_ext_hat_K)

            # ============================================================
            # State Machine: Switch Controllers
            # ============================================================
            if control_phase == ControlPhase.APPROACHING:
                # Use approach controller
                tau = approach_controller.update(robot_state)

                # Check if target reached
                if approach_controller.is_target_reached(robot_state):
                    print("\n" + "=" * 60)
                    print(f"TARGET REACHED at t={sim_time:.2f}s!")
                    print("PHASE 2: CIRCLE DRAWING")
                    print("=" * 60 + "\n")

                    control_phase = ControlPhase.CIRCLE_DRAWING
                    transition_time = sim_time
                    circle_controller.starting(sim_time, target_pos, target_quat)

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
                # Gravity compensation only
                g = pino.computeGeneralizedGravity(pino_model, pino_data, robot_state.q)
                tau = g
                break  # Exit the loop

            # ============================================================
            # Apply Control
            # ============================================================
            # Clip torques to safe limits
            tau = np.clip(tau, -87, 87)  # Franka torque limits

            # Send torque command
            robot.setTorques(tau.tolist())

            # ============================================================
            # Timing
            # ============================================================
            sim_time = time.time() - start_time

            # Maintain 1kHz rate
            elapsed = time.time() - loop_start
            if elapsed < common_config.dt:
                time.sleep(common_config.dt - elapsed)

    except KeyboardInterrupt:
        print("\n[MAIN] Control interrupted by user")
    except Exception as e:
        print(f"\n[MAIN] Error during control: {e}")
    finally:
        print("[MAIN] Stopping robot...")
        # Stop robot (automatically handles safe shutdown)
        del robot
        print("[MAIN] Robot stopped")

    # ============================================================
    # 9. Plot Results
    # ============================================================
    print("\n[MAIN] Generating plots...")
    plot_results(approach_controller, circle_controller, common_config.dt, transition_time)
    print("[MAIN] Done!")


if __name__ == "__main__":
    main()
