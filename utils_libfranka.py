import numpy as np
from scipy.linalg import pinv
from scipy.spatial.transform import Rotation


def generate_line_trajectory_delta(elapsed_time: float,
                             start_pos: np.ndarray,
                             end_pos: np.ndarray,
                             duration: float):
    """
    Generate desired position, velocity, and acceleration for minimum jerk trajectory.
    Uses the formula: x(t) = x_0 + [10σ³ - 15σ⁴ + 6σ⁵](x_f - x_0), σ = t/T
    Args:
        elapsed_time: Elapsed time since start
        start_pos: Starting position (3D)
        end_pos: Ending position (3D)
        duration: Total duration T
    Returns:
        Tuple of (position, velocity, acceleration)
    """
    # Clamp time to [0, T]
    t = np.clip(elapsed_time, 0.0, duration)
    sigma = t / duration

    # Position: x(t) = x_0 + [10σ³ - 15σ⁴ + 6σ⁵](x_f - x_0)
    s = 10 * sigma**3 - 15 * sigma**4 + 6 * sigma**5

    # # Velocity: dx/dt = [30σ² - 60σ³ + 30σ⁴] / T * (x_f - x_0)
    # ds_dt = (30 * sigma**2 - 60 * sigma**3 + 30 * sigma**4) / duration
    # x_dot_desired = ds_dt * (end_pos - start_pos)

    # # Acceleration: d²x/dt² = [60σ - 180σ² + 120σ³] / T² * (x_f - x_0)
    # d2s_dt2 = (60 * sigma - 180 * sigma**2 + 120 * sigma**3) / (duration**2)
    # x_ddot_desired = d2s_dt2 * (end_pos - start_pos)

    return start_pos + s * (end_pos - start_pos)

def task_space_inertiaM(M_inv, jac):
    """
    Compute the task-space inertia matrix from the joint-space inverse inertia matrix.
    """
    Mx_inv = jac @ M_inv @ jac.T
    if abs(np.linalg.det(Mx_inv)) >= 1e-2:
        Mx = np.linalg.inv(Mx_inv)
    else:
        Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)
    return Mx

def null_space_tau(q, dq, q0, Kp_null, Kd_null):
    """
    Compute the null-space torque to drive joints to a desired configuration q0 with PD control.
    """
    return Kp_null * (q0 - q) - Kd_null * dq

def euler_to_rot_matrix(euler):
    """
    Convert Euler angles (roll, pitch, yaw) to a rotation matrix.
    The input euler angles are in radians.
    The output rotation matrix is a 3x3 numpy array.
    """
    roll, pitch, yaw = euler
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    
    return R_z @ R_y @ R_x

def compute_ee_pose_error_quat(target_pos, current_pos, target_quat, current_mat, Kpos=0.95, Kori=0.95):
    twist = np.zeros(6)
    Kori: float = 0.95
    dx = target_pos - current_pos
    twist[:3] = Kpos * dx

    if np.all(current_mat == 0):
        rot_current = Rotation.from_matrix(np.eye(3))
    else:
        rot_current = Rotation.from_matrix(current_mat.reshape(3,3))
    rot_target = Rotation.from_quat(np.roll(target_quat, -1))
    
    R_error = rot_target * rot_current.inv()
    twist[3:] = Kori * R_error.as_rotvec()

    return twist

def compute_ee_pose_error(target_pos, current_pos, target_rot, current_mat, Kpos=0.95, Kori=0.95):
    twist = np.zeros(6)
    # site_quat = np.zeros(4)
    # site_quat_conj = np.zeros(4)
    # error_quat = np.zeros(4)
    # # Kpos Gains for the twist computation. These should be between 0 and 1. 0 means no
    # # movement, 1 means move the end-effector to the target in one integration step.
    # # Gain for the orientation component of the twist computation. This should be
    # # between 0 and 1. 0 means no movement, 1 means move the end-effector to the target
    # # orientation in one integrati on step.

    dx = target_pos - current_pos
    twist[:3] = Kpos * dx
    # mujoco.mju_mat2Quat(site_quat, current_mat)
    # mujoco.mju_negQuat(site_quat_conj, site_quat)
    # mujoco.mju_mulQuat(error_quat, target_quat, site_quat_conj)
    # mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
    # # twist[3:] *= Kori
    # return twist
    if np.all(current_mat == 0):
        rot_current = Rotation.from_matrix(np.eye(3))
    else:
        rot_current = Rotation.from_matrix(current_mat.reshape(3,3))
    rot_target = Rotation.from_matrix(target_rot)
    
    R_error = rot_target * rot_current.inv()
    twist[3:] = Kori * R_error.as_rotvec()

    return twist

def dynamically_consistent_inv(jac, M_inv):
    """
    Compute dynamically consistent pseudoinverse of jac
    J^{M+} = M^{-1} J^T (J M^{-1} J^T)^{-1}
    """
    Mx_inv = jac @ M_inv @ jac.T
    if abs(np.linalg.det(Mx_inv)) >= 1e-2:
        Mx = np.linalg.inv(Mx_inv)
    else:
        Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)
    return M_inv @ jac.T @ Mx

def feedforward_PD(x_acc_desired, x_delta, x_dot_delta, Kp, Kd):
    """
    Compute the feedforward PD control torque for the end-effector.
    Tracking desired acceleration.
    """
    # a_v = np.concatenate([x_ddot_desired, [0,0,0]]) @ S_v + Kp @ S_v * x_tilde + Kd @ S_v * x_dot_tilde
    # F_ctrl_x = Mx_motion @ a_v
    # tau_ctrl_x = J_motion.T @ F_ctrl_x
    a_v = x_acc_desired + Kp * x_delta + Kd * x_dot_delta
    return a_v

def compute_force_dot(
    S_f: np.ndarray,
    Compliance_matrix: np.ndarray,
    jac: np.ndarray,
    dq: np.ndarray
) -> np.ndarray:
    """
    Compute constraint-space force rate: λ˙ = Sf† K' J(q) q̇

    Args:
        S_f: Force selection matrix (6 x n_constraint)
        Compliance_matrix: Material compliance matrix (6 x 6), inverse of stiffness
        jac: End-effector Jacobian (6 x n_joints)
        dq: Joint velocities (n_joints,)

    Returns:
        F_dot: Force rate in constraint space
    """
    inner = S_f.T @ Compliance_matrix @ S_f
    K_effective = S_f @ np.linalg.inv(inner) @ S_f.T
    Sf_pinv = np.linalg.pinv(S_f, rcond=1e-6)
    return Sf_pinv @ K_effective @ jac @ dq


def force_ctrl_feedforward(F_desired: np.ndarray) -> np.ndarray:
    """
    Force control rule 1: pure feedforward.

    F_ctrl_constraint = F_desired

    Args:
        F_desired: Desired contact force in constraint space

    Returns:
        F_ctrl_constraint
    """
    return F_desired.copy()


def _build_compliance_matrix(k_normal: float) -> np.ndarray:
    """
    Build the 6x6 compliance matrix from a single normal stiffness value.

    Tangential stiffness = 0.1 * k_normal
    Rotational stiffness = 0.01 * k_normal
    Compliance = inv(K_material)
    """
    K_material = np.diag([
        k_normal * 0.1,   # x tangential
        k_normal * 0.1,   # y tangential
        k_normal * 0.1,   # z normal
        k_normal * 0.01,  # rx rotational
        k_normal * 0.01,  # ry rotational
        k_normal * 0.01,  # rz rotational
    ])
    return np.linalg.inv(K_material)


def force_ctrl_pd(
    F_desired: np.ndarray,
    F_ext_phi: np.ndarray,
    S_f: np.ndarray,
    jac: np.ndarray,
    dq: np.ndarray,
    k_normal: float = 5000.0,
    kp: float = 3.0,
    kd: float = 3.0
) -> np.ndarray:
    """
    Force control rule 2: PD force control (eq. 9.81).

    fλ = λ¨d + KDλ(λ˙d − λ˙) + KPλ(λd − λ)
    With λ¨d = 0, λ˙d = 0:
    F_ctrl_constraint = -Kd @ λ˙ - Kp @ (|F_desired| - |F_ext_phi|)

    The compliance matrix is built from k_normal:
        K_material = diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01]) * k_normal
        Compliance_matrix = inv(K_material)

    Args:
        F_desired: Desired contact force in constraint space
        F_ext_phi: Measured external force projected onto constraint space
        S_f: Force selection matrix (6 x n_constraint)
        jac: End-effector Jacobian (6 x n_joints)
        dq: Joint velocities (n_joints,)
        k_normal: Normal stiffness of the contact material (default 5000.0)
        kp: Proportional gain (default 3.0)
        kd: Derivative gain (default 3.0)

    Returns:
        F_ctrl_constraint
    """
    Compliance_matrix = _build_compliance_matrix(k_normal)
    F_dot = compute_force_dot(S_f, Compliance_matrix, jac, dq)
    n = F_dot.shape[0]
    Kd_force = np.eye(n) * kd
    Kp_force = np.eye(n) * kp
    return -Kd_force @ F_dot - Kp_force @ (np.abs(F_desired) - np.abs(F_ext_phi))


def generate_start_position(r, body_pos, size_z, R):
    theta = 0
    circle_local = np.zeros(3)
    circle_local[0] = r * np.cos(theta)  # x
    circle_local[1] = r * np.sin(theta)  # y
    circle_local[2] = size_z  # z
    return body_pos + (R @ circle_local.T).T