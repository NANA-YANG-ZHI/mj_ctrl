# ------------------------------------------------------------------------------
# Impedance Controller (MuJoCo-native kinematics)
#
# Phase 2 surface control law (from opspace_impedance.py) + force feedforward:
#   y          = jac_inv @ Md_inv @ (Kp * twist - Kd * (jac @ dq))
#   tau_motion = M @ y + tau_null
#   tau_f      = jac.T @ (force_mag * n)  [+ PI correction]
#   tau        = tau_motion + tau_f + qfrc_bias
# ------------------------------------------------------------------------------
import mujoco
import numpy as np
from dataclasses import dataclass
from typing import Optional

from utils_libfranka import PI_term, euler_to_rot_matrix
from src.controller_config import ControllerConfig
from src.trajectory import TrajectoryBase


@dataclass
class ImpedanceControllerConfig:
    """Configuration for MuJoCo-native impedance + feedforward force controller."""
    Kp: np.ndarray = None
    Kd: np.ndarray = None
    Kp_null: np.ndarray = None
    Kd_null: np.ndarray = None

    # Scaling applied to pose error before Kp/Kd (matches opspace_impedance.py)
    Kpos: float = 0.95
    Kori: float = 0.95
    integration_dt: float = 1.0

    # Force control
    force_mag: float = -8.0
    Kp_force: float = 2.0
    Ki_force: float = 5.0

    max_delta_tau: float = 1.0

    def __post_init__(self):
        if self.Kp is None:
            self.Kp = np.concatenate([np.array([100., 100., 100.]),
                                      np.array([50.,  50.,  50.])])
        if self.Kd is None:
            self.Kd = 2.0 * np.sqrt(self.Kp)
        if self.Kp_null is None:
            self.Kp_null = np.array([75., 75., 50., 50., 40., 25., 25.])
        if self.Kd_null is None:
            self.Kd_null = 2.0 * np.sqrt(self.Kp_null)


class ImpedanceController:
    """
    Surface controller using MuJoCo-native kinematics.

    Impedance law (opspace_impedance.py style):
        twist      = [Kpos/dt * dx,  Kori/dt * d_ori]
        y          = jac_inv @ Md_inv @ (Kp * twist - Kd * (jac @ dq))
        tau_motion = M @ y  +  tau_null

    Force feedforward:
        tau_f = jac.T @ [force_mag * n, 0, 0, 0]   [+ PI along normal]

    Total:
        tau = tau_motion + tau_f + qfrc_bias   (rate-limited)

    force_normal is set from euler (slope) in starting(), or updated each step
    from CylinderTrajectory.S_f for cylinder runs.
    """

    def __init__(
        self,
        config: ImpedanceControllerConfig,
        common_config: ControllerConfig,
        n_joints: int = 7,
        site_name: str = "attachment_site",
        trajectory: Optional[TrajectoryBase] = None,
    ):
        self.config = config
        self.common_config = common_config
        self.n_joints = n_joints
        self.site_name = site_name
        self.trajectory = trajectory

        # Set in starting()
        self.mj_model = None
        self.mj_data = None
        self.site_id: Optional[int] = None
        self.dof_ids: Optional[np.ndarray] = None

        self.target_pos: Optional[np.ndarray] = None
        self.target_rot: Optional[np.ndarray] = None
        self.x_dot_desired = np.zeros(3)
        self.x_ddot_desired = np.zeros(3)
        self.q0: Optional[np.ndarray] = None

        self.start_time: float = 0.0
        self.is_drawing: bool = False

        self.tau = np.zeros(n_joints)
        self.integral_force_error = np.zeros(1)
        self.force_normal = np.array([0., 0., 1.])

        # Pre-allocated (sized after model.nv is known)
        self._jac = None
        self._M = None
        self._M_inv = None
        self._twist = np.zeros(6)
        self._site_quat = np.zeros(4)
        self._site_quat_conj = np.zeros(4)
        self._error_quat = np.zeros(4)
        self._target_quat = np.zeros(4)

        # Logging (same fields as BaselineController for drop-in plotting)
        self.ee_positions: list = []
        self.target_positions: list = []
        self.contact_forces: list = []
        self.desired_forces: list = []
        self.normals: list = []
        self.joint_torques: list = []
        self.tau_motion_log: list = []
        self.tau_f_log: list = []

    # ------------------------------------------------------------------
    def starting(
        self,
        current_time: float,
        target_rot: np.ndarray,
        q0: np.ndarray,
        mj_model,
        mj_data,
        site_id: int,
        dof_ids: np.ndarray,
    ) -> None:
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.site_id = site_id
        self.dof_ids = dof_ids
        self.start_time = current_time
        self.is_drawing = True
        self.target_rot = target_rot.copy()
        self.q0 = q0.copy()

        nv = mj_model.nv
        self._jac   = np.zeros((6, nv))
        self._M     = np.zeros((nv, nv))
        self._M_inv = np.zeros((nv, nv))

        R_slope = euler_to_rot_matrix(self.common_config.euler)
        self.force_normal = R_slope @ np.array([0., 0., 1.])

        self.tau[:] = 0.0
        self.integral_force_error = np.zeros(1)
        self.ee_positions = []
        self.target_positions = []
        self.contact_forces = []
        self.desired_forces = []
        self.normals = []
        self.joint_torques = []
        self.tau_motion_log = []
        self.tau_f_log = []

        print(f"[IMPEDANCE START] t={current_time:.2f}s")
        print(f"[IMPEDANCE START] force_mag={self.config.force_mag} N")
        print(f"[IMPEDANCE START] force_normal={self.force_normal}")
        print(f"[IMPEDANCE START] use_pi={self.common_config.use_pi}")

    # ------------------------------------------------------------------
    def update(self, current_time: float, robot_state) -> np.ndarray:
        elapsed = current_time - self.start_time
        model = self.mj_model
        data  = self.mj_data

        q  = np.array(robot_state.q)
        dq = np.array(robot_state.dq)
        O_T_EE      = np.array(robot_state.O_T_EE).reshape(4, 4).T
        current_pos = O_T_EE[:3, 3]
        current_mat = O_T_EE[:3, :3]

        # ------------------------------------------------------------------
        # Trajectory
        # ------------------------------------------------------------------
        if self.trajectory is not None:
            self.target_pos, self.x_dot_desired, self.x_ddot_desired, \
                S_f, _S_v, target_rot, traj_done = \
                self.trajectory.step(elapsed, current_pos, current_mat)
            if traj_done:
                self.x_dot_desired[:] = 0.0
                self.x_ddot_desired[:] = 0.0
                self.is_drawing = False
                return data.qfrc_bias[self.dof_ids].copy()
            self.force_normal = S_f[:3, 0]
            target_rot = target_rot
        else:
            target_rot = self.target_rot

        # ------------------------------------------------------------------
        # Twist  (desired EE spatial velocity, as in opspace_impedance.py)
        # ------------------------------------------------------------------
        dx = self.target_pos - current_pos
        self._twist[:3] = self.config.Kpos * dx / self.config.integration_dt

        # Orientation error via quaternion
        mujoco.mju_mat2Quat(self._site_quat,      data.site(self.site_id).xmat)
        mujoco.mju_negQuat( self._site_quat_conj,  self._site_quat)
        mujoco.mju_mat2Quat(self._target_quat,     target_rot.flatten())
        mujoco.mju_mulQuat( self._error_quat,      self._target_quat, self._site_quat_conj)
        mujoco.mju_quat2Vel(self._twist[3:],       self._error_quat, 1.0)
        self._twist[3:] *= self.config.Kori / self.config.integration_dt

        # ------------------------------------------------------------------
        # Jacobian  (6 × nv → slice to 6 × n_joints)
        # ------------------------------------------------------------------
        mujoco.mj_jacSite(model, data, self._jac[:3], self._jac[3:], self.site_id)
        jac = self._jac[:, self.dof_ids]   # 6 × n_joints

        # ------------------------------------------------------------------
        # Inertia matrices
        # ------------------------------------------------------------------
        mujoco.mj_solveM(model, data, self._M_inv, np.eye(model.nv))
        mujoco.mj_fullM( model, self._M, data.qM)
        M7     = self._M    [np.ix_(self.dof_ids, self.dof_ids)]
        M_inv7 = self._M_inv[np.ix_(self.dof_ids, self.dof_ids)]

        # Task-space inertia (for dynamically-consistent null-space projector)
        Mx_inv = jac @ M_inv7 @ jac.T
        if abs(np.linalg.det(Mx_inv)) >= 1e-2:
            Mx = np.linalg.inv(Mx_inv)
        else:
            Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)

        # ------------------------------------------------------------------
        # Null-space posture torque
        # ------------------------------------------------------------------
        J_bar   = M_inv7 @ jac.T @ Mx                             # 7 × 6
        N       = np.eye(self.n_joints) - jac.T @ J_bar.T
        tau_null = N @ (self.config.Kp_null * (self.q0 - q)
                        - self.config.Kd_null * dq)

        # ------------------------------------------------------------------
        # Impedance  (opspace_impedance.py control law)
        #   y          = jac_inv @ Md_inv @ (Kp * twist - Kd * (jac @ dq))
        #   tau_motion = M @ y  +  tau_null
        # ------------------------------------------------------------------
        jac_inv = np.linalg.pinv(jac, rcond=1e-2)
        Md_inv  = np.eye(6)   # desired inertia = identity
        y = jac_inv @ Md_inv @ (
            self.config.Kp * self._twist
            - self.config.Kd * (jac @ dq)
        )
        tau_motion = M7 @ y + tau_null

        # ------------------------------------------------------------------
        # Force feedforward  (+ optional PI along surface normal)
        # ------------------------------------------------------------------
        F_des_6 = np.concatenate([self.config.force_mag * self.force_normal,
                                   np.zeros(3)])

        if self.common_config.use_pi:
            F_ext         = np.array(robot_state.O_F_ext_hat_K)
            F_ext_normal  = np.array([float(F_ext[:3] @ self.force_normal)])
            F_des_normal  = np.array([self.config.force_mag])
            pi_correction, self.integral_force_error = PI_term(
                F_ext_normal, F_des_normal,
                self.common_config.dt,
                self.integral_force_error,
                kp=self.config.Kp_force,
                ki=self.config.Ki_force,
            )
            F_des_6[:3] += pi_correction[0] * self.force_normal

        tau_f = jac.T @ F_des_6

        # ------------------------------------------------------------------
        # Gravity + Coriolis compensation (qfrc_bias = C(q,dq) + g(q))
        # ------------------------------------------------------------------
        g = data.qfrc_bias[self.dof_ids].copy()

        self.tau[:] = tau_motion + tau_f + g

        # Rate limiting
        last_tau  = np.array(robot_state.tau_J_d)
        delta_tau = np.clip(self.tau - last_tau,
                            -self.config.max_delta_tau,
                             self.config.max_delta_tau)
        self.tau[:] = last_tau + delta_tau

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        self.ee_positions.append(current_pos.copy())
        if self.target_pos is not None:
            self.target_positions.append(self.target_pos.copy())
        F_ext_raw = np.array(robot_state.O_F_ext_hat_K)
        self.contact_forces.append(F_ext_raw[:3].copy())
        self.desired_forces.append(np.array([self.config.force_mag]))
        self.normals.append(self.force_normal.copy())
        self.joint_torques.append(self.tau.copy())
        self.tau_motion_log.append(tau_motion.copy())
        self.tau_f_log.append(tau_f.copy())

        return self.tau.copy()

    # ------------------------------------------------------------------
    def is_finished(self) -> bool:
        return not self.is_drawing
