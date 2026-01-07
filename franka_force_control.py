"""
Operational Space Force Control for Franka Emika Panda Robot
Adapted from opspace_force.py to use libfranka instead of MuJoCo simulation.

This script implements:
- Cartesian impedance control
- Nullspace control for redundant joints
- Force feedback control for contact tasks
- Gravity compensation

Requirements:
    pip install panda-py numpy matplotlib

Note: Ensure the robot is in FCI mode and you have proper network connection.
"""

import numpy as np
import time
import sys
from typing import Optional
import matplotlib.pyplot as plt

try:
    from panda_py import Panda
except ImportError:
    print("Error: panda-py not installed. Install with: pip install panda-py")
    sys.exit(1)


class FrankaForceController:
    """Operational space force controller for Franka Panda robot."""

    def __init__(self, robot_ip: str = "172.16.0.2"):
        """
        Initialize the controller.

        Args:
            robot_ip: IP address of the Franka robot
        """
        # Cartesian impedance control gains
        self.impedance_pos = np.array([100.0, 100.0, 100.0])  # [N/m]
        self.impedance_ori = np.array([50.0, 50.0, 50.0])  # [Nm/rad]

        # Joint impedance control gains for nullspace
        self.Kp_null = np.array([75.0, 75.0, 50.0, 50.0, 40.0, 25.0, 25.0])

        # Damping ratio for both Cartesian and joint impedance control
        self.damping_ratio = 1.0

        # Gains for the twist computation (0-1 range)
        self.Kpos = 0.95  # Position tracking gain
        self.Kori = 0.95  # Orientation tracking gain

        # Integration timestep
        self.integration_dt = 1.0

        # Control loop frequency
        self.dt = 0.001  # 1 kHz control loop

        # Gravity compensation
        self.gravity_compensation = True

        # Force feedback parameters
        self.force_feedback = True
        self.Kp_force = 0.4
        self.Kd_force = 0.002
        self.Ki_force = 0.4
        self.desired_force = np.array([0.0, 0.0, 10.0])  # Desired contact force in world frame

        # Compute damping and stiffness matrices
        damping_pos = self.damping_ratio * 2 * np.sqrt(self.impedance_pos)
        damping_ori = self.damping_ratio * 2 * np.sqrt(self.impedance_ori)
        self.Kp = np.concatenate([self.impedance_pos, self.impedance_ori], axis=0)
        self.Kd = np.concatenate([damping_pos, damping_ori], axis=0)
        self.Kd_null = self.damping_ratio * 2 * np.sqrt(self.Kp_null)

        # Default home position (can be modified)
        self.q0 = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])

        # Target pose
        self.target_pos = np.array([0.5, 0.0, 0.3])  # [x, y, z] in meters
        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z] quaternion

        # Data logging
        self.contact_forces = []
        self.desired_forces = []
        self.force_errors = []
        self.tau_forces = []
        self.taus = []
        self.timestamps = []

        # Force feedback state
        self.force_error_prev = np.zeros(3)

        # Connect to robot
        print(f"Connecting to Franka robot at {robot_ip}...")
        try:
            self.robot = Panda(robot_ip)
            print("Successfully connected to robot!")
        except Exception as e:
            print(f"Failed to connect to robot: {e}")
            print("Please ensure:")
            print("  1. Robot is powered on and in FCI mode")
            print("  2. Network connection is properly configured")
            print("  3. IP address is correct")
            sys.exit(1)

    def quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        Multiply two quaternions (w, x, y, z format).

        Args:
            q1: First quaternion [w, x, y, z]
            q2: Second quaternion [w, x, y, z]

        Returns:
            Product quaternion [w, x, y, z]
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def quaternion_conjugate(self, q: np.ndarray) -> np.ndarray:
        """
        Compute quaternion conjugate.

        Args:
            q: Quaternion [w, x, y, z]

        Returns:
            Conjugate quaternion [w, -x, -y, -z]
        """
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def quat_to_angular_velocity(self, error_quat: np.ndarray) -> np.ndarray:
        """
        Convert quaternion error to angular velocity.

        Args:
            error_quat: Error quaternion [w, x, y, z]

        Returns:
            Angular velocity [wx, wy, wz]
        """
        # Extract vector part and scale by 2
        return 2.0 * error_quat[1:]

    def rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to quaternion (w, x, y, z format).

        Args:
            R: 3x3 rotation matrix

        Returns:
            Quaternion [w, x, y, z]
        """
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
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

    def compute_control_torques(self, state) -> np.ndarray:
        """
        Compute control torques using operational space control with force feedback.

        Args:
            state: Robot state from libfranka

        Returns:
            Control torques for 7 joints
        """
        # Get current joint positions and velocities
        q = np.array(state.q)
        dq = np.array(state.dq)

        # Get end-effector pose (4x4 transformation matrix)
        O_T_EE = np.array(state.O_T_EE).reshape(4, 4).T  # libfranka uses column-major
        ee_pos = O_T_EE[:3, 3]
        ee_rot = O_T_EE[:3, :3]
        ee_quat = self.rotation_matrix_to_quaternion(ee_rot)

        # Get Jacobian (6x7 matrix)
        jacobian = np.array(state.O_J_EE).reshape(6, 7).T  # libfranka uses column-major

        # Get mass matrix (7x7)
        mass_matrix = np.array(state.mass_matrix).reshape(7, 7)

        # Compute spatial velocity (twist)
        twist = np.zeros(6)

        # Position error
        dx = self.target_pos - ee_pos
        twist[:3] = self.Kpos * dx / self.integration_dt

        # Orientation error
        target_quat_conj = self.quaternion_conjugate(self.target_quat)
        error_quat = self.quaternion_multiply(self.target_quat,
                                              self.quaternion_conjugate(ee_quat))
        twist[3:] = self.quat_to_angular_velocity(error_quat) * self.Kori / self.integration_dt

        # Compute task-space inertia matrix
        M_inv = np.linalg.inv(mass_matrix)
        Mx_inv = jacobian @ M_inv @ jacobian.T

        # Use pseudo-inverse for numerical stability
        if abs(np.linalg.det(Mx_inv)) >= 1e-2:
            Mx = np.linalg.inv(Mx_inv)
        else:
            Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)

        # Compute generalized forces for task-space control
        tau = jacobian.T @ Mx @ (self.Kp * twist - self.Kd * (jacobian @ dq))

        # Add joint task in nullspace (redundancy resolution)
        Jbar = M_inv @ jacobian.T @ Mx
        ddq = self.Kp_null * (self.q0 - q) - self.Kd_null * dq
        tau += (np.eye(7) - jacobian.T @ Jbar.T) @ ddq

        # Add gravity compensation
        if self.gravity_compensation:
            gravity = np.array(state.gravity)
            tau += gravity

        # Add force feedback (if contact is detected)
        tau_force = np.zeros(7)
        if self.force_feedback:
            # Get external forces and torques from robot state
            # Note: In real robot, you might need force-torque sensor or estimate from tau_ext
            ext_forces = np.array(state.K_F_ext_hat_K)[:3]  # Estimated external force
            contact_force_world = -ext_forces  # Negate to get force on environment

            # Check if there's significant contact
            contact_threshold = 1.0  # Newtons
            if np.linalg.norm(contact_force_world) > contact_threshold:
                # PID force control
                force_error = self.desired_force - contact_force_world
                force_derivative = (force_error - self.force_error_prev) / self.dt

                # Compute force correction
                force_correction = (self.Kp_force * force_error +
                                  self.Kd_force * force_derivative)
                force_correction += self.desired_force

                # Add integral term if we have error history
                if len(self.force_errors) > 0:
                    force_error_sum = np.sum(self.force_errors, axis=0) * self.dt
                    force_correction += self.Ki_force * force_error_sum

                # Convert force to joint torques
                tau_force = jacobian.T[:, :3] @ force_correction
                tau -= tau_force  # Subtract to apply force

                self.force_error_prev = force_error

                # Log data
                self.contact_forces.append(contact_force_world.copy())
                self.force_errors.append(force_error.copy())
                self.desired_forces.append(self.desired_force.copy())
                self.tau_forces.append(tau_force.copy())
            else:
                # No contact
                self.contact_forces.append(np.zeros(3))
                self.force_errors.append(np.zeros(3))
                self.desired_forces.append(self.desired_force.copy())
                self.tau_forces.append(np.zeros(7))

        # Log total torques
        self.taus.append(tau.copy())

        return tau

    def run_control_loop(self, duration: float = 30.0):
        """
        Run the control loop for specified duration.

        Args:
            duration: Control duration in seconds
        """
        print("\nStarting control loop...")
        print(f"Target position: {self.target_pos}")
        print(f"Target orientation (quaternion): {self.target_quat}")
        print(f"Duration: {duration} seconds")
        print("\nPress Ctrl+C to stop early\n")

        start_time = time.time()

        try:
            # Move to home position first
            print("Moving to home position...")
            self.robot.move_to_joint_position(self.q0, speed=0.1)
            print("Reached home position!")

            # Start control loop
            print("Starting force control...")
            iteration = 0

            def control_callback(state):
                nonlocal iteration

                # Compute control torques
                tau = self.compute_control_torques(state)

                # Clip torques to safe limits
                tau_limits = np.array([87, 87, 87, 87, 12, 12, 12])  # Nm
                tau = np.clip(tau, -tau_limits, tau_limits)

                # Log timestamp
                current_time = time.time() - start_time
                self.timestamps.append(current_time)

                # Print status every 1000 iterations (1 second at 1kHz)
                if iteration % 1000 == 0:
                    ee_pos = np.array(state.O_T_EE).reshape(4, 4).T[:3, 3]
                    pos_error = np.linalg.norm(self.target_pos - ee_pos)
                    print(f"Time: {current_time:.2f}s | Pos error: {pos_error:.4f}m | " +
                          f"Contact force: {np.linalg.norm(self.contact_forces[-1]) if self.contact_forces else 0:.2f}N")

                iteration += 1

                # Check if duration exceeded
                if current_time >= duration:
                    return None  # Stop control

                return tau

            # Run control loop
            self.robot.control(control_callback)

        except KeyboardInterrupt:
            print("\nControl interrupted by user")
        except Exception as e:
            print(f"\nError during control: {e}")
        finally:
            print("\nControl loop finished")
            self.robot.stop_controller()

    def plot_results(self):
        """Plot the logged data."""
        if len(self.timestamps) == 0:
            print("No data to plot")
            return

        times = np.array(self.timestamps)

        # Plot force feedback data
        if self.force_feedback and len(self.contact_forces) > 0:
            contact_forces = np.array(self.contact_forces)
            desired_forces = np.array(self.desired_forces)
            force_errors = np.array(self.force_errors)

            fig, axes = plt.subplots(3, 1, figsize=(12, 8))
            axes_labels = ["X", "Y", "Z"]

            for i in range(3):
                axes[i].plot(times[:len(contact_forces)], contact_forces[:, i],
                           label=f"Contact Force {axes_labels[i]}")
                axes[i].plot(times[:len(desired_forces)], desired_forces[:, i],
                           label=f"Desired Force {axes_labels[i]}")
                axes[i].plot(times[:len(force_errors)], force_errors[:, i],
                           label=f"Force Error {axes_labels[i]}")
                axes[i].set_xlabel("Time [s]")
                axes[i].set_ylabel("Force [N]")
                axes[i].legend()
                axes[i].grid(True)

            plt.suptitle("Force Feedback Control")
            plt.tight_layout()

        # Plot joint torques
        if len(self.taus) > 0:
            taus = np.array(self.taus)

            fig, axes = plt.subplots(7, 1, figsize=(12, 14))

            for i in range(7):
                axes[i].plot(times[:len(taus)], taus[:, i])
                axes[i].set_xlabel("Time [s]")
                axes[i].set_ylabel(f"Joint {i+1} Torque [Nm]")
                axes[i].grid(True)

            plt.suptitle("Joint Torques")
            plt.tight_layout()

        plt.show()


def main():
    """Main function."""
    # Robot IP address - modify this to match your robot
    ROBOT_IP = "172.16.0.2"

    # Create controller
    controller = FrankaForceController(robot_ip=ROBOT_IP)

    # Optionally modify target pose
    # controller.target_pos = np.array([0.4, 0.0, 0.4])
    # controller.target_quat = np.array([1.0, 0.0, 0.0, 0.0])

    # Optionally modify force feedback settings
    # controller.desired_force = np.array([0.0, 0.0, 15.0])
    # controller.Kp_force = 0.5

    # Run control loop
    controller.run_control_loop(duration=30.0)

    # Plot results
    controller.plot_results()


if __name__ == "__main__":
    main()
