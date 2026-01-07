# Franka Robot Force Control Guide

This guide explains how to use the `franka_force_control.py` script for real-time force control on the Franka Emika Panda robot.

## Overview

The script implements operational space force control with:
- **Cartesian impedance control**: Control end-effector position and orientation
- **Nullspace control**: Control redundant joint configuration
- **Force feedback**: PID control for desired contact forces
- **Gravity compensation**: Automatic gravity compensation

This is adapted from the MuJoCo simulation (`opspace_force.py`) to work with real Franka hardware using libfranka.

## Installation

### Prerequisites

1. **Franka Robot Setup**:
   - Robot must be powered on and in FCI (Franka Control Interface) mode
   - Network connection configured (robot typically at `172.16.0.2`)
   - User has proper permissions and unlocked the robot

2. **Python Dependencies**:

```bash
# Install panda-py (Python bindings for libfranka)
pip install panda-py numpy matplotlib

# Alternative: If panda-py doesn't work, try:
# pip install frankx
# or
# pip install franky-panda
```

## Usage

### Basic Usage

```bash
python franka_force_control.py
```

### Customizing Parameters

Edit the script or modify the `main()` function:

```python
def main():
    # Set robot IP address
    ROBOT_IP = "172.16.0.2"  # Change to your robot's IP

    # Create controller
    controller = FrankaForceController(robot_ip=ROBOT_IP)

    # Customize target pose
    controller.target_pos = np.array([0.5, 0.0, 0.3])  # [x, y, z] in meters
    controller.target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]

    # Customize force feedback
    controller.desired_force = np.array([0.0, 0.0, 10.0])  # Force in Newtons
    controller.Kp_force = 0.4  # Force control proportional gain
    controller.Kd_force = 0.002  # Force control derivative gain
    controller.Ki_force = 0.4  # Force control integral gain

    # Customize impedance
    controller.impedance_pos = np.array([100.0, 100.0, 100.0])  # Stiffness [N/m]
    controller.impedance_ori = np.array([50.0, 50.0, 50.0])  # Rot. stiffness [Nm/rad]

    # Run for 30 seconds
    controller.run_control_loop(duration=30.0)

    # Plot results
    controller.plot_results()
```

## Key Differences from MuJoCo Version

| Feature | MuJoCo (`opspace_force.py`) | Franka (`franka_force_control.py`) |
|---------|----------------------------|-----------------------------------|
| **Robot State** | `mujoco.MjData` | `panda_py.RobotState` via `state` |
| **Control Loop** | `mujoco.mj_step()` at 500 Hz | `robot.control(callback)` at 1 kHz |
| **Contact Detection** | `data.ncon`, `mujoco.mj_contactForce()` | `state.K_F_ext_hat_K` (estimated ext. forces) |
| **Jacobian** | `mujoco.mj_jacSite()` | `state.O_J_EE` (end-effector Jacobian) |
| **Mass Matrix** | `mujoco.mj_solveM()` | `state.mass_matrix` |
| **Gravity** | `data.qfrc_bias` | `state.gravity` |
| **End-effector Pose** | `data.site(site_id).xpos/xmat` | `state.O_T_EE` (4x4 transform) |

## Robot State Information

The `state` object in the control callback provides:

```python
state.q          # Joint positions [7]
state.dq         # Joint velocities [7]
state.O_T_EE     # End-effector pose (4x4 matrix, column-major)
state.O_J_EE     # End-effector Jacobian (6x7, column-major)
state.mass_matrix  # Joint-space mass matrix (7x7)
state.gravity    # Gravity torques [7]
state.coriolis   # Coriolis torques [7]
state.K_F_ext_hat_K  # Estimated external wrench [6]
state.tau_J      # Measured joint torques [7]
```

## Safety Considerations

⚠️ **Important Safety Notes**:

1. **Emergency Stop**: Keep the emergency stop button accessible at all times
2. **Workspace**: Ensure the robot workspace is clear of obstacles and people
3. **Torque Limits**: The script clips torques to safe limits:
   - Joints 1-3: ±87 Nm
   - Joints 4-7: ±12 Nm
4. **Start Position**: Robot moves to home position before starting control
5. **Monitoring**: Watch the robot during operation and stop if behavior is unexpected

## Troubleshooting

### Connection Issues

```
Failed to connect to robot
```

**Solutions**:
- Verify robot is powered on and in FCI mode
- Check network connection: `ping 172.16.0.2`
- Ensure robot is unlocked in the web interface
- Check firewall settings

### Real-time Performance Issues

```
Control loop running slow
```

**Solutions**:
- Close unnecessary applications
- Run on a real-time Linux kernel (recommended)
- Increase process priority: `sudo nice -n -20 python franka_force_control.py`

### Force Control Not Working

```
No contact forces detected
```

**Solutions**:
- Check contact threshold (default: 1.0 N)
- Verify `state.K_F_ext_hat_K` is providing data
- Consider using external force-torque sensor for better force sensing
- Adjust `contact_threshold` in `compute_control_torques()`

## Alternative Python Bindings

If `panda-py` doesn't work for you, here are alternatives:

### Using frankx

```python
from frankx import Affine, Robot

robot = Robot("172.16.0.2")
robot.set_default_behavior()
# ... implement control loop
```

### Using franky-panda

```python
from franky import Robot, RobotState

robot = Robot("172.16.0.2")
# ... implement control loop
```

## Control Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Control Loop (1 kHz)                  │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  1. Read Robot State                                     │
│     ├─ Joint positions (q) and velocities (dq)          │
│     ├─ End-effector pose (position, orientation)        │
│     ├─ Jacobian, Mass matrix, Gravity                   │
│     └─ External forces (contact detection)              │
│                                                           │
│  2. Compute Desired Twist (Spatial Velocity)            │
│     ├─ Position error → Linear velocity                 │
│     └─ Orientation error → Angular velocity             │
│                                                           │
│  3. Operational Space Control                            │
│     ├─ Task-space impedance: τ_task = J^T M_x K_p twist │
│     └─ Nullspace control: τ_null = (I - J^T J̄^T) ddq   │
│                                                           │
│  4. Force Feedback (if contact detected)                 │
│     ├─ Measure contact force (from external wrench)     │
│     ├─ PID control: F_cmd = K_p e + K_d ė + K_i ∫e     │
│     └─ τ_force = J^T F_cmd                              │
│                                                           │
│  5. Gravity Compensation                                 │
│     └─ τ_gravity = g(q)                                  │
│                                                           │
│  6. Sum and Clip Torques                                 │
│     └─ τ = τ_task + τ_null + τ_gravity - τ_force        │
│                                                           │
│  7. Send Commands to Robot                               │
│     └─ robot.control(τ)                                  │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

## References

- [libfranka Documentation](https://frankaemika.github.io/libfranka/)
- [panda-py GitHub](https://github.com/JeanElsner/panda-py)
- Khatib, O. (1987). A Unified Approach for Motion and Force Control of Robot Manipulators
- Original MuJoCo implementation: `opspace_force.py`

## Support

For issues with:
- **This script**: Check the code comments and error messages
- **panda-py**: Visit https://github.com/JeanElsner/panda-py
- **Franka robot**: Consult the Franka Emika documentation
