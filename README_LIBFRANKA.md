# Hybrid Force-Impedance Control with libfranka-python

This document describes the adapted robot control script that uses **libfranka-python** for real Franka Panda robot control instead of mujoco simulation.

## Overview

The `hybrid_slope_ctrl_libfranka.py` script implements a two-phase hybrid force-impedance controller:

1. **Phase 1: Approach Control** - Cartesian space PD control to move the end-effector to a target position
2. **Phase 2: Circle Drawing** - Hybrid force/motion control to draw circles while maintaining contact force

## Key Changes from Original Script

### Original (hybrid_slope_ctrl_class_pin.py)
- Used **mujoco** for robot simulation
- Used **mujoco.viewer** for visualization
- Got robot state from `data.qpos`, `data.qvel`, `data.site()`
- Sent control signals via `data.ctrl`
- Simulated contact forces via `data.contact`

### Adapted (hybrid_slope_ctrl_libfranka.py)
- Uses **libfranka-python** for real robot control
- No visualization (real robot)
- Gets robot state from `robot.readOnce()`:
  - Joint positions/velocities: `state.q`, `state.dq`
  - End-effector pose: `state.O_T_EE` (4x4 transformation matrix)
  - External forces: `state.O_F_ext_hat_K` (estimated external wrench)
- Sends control signals via `robot.setTorques()`
- Uses estimated external forces instead of simulated contacts
- Runs at **1kHz real-time control loop**

### Preserved Components
- **Pinocchio** is still used for:
  - Jacobian computation
  - Mass matrix calculation
  - Gravity compensation
  - Forward kinematics
- **Control algorithms** remain unchanged:
  - Task-space impedance control
  - Nullspace control
  - Hybrid force/motion control

## Dependencies

### Required
```bash
pip install libfranka-python
pip install pin  # or pinocchio
pip install numpy
pip install matplotlib
```

### libfranka-python Installation
```bash
# From PyPI
pip install libfranka-python

# Or from source
git clone https://github.com/BarisYazici/libfranka-python
cd libfranka-python
pip install .
```

## Usage

### Basic Usage
```bash
python hybrid_slope_ctrl_libfranka.py --robot-ip 172.16.0.2
```

### Command Line Arguments
- `--robot-ip`: **Required**. IP address of the Franka robot (e.g., `172.16.0.2`)

### Configuration

Edit the configuration classes in the script to customize behavior:

#### ControllerConfig
```python
dt: float = 0.001  # 1kHz control loop
circle_center: np.ndarray = np.array([0.5, 0.0, 0.45])
circle_radius: float = 0.1
circle_duration: float = 10.0
angular_speed: float = np.pi
position_tolerance: float = 0.01  # 1cm
euler: np.ndarray = np.array([np.deg2rad(-10), 0, 0])  # Slope angle
```

#### CartesianSpacePDControlConfig
```python
Kpos: float = 0.95  # Position error gain
impedance_pos: np.ndarray = np.asarray([500.0, 500.0, 500.0])
impedance_ori: np.ndarray = np.asarray([250.0, 250.0, 250.0])
Kp_null: np.ndarray = np.asarray([75.0, 75.0, 50.0, 50.0, 40.0, 25.0, 25.0])
```

#### HybridControllerConfig
```python
impedance_pos: np.ndarray = np.asarray([500.0, 500.0, 500.0]) * 2
k_normal: float = 5000.0  # Material stiffness
F_desired_contact: np.ndarray = np.array([-10.0])  # Desired normal force
```

## Robot Setup

Before running the script:

1. **Unlock the robot** via the Franka Desk interface
2. **Activate FCI mode** (Franka Control Interface)
3. Ensure the robot is in a **safe workspace**
4. Configure the robot's **network settings** to allow FCI connections
5. Set appropriate **collision thresholds** for contact tasks

## Safety Features

The script includes several safety features:

1. **Torque Limits**: Commands are clipped to ±87 Nm (Franka limits)
2. **Collision Behavior**: Configurable collision thresholds
3. **Keyboard Interrupt**: Press Ctrl+C to safely stop
4. **Automatic Shutdown**: Robot stops gracefully on exit

## Control Flow

```
START
  ↓
Connect to Robot (libfranka.Robot)
  ↓
Initialize Controllers (Pinocchio models)
  ↓
PHASE 1: APPROACHING
  ├─ Read robot state (robot.readOnce())
  ├─ Compute control torques (approach_controller.update())
  ├─ Send torques (robot.setTorques())
  └─ Check if target reached → Transition to PHASE 2
  ↓
PHASE 2: CIRCLE DRAWING
  ├─ Read robot state
  ├─ Compute control torques (circle_controller.update())
  ├─ Send torques
  └─ Check if finished → STOPPED
  ↓
STOPPED (gravity compensation only)
  ↓
Plot Results
```

## Data Logging

The script logs:
- End-effector positions
- Target positions
- Contact forces (from external force estimation)
- Desired forces
- Control force compensation terms

Plots are saved to `plots/` directory:
- `combined_position_tracking.png`: Position tracking over time
- `contact_forces.png`: Contact force tracking during circle drawing

## Differences in State Access

### MuJoCo (Original)
```python
# Joint positions
q = data.qpos[dof_ids]

# Joint velocities
dq = data.qvel[dof_ids]

# End-effector position
ee_pos = data.site(site_id).xpos

# End-effector rotation matrix
ee_rot = data.site(site_id).xmat

# Contact forces
contact_force = data.contact[i].force
```

### libfranka (Adapted)
```python
# Joint positions
q = robot_state.q  # np.array (7,)

# Joint velocities
dq = robot_state.dq  # np.array (7,)

# End-effector pose (4x4 transformation matrix)
ee_transform = robot_state.O_T_EE.reshape(4, 4, order='F')
ee_pos = ee_transform[:3, 3]
ee_rot = ee_transform[:3, :3]

# External forces/torques (estimated)
F_ext = robot_state.O_F_ext_hat_K  # np.array (6,) [fx, fy, fz, tx, ty, tz]
```

## libfranka-python API Reference

### Robot Class
```python
robot = libfranka.Robot(robot_ip)
```

### Read State
```python
state = robot.readOnce()
# Returns RobotState with:
#   - q: joint positions [7]
#   - dq: joint velocities [7]
#   - tau_J: joint torques [7]
#   - O_T_EE: end-effector pose [16] (4x4 flattened, column-major)
#   - O_F_ext_hat_K: external wrench estimate [6]
```

### Send Torques
```python
robot.setTorques(tau_list)  # tau_list: list of 7 torques
```

### Set Collision Behavior
```python
robot.setCollisionBehavior(
    lower_torque_thresholds,  # [7]
    upper_torque_thresholds,  # [7]
    lower_force_thresholds,   # [6]
    upper_force_thresholds    # [6]
)
```

## Troubleshooting

### Connection Issues
- Verify robot IP address
- Check network connectivity: `ping <robot-ip>`
- Ensure FCI mode is activated in Franka Desk
- Check firewall settings

### Control Errors
- If robot stops unexpectedly, check collision thresholds
- If torques are too high, reduce controller gains
- If oscillations occur, increase damping (Kd values)

### Import Errors
- Ensure libfranka-python is installed: `pip install libfranka-python`
- Ensure pinocchio is installed: `pip install pin`
- Check Python version (3.7+ recommended)

## References

- [libfranka-python GitHub](https://github.com/BarisYazici/libfranka-python)
- [Official libfranka Documentation](https://frankarobotics.github.io/docs/libfranka.html)
- [Franka Control Interface (FCI) Documentation](https://frankarobotics.github.io/docs/franka_control.html)
- [Pinocchio Documentation](https://stack-of-tasks.github.io/pinocchio/)

## License

This code is adapted from the original hybrid_slope_ctrl_class_pin.py script.

## Author Notes

**Adaptation Summary:**
- Removed all mujoco dependencies
- Integrated libfranka-python for real robot control
- Maintained pinocchio for dynamics computations
- Converted simulation loop to real-time 1kHz control loop
- Replaced simulated contact forces with external force estimation
- Added command-line argument parsing for robot IP
- Preserved all control algorithms and logic
