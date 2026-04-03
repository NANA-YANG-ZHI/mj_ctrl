# 1. Introduction

## 1.1 Motivation

✏️ *Describe the real-world need for robots that simultaneously move fast and maintain precise contact force — e.g., surface polishing, surgical tools, inspection tasks.*

- Physical interaction between robots and environments requires simultaneous motion and force control
- Classical impedance control achieves indirect force through position set-point shifting — limited for explicit force targets
- High-speed constrained tasks (polishing, wiping, drawing) demand dynamic decoupling of force and motion subspaces
- Motivating application: drawing a circle on a flat/slope surface while maintaining force profile

## 1.2 Problem Statement

✏️ *Formally state the control problem: controlling end-effector contact force in the constrained (normal) direction while tracking a desired trajectory in the motion directions, on flat and inclined surfaces.*

## 1.3 Contributions

- Implementation and evaluation of the Hybrid Force-Impedance Control framework (Iskandar et al., 2023) in MuJoCo simulation and on real hardware (Franka FR3/Panda via libfranka)
- Development of a unified software architecture (MujocoRobotInterface) that allows the same controller code to run in simulation and on the real robot
- Systematic study of force control methods (F_desired feedforward, PD, velocity-term/paper method) on flat and slope surfaces across a range of end-effector speeds
- Extension of the controller from flat (φ = z) to inclined surface (slope 30°) using constraint-frame rotation and projected Jacobians
- Modified motion control law that avoids the dependency on the Coriolis matrix term Cx, enabling stable high-speed trajectory tracking in simulation
- Analysis of friction effects on contact force regulation
- Learning-based PPO compensation: comparison of PPO-based torque compensation and the paper method in simulation/real robot (motivation: paper method uses separate-space control but struggles with friction; PPO outputs a compensation torque to address force errors in friction environments)

## 1.4 Thesis Structure

✏️ *One paragraph summarising each chapter.*

---

# 2. Background and Related Work

✏️ *Literature on hybrid force/motion control, impedance control, operational-space dynamics, constraint Jacobians, and learning-based force compensation.*

---

# 3. Methodology and Controller Design

## 3.1 System Overview

### 3.1.1 Simulation Setup (MuJoCo)

*Describe the MuJoCo simulation environment: Franka FR3, Panda, and KUKA IIWA robots. Scene construction via XML with procedurally added slope geometry (`add_slope_xml`). Contact model settings (condim, friction). Simulation timestep (1 ms).*

### 3.1.2 Real-Robot Setup (Franka FR3 + libfranka)

*Describe the real-robot hardware interface: Franka Research 3 (FR3) and Panda controlled via libfranka. Torque control at 1 kHz. Pinocchio used for dynamics (Jacobian, inertia matrix, Coriolis, gravity) in both sim and real.*

### 3.1.3 Unified Software Architecture

*Describe the `MujocoRobotInterface` wrapper: it mimics the libfranka `Robot` and `RobotState` interface (fields: `q`, `dq`, `O_T_EE`, `O_F_ext_hat_K`, `tau_J_d`), enabling the same `HybridController` and `CartesianSpacePDController` code to run in both environments without modification.*

📊 `DIAGRAM` Software architecture: libfranka / MujocoRobotInterface → unified RobotState → controller stack (CartesianSpacePDController → HybridController)

## 3.2 Task Definition

*Define the two tasks:*
*(1) Circle-drawing on a flat horizontal surface (euler = [0,0,0]).*
*(2) Circle-drawing on a 30° inclined slope (euler = [30°,0,0]).*
*Trajectory parametrisation: angular speed swept from π/4 to 4π rad/s (EE linear speed 0.08–1.26 m/s at radius 0.1 m), circle radius = 0.1 m. Circle centre position and target orientation.*

## 3.3 Constraint Frame and Selection Matrices

*Explain how the constraint frame is constructed for flat surface (φ = z, S_fc = [0,0,1,0,0,0]ᵀ) and slope surface (block-diagonal rotation R applied to S_fc and S_vc). Define S_v (5-dim motion) and S_f (1-dim force).*

- Flat surface: `J_φ = S_fᵀ J`, `J_motion = S_vᵀ J`
- Slope 30°: 6×6 block-diagonal rotation matrix R from world to constraint frame; `S_f = R · S_fc`, `S_v = R · S_vc`
- Key finding: S_f and R are constant for planar/slope surfaces, so `J̇_φ ≈ 0` (not true for curved surfaces e.g. sphere)
- S_vc structure: x tangential (row 0), y tangential (row 1), rx (row 3), ry (row 4), rz (row 5) — rz kept in motion space to avoid orientation drift

📊 `PLOT` Visual: selection matrix S_f (6×1) and S_v (6×5) structures for flat vs. slope

## 3.4 Force Control Strategies Investigated

*Describe each force control strategy tested, from simplest to most complete. All methods compute a scalar force command `F_ctrl_φ` in the constraint direction, which is then mapped to joint torques via `τ_ctrl_φ = J_φᵀ F_ctrl_φ`.*

### 3.4.1 Baseline: F_desired Feed-forward

*`F_ctrl_φ = F_desired`. No feedback. Establishes performance lower bound. Works well at low speed but degrades at high speed due to unmodelled dynamic coupling.*

### 3.4.2 PD Force Controller

*`F_ctrl_φ = F_desired – Kp·(|F_desired| – |F_ext_φ|) – Kd·Ḟ_ext_φ`. Sign convention: force error is defined as magnitude difference to handle the signed nature of contact forces (robot pushes down, environment pushes up). Ḟ is numerically differentiated from consecutive measurements.*

### 3.4.3 Hybrid Force-Impedance with Velocity-Term Compensation (Paper Method)

*`F_ctrl_φ = F_desired + control_force_compensation + contact_force_compensation + velocity_term`.*

- `control_force_compensation = Λ_φ J_φ M⁻¹ (–τ_ctrl_x – τ_ctrl_v)`: cancels the coupling from motion-space and null-space torques into the constraint direction
- `contact_force_compensation = Λ_φ J_φ M⁻¹ Jᵀ_motion F_ext_x`: accounts for tangential contact forces coupling back into the constraint space
- `velocity_term = Λ_φ (J_φ M⁻¹ C – J̇_φ) q̇`: compensates dynamic coupling from Coriolis forces and Jacobian time variation

*Explain the negative sign in the control compensation: τ_ctrl_x opposes the external forces from the surface, so subtracting it cancels the coupling into the constraint force subspace.*

📊 `PLOT` Decomposition plot: F_desired, control_force_compensation, contact_force_compensation, velocity_term, and their sum (F_ctrl_φ) vs. actual contact force over time

**Note:** A PI force controller (adding integral term to eliminate steady-state error) was considered but not implemented in this work.

## 3.5 Motion Control Law

*The paper (Iskandar et al.) derives the motion control law using the full operational-space dynamics, which requires the Coriolis matrix term Cx = (Λ_motion J_motion M⁻¹ C – Λ_motion J̇_motion) to cancel dynamic coupling in the motion space. In this implementation, Cx is not available (requires additional Pinocchio calls and careful bookkeeping for the projected space), and omitting Cx caused instability — the end-effector diverged rapidly ("flying") at higher speeds.*

*Modified law used in this work: Cartesian-space feedforward PD with motion-space inertia Λ_motion:*

`a_motion = ẍ_desired_sel + Kp·(S_vᵀ·x̃) + Kd·(S_vᵀ·ẋ̃)`
`F_ctrl_x = Λ_motion · a_motion`
`τ_ctrl_x = J_motionᵀ · F_ctrl_x`

*where Λ_motion = (J_motion M⁻¹ J_motionᵀ)⁻¹ is the projected task-space inertia, x̃ is the 6D pose error in the motion subspace, and ẋ̃ = (ẋ_desired – J q̇) projected onto S_v. The desired acceleration ẍ_desired_sel and velocity error ẋ̃ are projected onto the motion subspace via S_vᵀ before computing the PD law.*

*Comparison: The paper's approach cancels model nonlinearities via Cx, achieving better decoupling. The modified law relies on high Kp/Kd to suppress tracking error, and is sufficient for stable circle drawing across the tested speed range.*

📊 `PLOT` EE position tracking: X, Y, Z vs. time (desired vs. actual) for slow (π/4) and fast (π) circle speeds

## 3.6 Null-Space Control

*Explain null-space torque `τ_ns = N₂ · τ₀` for posture control / joint limit avoidance, where N₂ = (I – J̃ᵀ J̃_inv) projects onto the null space of the stacked Jacobian [J_φ; J_motion], ensuring null-space motion does not disturb the primary task. Discuss its negligible effect on contact force in practice.*

## 3.7 Approach Phase Controller

*Describe the `CartesianSpacePDController` used in Phase 1 to move the robot from rest to the surface contact point before engaging hybrid control. Uses task-space impedance control with minimum-jerk trajectory generation (`generate_line_trajectory_delta`). Transition logic: switches to HybridController when EE position is within 1 cm of target.*

*Control law: `τ = Jᵀ Λ_x (Kp · twist – Kd · Jq̇) + N τ_null`*

📊 `PLOT` Time-series: approach phase (position control) transitioning to hybrid control phase, showing contact force rising from 0 to F_desired

---

# 4. Simulation Experiments and Results

## 4.1 Experimental Setup

*Franka FR3 or Panda in MuJoCo. Flat board (condim=1, frictionless). Circle radius = 0.1 m. F_desired = –8 N. Impedance gains: impedance_pos = [100,100,100], impedance_ori = [50,50,50]. Simulation timestep = 1 ms. Torque rate limiting: max_delta_tau = 1 Nm/step.*

📊 `PLOT` MuJoCo scene rendering: robot drawing circle on flat board with trajectory trace

## 4.2 Force Control Method Comparison (Frictionless Flat Surface)

*Compare F_desired feedforward, PD, and paper method (velocity-term) at varying end-effector speeds. The angular_speed_sweep experiment parametrically varies angular speed ω from π/4 to 4π rad/s (corresponding to EE linear speeds v = r·ω from ~0.08 to ~1.26 m/s at r = 0.1 m).*

*Metrics recorded for each (method, speed) combination:*
- *Average force error: |F_desired – F_ext_z| averaged over the circle*
- *Variance of force error*
- *Average position error: 3D EE position deviation from desired circle*
- *Variance of position error*

📊 `PLOT` EE linear speed (m/s) vs. average force error — one curve per method (feedforward, PD, paper)
📊 `PLOT` EE linear speed (m/s) vs. force error variance — one curve per method
📊 `PLOT` EE linear speed (m/s) vs. average position error — one curve per method
📊 `PLOT` EE linear speed (m/s) vs. position error variance — one curve per method

*Expected result: feedforward degrades proportionally with speed; PD shows oscillations at high speed; paper method maintains lower force error across speeds due to velocity-term compensation.*

## 4.3 Effect of Friction

*Repeat experiments with friction enabled (condim=3, friction='0.7 0.02 0.01' and condim=6, friction='1 0.02 0.01'). Show friction induces stick-slip instability at friction='0.85'. Analyse how velocity-term compensation interacts with tangential friction.*

📊 `PLOT` Force tracking comparison: paper method with no friction vs. low friction vs. full friction
📊 `PLOT` Tangential friction force norm over time — showing non-constant tangential force but relatively stable z-component

## 4.4 Contact Force Compensation Analysis

*Deep dive into the coupling compensation terms in the paper method: `control_force_compensation` (cancels motion-torque coupling), `contact_force_compensation` (cancels tangential force coupling), `velocity_term` (cancels dynamic coupling). Show effect when each term is included (+1), excluded (0), or negated (–1).*

📊 `PLOT` Component visualisation: control_force_compensation, contact_force_compensation, velocity_term over time in the force subspace

## 4.5 Impedance Gain Sensitivity

*Show how impedance_pos values (100, 500, 1000, 1500) affect EE position tracking stability and force tracking accuracy. Identify stable operating range.*

📊 `PLOT` Force tracking and EE position error for impedance_pos ∈ {100, 500, 1000, 1500} — comparison subplots

---

# 5. Learning-Based Compensation (PPO)

## 5.1 Motivation

*The paper method uses decoupled subspace control and performs well in frictionless conditions, but force errors increase with friction and at high speed. A PPO-based residual compensation policy is trained to output an additive torque correction on top of the hybrid controller, targeting the residual force error.*

## 5.2 PPO Policy Design

✏️ *State and action space, reward function (force error penalty + position tracking reward), network architecture.*

## 5.3 Simulation Results

✏️ *Compare PPO compensation vs. paper method in simulation with friction.*

## 5.4 Real-Robot Transfer

✏️ *Describe sim-to-real transfer. Results on FR3/Panda.*

---

# 6. Conclusion

## 6.1 Summary

✏️ *Recap contributions and main findings.*

## 6.2 Limitations and Future Work

✏️ *PI controller not implemented. Cx-based full dynamic cancellation not implemented. Sim-to-real gap. Extension to curved surfaces.*
