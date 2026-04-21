# Hybrid Force-Impedance Controller on a Curved Surface

## Overview

The controller has two sequential phases:

1. **Phase 1 — Approach**: Cartesian-space PD controller moves the end-effector (EE) to a point slightly above the cylinder surface.
2. **Phase 2 — Hybrid Control**: Once contact is made, a hybrid force/motion controller takes over. It controls **force** in the surface-normal direction and **position** in the tangential/rotational directions simultaneously.

---

## Cylinder Geometry (fixed parameters)

```
Center : [0.5,  0.0,  0.45]  m
Axis   : [1,    0,    0  ]    (horizontal, world X)
Radius : 0.1 m
```

The EE sweeps around the cylinder in the **Y–Z plane**. The angle θ is measured from the top of the cylinder (θ = 0 → pointing straight up).

---

## Phase 1: Approach (CartesianSpacePD)

The robot moves to a standoff position **1 cm above** the surface at the starting angle θ₀ = 0:

```
approach_normal     = [0, sin(0), cos(0)] = [0, 0, 1]
approach_target_pos = center + (radius + 0.002) * normal
                    = [0.5, 0, 0.45] + 0.102 * [0, 0, 1]
                    = [0.5, 0.0, 0.552]
```

A minimum-jerk trajectory (10σ³ − 15σ⁴ + 6σ⁵, σ = t/T) is used to smoothly reach this goal. When the EE is within position tolerance, Phase 2 begins.

---

## Phase 2: Hybrid Force-Impedance Control

Every control step (dt = 1 ms) the following pipeline runs.

### Step 0 — Arc Trajectory

The desired trajectory on the cylinder surface at time `t` is:

```
θ(t) = θ_start + ω · t

target_pos = center + radius · [0, sin θ, cos θ]
x_dot      = radius · ω · [0, cos θ, −sin θ]
x_ddot     = −radius · ω² · [0, sin θ,  cos θ]
```

Default angular speed ω = π/4 rad/s (≈ 45°/s).

---

### Numerical Example: θ = 30° (π/6 rad)

With ω = π/4 rad/s, the EE reaches 30° after **t = (π/6)/(π/4) = 0.667 s**.

**Surface normal and tangents:**

```
normal  n̂ = [0,  sin 30°,  cos 30°] = [0,  0.500,  0.866]
tangent t̂ = [0,  cos 30°, −sin 30°] = [0,  0.866, −0.500]   (circumferential)
axis    â  = [1,  0,        0       ]                          (along cylinder X)
```

**Target EE position and velocity:**

```
target_pos = [0.5, 0, 0.45] + 0.1 · [0, 0.500, 0.866]
           = [0.500,  0.050,  0.537]  m

x_dot  = 0.1 · (π/4) · [0,  0.866, −0.500]
       = [0,  0.068,  −0.039]  m/s

x_ddot = −0.1 · (π/4)² · [0,  0.500,  0.866]
       = [0,  −0.031,  −0.053]  m/s²
```

---

### Step 1 — Build Selection Matrices

Hybrid control splits the 6-DOF task space into:
- **Force space** (1 DOF): along the outward surface normal → `S_f` (6×1)
- **Motion space** (5 DOF): axial, circumferential, and 3 rotational → `S_v` (6×5)

At θ = 30°:

```
S_f[:3, 0] = n̂ = [0,  0.500,  0.866]          ← force direction

S_v[:3, 0] = â  = [1,  0,     0    ]            ← axial translation
S_v[:3, 1] = t̂  = [0,  0.866, −0.500]          ← circumferential translation
S_v[3,  2] = 1                                   ← roll
S_v[4,  3] = 1                                   ← pitch
S_v[5,  4] = 1                                   ← yaw
```

`S_f` and `S_v` are orthogonal complements: they span the full 6-DOF space without overlap.

---

### Step 2 — Target EE Orientation

The EE is oriented so it always faces the surface squarely:

```
x_ee = â         = [1,  0,      0    ]   (EE x along cylinder axis)
z_ee = −n̂        = [0, −0.500, −0.866]   (EE z presses into surface)
y_ee = z_ee × x_ee = [0, −0.866,  0.500]
```

Rotation matrix  `R_target = [x_ee | y_ee | z_ee]`:

```
R_target = [[1,  0,      0    ],
            [0, −0.866, −0.500],
            [0,  0.500, −0.866]]
```

---

### Step 3 — Kinematics & Dynamics (Pinocchio)

For the current joint angles **q** and velocities **dq**:

```
J        = full 6×7 Jacobian (LOCAL_WORLD_ALIGNED frame)
M_inv    = inverse of 7×7 joint-space inertia matrix

J_phi    = S_f^T  @ J    →  1×7  (force-space Jacobian)
J_motion = S_v^T  @ J    →  5×7  (motion-space Jacobian)

Mx_constraint = (J_phi   @ M_inv @ J_phi^T   )⁻¹   ← 1×1 task-space inertia
Mx_motion     = (J_motion @ M_inv @ J_motion^T)⁻¹   ← 5×5 task-space inertia
```

---

### Step 4 — Null-Space Torque

A PD controller drives the joints back toward the home configuration **q₀** while staying in the null space of the constrained Jacobian (so it does not disturb task-space behaviour):

```
τ_null_raw = Kp_null · (q₀ − q) − Kd_null · dq

J_stack = vstack([J_phi, J_motion])   # 6×7
J⁺      = M_inv @ J_stack^T @ (J_stack @ M_inv @ J_stack^T)⁻¹   (dynamically consistent pseudoinverse)
N       = I − J_stack^T @ J⁺^T       # 7×7 null-space projector

τ_null  = N @ τ_null_raw
```

---

### Step 5 — Motion-Space Control (5 DOF)

A feedforward PD law tracks the desired trajectory in the 5 motion directions:

```
twist        = pose error (6-vector: [Δpos·Kpos, rotvec_error·Kori])
x̃            = twist @ S_v          (5-vector, projected onto motion space)
ẋ̃            = (ẋ_des_full − J·dq) @ S_v   (velocity error, projected)
ẍ_sel        = x_ddot_des_full @ S_v        (desired acceleration, projected)

a_motion     = ẍ_sel + Kp · x̃ + Kd · ẋ̃    (desired task-space acceleration)
F_motion     = Mx_motion @ a_motion          (force in 5-DOF motion space)
τ_motion     = J_motion^T @ F_motion
```

**Example numbers** (simplified, at θ = 30°, assuming small errors):

```
ẍ_sel ≈ [−0.031, −0.053, 0, 0, 0]   (centripetal, from trajectory)
Kp    = diag([100, 100, 100, 50, 50])
Kd    = diag([20,  20,  20,  14, 14])

If position error along t̂ = 2 mm = 0.002 m:
   a_motion[1] ≈ −0.053 + 100·0.002 + Kd·ẋ̃
             ≈ −0.053 + 0.200 = +0.147 m/s²
```

---

### Step 6 — Force Control (1 DOF, "paper" method)

Force is controlled along the surface normal. The desired force is **F_desired = −10 N** (negative = pressing into the surface).

The force command is built from four terms:

```
F_ctrl = F_desired
       + ctrl_comp      (decouples force from the motion torques)
       + contact_comp   (accounts for tangential contact forces)
       + vel_term       (Coriolis/velocity-dependent correction)
```

Where:

```
ctrl_comp    = −Mx_constraint @ J_phi @ M_inv @ (τ_motion + τ_null)
contact_comp =  Mx_constraint @ J_phi @ M_inv @ J_motion^T @ F_ext_tangential
vel_term     =  Mx_constraint @ (J_phi @ M_inv @ C − J̇_phi) @ dq
```

**PI correction** is added on top to eliminate steady-state force error:

```
f_error              = F_ext_phi − F_desired          (measured minus desired)
integral_f_error    += f_error · dt
PI                   = −Kp_force · f_error − Ki_force · integral_f_error

F_ctrl              += PI
```

**Example numbers** (at θ = 30°):

```
Assume:
  F_ext_phi (measured projection onto n̂) = −8 N
  F_desired                               = −10 N
  Kp_force = 0.8,  Ki_force = 0.8

f_error = −8 − (−10) = +2 N   (pressing 2 N too little)
PI      = −0.8 · 2 − 0.8 · integral
        ≈ −1.6 N               (on first step, integral ≈ 0)

F_ctrl ≈ −10 + (small coupling terms) + (−1.6)
        ≈ −11.6 N              (commands harder push to close the gap)
```

The constraint torque:

```
τ_force = J_phi^T @ F_ctrl
```

---

### Step 7 — Total Torque

All components are summed:

```
τ_total = τ_force + τ_motion + τ_null + g(q)
```

Where `g(q)` is the gravity compensation vector from Pinocchio.

A **torque rate limiter** clamps the per-joint change from the previous command to ±1 Nm/step, preventing abrupt jumps.

---

## Summary Diagram

```
                        ┌──────────────────┐
Arc trajectory ──────►  │  θ(t), pos, vel, │
                        │  accel           │
                        └───────┬──────────┘
                                │
                    ┌───────────▼──────────────┐
                    │  Cylinder geometry        │
                    │  normal n̂, tangents       │
                    │  S_f (1×6), S_v (5×6)     │
                    └───┬──────────┬────────────┘
                        │          │
            ┌───────────▼───┐  ┌───▼────────────────┐
            │ Force control │  │  Motion control     │
            │ (normal, 1D)  │  │  (tangent+rot, 5D)  │
            │               │  │                     │
            │ F_desired      │  │  PD + feedforward   │
            │ + coupling     │  │  on pose error      │
            │ + PI           │  │                     │
            └───────┬───────┘  └──────────┬──────────┘
                    │                     │
            τ_force = J_phi^T @ F_ctrl    │
                    │         τ_motion = J_motion^T @ F_motion
                    │                     │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  + τ_null           │
                    │  + g(q)             │
                    │  + rate limiter     │
                    └──────────┬──────────┘
                               │
                           τ_total → robot joints
```
