# ------------------------------------------------------------------------------
# Cylinder surface helpers for run_approach_then_hybrid_mujoco.py (--cylinder).
#
# Geometry constants and the CylinderTrajectory class live in src/trajectory.py.
# This module provides the robot-config map, approach-target setup, and plots.
# ------------------------------------------------------------------------------
import numpy as np

from src.trajectory import (
    CYLINDER_CENTER,
    CYLINDER_AXIS,
    CYLINDER_RADIUS,
    _cylinder_ee_rotation,
)

CYLINDER_CONFIG_MAP = {
    "fr3":              "fr3_cylinder",
    "kuka":             "kuka_cylinder",
    "fr3_friction":     "fr3_friction_cylinder",
    "fr3_jointf":       "fr3_jointf_cylinder",
    "fr3_jointf_surff": "fr3_jointf_surff_cylinder",
}


def get_cylinder_approach_target(theta_start: float, size_z: float):
    """Return (target_pos, target_rot) for the approach phase."""
    normal     = np.array([0.0, np.sin(theta_start), np.cos(theta_start)])
    target_pos = CYLINDER_CENTER + (CYLINDER_RADIUS + size_z) * normal
    target_rot = _cylinder_ee_rotation(normal)
    return target_pos, target_rot


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────

def plot_cylinder_position_tracking(t, ee_pos, tgt_pos, pos_err, save_dir=None):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 11))
    fig.suptitle("Position Tracking on Cylinder Surface", fontsize=13, fontweight="bold")
    gs  = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.35)

    ax_x  = fig.add_subplot(gs[0, 0])
    ax_y  = fig.add_subplot(gs[1, 0])
    ax_z  = fig.add_subplot(gs[2, 0])
    ax_e  = fig.add_subplot(gs[3, 0])
    ax_3d = fig.add_subplot(gs[:, 1], projection="3d")

    for ax, col, lbl in zip([ax_x, ax_y, ax_z], range(3), ["X (m)", "Y (m)", "Z (m)"]):
        ax.plot(t, ee_pos[:, col],  "b-",  lw=1.5, label="EE")
        ax.plot(t, tgt_pos[:, col], "r--", lw=1.5, label="Target")
        ax.set_ylabel(lbl)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    ax_e.plot(t, pos_err * 1e3, "m-", lw=1.5)
    ax_e.set_ylabel("Position error (mm)")
    ax_e.set_xlabel("Time (s)")
    ax_e.grid(True, alpha=0.3)
    ax_e.set_title(f"Mean error: {np.mean(pos_err)*1e3:.2f} mm", fontsize=9)

    theta_wire = np.linspace(0, 2 * np.pi, 60)
    for x_end in [CYLINDER_CENTER[0] - CYLINDER_RADIUS, CYLINDER_CENTER[0] + CYLINDER_RADIUS]:
        ax_3d.plot(
            x_end * np.ones_like(theta_wire),
            CYLINDER_CENTER[1] + CYLINDER_RADIUS * np.sin(theta_wire),
            CYLINDER_CENTER[2] + CYLINDER_RADIUS * np.cos(theta_wire),
            "k-", alpha=0.15, lw=0.8,
        )
    for th in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        ax_3d.plot(
            [CYLINDER_CENTER[0] - CYLINDER_RADIUS, CYLINDER_CENTER[0] + CYLINDER_RADIUS],
            [CYLINDER_CENTER[1] + CYLINDER_RADIUS * np.sin(th)] * 2,
            [CYLINDER_CENTER[2] + CYLINDER_RADIUS * np.cos(th)] * 2,
            "k-", alpha=0.15, lw=0.8,
        )
    ax_3d.plot(tgt_pos[:, 0], tgt_pos[:, 1], tgt_pos[:, 2], "r--", lw=1.5, label="Target arc")
    ax_3d.plot(ee_pos[:, 0],  ee_pos[:, 1],  ee_pos[:, 2],  "b-",  lw=1.5, label="EE path")
    ax_3d.set_xlabel("X (m)"); ax_3d.set_ylabel("Y (m)"); ax_3d.set_zlabel("Z (m)")
    ax_3d.legend(fontsize=8)
    ax_3d.set_title("3-D trajectory", fontsize=9)

    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        path = f"{save_dir}/position_tracking.png"
        fig.savefig(path, dpi=150)
        print(f"[PLOT] Position tracking → {path}")


def plot_cylinder_contact_force(t, cf, normals, f_proj, f_desired, force_err, save_dir=None):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    fig.suptitle("Contact Force – Hybrid Phase on Cylinder", fontsize=13, fontweight="bold")

    axes[0].plot(t, f_proj, "b-", lw=1.5, label="Normal force (meas.)")
    axes[0].axhline(f_desired[0], color="r", lw=1.5, ls="--", label=f"Desired ({f_desired[0]:.1f} N)")
    axes[0].set_ylabel("Force along n̂ (N)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)
    axes[0].set_title(f"Mean measured: {np.mean(f_proj):.2f} N", fontsize=9)

    axes[1].plot(t, force_err, "m-", lw=1.3, label="Error = meas − desired")
    axes[1].axhline(0, color="k", lw=0.8, ls="--")
    axes[1].fill_between(t, force_err, alpha=0.15, color="m")
    axes[1].set_ylabel("Force error (N)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)
    axes[1].set_title(f"Mean |error|: {np.mean(np.abs(force_err)):.3f} N", fontsize=9)

    for i, (lbl, col) in enumerate(zip(["Fx", "Fy", "Fz"], ["tab:blue", "tab:orange", "tab:green"])):
        axes[2].plot(t, cf[:, i], color=col, lw=1.2, label=lbl)
    axes[2].set_ylabel("World-frame force (N)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=9, ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        path = f"{save_dir}/contact_force.png"
        fig.savefig(path, dpi=150)
        print(f"[PLOT] Contact force → {path}")
