"""Plot trajectory tracking and force error for selected speed and friction conditions.

For each method (baseline, ff, ff_pi, pd, paper, paper_pi) generates:
  - Left panel: Force error (Z-axis) vs time
  - Right panel: Position tracking in Y-Z plane with desired trajectory

Usage:
    python plot_trajectory_and_force.py \\
        --robot fr3_friction \\
        --angular-speed 3.0 \\
        --output sweep_friction_results/plots/trajectory_force_plot.png

Or interactive mode:
    python plot_trajectory_and_force.py --interactive
"""

import argparse
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Data path configuration
DATA_BASE_DIR = Path(__file__).parent / "data_trajectory_2"

METHODS = ["baseline", "ff", "ff_pi", "pd", "paper", "paper_pi"]
ROBOTS = ["fr3_friction", "fr3_jointf_surff"]

METHOD_COLORS = {
    "baseline": "tab:gray",
    "ff": "tab:blue",
    "ff_pi": "tab:purple",
    "pd": "tab:green",
    "paper": "tab:orange",
    "paper_pi": "tab:red",
}

METHOD_LABELS = {
    "baseline": "Baseline",
    "ff": "Feedforward",
    "ff_pi": "Feedforward + PI",
    "pd": "PD",
    "paper": "HFDC",
    "paper_pi": "HFDC + PI",
}


def find_closest_speed(robot: str, target_speed: float) -> tuple:
    """Find the closest available angular speed to target.
    
    Returns:
        (actual_speed_value, speed_filename_suffix, is_all_format)
        where is_all_format indicates if the filename uses "_all" suffix
    """
    # Get all available data files for baseline method (all methods have same speeds)
    data_dir = DATA_BASE_DIR / robot / "baseline"
    if not data_dir.exists():
        raise ValueError(f"No data found for robot: {robot}")
    
    data_files = sorted(glob.glob(str(data_dir / "data_*.npz")))
    if not data_files:
        raise ValueError(f"No data files found in {data_dir}")
    
    # Extract speed values from filenames
    available_speeds = []
    for f in data_files:
        # Filename format: data_<speed>.npz
        suffix = Path(f).stem.replace("data_", "")
        try:
            speed = float(suffix)
            available_speeds.append((speed, suffix, False))
        except ValueError:
            continue
    
    if not available_speeds:
        raise ValueError("Could not parse any speed values from filenames")
    
    # Find closest speed
    closest = min(available_speeds, key=lambda x: abs(x[0] - target_speed))
    return closest


def load_data(robot: str, method: str, speed_suffix: str) -> dict:
    """Load data from npz file. Handles both 'speed' and 'speed_all' filename formats."""
    # Try multiple filename formats
    npz_paths_to_try = [
        DATA_BASE_DIR / robot / method / f"data_{speed_suffix}.npz",
        DATA_BASE_DIR / robot / method / f"data_{speed_suffix}_all.npz",
    ]
    
    npz_path = None
    for candidate in npz_paths_to_try:
        if candidate.exists():
            npz_path = candidate
            break
    
    if npz_path is None:
        raise FileNotFoundError(f"Data file not found for {robot}/{method}/{speed_suffix}")
    
    data = np.load(npz_path)
    return {
        "actual_pos": data["actual_positions"],      # (N, 3): x, y, z
        "desired_pos": data["desired_positions"],    # (N, 3): x, y, z
        "force_error": data["force_error"],          # (N,): scalar force error
        "position_error": data["position_error"],    # (N,): scalar position error
        "angular_speed": float(data["angular_speed_rad_s"]),
        "friction_coeff": float(data["multiplier"]),
    }


def plot_single_method(robot: str, method: str, speed_suffix: str, output_path: str = None):
    """Create a 2-panel plot for a single method."""
    
    try:
        data = load_data(robot, method, speed_suffix)
    except (FileNotFoundError, Exception) as e:
        print(f"[ERROR] Failed to load data for {robot}/{method}: {e}")
        return None
    
    actual_pos = data["actual_pos"]
    desired_pos = data["desired_pos"]
    force_error = data["force_error"]
    angular_speed = data["angular_speed"]
    friction_coeff = data["friction_coeff"]
    
    # Time array (assuming 0.002s timestep, which is typical)
    dt = 0.002
    time = np.arange(len(force_error)) * dt
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # LEFT PANEL: Force error vs time
    ax1.plot(time, force_error, linewidth=1.5, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
    ax1.set_xlabel("Time (s)", fontsize=11)
    ax1.set_ylabel("Force Error - Z (N)", fontsize=11)
    ax1.set_title(f"Force Control Error\n{METHOD_LABELS[method]}", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")
    
    # RIGHT PANEL: Y-Z plane trajectory (position tracking)
    ax2.plot(desired_pos[:, 1], desired_pos[:, 2], "k--", linewidth=2, label="Desired", alpha=0.7)
    ax2.plot(actual_pos[:, 1], actual_pos[:, 2], color=METHOD_COLORS[method], 
             linewidth=1.5, label="Actual", alpha=0.8)
    ax2.set_xlabel("Y Position (m)", fontsize=11)
    ax2.set_ylabel("Z Position (m)", fontsize=11)
    ax2.set_title(f"Position Tracking (Y-Z Plane)\n{METHOD_LABELS[method]}", 
                  fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")
    ax2.axis("equal")
    
    # Overall title with experimental conditions
    fig.suptitle(
        f"Robot: {robot.replace('_', ' ').title()} | "
        f"ω = {angular_speed:.3f} rad/s | μ = {friction_coeff:.1f}",
        fontsize=11, fontweight="bold", y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[SAVED] {output_path}")
    
    return fig


def plot_all_methods(robot: str, speed_suffix: str, output_dir: str = None):
    """Create individual plots for all methods."""
    
    output_dir = output_dir or Path(__file__).parent / "plots"
    os.makedirs(output_dir, exist_ok=True)
    
    for method in METHODS:
        output_path = os.path.join(output_dir, f"trajectory_force_{robot}_{speed_suffix}_{method}.png")
        plot_single_method(robot, method, speed_suffix, output_path)


def plot_comparison_all_methods(robot: str, speed_suffix: str, output_path: str = None):
    """Create a 2x3 grid showing all methods side-by-side."""
    
    output_path = output_path or str(Path(__file__).parent / "plots" / f"trajectory_force_comparison_{robot}_{speed_suffix}.png")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, method in enumerate(METHODS):
        try:
            data = load_data(robot, method, speed_suffix)
        except (FileNotFoundError, Exception) as e:
            print(f"[SKIP] {robot}/{method}: {e}")
            axes[idx].text(0.5, 0.5, f"No data for\n{method}", ha="center", va="center")
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])
            continue
        
        actual_pos = data["actual_pos"]
        desired_pos = data["desired_pos"]
        force_error = data["force_error"]
        
        # Create a small 2-panel subplot within the grid cell
        # Show both force and trajectory overlaid or split
        ax = axes[idx]
        
        # Plot trajectory in the main axis (Y-Z plane)
        ax.plot(desired_pos[:, 1], desired_pos[:, 2], "k--", linewidth=1.5, alpha=0.6, label="Desired")
        ax.plot(actual_pos[:, 1], actual_pos[:, 2], color=METHOD_COLORS[method], 
                linewidth=1.2, alpha=0.9, label="Actual")
        
        # Add force error info as text
        mean_force_err = np.mean(force_error)
        std_force_err = np.std(force_error)
        
        ax.text(0.98, 0.02, f"Force Error: {mean_force_err:.2f} ± {std_force_err:.2f} N",
                transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        
        ax.set_xlabel("Y (m)", fontsize=9)
        ax.set_ylabel("Z (m)", fontsize=9)
        ax.set_title(METHOD_LABELS[method], fontsize=10, fontweight="bold", color=METHOD_COLORS[method])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
        ax.axis("equal")
    
    fig.suptitle(
        f"All Methods Comparison - {robot.replace('_', ' ').title()} | "
        f"Speed: {speed_suffix} rad/s",
        fontsize=13, fontweight="bold"
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[SAVED] {output_path}")


def list_available_options():
    """Print available robots and speeds."""
    print("\n=== AVAILABLE ROBOTS ===")
    for robot in ROBOTS:
        robot_dir = DATA_BASE_DIR / robot
        if robot_dir.exists():
            print(f"  {robot}")
    
    print("\n=== AVAILABLE ANGULAR SPEEDS (rad/s) ===")
    for robot in ROBOTS:
        robot_dir = DATA_BASE_DIR / robot / "baseline"
        if robot_dir.exists():
            speeds = []
            for f in sorted(glob.glob(str(robot_dir / "data_*.npz"))):
                suffix = Path(f).stem.replace("data_", "")
                try:
                    speed = float(suffix)
                    speeds.append(f"{speed:.1f}")
                except ValueError:
                    pass
            if speeds:
                print(f"  {robot}: {', '.join(speeds)}")


def interactive_mode():
    """Interactive selection of robot and speed."""
    list_available_options()
    
    print("\n=== SELECT OPTIONS ===")
    robot = input(f"Enter robot name [{ROBOTS[0]}]: ").strip() or ROBOTS[0]
    
    if robot not in ROBOTS:
        print(f"[ERROR] Unknown robot. Available: {ROBOTS}")
        return
    
    try:
        target_speed = float(input("Enter target angular speed in rad/s [1.0]: ") or "1.0")
    except ValueError:
        print("[ERROR] Invalid speed")
        return
    
    try:
        actual_speed, speed_suffix, _ = find_closest_speed(robot, target_speed)
        print(f"[INFO] Using closest available speed: {actual_speed:.3f} rad/s (file suffix: {speed_suffix})")
    except ValueError as e:
        print(f"[ERROR] {e}")
        return
    
    plot_type = input("Plot type? [1=single method, 2=all methods separate, 3=comparison grid]: ").strip() or "3"
    
    output_dir = Path(__file__).parent / "plots"
    
    if plot_type == "1":
        method = input(f"Enter method [{METHODS[0]}]: ").strip() or METHODS[0]
        if method not in METHODS:
            print(f"[ERROR] Unknown method. Available: {METHODS}")
            return
        output_path = str(output_dir / f"trajectory_force_{robot}_{speed_suffix}_{method}.png")
        plot_single_method(robot, method, speed_suffix, output_path)
    elif plot_type == "2":
        plot_all_methods(robot, speed_suffix, str(output_dir))
    else:  # "3" or default
        plot_comparison_all_methods(robot, speed_suffix, str(output_dir / f"trajectory_force_comparison_{robot}_{speed_suffix}.png"))
    
    print("\n[SUCCESS] Plots complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Plot trajectory tracking and force error for cylinder experiment."
    )
    parser.add_argument("--robot", type=str, choices=ROBOTS, 
                       help=f"Robot type (default: {ROBOTS[0]})")
    parser.add_argument("--angular-speed", type=float, 
                       help="Target angular speed in rad/s (finds closest available)")
    parser.add_argument("--method", type=str, choices=METHODS,
                       help="Control method (if not specified, plots all or comparison)")
    parser.add_argument("--output", type=str,
                       help="Output file path")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for multiple plots")
    parser.add_argument("--comparison", action="store_true",
                       help="Generate comparison grid of all methods")
    parser.add_argument("--list", action="store_true",
                       help="List available robots and speeds")
    parser.add_argument("--interactive", action="store_true",
                       help="Interactive mode")
    
    args = parser.parse_args()
    
    # Handle list and interactive modes
    if args.list:
        list_available_options()
        return
    
    if args.interactive or (not args.robot and not args.angular_speed):
        interactive_mode()
        return
    
    # Validate required arguments
    if not args.robot or args.angular_speed is None:
        parser.print_help()
        sys.exit(1)
    
    # Find closest speed
    try:
        actual_speed, speed_suffix, _ = find_closest_speed(args.robot, args.angular_speed)
        print(f"[INFO] Using speed: {actual_speed:.3f} rad/s (file suffix: {speed_suffix})")
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    
    # Generate plots
    if args.method:
        # Single method
        output_path = args.output or str(
            Path(__file__).parent / "plots" / 
            f"trajectory_force_{args.robot}_{speed_suffix}_{args.method}.png"
        )
        plot_single_method(args.robot, args.method, speed_suffix, output_path)
    elif args.comparison:
        # Comparison grid
        output_path = args.output or str(
            Path(__file__).parent / "plots" / 
            f"trajectory_force_comparison_{args.robot}_{speed_suffix}.png"
        )
        plot_comparison_all_methods(args.robot, speed_suffix, output_path)
    else:
        # All methods separate
        output_dir = args.output_dir or str(Path(__file__).parent / "plots")
        plot_all_methods(args.robot, speed_suffix, output_dir)


if __name__ == "__main__":
    main()
