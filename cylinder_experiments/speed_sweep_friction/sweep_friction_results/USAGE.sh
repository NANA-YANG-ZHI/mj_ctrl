#!/bin/bash
# Quick usage examples for plot_trajectory_and_force.py

# Example 1: Generate comparison plot for all methods at 3π rad/s
# Shows all 6 methods in a 2x3 grid with Y-Z plane tracking and force error
python plot_trajectory_and_force.py \
    --robot fr3_friction \
    --angular-speed 9.42 \
    --comparison

# Example 2: Single method plot
# Generates a detailed 2-panel plot (force error + Y-Z tracking) for one method
python plot_trajectory_and_force.py \
    --robot fr3_friction \
    --angular-speed 3.0 \
    --method paper_pi

# Example 3: Generate plots for all methods separately
# Creates one plot file for each method
python plot_trajectory_and_force.py \
    --robot fr3_jointf_surff \
    --angular-speed 1.5

# Example 4: Interactive mode
# Prompts you to select robot, speed, and plot type interactively
python plot_trajectory_and_force.py --interactive

# Example 5: List available options
# Shows all available robots and angular speeds
python plot_trajectory_and_force.py --list

# Example 6: Custom output path for single method
python plot_trajectory_and_force.py \
    --robot fr3_friction \
    --angular-speed 2.5 \
    --method baseline \
    --output custom_output.png

# Angular speeds available: 0.1 to 3.0 rad/s (in 0.1 increments)
# Robots available: fr3_friction, fr3_jointf_surff
# Methods available: baseline, ff, ff_pi, pd, paper, paper_pi

# Note: The script automatically finds the closest available speed
# For example, requesting 3π ≈ 9.42 rad/s will use 3.0 rad/s
