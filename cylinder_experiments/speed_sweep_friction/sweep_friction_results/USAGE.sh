#!/bin/bash
# Quick usage examples for plot_trajectory_and_force.py

# Example 1: Generate comparison plot for all methods at multiplier 1.0
# Produces a 2×3 grid of Y-Z trajectory subplots + a combined force error plot.
python cylinder_experiments/speed_sweep_friction/plot_trajectory_and_force.py \
    --multiplier 1.0 \
    --robot fr3_friction

# Example 2: Single method plot
# Generates a 2-panel plot (Y-Z trajectory + force error vs time) for one method.
python cylinder_experiments/speed_sweep_friction/plot_trajectory_and_force.py \
    --multiplier 1.0 \
    --method paper \
    --robot fr3_friction

# Example 3: Generate plots for all methods separately
# Creates one 2-panel plot file for each method.
python cylinder_experiments/speed_sweep_friction/plot_trajectory_and_force.py \
    --multiplier 1.0 \
    --all-separate \
    --robot fr3_friction

# Robots available: fr3_friction, fr3_jointf_surff
# Methods available: baseline, ff, ff_pi, pd, paper, paper_pi
