# ------------------------------------------------------------------------------
# Profiling wrapper for real-time control loop diagnostics
# Use this to identify which operations exceed the 1ms budget
# ------------------------------------------------------------------------------
import time
import numpy as np
from collections import defaultdict
from typing import Dict, List, Callable
import functools


class ControlLoopProfiler:
    """
    Lightweight profiler for real-time control loops.

    Usage:
        profiler = ControlLoopProfiler()

        # In your control loop:
        with profiler.measure("pinocchio_fk"):
            pino.forwardKinematics(...)

        with profiler.measure("jacobian"):
            jac = pino.getFrameJacobian(...)

        # After N iterations:
        profiler.print_stats()
    """

    def __init__(self, warmup_iterations: int = 10):
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.iteration_count = 0
        self.warmup_iterations = warmup_iterations
        self.total_loop_times: List[float] = []
        self._loop_start: float = 0.0

    def start_iteration(self):
        """Call at the start of each control loop iteration."""
        self.iteration_count += 1
        self._loop_start = time.perf_counter()

    def end_iteration(self):
        """Call at the end of each control loop iteration."""
        if self.iteration_count > self.warmup_iterations:
            elapsed = (time.perf_counter() - self._loop_start) * 1e6  # microseconds
            self.total_loop_times.append(elapsed)

    class _MeasureContext:
        def __init__(self, profiler: 'ControlLoopProfiler', name: str):
            self.profiler = profiler
            self.name = name
            self.start = 0.0

        def __enter__(self):
            self.start = time.perf_counter()
            return self

        def __exit__(self, *args):
            if self.profiler.iteration_count > self.profiler.warmup_iterations:
                elapsed = (time.perf_counter() - self.start) * 1e6  # microseconds
                self.profiler.timings[self.name].append(elapsed)

    def measure(self, name: str) -> _MeasureContext:
        """Context manager to measure a code block."""
        return self._MeasureContext(self, name)

    def print_stats(self, top_n: int = 20):
        """Print timing statistics."""
        print("\n" + "=" * 80)
        print("CONTROL LOOP PROFILING RESULTS")
        print("=" * 80)

        if self.total_loop_times:
            loop_arr = np.array(self.total_loop_times)
            print(f"\nTotal loop time (μs):")
            print(f"  Mean:   {np.mean(loop_arr):8.1f}")
            print(f"  Std:    {np.std(loop_arr):8.1f}")
            print(f"  Min:    {np.min(loop_arr):8.1f}")
            print(f"  Max:    {np.max(loop_arr):8.1f}")
            print(f"  p95:    {np.percentile(loop_arr, 95):8.1f}")
            print(f"  p99:    {np.percentile(loop_arr, 99):8.1f}")

            violations = np.sum(loop_arr > 1000)
            print(f"\n  Iterations > 1ms: {violations} / {len(loop_arr)} "
                  f"({100*violations/len(loop_arr):.1f}%)")

        print(f"\nComponent breakdown (μs):")
        print(f"{'Component':<40} {'Mean':>8} {'Std':>8} {'Max':>8} {'p99':>8}")
        print("-" * 80)

        # Sort by mean time, descending
        sorted_items = sorted(
            self.timings.items(),
            key=lambda x: np.mean(x[1]) if x[1] else 0,
            reverse=True
        )

        total_mean = 0.0
        for name, times in sorted_items[:top_n]:
            if times:
                arr = np.array(times)
                mean_t = np.mean(arr)
                total_mean += mean_t
                print(f"{name:<40} {mean_t:8.1f} {np.std(arr):8.1f} "
                      f"{np.max(arr):8.1f} {np.percentile(arr, 99):8.1f}")

        print("-" * 80)
        print(f"{'Sum of measured components':<40} {total_mean:8.1f}")

        if self.total_loop_times:
            unmeasured = np.mean(self.total_loop_times) - total_mean
            print(f"{'Unmeasured overhead':<40} {unmeasured:8.1f}")

        print("=" * 80 + "\n")

    def get_violations(self, threshold_us: float = 1000.0) -> Dict[str, int]:
        """Get count of timing violations per component."""
        violations = {}
        for name, times in self.timings.items():
            arr = np.array(times)
            violations[name] = int(np.sum(arr > threshold_us))
        return violations


def profile_function(profiler: ControlLoopProfiler, name: str):
    """Decorator to profile a function."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with profiler.measure(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Example usage showing how to instrument your control loop
EXAMPLE_USAGE = """
# In hybrid_slope_libfranka_pin.py:

from profile_control_loop import ControlLoopProfiler

# Create profiler (global or pass to controllers)
profiler = ControlLoopProfiler()

# In your main control loop:
while True:
    profiler.start_iteration()

    with profiler.measure("readOnce"):
        robot_state, duration = active_control.readOnce()

    # ... (remove all print statements!)

    if control_phase == ControlPhase.APPROACHING:
        with profiler.measure("approach_update"):
            tau = approach_controller.update(robot_state)

    with profiler.measure("writeOnce"):
        active_control.writeOnce(torque_cmd)

    profiler.end_iteration()

    # Print stats every 1000 iterations
    if profiler.iteration_count % 1000 == 0:
        profiler.print_stats()

# In CartesianSpacePDController.update(), instrument sections:
def update(self, robot_state, profiler=None):
    with profiler.measure("pose_error"):
        twist = compute_ee_pose_error(...)

    with profiler.measure("pinocchio_fk"):
        pino.forwardKinematics(...)
        pino.computeJointJacobians(...)
        pino.updateFramePlacements(...)

    with profiler.measure("jacobian"):
        jac = pino.getFrameJacobian(...)

    with profiler.measure("mass_matrix"):
        M_inv = pino.computeMinverse(...)
        Mx = task_space_inertiaM(M_inv, jac)

    with profiler.measure("control_law"):
        self.tau[:] = jac.T @ Mx @ (...)

    with profiler.measure("nullspace"):
        # nullspace computation

    with profiler.measure("gravity"):
        g = pino.computeGeneralizedGravity(...)
"""

if __name__ == "__main__":
    print(EXAMPLE_USAGE)
