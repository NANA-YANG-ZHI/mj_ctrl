"""
ppo_franka_eval — PPO evaluation utilities for real Franka FR3.

Provides real-time-safe PPO actor and evaluator for deployment on the
physical robot.  The key design decision is that torch.nn is NEVER imported
(not even indirectly), which prevents PyTorch from spawning its persistent
OS-level thread pool and interfering with libfranka's 1 ms control budget.

Key classes
-----------
PPOActorInference
    Loads an actor checkpoint, extracts weights as float32 numpy arrays,
    and performs deterministic inference using pure NumPy matmul + tanh.
    No torch.nn, no thread pool, no allocation in the hot path.

WelfordNormalizerInference
    Read-only normalizer loaded from a .npz checkpoint.  Statistics are
    fixed at load time; std is pre-computed so normalize() is a single
    multiply-add.

PPOFrankaEvaluator
    Ties together the normalizer and actor.  Builds the 25-dim observation
    from live robot_state + Pinocchio FK, normalises it, and runs the actor
    at 50 Hz (every action_repeat=20 control cycles, matching training).
    Between refreshes the last delta_tau is held constant (zero-order hold).
"""

from .ppo_actor import PPOActorInference
from .ppo_franka_evaluator import PPOFrankaEvaluator, WelfordNormalizerInference

__all__ = [
    "PPOActorInference",
    "PPOFrankaEvaluator",
    "WelfordNormalizerInference",
]
