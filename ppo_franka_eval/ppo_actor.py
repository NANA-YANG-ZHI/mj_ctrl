"""
Real-time-safe PPO actor using pure NumPy inference.

Why NOT torch.nn
----------------
`import torch.nn as nn` causes PyTorch (via pthreadpool / OpenMP) to spawn a
pool of persistent OS-level C++ threads — one per CPU core.  These threads
live for the entire process lifetime and compete with libfranka's real-time
control thread for CPU time.  Even though they sit idle, their presence
causes the scheduler to occasionally delay the control thread, causing
libfranka's 1 ms deadline to be missed and triggering:

    communication_constraints_violation

The fix: never import torch.nn.  For inference we only need the weight
matrices as float32 numpy arrays.  The forward pass is three matmuls and two
tanh calls — all in NumPy with BLAS-optimised routines.

`import torch` (without .nn) is used *once*, inside ``from_checkpoint()``,
solely to deserialise the .pt file.  It does NOT spawn the thread pool.
All torch objects are discarded immediately after weight extraction, and
torch is never referenced again.

Network layout (matches ppo_friction_compensation/ppo_agent.Actor)
------------------------------------------------------------------
  Linear(obs_dim → hidden)  weight: mean_net.0.weight  bias: mean_net.0.bias
  Tanh
  Linear(hidden  → hidden)  weight: mean_net.2.weight  bias: mean_net.2.bias
  Tanh
  Linear(hidden  → act_dim) weight: mean_net.4.weight  bias: mean_net.4.bias
  (log_std parameter present in state-dict but not used for inference)
"""

from __future__ import annotations

import numpy as np


class PPOActorInference:
    """
    Deterministic 2-layer MLP actor — pure NumPy, zero torch.nn.

    All weight tensors are stored as contiguous float32 C-arrays.
    Intermediate activations and the output buffer are pre-allocated
    once at construction so that ``infer()`` performs no heap allocation.

    Parameters
    ----------
    w1, b1 : ndarray  shape (hidden, obs_dim), (hidden,)
    w2, b2 : ndarray  shape (hidden, hidden),  (hidden,)
    w3, b3 : ndarray  shape (act_dim, hidden), (act_dim,)
        Weight matrices and biases extracted from the training checkpoint.
    act_limit : float
        Symmetric clip applied to the output [Nm].
    """

    def __init__(
        self,
        w1: np.ndarray,
        b1: np.ndarray,
        w2: np.ndarray,
        b2: np.ndarray,
        w3: np.ndarray,
        b3: np.ndarray,
        act_limit: float = 5.0,
    ) -> None:
        # Store as contiguous float32 arrays for fast BLAS matmul
        self._w1 = np.ascontiguousarray(w1, dtype=np.float32)
        self._b1 = np.ascontiguousarray(b1, dtype=np.float32)
        self._w2 = np.ascontiguousarray(w2, dtype=np.float32)
        self._b2 = np.ascontiguousarray(b2, dtype=np.float32)
        self._w3 = np.ascontiguousarray(w3, dtype=np.float32)
        self._b3 = np.ascontiguousarray(b3, dtype=np.float32)
        self._act_limit = np.float32(act_limit)

        hidden  = w1.shape[0]
        act_dim = w3.shape[0]

        # Pre-allocate all intermediate buffers — infer() does zero allocation
        self._h1  = np.empty(hidden,  dtype=np.float32)
        self._h2  = np.empty(hidden,  dtype=np.float32)
        self._out = np.empty(act_dim, dtype=np.float32)

    # ------------------------------------------------------------------
    # Construction helper
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_prefix: str,
        obs_dim: int = 25,     # kept for API symmetry; verified against weights
        act_dim: int = 7,
        hidden:  int = 64,
        act_limit: float = 5.0,
    ) -> "PPOActorInference":
        """
        Load actor weights from ``<checkpoint_prefix>_actor.pt``.

        ``import torch`` is used *here only* for deserialisation; it is NOT
        imported at module level and does NOT trigger torch.nn thread-pool
        initialisation.  All torch tensors are released immediately after
        weight extraction.

        Parameters
        ----------
        checkpoint_prefix : str
            E.g. ``"ppo_checkpoints/final"`` or
            ``"experiments/run_xxx/checkpoints/epoch_0050"``.
        """
        # Late import — torch alone does not spawn the thread pool.
        # torch.nn is never imported.
        import torch  # noqa: PLC0415

        path = f"{checkpoint_prefix}_actor.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)

        # Verify shapes match requested architecture
        actual_obs  = state["mean_net.0.weight"].shape[1]
        actual_hid  = state["mean_net.0.weight"].shape[0]
        actual_act  = state["mean_net.4.weight"].shape[0]
        if actual_obs != obs_dim:
            print(
                f"[PPOActorInference] WARNING: checkpoint obs_dim={actual_obs}, "
                f"requested obs_dim={obs_dim}. Using checkpoint shape."
            )
        if actual_hid != hidden:
            print(
                f"[PPOActorInference] WARNING: checkpoint hidden={actual_hid}, "
                f"requested hidden={hidden}. Using checkpoint shape."
            )
        if actual_act != act_dim:
            print(
                f"[PPOActorInference] WARNING: checkpoint act_dim={actual_act}, "
                f"requested act_dim={act_dim}. Using checkpoint shape."
            )

        # Extract weight matrices as plain numpy float32 copies
        w1 = state["mean_net.0.weight"].numpy().copy()
        b1 = state["mean_net.0.bias"].numpy().copy()
        w2 = state["mean_net.2.weight"].numpy().copy()
        b2 = state["mean_net.2.bias"].numpy().copy()
        w3 = state["mean_net.4.weight"].numpy().copy()
        b3 = state["mean_net.4.bias"].numpy().copy()

        # Release torch state dict immediately — we no longer need torch
        del state

        print(
            f"[PPOActorInference] Loaded from {path!r}  "
            f"obs={actual_obs} → hidden={actual_hid} → act={actual_act}  "
            f"(pure NumPy inference, no torch.nn)"
        )
        return cls(w1, b1, w2, b2, w3, b3, act_limit=act_limit)

    # ------------------------------------------------------------------
    # Pre-warming
    # ------------------------------------------------------------------

    def warmup(self, n_calls: int = 20) -> None:
        """
        Pre-warm NumPy/BLAS caches before the real-time loop.

        Performs ``n_calls`` dummy forward passes so that BLAS library
        initialisation and CPU cache warming happen before the real-time
        loop starts and before ``gc.disable()`` is called.
        """
        dummy = np.zeros(self._w1.shape[1], dtype=np.float32)
        for _ in range(n_calls):
            self.infer(dummy)
        print(f"[PPOActorInference] Warmed up ({n_calls} dummy inferences).")

    # ------------------------------------------------------------------
    # Inference  (called every action_repeat=20 ms, NOT every 1 ms)
    # ------------------------------------------------------------------

    def infer(self, obs_norm_np: np.ndarray) -> np.ndarray:
        """
        Deterministic forward pass: tanh-MLP, 3 layers, pure NumPy.

        Called at **50 Hz** (every action_repeat control cycles).
        Do NOT call at 1 kHz — not because of torch (there is none), but
        because Pinocchio FK in PPOFrankaEvaluator._build_obs_raw() is also
        only needed at 50 Hz.

        Parameters
        ----------
        obs_norm_np : np.ndarray, shape (obs_dim,), dtype float32
            Normalised observation from WelfordNormalizerInference.

        Returns
        -------
        np.ndarray, shape (act_dim,), dtype float32
            Torque corrections clipped to ±act_limit [Nm].
            The pre-allocated internal buffer is returned; copy if you
            need to keep the value across subsequent calls.
        """
        # Layer 1: h1 = tanh(W1 @ x + b1)
        np.dot(self._w1, obs_norm_np, out=self._h1)
        self._h1 += self._b1
        np.tanh(self._h1, out=self._h1)

        # Layer 2: h2 = tanh(W2 @ h1 + b2)
        np.dot(self._w2, self._h1, out=self._h2)
        self._h2 += self._b2
        np.tanh(self._h2, out=self._h2)

        # Layer 3: out = W3 @ h2 + b3  (linear output)
        np.dot(self._w3, self._h2, out=self._out)
        self._out += self._b3

        # Clip to action limit (in-place, no allocation)
        np.clip(self._out, -self._act_limit, self._act_limit, out=self._out)
        return self._out
