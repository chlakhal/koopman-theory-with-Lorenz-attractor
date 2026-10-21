"""Small shared helpers."""

from __future__ import annotations

import numpy as np


def as_trajectory_array(trajectories) -> np.ndarray:
    """Return trajectories as a 3-D array ``(n_traj, n_steps, state_dim)``.

    Accepts a single trajectory ``(n_steps, state_dim)``, an array
    ``(n_traj, n_steps, state_dim)`` or a list of equally long trajectories.
    """
    arr = np.asarray(trajectories, dtype=float)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3:
        raise ValueError(
            "trajectories must have shape (n_steps, d) or (n_traj, n_steps, d)"
        )
    return arr


def snapshot_pairs(trajectories):
    """Consecutive snapshot pairs ``(X, Y)`` with ``Y[i]`` one step after ``X[i]``.

    Pairs never straddle the boundary between two trajectories. Both outputs
    have shape ``(n_pairs, state_dim)``.
    """
    arr = as_trajectory_array(trajectories)
    X = arr[:, :-1, :].reshape(-1, arr.shape[-1])
    Y = arr[:, 1:, :].reshape(-1, arr.shape[-1])
    return X, Y


def set_seed(seed: int) -> None:
    """Seed NumPy and (if installed) PyTorch for reproducible runs."""
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass
