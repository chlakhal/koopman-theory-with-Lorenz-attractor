"""Exact Dynamic Mode Decomposition (DMD) computed through the SVD.

Given snapshot pairs ``(x_k, y_k = F(x_k))`` DMD finds the best-fit linear
operator ``A`` with ``y ~ A x`` (in the least-squares sense) and its spectral
decomposition. With the thin SVD ``X = U S V^*`` truncated to ``r`` modes,

* the reduced operator is ``A_r = U^* Y V S^{-1}``  (``r x r``),
* its eigenpairs ``A_r w = lambda w`` give the eigenvalues of ``A``,
* the *exact* DMD modes are ``Phi = Y V S^{-1} W`` (Tu et al., 2014), which are
  eigenvectors of ``A = Y V S^{-1} U^*`` with the same eigenvalues.

All public arrays use the "one row per sample" convention ``(n_samples, d)``.
"""

from __future__ import annotations

import numpy as np


class DMD:
    """Exact DMD with optional rank truncation.

    Parameters
    ----------
    rank : int, optional
        Number of SVD modes to keep. ``None`` keeps all numerically non-zero ones.
    energy : float in (0, 1], optional
        Alternative to ``rank``: keep the smallest number of modes whose squared
        singular values capture this fraction of the total.
    dt : float
        Time between ``x_k`` and ``y_k``; only used for the continuous-time
        spectrum (``omega = log(lambda) / dt``).
    """

    def __init__(self, rank: int | None = None, energy: float | None = None, dt: float = 1.0):
        if rank is not None and energy is not None:
            raise ValueError("give either rank or energy, not both")
        if rank is not None and rank < 1:
            raise ValueError("rank must be >= 1")
        if energy is not None and not (0.0 < energy <= 1.0):
            raise ValueError("energy must lie in (0, 1]")
        self.rank = rank
        self.energy = energy
        self.dt = float(dt)
        self._fitted = False

    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, Y: np.ndarray) -> "DMD":
        """Fit on snapshot pairs ``X, Y`` of shape ``(n_samples, d)``."""
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if X.shape != Y.shape or X.ndim != 2:
            raise ValueError("X and Y must be 2-D arrays of identical shape")
        Xc, Yc = X.T, Y.T  # columns are snapshots, as in the DMD literature
        U, S, Vh = np.linalg.svd(Xc, full_matrices=False)
        r = self._select_rank(S)
        U, S, Vh = U[:, :r], S[:r], Vh[:r]

        V = Vh.conj().T
        YVSinv = Yc @ V @ np.diag(1.0 / S)
        A_tilde = U.conj().T @ YVSinv
        evals, W = np.linalg.eig(A_tilde)

        self.singular_values_ = np.linalg.svd(Xc, compute_uv=False)
        self.rank_ = r
        self.U_ = U
        self.A_tilde_ = A_tilde
        self.eigenvalues = evals
        self.modes = YVSinv @ W  # exact DMD modes, shape (d, r)
        self.A_ = (YVSinv @ U.conj().T)  # rank-r operator, shape (d, d)
        self._fitted = True
        return self

    def _select_rank(self, S: np.ndarray) -> int:
        numerical = int(np.sum(S > S[0] * max(S.shape) * np.finfo(float).eps))
        if self.rank is not None:
            return min(self.rank, numerical)
        if self.energy is not None:
            cum = np.cumsum(S**2) / np.sum(S**2)
            return min(int(np.searchsorted(cum, self.energy) + 1), numerical)
        return numerical

    def _check(self):
        if not self._fitted:
            raise RuntimeError("call fit() first")

    # ------------------------------------------------------------------ #
    @property
    def operator(self) -> np.ndarray:
        """The rank-``r`` linear operator ``A`` (``y ~ A x`` for column vectors)."""
        self._check()
        return self.A_

    @property
    def continuous_eigenvalues(self) -> np.ndarray:
        """``log(lambda) / dt``: growth rate (real part) and angular frequency (imag)."""
        self._check()
        return np.log(self.eigenvalues.astype(complex)) / self.dt

    @property
    def frequencies(self) -> np.ndarray:
        """Oscillation frequencies in cycles per unit time."""
        return self.continuous_eigenvalues.imag / (2.0 * np.pi)

    def amplitudes(self, x0: np.ndarray) -> np.ndarray:
        """Mode amplitudes ``b`` such that ``Phi b ~ x0`` (least squares)."""
        self._check()
        return np.linalg.lstsq(self.modes, np.asarray(x0, dtype=complex), rcond=None)[0]

    def mode_energies(self, x0: np.ndarray) -> np.ndarray:
        """``|b_j| ||Phi_j||``: how strongly each mode is excited by ``x0``."""
        b = self.amplitudes(x0)
        return np.abs(b) * np.linalg.norm(self.modes, axis=0)

    # ------------------------------------------------------------------ #
    def predict(self, x0: np.ndarray, n_steps: int) -> np.ndarray:
        """Roll out ``n_steps`` steps from ``x0`` using the modal expansion.

        Returns an array of shape ``(n_steps + 1, d)`` whose first row is ``x0``.
        """
        self._check()
        x0 = np.asarray(x0, dtype=float)
        b = self.amplitudes(x0)
        k = np.arange(n_steps + 1)
        dynamics = self.eigenvalues[None, :] ** k[:, None] * b[None, :]  # (n+1, r)
        return (dynamics @ self.modes.T).real

    def predict_one_step(self, X: np.ndarray) -> np.ndarray:
        """Apply the operator once to every row of ``X``."""
        self._check()
        return np.asarray(X, dtype=float) @ self.A_.T
