"""Extended Dynamic Mode Decomposition (EDMD).

EDMD lifts the state with a dictionary ``psi: R^d -> R^p`` and fits a *finite
dimensional* approximation ``K`` of the (infinite dimensional) Koopman operator
on the span of the dictionary:

    psi(x_{k+1})  ~  psi(x_k) K        (row-vector convention),

by (ridge-regularised) least squares over all snapshot pairs. The state is
recovered from a lifted vector with a linear read-out ``B`` fitted by least
squares (``x ~ psi(x) B``); for state-inclusive dictionaries ``B`` is exactly the
projection onto the leading ``d`` coordinates.
"""

from __future__ import annotations

import numpy as np

from .observables import Dictionary


class EDMD:
    """EDMD with a user supplied dictionary.

    Parameters
    ----------
    dictionary : :class:`~koopman.observables.Dictionary`
    reg : float
        Ridge penalty (on standardised features); ``0`` gives plain least squares.
    scale : bool
        Rescale every feature by its standard deviation before fitting. This
        is a similarity transform of ``K`` (same spectrum) but keeps the problem
        well conditioned when features span many orders of magnitude.
    rcond : float
        Cut-off for small singular values in the least-squares solve.
    """

    def __init__(
        self,
        dictionary: Dictionary,
        reg: float = 1e-8,
        scale: bool = True,
        rcond: float = 1e-12,
    ):
        self.dictionary = dictionary
        self.reg = float(reg)
        self.scale = scale
        self.rcond = rcond
        self._fitted = False

    # ------------------------------------------------------------------ #
    def _lstsq(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Solve ``min ||A W - B||^2 + reg * n * ||W||^2``."""
        if self.reg > 0.0:
            n, p = A.shape
            A = np.vstack([A, np.sqrt(self.reg * n) * np.eye(p)])
            B = np.vstack([B, np.zeros((p, B.shape[1]))])
        return np.linalg.lstsq(A, B, rcond=self.rcond)[0]

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "EDMD":
        """Fit on snapshot pairs ``X, Y`` of shape ``(n_samples, d)``."""
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if X.shape != Y.shape or X.ndim != 2:
            raise ValueError("X and Y must be 2-D arrays of identical shape")
        PX = self.dictionary(X)
        PY = self.dictionary(Y)
        if self.scale:
            std = PX.std(axis=0)
            self.feature_scale_ = np.divide(
                1.0, std, out=np.ones_like(std), where=std > 1e-12
            )
        else:
            self.feature_scale_ = np.ones(PX.shape[1])
        PXs, PYs = PX * self.feature_scale_, PY * self.feature_scale_
        self.K_ = self._lstsq(PXs, PYs)  # (p, p)
        self.readout_ = np.linalg.lstsq(PXs, X, rcond=self.rcond)[0]  # (p, d)
        self.eigenvalues = np.linalg.eigvals(self.K_)
        self._fitted = True
        return self

    def _check(self):
        if not self._fitted:
            raise RuntimeError("call fit() first")

    # ------------------------------------------------------------------ #
    @property
    def operator(self) -> np.ndarray:
        """Koopman matrix ``K`` in the (standardised) feature basis, ``psi' = psi K``."""
        self._check()
        return self.K_

    def lift(self, X: np.ndarray) -> np.ndarray:
        """Standardised features ``psi(x)`` of states ``(n, d)``."""
        self._check()
        return self.dictionary(X) * self.feature_scale_

    def project(self, Psi: np.ndarray) -> np.ndarray:
        """Map standardised features back to the state."""
        self._check()
        return Psi @ self.readout_

    def predict_one_step(self, X: np.ndarray) -> np.ndarray:
        """One-step prediction ``x_{k+1}`` for every row of ``X``."""
        return self.project(self.lift(X) @ self.K_)

    def predict(self, x0: np.ndarray, n_steps: int) -> np.ndarray:
        """Roll out ``n_steps`` steps from ``x0`` in the lifted space.

        Returns an array of shape ``(n_steps + 1, d)`` whose first row is the
        read-out of ``psi(x0)`` (equal to ``x0`` for state-inclusive dictionaries).
        """
        self._check()
        psi = self.lift(np.asarray(x0, dtype=float))  # (1, p)
        out = np.empty((n_steps + 1, self.readout_.shape[1]))
        out[0] = (psi @ self.readout_)[0]
        for k in range(1, n_steps + 1):
            psi = psi @ self.K_
            out[k] = (psi @ self.readout_)[0]
        return out
