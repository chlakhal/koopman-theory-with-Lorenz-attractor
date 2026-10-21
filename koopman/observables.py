"""Dictionaries of observables ``psi(x)`` used by EDMD.

Every dictionary maps an array of states ``(n, d)`` to features ``(n, p)``.
The first ``d`` features are always the state itself, so the state can be read
back from a lifted vector by taking its first ``d`` entries.

* :class:`PolynomialDictionary`   -- all monomials up to a given degree
  (generic; used for Lorenz, whose vector field is quadratic).
* :class:`PendulumDictionary`     -- physics-informed trigonometric observables
  ``theta_dot^p * {1, sin(k theta), cos(k theta)}``: they respect the
  ``2 pi``-periodicity of the angle and are closed under differentiation up to
  the truncation order (``d/dt sin(theta) = theta_dot cos(theta)``, ...).
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from typing import Sequence

import numpy as np


class Dictionary:
    """Base class: subclasses fill ``state_dim`` and ``names``."""

    state_dim: int
    names: list[str]

    @property
    def n_features(self) -> int:
        return len(self.names)

    def __call__(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    @staticmethod
    def _as_2d(X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X[None] if X.ndim == 1 else X


class PolynomialDictionary(Dictionary):
    """All monomials of the state variables up to ``degree``.

    Degree-1 terms come first (so the state is the leading block), then
    degree 2, and so on. ``include_constant`` appends the constant ``1`` at the end.
    """

    def __init__(
        self,
        state_dim: int,
        degree: int = 2,
        include_constant: bool = False,
        state_names: Sequence[str] | None = None,
    ):
        if degree < 1:
            raise ValueError("degree must be >= 1")
        self.state_dim = state_dim
        self.degree = degree
        self.include_constant = include_constant
        var = list(state_names) if state_names else [f"x{i}" for i in range(state_dim)]
        if len(var) != state_dim:
            raise ValueError("state_names must have one name per state variable")
        self._exponents: list[tuple[int, ...]] = []
        self.names = []
        for deg in range(1, degree + 1):
            for combo in combinations_with_replacement(range(state_dim), deg):
                exps = [0] * state_dim
                for i in combo:
                    exps[i] += 1
                self._exponents.append(tuple(exps))
                self.names.append("*".join(var[i] for i in combo))
        if include_constant:
            self.names.append("1")

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = self._as_2d(X)
        cols = [np.prod(X ** np.array(e), axis=1) for e in self._exponents]
        if self.include_constant:
            cols.append(np.ones(len(X)))
        return np.stack(cols, axis=1)


class PendulumDictionary(Dictionary):
    """Trigonometric observables for the state ``(theta, theta_dot)``.

    Features (in this order): ``theta``, ``theta_dot``, the constant ``1``,
    then ``theta_dot^p sin(k theta)`` and ``theta_dot^p cos(k theta)`` for
    ``p = 0..velocity_degree`` and ``k = 1..n_harmonics``, then the pure
    velocity powers ``theta_dot^p`` for ``p = 2..velocity_degree``.
    """

    def __init__(self, n_harmonics: int = 2, velocity_degree: int = 2):
        if n_harmonics < 1 or velocity_degree < 1:
            raise ValueError("n_harmonics and velocity_degree must be >= 1")
        self.state_dim = 2
        self.n_harmonics = n_harmonics
        self.velocity_degree = velocity_degree
        names = ["theta", "omega", "1"]
        for p in range(velocity_degree + 1):
            for k in range(1, n_harmonics + 1):
                pre = "" if p == 0 else (f"omega^{p} " if p > 1 else "omega ")
                names += [f"{pre}sin({k}theta)", f"{pre}cos({k}theta)"]
        names += [f"omega^{p}" for p in range(2, velocity_degree + 1)]
        self.names = names

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = self._as_2d(X)
        theta, omega = X[:, 0], X[:, 1]
        cols = [theta, omega, np.ones_like(theta)]
        for p in range(self.velocity_degree + 1):
            for k in range(1, self.n_harmonics + 1):
                cols.append(omega**p * np.sin(k * theta))
                cols.append(omega**p * np.cos(k * theta))
        cols += [omega**p for p in range(2, self.velocity_degree + 1)]
        return np.stack(cols, axis=1)


def lorenz_dictionary(degree: int = 2) -> PolynomialDictionary:
    """Polynomial observables in ``(x, y, z)`` with readable feature names."""
    return PolynomialDictionary(3, degree=degree, state_names=("x", "y", "z"))
