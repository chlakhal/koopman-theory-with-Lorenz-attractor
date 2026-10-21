"""Benchmark dynamical systems and data generation.

* :class:`Pendulum` -- (optionally damped) nonlinear pendulum, state ``(theta, omega)``.
* :class:`Lorenz`   -- the Lorenz-63 system, state ``(x, y, z)``.

Data are always produced with the custom solvers of :mod:`koopman.solvers`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .solvers import integrate

# Largest Lyapunov exponent of Lorenz-63 for (sigma, rho, beta) = (10, 28, 8/3).
LORENZ_LYAPUNOV = 0.9056


# --------------------------------------------------------------------------- #
# Systems
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Pendulum:
    """Nonlinear pendulum ``theta'' = -(g/L) sin(theta) - c theta'``.

    The state is ``(theta, omega)`` with ``omega = theta'``. ``theta`` is *not*
    wrapped to ``[-pi, pi]`` so that trajectories stay continuous.
    """

    g: float = 9.81
    length: float = 1.0
    damping: float = 0.0
    state_dim: int = 2

    @property
    def omega0_sq(self) -> float:
        return self.g / self.length

    @property
    def omega0(self) -> float:
        return math.sqrt(self.omega0_sq)

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        theta, omega = y
        return np.array([omega, -self.omega0_sq * math.sin(theta) - self.damping * omega])

    def energy(self, states: np.ndarray) -> np.ndarray:
        """Mechanical energy per unit ``L^2`` (zero at the stable equilibrium)."""
        s = np.asarray(states)
        return 0.5 * s[..., 1] ** 2 + self.omega0_sq * (1.0 - np.cos(s[..., 0]))

    def energy_fraction(self, states: np.ndarray) -> np.ndarray:
        """Energy relative to the separatrix; ``< 1`` means the motion is a libration."""
        return self.energy(states) / (2.0 * self.omega0_sq)

    def linearized_eigenvalues(self) -> np.ndarray:
        """Continuous-time eigenvalues of the linearisation at ``theta = 0``."""
        c, w2 = self.damping, self.omega0_sq
        disc = complex(c * c / 4.0 - w2)
        root = np.sqrt(disc)
        return np.array([-c / 2.0 + root, -c / 2.0 - root])


@dataclass(frozen=True)
class Lorenz:
    """Lorenz-63 system with the classical chaotic parameters by default."""

    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0
    state_dim: int = 3

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        x, yy, z = y
        return np.array(
            [self.sigma * (yy - x), x * (self.rho - z) - yy, x * yy - self.beta * z]
        )


# --------------------------------------------------------------------------- #
# Data generation
# --------------------------------------------------------------------------- #
def generate_trajectory(
    system,
    x0,
    t_final: float,
    dt: float,
    method: str = "rk4",
    max_h: float = 0.005,
):
    """Simulate one trajectory on the grid ``0, dt, 2 dt, ..., t_final``.

    For fixed-step methods the output interval ``dt`` is subdivided so that the
    internal step never exceeds ``max_h``.

    Returns ``(t, X)`` with ``X`` of shape ``(n_steps + 1, state_dim)``.
    """
    n = int(round(t_final / dt))
    t = np.arange(n + 1) * dt
    substeps = max(1, int(math.ceil(dt / max_h)))
    X = integrate(system.rhs, np.asarray(x0, float), t, method=method, substeps=substeps)
    return t, X


def sample_pendulum_initial_conditions(
    system: Pendulum,
    n: int,
    rng: np.random.Generator,
    theta_max: float = 2.7,
    omega_max: float = 4.0,
    max_energy_fraction: float = 0.95,
) -> np.ndarray:
    """Draw ``n`` initial conditions that stay in the libration regime.

    Rejection sampling on a box ``|theta| <= theta_max``, ``|omega| <= omega_max``,
    keeping only states whose energy is below ``max_energy_fraction`` of the
    separatrix energy (damping can only lower the energy afterwards).
    """
    out = []
    while len(out) < n:
        cand = np.array(
            [rng.uniform(-theta_max, theta_max), rng.uniform(-omega_max, omega_max)]
        )
        if system.energy_fraction(cand) < max_energy_fraction:
            out.append(cand)
    return np.array(out)


def generate_pendulum_dataset(
    system: Pendulum,
    n_traj: int,
    t_final: float,
    dt: float,
    seed: int = 0,
    method: str = "rk4",
    **ic_kwargs,
):
    """Many pendulum trajectories from random libration initial conditions.

    Returns ``(t, trajectories)`` with ``trajectories`` of shape
    ``(n_traj, n_steps + 1, 2)``.
    """
    rng = np.random.default_rng(seed)
    ics = sample_pendulum_initial_conditions(system, n_traj, rng, **ic_kwargs)
    trajs = []
    t = None
    for x0 in ics:
        t, X = generate_trajectory(system, x0, t_final, dt, method=method)
        trajs.append(X)
    return t, np.stack(trajs)


def generate_lorenz_long(
    system: Lorenz,
    n_steps: int,
    dt: float,
    spinup: float = 20.0,
    x0=(1.0, 1.0, 1.0),
    method: str = "rk4",
):
    """One long Lorenz trajectory on the attractor (the spin-up transient is dropped).

    Returns ``(t, X)`` with ``X`` of shape ``(n_steps + 1, 3)`` and ``t[0] = 0``.
    """
    _, burn = generate_trajectory(system, x0, spinup, dt, method=method)
    t, X = generate_trajectory(system, burn[-1], n_steps * dt, dt, method=method)
    return t, X
