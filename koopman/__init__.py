"""Koopman operator learning for nonlinear dynamics.

Exact DMD, EDMD with physics-informed observables and Deep Koopman autoencoders,
demonstrated on the nonlinear pendulum and the Lorenz attractor, with custom ODE
solvers for data generation.
"""

from .dmd import DMD
from .edmd import EDMD
from .metrics import error_curve, rmse, state_scale, valid_prediction_time
from .observables import PendulumDictionary, PolynomialDictionary, lorenz_dictionary
from .solvers import integrate
from .systems import (
    LORENZ_LYAPUNOV,
    Lorenz,
    Pendulum,
    generate_lorenz_long,
    generate_pendulum_dataset,
    generate_trajectory,
)
from .utils import set_seed, snapshot_pairs

__version__ = "1.0.0"

__all__ = [
    "DMD",
    "EDMD",
    "LORENZ_LYAPUNOV",
    "Lorenz",
    "Pendulum",
    "PendulumDictionary",
    "PolynomialDictionary",
    "error_curve",
    "generate_lorenz_long",
    "generate_pendulum_dataset",
    "generate_trajectory",
    "integrate",
    "lorenz_dictionary",
    "rmse",
    "set_seed",
    "snapshot_pairs",
    "state_scale",
    "valid_prediction_time",
]
