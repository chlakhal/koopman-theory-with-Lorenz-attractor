"""Custom ODE solvers used to generate data.

Three integrators are provided, all written from scratch on top of NumPy:

* ``euler``  -- explicit Euler (order 1), kept as a convergence-order baseline;
* ``rk4``    -- classical Runge-Kutta (order 4) with a fixed step;
* ``rk45``   -- Dormand-Prince 5(4) embedded pair with adaptive step control.

All of them integrate ``dy/dt = f(t, y)`` and return the solution *exactly* on a
user supplied time grid ``t_eval`` (the adaptive scheme clips its steps so that
it lands on every grid point, so no dense output / interpolation is needed).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

RHS = Callable[[float, np.ndarray], np.ndarray]


# --------------------------------------------------------------------------- #
# Single steps
# --------------------------------------------------------------------------- #
def euler_step(f: RHS, t: float, y: np.ndarray, h: float) -> np.ndarray:
    """One explicit Euler step."""
    return y + h * f(t, y)


def rk4_step(f: RHS, t: float, y: np.ndarray, h: float) -> np.ndarray:
    """One classical fourth-order Runge-Kutta step."""
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(t + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# Dormand-Prince 5(4) tableau (Dormand & Prince, 1980).
_DP_C = np.array([0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0])
_DP_A = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
]
_DP_B5 = np.array([35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0])
_DP_B4 = np.array(
    [5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40]
)
_DP_E = _DP_B5 - _DP_B4  # error-estimate weights


def dopri5_step(f: RHS, t: float, y: np.ndarray, h: float):
    """One Dormand-Prince step. Returns ``(y_new, error_estimate)``."""
    k = np.empty((7,) + y.shape, dtype=float)
    k[0] = f(t, y)
    for i in range(1, 7):
        incr = sum(a * k[j] for j, a in enumerate(_DP_A[i]))
        k[i] = f(t + _DP_C[i] * h, y + h * incr)
    y_new = y + h * np.tensordot(_DP_B5, k, axes=1)
    err = h * np.tensordot(_DP_E, k, axes=1)
    return y_new, err


# --------------------------------------------------------------------------- #
# Integrators on a grid
# --------------------------------------------------------------------------- #
_FIXED_STEPPERS = {"euler": euler_step, "rk4": rk4_step}


def _integrate_fixed(f, y0, t_eval, stepper, substeps):
    y = np.asarray(y0, dtype=float)
    out = np.empty((len(t_eval),) + y.shape)
    out[0] = y
    for i in range(1, len(t_eval)):
        t0, t1 = t_eval[i - 1], t_eval[i]
        h = (t1 - t0) / substeps
        t = t0
        for _ in range(substeps):
            y = stepper(f, t, y, h)
            t += h
        out[i] = y
    return out


def _integrate_adaptive(f, y0, t_eval, rtol, atol, max_steps, h0):
    y = np.asarray(y0, dtype=float)
    out = np.empty((len(t_eval),) + y.shape)
    out[0] = y
    h = h0 if h0 is not None else (t_eval[1] - t_eval[0]) / 4.0
    n_steps = 0
    for i in range(1, len(t_eval)):
        t, t_end = t_eval[i - 1], t_eval[i]
        while t < t_end:
            h = min(h, t_end - t)  # land exactly on the grid point
            y_new, err = dopri5_step(f, t, y, h)
            scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
            err_norm = float(np.sqrt(np.mean((err / scale) ** 2)))
            if err_norm <= 1.0:  # accept
                t += h
                y = y_new
            factor = 0.9 * err_norm ** (-0.2) if err_norm > 0 else 10.0
            h *= min(10.0, max(0.2, factor))
            n_steps += 1
            if n_steps > max_steps:
                raise RuntimeError("rk45: maximum number of steps exceeded")
        out[i] = y
    return out


def integrate(
    f: RHS,
    y0,
    t_eval,
    method: str = "rk4",
    substeps: int = 1,
    rtol: float = 1e-9,
    atol: float = 1e-11,
    max_steps: int = 5_000_000,
    h0: float | None = None,
) -> np.ndarray:
    """Integrate ``dy/dt = f(t, y)`` and return the solution on ``t_eval``.

    Parameters
    ----------
    f : callable ``f(t, y) -> dy/dt`` (NumPy array in, NumPy array out).
    y0 : initial state at ``t_eval[0]``.
    t_eval : strictly increasing 1-D array of output times.
    method : ``"euler"``, ``"rk4"`` or ``"rk45"``.
    substeps : number of internal fixed steps per output interval
        (``euler`` / ``rk4`` only).
    rtol, atol, max_steps, h0 : adaptive-step controls (``rk45`` only).

    Returns
    -------
    ndarray of shape ``(len(t_eval), len(y0))``.
    """
    t_eval = np.asarray(t_eval, dtype=float)
    if t_eval.ndim != 1 or len(t_eval) < 2:
        raise ValueError("t_eval must be a 1-D array with at least two entries")
    if np.any(np.diff(t_eval) <= 0):
        raise ValueError("t_eval must be strictly increasing")
    if method in _FIXED_STEPPERS:
        if substeps < 1:
            raise ValueError("substeps must be >= 1")
        return _integrate_fixed(f, y0, t_eval, _FIXED_STEPPERS[method], substeps)
    if method == "rk45":
        return _integrate_adaptive(f, y0, t_eval, rtol, atol, max_steps, h0)
    raise ValueError(f"unknown method {method!r}; use 'euler', 'rk4' or 'rk45'")
