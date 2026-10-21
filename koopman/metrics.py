"""Error metrics for comparing predicted and true trajectories."""

from __future__ import annotations

import numpy as np


def rmse(true: np.ndarray, pred: np.ndarray) -> float:
    """Root-mean-square error over every entry."""
    true, pred = np.asarray(true, float), np.asarray(pred, float)
    if true.shape != pred.shape:
        raise ValueError("true and pred must have the same shape")
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def state_scale(reference: np.ndarray) -> float:
    """RMS distance of ``reference`` states (``(..., d)``) from their mean.

    Used to normalise errors so that they are comparable across systems:
    a normalised error of ``1`` means "as wrong as predicting the mean state".
    """
    ref = np.asarray(reference, float).reshape(-1, np.shape(reference)[-1])
    return float(np.sqrt(np.mean(np.sum((ref - ref.mean(axis=0)) ** 2, axis=1))))


def error_curve(true: np.ndarray, pred: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Euclidean error at every time step, divided by ``scale``.

    Inputs have shape ``(..., n_steps, d)``; the output has shape ``(..., n_steps)``.
    """
    true, pred = np.asarray(true, float), np.asarray(pred, float)
    if true.shape != pred.shape:
        raise ValueError("true and pred must have the same shape")
    return np.linalg.norm(true - pred, axis=-1) / scale


def valid_prediction_time(curve: np.ndarray, t: np.ndarray, threshold: float = 0.2):
    """First time at which a normalised error curve exceeds ``threshold``.

    Returns ``(time, censored)``. If the error never exceeds the threshold the
    last time is returned with ``censored = True`` (the true value is *at least*
    that large).
    """
    curve, t = np.asarray(curve, float), np.asarray(t, float)
    if curve.shape != t.shape:
        raise ValueError("curve and t must have the same shape")
    above = np.nonzero(curve > threshold)[0]
    if len(above) == 0:
        return float(t[-1]), True
    return float(t[above[0]]), False
