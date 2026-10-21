import numpy as np
import pytest

from koopman.dmd import DMD
from koopman.utils import as_trajectory_array, snapshot_pairs


def test_recovers_a_linear_operator_exactly(rotation_map):
    A, x, _ = rotation_map
    X, Y = x[:-1], x[1:]
    dmd = DMD().fit(X, Y)
    assert np.allclose(dmd.operator, A, atol=1e-10)
    assert dmd.rank_ == 2


def test_eigenvalues_are_0p95_exp_pm_i_theta(rotation_map):
    _, x, theta = rotation_map
    dmd = DMD().fit(x[:-1], x[1:])
    expected = 0.95 * np.exp(1j * theta * np.array([1, -1]))
    assert np.allclose(sorted(dmd.eigenvalues, key=np.imag), sorted(expected, key=np.imag))


def test_continuous_time_spectrum_and_frequency(rotation_map):
    _, x, theta = rotation_map
    dt = 0.1
    dmd = DMD(dt=dt).fit(x[:-1], x[1:])
    omega = dmd.continuous_eigenvalues
    assert np.allclose(sorted(omega.real), [np.log(0.95) / dt] * 2)
    assert np.allclose(sorted(np.abs(omega.imag)), [theta / dt] * 2)
    assert np.allclose(np.abs(dmd.frequencies), theta / dt / (2 * np.pi))


def test_predict_reproduces_the_trajectory(rotation_map):
    _, x, _ = rotation_map
    dmd = DMD().fit(x[:-1], x[1:])
    pred = dmd.predict(x[0], len(x) - 1)
    assert pred.shape == x.shape
    assert np.allclose(pred, x, atol=1e-9)


def test_exact_modes_are_eigenvectors_of_the_operator(rotation_map):
    _, x, _ = rotation_map
    dmd = DMD().fit(x[:-1], x[1:])
    lhs = dmd.operator @ dmd.modes
    rhs = dmd.modes * dmd.eigenvalues[None, :]
    assert np.allclose(lhs, rhs, atol=1e-10)


def test_amplitudes_reconstruct_the_initial_state(rotation_map):
    _, x, _ = rotation_map
    dmd = DMD().fit(x[:-1], x[1:])
    b = dmd.amplitudes(x[0])
    assert np.allclose((dmd.modes @ b).real, x[0], atol=1e-10)
    assert dmd.mode_energies(x[0]).shape == (2,)


def test_numerical_rank_is_detected_for_low_dimensional_data():
    rng = np.random.default_rng(0)
    basis = np.linalg.qr(rng.normal(size=(5, 2)))[0]  # data lives in a 2-D subspace of R^5
    Ar = 0.9 * np.array([[np.cos(0.4), -np.sin(0.4)], [np.sin(0.4), np.cos(0.4)]])
    z = [np.array([1.0, 1.0])]
    for _ in range(40):
        z.append(Ar @ z[-1])
    x = np.array(z) @ basis.T
    dmd = DMD().fit(x[:-1], x[1:])
    assert dmd.rank_ == 2
    assert np.allclose(dmd.predict(x[0], 40), x, atol=1e-8)


def test_rank_and_energy_truncation():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 6)) * np.array([10, 5, 1, 0.1, 0.01, 0.001])
    Y = X @ rng.normal(size=(6, 6)) * 0.1
    assert DMD(rank=3).fit(X, Y).rank_ == 3
    assert DMD(rank=100).fit(X, Y).rank_ == 6  # clipped to what exists
    assert DMD(energy=0.99).fit(X, Y).rank_ == 2
    assert DMD(energy=1.0).fit(X, Y).rank_ == 6


def test_truncated_dmd_is_lower_rank_than_full():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(100, 4)) * np.array([5, 2, 0.5, 0.1])
    Y = X @ rng.normal(size=(4, 4)) * 0.3
    full = DMD().fit(X, Y).operator
    low = DMD(rank=2).fit(X, Y).operator
    assert np.linalg.matrix_rank(low) == 2
    assert np.linalg.matrix_rank(full) == 4


def test_predict_one_step_matches_operator(rotation_map):
    A, x, _ = rotation_map
    dmd = DMD().fit(x[:-1], x[1:])
    assert np.allclose(dmd.predict_one_step(x[:-1]), x[1:], atol=1e-10)


def test_nonlinear_data_is_not_fitted_exactly():
    # x' = x - 0.1 x^3 integrated naively: a purely linear model must leave a residual.
    x = np.linspace(-2, 2, 200)[:, None]
    y = x - 0.1 * x**3
    dmd = DMD().fit(x, y)
    assert np.max(np.abs(dmd.predict_one_step(x) - y)) > 0.1


@pytest.mark.parametrize("kwargs", [dict(rank=0), dict(energy=0.0), dict(energy=1.5), dict(rank=2, energy=0.9)])
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        DMD(**kwargs)


def test_shape_mismatch_and_unfitted_use_raise():
    with pytest.raises(ValueError):
        DMD().fit(np.zeros((5, 2)), np.zeros((4, 2)))
    with pytest.raises(RuntimeError):
        DMD().predict(np.zeros(2), 3)
    with pytest.raises(RuntimeError):
        _ = DMD().operator


def test_snapshot_pairs_never_straddle_trajectories():
    a = np.arange(5)[:, None] + 0.0
    b = np.arange(5)[:, None] + 100.0
    X, Y = snapshot_pairs(np.stack([a, b]))
    assert X.shape == (8, 1) and Y.shape == (8, 1)
    assert np.all(Y - X == 1.0)  # a 4 -> 100 jump would give 96


def test_as_trajectory_array_accepts_single_and_list():
    one = np.zeros((10, 3))
    assert as_trajectory_array(one).shape == (1, 10, 3)
    assert as_trajectory_array([one, one]).shape == (2, 10, 3)
    with pytest.raises(ValueError):
        as_trajectory_array(np.zeros(5))
