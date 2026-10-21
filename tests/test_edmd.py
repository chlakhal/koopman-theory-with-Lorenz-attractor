import numpy as np
import pytest

from koopman.dmd import DMD
from koopman.edmd import EDMD
from koopman.observables import PendulumDictionary, PolynomialDictionary, lorenz_dictionary
from koopman.utils import snapshot_pairs

MU, LAM, C = 0.9, 0.5, 1.0


def closed_map(x):
    """A nonlinear map with an exactly invariant 3-D observable subspace {x1, x2, x1^2}."""
    return np.array([MU * x[0], LAM * x[1] + C * x[0] ** 2])


@pytest.fixture
def closed_data():
    rng = np.random.default_rng(0)
    trajs = []
    for _ in range(8):
        x = [rng.uniform(-1.5, 1.5, size=2)]
        for _ in range(25):
            x.append(closed_map(x[-1]))
        trajs.append(np.array(x))
    return np.array(trajs)


# ----------------------------------------------------------------------------- #
# EDMD on a system with an exactly invariant subspace
# ----------------------------------------------------------------------------- #
def test_edmd_is_exact_when_the_dictionary_spans_an_invariant_subspace(closed_data):
    X, Y = snapshot_pairs(closed_data)
    edmd = EDMD(PolynomialDictionary(2, degree=2), reg=0.0).fit(X, Y)
    assert np.max(np.abs(edmd.predict_one_step(X) - Y)) < 1e-9
    x0 = np.array([1.0, -0.5])
    truth = [x0]
    for _ in range(30):
        truth.append(closed_map(truth[-1]))
    assert np.allclose(edmd.predict(x0, 30), np.array(truth), atol=1e-9)


def test_linear_dmd_cannot_capture_the_same_system(closed_data):
    X, Y = snapshot_pairs(closed_data)
    x0 = np.array([1.0, -0.5])
    truth = [x0]
    for _ in range(30):
        truth.append(closed_map(truth[-1]))
    dmd_err = np.max(np.abs(DMD().fit(X, Y).predict(x0, 30) - np.array(truth)))
    assert dmd_err > 1e-3


def test_edmd_spectrum_contains_the_koopman_eigenvalues(closed_data):
    X, Y = snapshot_pairs(closed_data)
    ev = EDMD(PolynomialDictionary(2, degree=2), reg=0.0).fit(X, Y).eigenvalues
    for expected in (MU, LAM, MU**2):  # eigenfunctions x1, x2 - (...)x1^2, x1^2
        assert np.min(np.abs(ev - expected)) < 1e-8


def test_feature_scaling_does_not_change_the_predictions(closed_data):
    X, Y = snapshot_pairs(closed_data)
    x0 = np.array([0.7, 0.2])
    a = EDMD(PolynomialDictionary(2, 2), reg=0.0, scale=True).fit(X, Y).predict(x0, 15)
    b = EDMD(PolynomialDictionary(2, 2), reg=0.0, scale=False).fit(X, Y).predict(x0, 15)
    assert np.allclose(a, b, atol=1e-7)


def test_first_prediction_row_is_the_initial_state(closed_data):
    X, Y = snapshot_pairs(closed_data)
    x0 = np.array([0.3, 0.9])
    pred = EDMD(PolynomialDictionary(2, 2)).fit(X, Y).predict(x0, 5)
    assert pred.shape == (6, 2)
    assert np.allclose(pred[0], x0, atol=1e-6)


def test_ridge_shrinks_the_operator(closed_data):
    X, Y = snapshot_pairs(closed_data)
    K0 = EDMD(PolynomialDictionary(2, 2), reg=0.0).fit(X, Y).operator
    K1 = EDMD(PolynomialDictionary(2, 2), reg=1.0).fit(X, Y).operator
    assert np.linalg.norm(K1) < np.linalg.norm(K0)


def test_constant_feature_does_not_break_scaling(closed_data):
    X, Y = snapshot_pairs(closed_data)
    d = PolynomialDictionary(2, degree=2, include_constant=True)
    edmd = EDMD(d, reg=1e-10).fit(X, Y)
    assert np.all(np.isfinite(edmd.operator))
    assert np.max(np.abs(edmd.predict_one_step(X) - Y)) < 1e-6


def test_unfitted_and_bad_shapes_raise():
    with pytest.raises(RuntimeError):
        EDMD(PolynomialDictionary(2)).predict(np.zeros(2), 3)
    with pytest.raises(ValueError):
        EDMD(PolynomialDictionary(2)).fit(np.zeros((5, 2)), np.zeros((4, 2)))


# ----------------------------------------------------------------------------- #
# Dictionaries
# ----------------------------------------------------------------------------- #
def test_polynomial_feature_counts_and_state_first():
    assert PolynomialDictionary(3, degree=1).n_features == 3
    assert PolynomialDictionary(3, degree=2).n_features == 9
    assert PolynomialDictionary(3, degree=3).n_features == 19
    assert PolynomialDictionary(3, degree=2, include_constant=True).n_features == 10
    X = np.random.default_rng(0).normal(size=(7, 3))
    psi = PolynomialDictionary(3, degree=3)(X)
    assert psi.shape == (7, 19)
    assert np.allclose(psi[:, :3], X)


def test_polynomial_values_and_names():
    d = lorenz_dictionary(2)
    assert d.names[:3] == ["x", "y", "z"]
    assert "x*y" in d.names and "z*z" in d.names
    X = np.array([[2.0, 3.0, 5.0]])
    psi = dict(zip(d.names, d(X)[0]))
    assert psi["x*y"] == 6.0 and psi["y*z"] == 15.0 and psi["z*z"] == 25.0


def test_polynomial_constant_is_last_and_single_state_input_works():
    d = PolynomialDictionary(2, degree=2, include_constant=True)
    out = d(np.array([2.0, 3.0]))
    assert out.shape == (1, 6)
    assert out[0, -1] == 1.0 and d.names[-1] == "1"


def test_polynomial_validation():
    with pytest.raises(ValueError):
        PolynomialDictionary(2, degree=0)
    with pytest.raises(ValueError):
        PolynomialDictionary(2, state_names=("a",))


@pytest.mark.parametrize("K, P", [(1, 1), (2, 2), (3, 5)])
def test_pendulum_dictionary_size_and_state_first(K, P):
    d = PendulumDictionary(K, P)
    X = np.random.default_rng(1).normal(size=(9, 2))
    psi = d(X)
    assert psi.shape == (9, 3 + 2 * K * (P + 1) + (P - 1))
    assert d.n_features == psi.shape[1] == len(d.names)
    assert np.allclose(psi[:, :2], X)
    assert np.allclose(psi[:, 2], 1.0)


def test_pendulum_dictionary_values():
    d = PendulumDictionary(2, 2)
    theta, omega = 0.7, -1.3
    psi = dict(zip(d.names, d(np.array([[theta, omega]]))[0]))
    assert psi["sin(1theta)"] == pytest.approx(np.sin(theta))
    assert psi["cos(2theta)"] == pytest.approx(np.cos(2 * theta))
    assert psi["omega sin(1theta)"] == pytest.approx(omega * np.sin(theta))
    assert psi["omega^2 cos(2theta)"] == pytest.approx(omega**2 * np.cos(2 * theta))
    assert psi["omega^2"] == pytest.approx(omega**2)


def test_pendulum_dictionary_is_2pi_periodic_in_the_trigonometric_features():
    d = PendulumDictionary(2, 2)
    a = d(np.array([[0.4, 1.0]]))
    b = d(np.array([[0.4 + 2 * np.pi, 1.0]]))
    assert np.allclose(a[:, 2:], b[:, 2:])  # everything except the raw angle itself


def test_pendulum_dictionary_validation():
    with pytest.raises(ValueError):
        PendulumDictionary(0, 2)
    with pytest.raises(ValueError):
        PendulumDictionary(2, 0)
