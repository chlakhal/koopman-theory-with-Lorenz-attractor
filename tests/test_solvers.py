import numpy as np
import pytest
from scipy.integrate import solve_ivp

from koopman.solvers import dopri5_step, integrate, rk4_step
from koopman.systems import Lorenz, Pendulum


def decay(t, y):
    return -y


def final_error(method, n_intervals):
    t = np.linspace(0.0, 1.0, n_intervals + 1)
    y = integrate(decay, [1.0], t, method=method)
    return abs(y[-1, 0] - np.exp(-1.0))


def test_euler_is_first_order():
    ratio = final_error("euler", 50) / final_error("euler", 100)
    assert 1.8 < ratio < 2.2


def test_rk4_is_fourth_order():
    ratio = final_error("rk4", 10) / final_error("rk4", 20)
    assert 14 < ratio < 18


def test_rk45_meets_requested_tolerance():
    t = np.linspace(0, 5, 11)
    y = integrate(decay, [1.0], t, method="rk45", rtol=1e-10, atol=1e-12)
    assert np.max(np.abs(y[:, 0] - np.exp(-t))) < 1e-9


def test_rk45_lands_exactly_on_grid_and_returns_all_points():
    t = np.array([0.0, 0.013, 0.5, 0.5001, 2.0])
    y = integrate(decay, [2.0], t, method="rk45")
    assert y.shape == (5, 1)
    assert np.allclose(y[:, 0], 2.0 * np.exp(-t), atol=1e-8)


def test_rk4_substeps_improve_accuracy():
    t = np.linspace(0, 1, 6)
    coarse = integrate(decay, [1.0], t, method="rk4", substeps=1)
    fine = integrate(decay, [1.0], t, method="rk4", substeps=8)
    assert abs(fine[-1, 0] - np.exp(-1)) < abs(coarse[-1, 0] - np.exp(-1))


def test_matches_scipy_on_lorenz():
    lz = Lorenz()
    t = np.linspace(0, 2, 101)
    x0 = [1.0, 1.0, 1.0]
    ours = integrate(lz.rhs, x0, t, method="rk45", rtol=1e-10, atol=1e-12)
    ref = solve_ivp(lz.rhs, (0, 2), x0, t_eval=t, method="DOP853", rtol=1e-12, atol=1e-13).y.T
    assert np.max(np.abs(ours - ref)) < 1e-6


def test_rk4_conserves_pendulum_energy():
    p = Pendulum(damping=0.0)
    t = np.arange(0, 10, 0.05)
    x = integrate(p.rhs, [1.2, 0.0], t, method="rk4", substeps=10)
    e = p.energy(x)
    assert np.max(np.abs(e - e[0])) / e[0] < 1e-6


def test_single_steps_are_consistent_with_the_integrator():
    h = 0.1
    y1 = rk4_step(decay, 0.0, np.array([1.0]), h)
    y2 = integrate(decay, [1.0], np.array([0.0, h]), method="rk4")[-1]
    assert np.allclose(y1, y2)
    y5, err = dopri5_step(decay, 0.0, np.array([1.0]), h)
    assert abs(y5[0] - np.exp(-h)) < 1e-9
    assert abs(err[0]) < 1e-7


def test_solvers_integrate_vector_states():
    # harmonic oscillator: x'' = -x
    f = lambda t, y: np.array([y[1], -y[0]])
    t = np.linspace(0, 2 * np.pi, 41)
    y = integrate(f, [1.0, 0.0], t, method="rk45")
    assert np.allclose(y[:, 0], np.cos(t), atol=1e-7)
    assert np.allclose(y[:, 1], -np.sin(t), atol=1e-7)


@pytest.mark.parametrize(
    "t, kwargs",
    [
        (np.array([0.0]), {}),
        (np.array([0.0, 1.0, 0.5]), {}),
        (np.array([0.0, 0.0, 1.0]), {}),
        (np.linspace(0, 1, 5), {"method": "bogus"}),
        (np.linspace(0, 1, 5), {"method": "rk4", "substeps": 0}),
    ],
)
def test_invalid_input_raises(t, kwargs):
    with pytest.raises(ValueError):
        integrate(decay, [1.0], t, **kwargs)


def test_rk45_step_limit_is_enforced():
    with pytest.raises(RuntimeError):
        integrate(decay, [1.0], np.linspace(0, 100, 3), method="rk45", rtol=1e-14, atol=1e-16, max_steps=5)
