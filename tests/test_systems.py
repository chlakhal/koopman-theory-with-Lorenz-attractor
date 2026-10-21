import numpy as np
import pytest

from koopman.systems import (
    Lorenz,
    Pendulum,
    generate_lorenz_long,
    generate_pendulum_dataset,
    generate_trajectory,
    sample_pendulum_initial_conditions,
)


def test_pendulum_equilibrium_and_horizontal_release():
    p = Pendulum(g=9.81, length=1.0)
    assert np.allclose(p.rhs(0, np.array([0.0, 0.0])), 0.0)
    assert np.allclose(p.rhs(0, np.array([np.pi / 2, 0.0])), [0.0, -9.81])
    assert np.allclose(p.rhs(0, np.array([np.pi, 0.0])), 0.0, atol=1e-12)


def test_pendulum_linearised_eigenvalues():
    undamped = Pendulum(damping=0.0)
    ev = undamped.linearized_eigenvalues()
    assert np.allclose(ev.real, 0.0)
    assert np.allclose(sorted(ev.imag), [-undamped.omega0, undamped.omega0])
    damped = Pendulum(damping=0.4)
    ev = damped.linearized_eigenvalues()
    assert np.allclose(ev.real, -0.2)
    # the Jacobian at the origin has exactly these eigenvalues
    J = np.array([[0, 1], [-damped.omega0_sq, -damped.damping]])
    assert np.allclose(sorted(np.linalg.eigvals(J), key=lambda z: z.imag), sorted(ev, key=lambda z: z.imag))


def test_overdamped_pendulum_has_real_eigenvalues():
    ev = Pendulum(damping=10.0).linearized_eigenvalues()
    assert np.allclose(ev.imag, 0.0)
    assert np.all(ev.real < 0)


def test_energy_is_zero_at_rest_and_fraction_one_on_separatrix():
    p = Pendulum()
    assert p.energy(np.array([0.0, 0.0])) == pytest.approx(0.0)
    assert p.energy_fraction(np.array([np.pi, 0.0])) == pytest.approx(1.0)


def test_damped_pendulum_loses_energy_monotonically():
    p = Pendulum(damping=0.3)
    _, x = generate_trajectory(p, [2.0, 0.0], 8.0, 0.02)
    e = p.energy(x)
    assert np.all(np.diff(e) < 1e-9)
    assert e[-1] < 0.2 * e[0]


def test_undamped_pendulum_conserves_energy():
    p = Pendulum(damping=0.0)
    _, x = generate_trajectory(p, [1.5, 0.0], 10.0, 0.02)
    e = p.energy(x)
    assert np.max(np.abs(e - e[0])) / e[0] < 1e-6


def test_generate_trajectory_grid_and_initial_state():
    t, x = generate_trajectory(Pendulum(), [0.5, 0.1], 3.0, 0.05)
    assert t.shape == (61,) and x.shape == (61, 2)
    assert np.allclose(np.diff(t), 0.05)
    assert np.allclose(x[0], [0.5, 0.1])


def test_initial_conditions_stay_in_libration_regime_and_are_reproducible():
    p = Pendulum()
    a = sample_pendulum_initial_conditions(p, 50, np.random.default_rng(3), max_energy_fraction=0.9)
    b = sample_pendulum_initial_conditions(p, 50, np.random.default_rng(3), max_energy_fraction=0.9)
    assert np.array_equal(a, b)
    assert a.shape == (50, 2)
    assert np.all(p.energy_fraction(a) < 0.9)
    assert np.all(np.abs(a[:, 0]) <= 2.7) and np.all(np.abs(a[:, 1]) <= 4.0)


def test_pendulum_dataset_shapes_and_seed_dependence():
    p = Pendulum(damping=0.2)
    t, d1 = generate_pendulum_dataset(p, 5, 2.0, 0.05, seed=1)
    _, d2 = generate_pendulum_dataset(p, 5, 2.0, 0.05, seed=1)
    _, d3 = generate_pendulum_dataset(p, 5, 2.0, 0.05, seed=2)
    assert d1.shape == (5, 41, 2) and t.shape == (41,)
    assert np.array_equal(d1, d2)
    assert not np.allclose(d1, d3)


def test_lorenz_fixed_points():
    lz = Lorenz()
    assert np.allclose(lz.rhs(0, np.zeros(3)), 0.0)
    c = np.sqrt(lz.beta * (lz.rho - 1))
    for s in (+1, -1):
        assert np.allclose(lz.rhs(0, np.array([s * c, s * c, lz.rho - 1])), 0.0, atol=1e-12)


def test_lorenz_trajectory_stays_on_bounded_attractor():
    t, x = generate_lorenz_long(Lorenz(), 1500, 0.02)
    assert x.shape == (1501, 3)
    assert t[0] == 0.0 and t[-1] == pytest.approx(30.0)
    assert np.all(np.abs(x[:, :2]) < 30) and np.all((x[:, 2] > 0) & (x[:, 2] < 55))
    # chaotic motion visits both lobes (x changes sign)
    assert x[:, 0].min() < -5 and x[:, 0].max() > 5


def test_lorenz_sensitive_dependence_on_initial_conditions():
    # Starting from (1, 1, 1) the perturbation grows slowly through the initial transient
    # (finite-time exponent ~0.6) and saturates at the attractor size (~10-20) near t = 50;
    # cross-checked against SciPy's DOP853 (separation ~15 at t = 50).
    lz = Lorenz()
    _, a = generate_trajectory(lz, [1.0, 1.0, 1.0], 50.0, 0.02)
    _, b = generate_trajectory(lz, [1.0 + 1e-8, 1.0, 1.0], 50.0, 0.02)
    sep = np.linalg.norm(a - b, axis=1)
    assert sep[0] == pytest.approx(1e-8)
    assert sep[-1] > 1.0  # amplified by more than eight orders of magnitude
