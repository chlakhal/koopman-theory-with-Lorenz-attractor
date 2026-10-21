import numpy as np
import pytest
import torch

from koopman.deep_koopman import (
    DeepKoopman,
    TrainConfig,
    koopman_losses,
    make_windows,
    train_deep_koopman,
)


def spiral_data(n_traj=8, steps=60, seed=0):
    """Damped linear oscillator trajectories: a system with an exact linear latent model."""
    rng = np.random.default_rng(seed)
    A = 0.97 * np.array([[np.cos(0.25), -np.sin(0.25)], [np.sin(0.25), np.cos(0.25)]])
    trajs = []
    for _ in range(n_traj):
        x = [rng.normal(size=2)]
        for _ in range(steps):
            x.append(A @ x[-1])
        trajs.append(np.array(x))
    return np.array(trajs)


def test_make_windows_shapes_and_no_cross_trajectory_windows():
    # value = 1000 * trajectory_id + step, so any window straddling two trajectories is detectable
    T = 12
    arr = np.stack([np.stack([1000.0 * i + np.arange(T)] * 2, axis=1) for i in range(3)])
    w = make_windows(arr, n_steps=4)
    assert w.shape == (3 * (T - 4), 5, 2)
    ids = torch.round(w[:, :, 0] / 1000.0)
    assert torch.all(ids == ids[:, :1])  # one trajectory per window
    assert torch.all(w[:, 1:, 0] - w[:, :-1, 0] == 1.0)  # consecutive steps


def test_make_windows_rejects_short_trajectories():
    with pytest.raises(ValueError):
        make_windows(np.zeros((2, 3, 2)), n_steps=5)


def test_forward_shapes_and_orthogonal_koopman_initialisation():
    m = DeepKoopman(3, latent_dim=6, hidden=(16,), seed=0)
    x = torch.randn(10, 3)
    rec, z, z_next = m(x)
    assert rec.shape == (10, 3) and z.shape == (10, 6) and z_next.shape == (10, 6)
    K = m.K.detach()
    assert torch.allclose(K @ K.T, torch.eye(6), atol=1e-5)
    assert len(m.eigenvalues) == 6


def test_same_seed_gives_identical_initialisation_and_different_seed_does_not():
    a, b, c = (DeepKoopman(2, 4, seed=s) for s in (5, 5, 6))
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
    assert not torch.equal(a.K, c.K)


def test_normalisation_statistics_and_roundtrip():
    data = spiral_data() * np.array([5.0, 0.5]) + np.array([2.0, -1.0])
    m = DeepKoopman(2, 4, seed=0)
    m.set_normalization(data)
    xn = m.normalize(torch.as_tensor(data.reshape(-1, 2), dtype=torch.float32))
    assert torch.allclose(xn.mean(0), torch.zeros(2), atol=1e-4)
    assert torch.allclose(xn.std(0), torch.ones(2), atol=1e-2)
    assert torch.allclose(m.denormalize(xn), torch.as_tensor(data.reshape(-1, 2), dtype=torch.float32), atol=1e-4)


def test_losses_are_nonnegative_scalars():
    m = DeepKoopman(2, 4, hidden=(8,), seed=0)
    w = make_windows(spiral_data(2, 20), 3)
    rec, lin, pred = koopman_losses(m, w)
    for v in (rec, lin, pred):
        assert v.ndim == 0 and v.item() >= 0.0


def test_training_is_reproducible_and_reduces_the_loss():
    data = spiral_data()

    def fit():
        m = DeepKoopman(2, latent_dim=4, hidden=(16,), seed=1)
        cfg = TrainConfig(n_steps=4, iterations=400, batch_size=64, lr=1e-2, log_every=50, seed=1)
        return m, train_deep_koopman(m, data[:6], cfg, val_trajectories=data[6:])

    m1, h1 = fit()
    m2, h2 = fit()
    assert h1["loss"] == h2["loss"]
    assert torch.equal(m1.K, m2.K)
    assert h1["loss"][-1] < 0.5 * h1["loss"][0]
    assert set(h1) == {"iteration", "loss", "rec", "lin", "pred", "val_pred"}
    assert len(h1["iteration"]) == len(h1["val_pred"]) == len(h1["loss"])
    assert h1["iteration"][0] == 0 and h1["iteration"][-1] == 399


def test_learns_to_predict_a_linear_system():
    # Needs enough trajectories: with only a handful the encoder memorises the few curves
    # instead of learning the global linear map (large train/validation gap).
    data = spiral_data(n_traj=46, steps=40)
    train, held_out = data[:40], data[40:]
    m = DeepKoopman(2, latent_dim=4, hidden=(32,), seed=0)
    cfg = TrainConfig(n_steps=6, iterations=500, batch_size=128, lr=1e-2, log_every=100, seed=0)
    train_deep_koopman(m, train, cfg)
    scale = np.sqrt(np.mean(np.sum(train.reshape(-1, 2) ** 2, axis=1)))
    errs = [np.linalg.norm(m.predict(x[0], 20) - x[:21], axis=1) / scale for x in held_out]
    assert np.mean(errs) < 0.1  # normalised 20-step rollout error on unseen initial states (~0.02 observed)


def test_numpy_interfaces():
    m = DeepKoopman(3, latent_dim=5, hidden=(8,), seed=0)
    m.set_normalization(np.random.default_rng(0).normal(size=(4, 20, 3)))
    x0 = np.array([0.1, 0.2, 0.3])
    pred = m.predict(x0, 7)
    assert isinstance(pred, np.ndarray) and pred.shape == (8, 3)
    X = np.random.default_rng(1).normal(size=(11, 3))
    assert m.reconstruct(X).shape == (11, 3)
    assert m.predict_one_step(X).shape == (11, 3)
    assert m.latent(X).shape == (11, 5)
    # the first row of a rollout is the reconstruction of x0
    assert np.allclose(pred[0], m.reconstruct(x0[None])[0], atol=1e-6)
    # one rollout step equals encode -> K -> decode
    assert np.allclose(pred[1], m.predict_one_step(x0[None])[0], atol=1e-5)


def test_stabilize_shrinks_only_the_unstable_eigenvalues():
    m = DeepKoopman(2, latent_dim=4, hidden=(8,), seed=0)
    # real spectrum with one unstable and one marginal-ish mode, in a non-trivial basis
    rng = np.random.default_rng(0)
    V = rng.normal(size=(4, 4)) + 3 * np.eye(4)
    K = V @ np.diag([1.2, 0.9, -0.5, 0.3]) @ np.linalg.inv(V)
    m.K.data = torch.as_tensor(K, dtype=torch.float32)
    before = m.stabilize()
    assert before == pytest.approx(1.2, rel=1e-4)
    ev = np.sort(np.real(m.eigenvalues))
    assert np.allclose(ev, np.sort([1.0, 0.9, -0.5, 0.3]), atol=1e-3)
    assert np.max(np.abs(m.eigenvalues)) <= 1.0 + 1e-4


def test_stabilize_handles_complex_pairs_and_leaves_stable_models_untouched():
    m = DeepKoopman(2, latent_dim=2, hidden=(8,), seed=0)
    a, b = 1.1 * np.cos(0.4), 1.1 * np.sin(0.4)
    m.K.data = torch.tensor([[a, -b], [b, a]], dtype=torch.float32)  # radius 1.1
    m.stabilize(1.0)
    assert np.allclose(np.abs(m.eigenvalues), 1.0, atol=1e-4)
    assert np.allclose(np.abs(np.angle(m.eigenvalues)), 0.4, atol=1e-4)  # the rotation angle is preserved
    stable = DeepKoopman(2, latent_dim=3, hidden=(8,), seed=0)
    stable.K.data = 0.5 * torch.eye(3)
    snapshot = stable.K.detach().clone()
    assert stable.stabilize() == pytest.approx(0.5)
    assert torch.equal(stable.K, snapshot)


def test_invalid_activation_raises():
    with pytest.raises(ValueError):
        DeepKoopman(2, activation="nope")
