"""Deep Koopman autoencoder in PyTorch.

An encoder ``phi: R^d -> R^L`` learns coordinates in which the dynamics are
*linear*, ``phi(x_{k+1}) ~ K phi(x_k)``, and a decoder ``psi: R^L -> R^d`` maps
back to the state (Lusch, Kutz & Brunton, 2018). Training combines three terms
over windows ``x_k, ..., x_{k+M}``:

* reconstruction  ``|| x_k - psi(phi(x_k)) ||^2``;
* latent linearity ``|| phi(x_{k+m}) - K^m phi(x_k) ||^2``,  ``m = 1..M``;
* multi-step prediction ``|| x_{k+m} - psi(K^m phi(x_k)) ||^2``,  ``m = 1..M``.

The prediction term is what makes a *rollout* from a single initial state
meaningful: a model trained on reconstruction alone can reproduce every point of
a trajectory without having learned any dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from .utils import as_trajectory_array

_ACTIVATIONS = {"tanh": nn.Tanh, "silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU}


def _mlp(sizes, activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(_ACTIVATIONS[activation]())
    return nn.Sequential(*layers)


class DeepKoopman(nn.Module):
    """Autoencoder with a linear (bias-free) Koopman matrix in latent space.

    Inputs and outputs are in physical units: the model stores the training mean
    and standard deviation of the state and normalises internally.
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 8,
        hidden: tuple[int, ...] = (64, 64),
        activation: str = "tanh",
        seed: int | None = 0,
    ):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(_ACTIVATIONS)}")
        if seed is not None:
            torch.manual_seed(seed)
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.encoder = _mlp([state_dim, *hidden, latent_dim], activation)
        self.decoder = _mlp([latent_dim, *hidden, state_dim], activation)
        K = torch.empty(latent_dim, latent_dim)
        nn.init.orthogonal_(K)
        self.K = nn.Parameter(K)  # latent update: z_next = z @ K.T
        self.register_buffer("x_mean", torch.zeros(state_dim))
        self.register_buffer("x_std", torch.ones(state_dim))

    # ------------------------------------------------------------------ #
    # normalisation
    # ------------------------------------------------------------------ #
    def set_normalization(self, trajectories) -> None:
        """Store the mean / std of the training states."""
        X = as_trajectory_array(trajectories).reshape(-1, self.state_dim)
        self.x_mean.copy_(torch.as_tensor(X.mean(axis=0), dtype=torch.float32))
        self.x_std.copy_(torch.as_tensor(X.std(axis=0) + 1e-8, dtype=torch.float32))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.x_mean) / self.x_std

    def denormalize(self, xn: torch.Tensor) -> torch.Tensor:
        return xn * self.x_std + self.x_mean

    # ------------------------------------------------------------------ #
    # network pieces (physical units in / out)
    # ------------------------------------------------------------------ #
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.normalize(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.denormalize(self.decoder(z))

    def step(self, z: torch.Tensor) -> torch.Tensor:
        """One latent step ``z -> K z`` (rows are samples)."""
        return z @ self.K.T

    def forward(self, x: torch.Tensor):
        """Returns ``(x_reconstructed, z, z_next_predicted)``."""
        z = self.encode(x)
        return self.decode(z), z, self.step(z)

    # ------------------------------------------------------------------ #
    # numpy conveniences
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self, x0: np.ndarray, n_steps: int) -> np.ndarray:
        """Roll out ``n_steps`` latent steps from a single state ``x0``.

        Returns ``(n_steps + 1, d)``; row 0 is ``decode(encode(x0))``.
        """
        z = self.encode(torch.as_tensor(np.asarray(x0), dtype=torch.float32).reshape(1, -1))
        zs = [z]
        for _ in range(n_steps):
            z = self.step(z)
            zs.append(z)
        out = self.decode(torch.cat(zs, dim=0))
        return out.numpy().astype(float)

    @torch.no_grad()
    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Autoencode states ``(n, d)`` (no dynamics involved)."""
        x = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        return self.decode(self.encode(x)).numpy().astype(float)

    @torch.no_grad()
    def predict_one_step(self, X: np.ndarray) -> np.ndarray:
        """Encode, advance once with ``K``, decode; rows are samples."""
        x = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        return self.decode(self.step(self.encode(x))).numpy().astype(float)

    @torch.no_grad()
    def latent(self, X: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        return self.encode(x).numpy().astype(float)

    @property
    def eigenvalues(self) -> np.ndarray:
        """Eigenvalues of the learned Koopman matrix."""
        return np.linalg.eigvals(self.K.detach().numpy().astype(float))

    @torch.no_grad()
    def stabilize(self, max_radius: float = 1.0) -> float:
        """Rescale eigenvalues of ``K`` with ``|lambda| > max_radius`` onto that circle.

        A learned ``K`` can have a slightly unstable eigenvalue (e.g. 1.013) that is
        harmless over the short windows seen in training but explodes in a long
        rollout. This post-hoc fix keeps the eigenvectors and all eigenvalues that are
        already inside the disc, and only shrinks the offending moduli. Returns the
        spectral radius *before* the correction (``K`` is left untouched if it was
        already within ``max_radius``).
        """
        K = self.K.detach().double().numpy()
        w, V = np.linalg.eig(K)
        radius = float(np.abs(w).max())
        if radius <= max_radius:
            return radius
        w_new = w / np.maximum(1.0, np.abs(w) / max_radius)
        K_new = np.linalg.solve(V.T, (V * w_new[None, :]).T).T.real
        self.K.copy_(torch.as_tensor(K_new, dtype=self.K.dtype))
        return radius


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    n_steps: int = 8  # prediction horizon M inside the loss
    iterations: int = 3000
    batch_size: int = 256
    lr: float = 2e-3
    lr_final: float = 1e-4
    w_rec: float = 1.0
    w_lin: float = 1.0
    w_pred: float = 1.0
    grad_clip: float = 5.0
    log_every: int = 100
    seed: int = 0
    verbose: bool = False


def make_windows(trajectories, n_steps: int) -> torch.Tensor:
    """Sliding windows of ``n_steps + 1`` consecutive states: ``(N, M + 1, d)``.

    Windows never straddle two different trajectories.
    """
    arr = as_trajectory_array(trajectories)
    n_traj, T, d = arr.shape
    if T < n_steps + 1:
        raise ValueError("trajectories are shorter than the prediction window")
    idx = np.arange(T - n_steps)[:, None] + np.arange(n_steps + 1)[None, :]
    windows = arr[:, idx, :].reshape(-1, n_steps + 1, d)
    return torch.as_tensor(windows, dtype=torch.float32)


def koopman_losses(model: DeepKoopman, windows: torch.Tensor):
    """Reconstruction, latent-linearity and prediction losses on a batch.

    All terms are mean-squared errors in *normalised* coordinates, so that state
    components with different physical scales contribute equally.
    """
    xn = model.normalize(windows)  # (B, M+1, d)
    z = model.encoder(xn)  # (B, M+1, L)
    rec = torch.mean((model.decoder(z) - xn) ** 2)

    M = windows.shape[1] - 1
    zk = z[:, 0]
    lin = 0.0
    pred = 0.0
    for m in range(1, M + 1):
        zk = model.step(zk)
        lin = lin + torch.mean((zk - z[:, m]) ** 2)
        pred = pred + torch.mean((model.decoder(zk) - xn[:, m]) ** 2)
    return rec, lin / M, pred / M


def train_deep_koopman(
    model: DeepKoopman,
    trajectories,
    config: TrainConfig | None = None,
    val_trajectories=None,
) -> dict[str, list[float]]:
    """Train ``model`` in place and return the loss history.

    ``history`` has the keys ``iteration, loss, rec, lin, pred`` and, when
    ``val_trajectories`` is given, ``val_pred`` (validation multi-step prediction
    loss) -- all logged every ``config.log_every`` iterations.
    """
    cfg = config or TrainConfig()
    torch.manual_seed(cfg.seed)
    model.set_normalization(trajectories)
    windows = make_windows(trajectories, cfg.n_steps)
    val_windows = (
        make_windows(val_trajectories, cfg.n_steps) if val_trajectories is not None else None
    )

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    gamma = (cfg.lr_final / cfg.lr) ** (1.0 / max(cfg.iterations, 1))
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
    gen = torch.Generator().manual_seed(cfg.seed)

    history: dict[str, list[float]] = {k: [] for k in ("iteration", "loss", "rec", "lin", "pred")}
    if val_windows is not None:
        history["val_pred"] = []

    iterator = range(cfg.iterations)
    if cfg.verbose:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="DeepKoopman", leave=False)
        except ImportError:  # pragma: no cover
            pass

    for it in iterator:
        idx = torch.randint(0, len(windows), (min(cfg.batch_size, len(windows)),), generator=gen)
        rec, lin, pred = koopman_losses(model, windows[idx])
        loss = cfg.w_rec * rec + cfg.w_lin * lin + cfg.w_pred * pred
        opt.zero_grad()
        loss.backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()

        if it % cfg.log_every == 0 or it == cfg.iterations - 1:
            history["iteration"].append(it)
            history["loss"].append(loss.item())
            history["rec"].append(rec.item())
            history["lin"].append(lin.item())
            history["pred"].append(pred.item())
            if val_windows is not None:
                with torch.no_grad():
                    _, _, vp = koopman_losses(model, val_windows)
                history["val_pred"].append(vp.item())
    return history
