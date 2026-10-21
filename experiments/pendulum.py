"""Pendulum experiment: DMD vs EDMD vs Deep Koopman.

A damped nonlinear pendulum is simulated from many random initial conditions
(libration regime). All three models are fitted on the *same* training
trajectories, hyper-parameters are chosen on a separate validation set, and
every model is evaluated on held-out test initial conditions by **rolling out
from the initial state alone** (no model ever sees the true trajectory at test time).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from koopman import (
    DMD,
    EDMD,
    Pendulum,
    PendulumDictionary,
    generate_pendulum_dataset,
    generate_trajectory,
    snapshot_pairs,
    viz,
)
from koopman.deep_koopman import DeepKoopman, TrainConfig, train_deep_koopman
from koopman.metrics import error_curve, state_scale, valid_prediction_time
from koopman.utils import set_seed

# --- experiment constants ------------------------------------------------- #
DT = 0.05  # sampling interval (s)
T_FINAL = 12.0  # rollout horizon (s)
DAMPING = 0.2  # c in theta'' = -(g/L) sin(theta) - c theta'
THRESHOLD = 0.2  # normalised error that defines "valid prediction"
SHOWCASE = {"small": 0.4, "medium": 1.5, "large": 2.5}  # theta0 (rad), omega0 = 0

# Deep Koopman settings chosen on the validation set (latent 4/8/16/32, horizon 10/30/60;
# the longest horizon and a 16-dimensional latent space gave the lowest validation error).
DK_LATENT = 16
DK_HORIZON = 60  # steps in the multi-step loss (= 3 s)
DK_ITERATIONS = 4000


def _rollout_curves(models, trajs, scale):
    """``{name: (n_traj, T+1)}`` normalised error curves of full rollouts."""
    n = trajs.shape[1] - 1
    return {
        name: np.stack([error_curve(x, f(x[0], n), scale) for x in trajs])
        for name, f in models.items()
    }


def run(out_dir="results", quick: bool = False, seed: int = 0, verbose: bool = True) -> dict:
    """Run the full pendulum experiment, write figures + metrics, return the metrics."""
    t_start = time.time()
    out = Path(out_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    log = print if verbose else (lambda *a, **k: None)
    set_seed(seed)

    n_train, n_val, n_test = (12, 4, 6) if quick else (60, 12, 30)
    dk_iters = 150 if quick else DK_ITERATIONS
    grid_K = [1, 2] if quick else [1, 2, 3, 4]
    grid_P = [1, 2, 3] if quick else [1, 2, 3, 4, 5, 6]

    system = Pendulum(damping=DAMPING)
    t, train = generate_pendulum_dataset(system, n_train, T_FINAL, DT, seed=seed)
    _, val = generate_pendulum_dataset(system, n_val, T_FINAL, DT, seed=seed + 1)
    _, test = generate_pendulum_dataset(system, n_test, T_FINAL, DT, seed=seed + 2)
    n_steps = train.shape[1] - 1
    X, Y = snapshot_pairs(train)
    scale = state_scale(train)
    log(f"[pendulum] data: {n_train} train / {n_val} val / {n_test} test trajectories, "
        f"{n_steps} steps of dt={DT}")

    # ---- DMD --------------------------------------------------------------
    dmd = DMD(rank=2, dt=DT).fit(X, Y)

    # ---- EDMD: pick the dictionary on the validation set -------------------
    def val_error(model):
        c = np.stack([error_curve(x, model.predict(x[0], n_steps), scale) for x in val])
        return float(c.mean())

    grid = np.full((len(grid_K), len(grid_P)), np.nan)
    for i, K in enumerate(grid_K):
        for j, P in enumerate(grid_P):
            grid[i, j] = val_error(EDMD(PendulumDictionary(K, P), reg=1e-8).fit(X, Y))
    best = np.unravel_index(np.nanargmin(grid), grid.shape)
    K_sel, P_sel = grid_K[best[0]], grid_P[best[1]]
    edmd = EDMD(PendulumDictionary(K_sel, P_sel), reg=1e-8).fit(X, Y)
    log(f"[pendulum] EDMD dictionary selected on validation: harmonics={K_sel}, "
        f"velocity degree={P_sel} ({edmd.dictionary.n_features} features)")

    # ---- Deep Koopman ------------------------------------------------------
    model = DeepKoopman(2, latent_dim=DK_LATENT, seed=seed)
    cfg = TrainConfig(n_steps=min(DK_HORIZON, n_steps - 1), iterations=dk_iters, seed=seed,
                      log_every=max(1, dk_iters // 60), verbose=verbose)
    history = train_deep_koopman(model, train, cfg, val_trajectories=val)

    # ---- evaluation on held-out initial conditions -------------------------
    predictors = {
        "DMD": lambda x0, n: dmd.predict(x0, n),
        "EDMD": lambda x0, n: edmd.predict(x0, n),
        "Deep Koopman": lambda x0, n: model.predict(x0, n),
    }
    curves = _rollout_curves(predictors, test, scale)
    one_step = {
        "DMD": dmd.predict_one_step,
        "EDMD": edmd.predict_one_step,
        "Deep Koopman": model.predict_one_step,
    }
    Xt, Yt = snapshot_pairs(test)
    metrics: dict = {"config": {
        "dt": DT, "t_final": T_FINAL, "damping": DAMPING, "threshold": THRESHOLD,
        "n_train": n_train, "n_val": n_val, "n_test": n_test, "seed": seed, "quick": quick,
        "error_scale": scale, "dk_latent": DK_LATENT, "dk_horizon": cfg.n_steps,
        "dk_iterations": dk_iters,
    }, "methods": {}}
    period = 2 * np.pi / system.omega0
    valid_times, censored = {}, {}
    for name, c in curves.items():
        vt = [valid_prediction_time(ci, t, THRESHOLD) for ci in c]
        med = float(np.median([v for v, _ in vt]))
        valid_times[name] = med
        frac_capped = float(np.mean([flag for _, flag in vt]))  # runs that never exceeded the threshold
        censored[name] = bool(med >= t[-1] - 1e-9)  # the median itself is capped at the horizon
        metrics["methods"][name] = {
            "mean_rollout_error": float(c.mean()),
            "final_time_error": float(c[:, -1].mean()),
            "median_valid_time_s": med,
            "median_valid_time_periods": med / period,
            "fraction_never_exceeded": frac_capped,
            "one_step_rmse": float(np.sqrt(np.mean((one_step[name](Xt) - Yt) ** 2))),
        }
    metrics["methods"]["Deep Koopman"]["reconstruction_rmse"] = float(
        np.sqrt(np.mean((model.reconstruct(Xt) - Xt) ** 2)))
    metrics["edmd_selected"] = {"harmonics": K_sel, "velocity_degree": P_sel,
                                "n_features": edmd.dictionary.n_features,
                                "validation_error": float(grid[best])}
    metrics["dmd"] = {
        "continuous_eigenvalues": [[float(z.real), float(z.imag)] for z in dmd.continuous_eigenvalues],
        "linearised_eigenvalues": [[float(z.real), float(z.imag)] for z in system.linearized_eigenvalues()],
    }

    # ---- amplitude sweep -----------------------------------------------------
    amps = np.linspace(0.2, 2.6, 5 if quick else 13)
    sweep = {name: [] for name in predictors}
    for a in amps:
        _, ref = generate_trajectory(system, [a, 0.0], T_FINAL, DT)
        for name, f in predictors.items():
            sweep[name].append(float(error_curve(ref, f(ref[0], n_steps), scale).mean()))
    metrics["amplitude_sweep"] = {"theta0": amps.tolist(), **{k: v for k, v in sweep.items()}}

    # ---- figures ---------------------------------------------------------------
    show_trajs, titles = [], []
    for label, a in SHOWCASE.items():
        _, ref = generate_trajectory(system, [a, 0.0], T_FINAL, DT)
        show_trajs.append(ref)
        titles.append(rf"{label} amplitude  ($\theta_0$ = {a:g} rad)")
    show_preds = {name: [f(r[0], n_steps) for r in show_trajs] for name, f in predictors.items()}
    viz.plot_phase_portraits(show_trajs, show_preds, titles, path=fig_dir / "pendulum_phase_portraits.png")
    big = show_trajs[-1]
    viz.plot_time_series(
        t, big, {k: v[-1] for k, v in show_preds.items()}, ["θ  (rad)", "ω  (rad/s)"],
        xlabel="time  (s)",
        title=rf"Large-amplitude release ($\theta_0$ = {SHOWCASE['large']:g} rad): rollout from the initial state",
        path=fig_dir / "pendulum_time_series.png", figsize=(9, 5.4))
    viz.plot_error_curves(
        t, curves, threshold=THRESHOLD, xlabel="time  (s)",
        ylabel="normalised error  ‖x̂ − x‖ / σ",
        title="Pendulum: rollout error on held-out initial conditions (median and IQR)",
        path=fig_dir / "pendulum_error_curves.png", ylim=(1e-4, 5))
    viz.plot_valid_time_bars(valid_times, censored, xlabel="median valid prediction time  (s)",
                             title=f"Pendulum: time until error exceeds {THRESHOLD:g}",
                             path=fig_dir / "pendulum_valid_time.png")
    viz.plot_error_vs_amplitude(amps, sweep, title="Pendulum: error vs. initial amplitude",
                                path=fig_dir / "pendulum_error_vs_amplitude.png")
    ref_disc = np.exp(system.linearized_eigenvalues() * DT)
    viz.plot_spectrum({"DMD": dmd.eigenvalues, "EDMD": edmd.eigenvalues, "Deep Koopman": model.eigenvalues},
                      reference=ref_disc, path=fig_dir / "pendulum_spectrum.png",
                      title="Pendulum: Koopman eigenvalues (Δt = 0.05 s)", zoom=(0.9, 1.02, -0.35, 0.35))
    viz.plot_training_history(history, title="Pendulum: Deep Koopman training",
                              path=fig_dir / "pendulum_training.png")
    viz.plot_latent_pca([model.latent(x) for x in test[:10]], title="Pendulum: latent trajectories (PCA)",
                        path=fig_dir / "pendulum_latent.png")
    viz.plot_dictionary_ablation(grid, grid_K, grid_P, chosen=best,
                                 title="Pendulum: EDMD dictionary ablation (validation error)",
                                 path=fig_dir / "pendulum_edmd_ablation.png")

    metrics["runtime_s"] = time.time() - t_start
    (out / "pendulum_metrics.json").write_text(json.dumps(metrics, indent=2))

    log("\n[pendulum] results on held-out initial conditions")
    log(f"{'method':14s} {'mean err':>9s} {'valid time (s)':>15s} {'one-step RMSE':>14s}")
    for name, m in metrics["methods"].items():
        log(f"{name:14s} {m['mean_rollout_error']:9.3f} {m['median_valid_time_s']:15.2f} {m['one_step_rmse']:14.2e}")
    return metrics
