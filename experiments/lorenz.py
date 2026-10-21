"""Lorenz experiment: DMD vs EDMD vs Deep Koopman on a chaotic attractor.

One long trajectory on the attractor is split *chronologically* into train /
validation / test segments. Models are trained on the training segment only and
evaluated by rolling out from many different test initial states over several
Lyapunov times. Because Lorenz-63 is chaotic, accurate prediction is limited to
a few Lyapunov times at best (``t_lyap = 1 / 0.9056 ~ 1.1`` time units), and a
single finite-dimensional *linear* model cannot reproduce a chaotic attractor
in the long run -- the experiment quantifies how far each method gets.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np

from koopman import (
    DMD,
    EDMD,
    LORENZ_LYAPUNOV,
    Lorenz,
    generate_lorenz_long,
    lorenz_dictionary,
    snapshot_pairs,
    viz,
)
from koopman.deep_koopman import DeepKoopman, TrainConfig, train_deep_koopman
from koopman.metrics import error_curve, state_scale, valid_prediction_time
from koopman.utils import set_seed

# --- experiment constants ------------------------------------------------- #
DT = 0.02
THRESHOLD = 0.2  # normalised error defining "valid prediction"
HORIZON_LYAP = 6.0  # rollout length in Lyapunov times
DEMO_LYAP = 3.0  # horizon drawn in the 3-D comparison figure
N_TRAIN, N_VAL, N_TEST = 10_000, 2_000, 4_000
STRIDE = 80  # spacing (steps) between test initial states

# Deep Koopman settings chosen on the validation segment from a small grid
# (horizon 10/20/40 steps, latent 16/32): the longest horizon with a 32-dimensional
# latent space gave the longest median valid prediction time.
DK_LATENT = 32
DK_HIDDEN = (128, 128)
DK_HORIZON = 40  # steps in the multi-step loss (= 0.8 time units ~ 0.7 Lyapunov times)
DK_ITERATIONS = 3000


def run(out_dir="results", quick: bool = False, seed: int = 0, verbose: bool = True) -> dict:
    """Run the full Lorenz experiment, write figures + metrics, return the metrics."""
    t_start = time.time()
    out = Path(out_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    log = print if verbose else (lambda *a, **k: None)
    set_seed(seed)

    n_train, n_val, n_test = (1500, 400, 900) if quick else (N_TRAIN, N_VAL, N_TEST)
    horizon_lyap = 1.5 if quick else HORIZON_LYAP
    dk_iters = 150 if quick else DK_ITERATIONS
    degrees = [2, 3] if quick else [2, 3, 4]

    system = Lorenz()
    _, traj = generate_lorenz_long(system, n_train + n_val + n_test, DT)
    train = traj[: n_train + 1]
    val = traj[n_train + 1 : n_train + n_val + 1]
    test = traj[n_train + n_val + 1 :]
    scale = state_scale(train)
    H = int(round(horizon_lyap / LORENZ_LYAPUNOV / DT))
    t = np.arange(H + 1) * DT
    t_lyap = t * LORENZ_LYAPUNOV
    log(f"[lorenz] {n_train}/{n_val}/{n_test} train/val/test steps (dt={DT}); rollout horizon "
        f"{horizon_lyap:g} Lyapunov times = {H} steps")

    X, Y = snapshot_pairs(train)

    def starts_of(seg):
        return np.arange(0, len(seg) - H - 1, STRIDE if not quick else 100)

    def curves_of(seg, predict):
        return np.stack([error_curve(seg[s : s + H + 1], predict(seg[s], H), scale)
                         for s in starts_of(seg)])

    # ---- DMD ---------------------------------------------------------------
    dmd = DMD(rank=3, dt=DT).fit(X, Y)

    # ---- EDMD: pick the polynomial degree on the validation segment --------
    val_err = {}
    for deg in degrees:
        m = EDMD(lorenz_dictionary(deg), reg=1e-6).fit(X, Y)
        val_err[deg] = float(curves_of(val, m.predict).mean())
    deg_sel = min(val_err, key=val_err.get)
    edmd = EDMD(lorenz_dictionary(deg_sel), reg=1e-6).fit(X, Y)
    log(f"[lorenz] EDMD polynomial degree selected on validation: {deg_sel} "
        f"({edmd.dictionary.n_features} features); validation errors {val_err}")

    # ---- Deep Koopman -------------------------------------------------------
    model = DeepKoopman(3, latent_dim=DK_LATENT, hidden=DK_HIDDEN, seed=seed)
    cfg = TrainConfig(n_steps=DK_HORIZON, iterations=dk_iters, seed=seed, batch_size=512,
                      log_every=max(1, dk_iters // 60), verbose=verbose)
    history = train_deep_koopman(model, train, cfg, val_trajectories=val)

    # A learned K can have a slightly unstable eigenvalue that explodes in long rollouts.
    # Keep the stabilised K only if it lowers the *validation* rollout error.
    stabilised = copy.deepcopy(model)
    radius_before = stabilised.stabilize()
    val_raw = float(curves_of(val, model.predict).mean())
    val_stab = float(curves_of(val, stabilised.predict).mean())
    use_stab = radius_before > 1.0 and val_stab < val_raw
    if use_stab:
        model = stabilised
    log(f"[lorenz] Deep Koopman spectral radius {radius_before:.4f}; validation error "
        f"{val_raw:.3f} (raw) vs {val_stab:.3f} (stabilised) -> "
        f"{'stabilised' if use_stab else 'raw'} K kept")

    # ---- evaluation on the test segment --------------------------------------
    predictors = {"DMD": dmd.predict, "EDMD": edmd.predict, "Deep Koopman": model.predict}
    curves = {name: curves_of(test, f) for name, f in predictors.items()}
    one_step = {"DMD": dmd.predict_one_step, "EDMD": edmd.predict_one_step,
                "Deep Koopman": model.predict_one_step}
    Xt, Yt = snapshot_pairs(test)

    metrics: dict = {"config": {
        "dt": DT, "lyapunov_exponent": LORENZ_LYAPUNOV, "threshold": THRESHOLD,
        "horizon_lyapunov_times": horizon_lyap, "n_train": n_train, "n_val": n_val,
        "n_test": n_test, "n_test_starts": int(len(curves["DMD"])), "seed": seed, "quick": quick,
        "error_scale": scale, "dk_latent": DK_LATENT, "dk_hidden": list(DK_HIDDEN),
        "dk_horizon": cfg.n_steps, "dk_iterations": dk_iters,
    }, "methods": {}}
    valid_med, valid_mean, censored = {}, {}, {}
    for name, c in curves.items():
        vt = np.array([valid_prediction_time(ci, t_lyap, THRESHOLD)[0] for ci in c])
        fl = np.array([valid_prediction_time(ci, t_lyap, THRESHOLD)[1] for ci in c])
        valid_med[name], valid_mean[name] = float(np.median(vt)), float(vt.mean())
        censored[name] = bool(valid_med[name] >= t_lyap[-1] - 1e-9)  # median capped at the horizon
        metrics["methods"][name] = {
            "mean_rollout_error": float(c.mean()),
            "median_valid_time_lyapunov": valid_med[name],
            "mean_valid_time_lyapunov": valid_mean[name],
            "fraction_never_exceeded": float(fl.mean()),
            "one_step_rmse": float(np.sqrt(np.mean((one_step[name](Xt) - Yt) ** 2))),
        }
    metrics["methods"]["Deep Koopman"]["reconstruction_rmse"] = float(
        np.sqrt(np.mean((model.reconstruct(Xt) - Xt) ** 2)))
    metrics["edmd_selected"] = {"degree": deg_sel, "n_features": edmd.dictionary.n_features,
                                "validation_errors": {str(k): v for k, v in val_err.items()}}
    metrics["deep_koopman_stabilisation"] = {
        "spectral_radius_before": radius_before, "validation_error_raw": val_raw,
        "validation_error_stabilised": val_stab, "stabilised_K_used": bool(use_stab)}

    # error ladder for Deep Koopman: reconstruction (no dynamics) vs prediction
    dk = curves["Deep Koopman"]
    rec_err = float(np.mean(np.linalg.norm(model.reconstruct(Xt) - Xt, axis=1)) / scale)
    ladder = {
        "reconstruction (no dynamics)": rec_err,
        "1 step ahead": float(np.mean(np.linalg.norm(model.predict_one_step(Xt) - Yt, axis=1)) / scale),
        "1 Lyapunov time": float(dk[:, int(round(1.0 / LORENZ_LYAPUNOV / DT))].mean()) if horizon_lyap >= 1 else float("nan"),
        "3 Lyapunov times": float(dk[:, int(round(3.0 / LORENZ_LYAPUNOV / DT))].mean()) if horizon_lyap >= 3 else float("nan"),
    }
    ladder = {k: v for k, v in ladder.items() if np.isfinite(v)}
    metrics["deep_koopman_error_ladder"] = ladder

    # ---- figures ----------------------------------------------------------------
    s0 = starts_of(test)[0]  # first test start: deliberately not cherry-picked
    ref = test[s0 : s0 + H + 1]
    rolled = {name: f(ref[0], H) for name, f in predictors.items()}
    h_demo = int(round(min(DEMO_LYAP, horizon_lyap) / LORENZ_LYAPUNOV / DT))
    notes = {}
    for name in predictors:
        c0 = error_curve(ref, rolled[name], scale)
        vt, _ = valid_prediction_time(c0, t_lyap, THRESHOLD)
        notes[name] = f"valid for {vt:.2f} Lyapunov times (threshold {THRESHOLD:g})"
    viz.plot_attractor_comparison(
        ref[: h_demo + 1], {k: v[: h_demo + 1] for k, v in rolled.items()},
        panel_notes=notes, background=train[: min(len(train), 5000)],
        path=fig_dir / "lorenz_attractor_comparison.png")
    viz.plot_time_series(t_lyap, ref, rolled, ["x", "y", "z"], xlabel="time  (Lyapunov times)",
                         title="Lorenz: rollout from a single test initial state",
                         path=fig_dir / "lorenz_time_series.png", figsize=(9, 6.2))
    viz.plot_error_curves(t_lyap, curves, threshold=THRESHOLD, xlabel="time  (Lyapunov times)",
                          ylabel="normalised error  ‖x̂ − x‖ / σ",
                          title="Lorenz: rollout error over many test initial states (median and IQR)",
                          path=fig_dir / "lorenz_error_curves.png", ylim=(1e-4, 5))
    viz.plot_valid_time_bars(valid_med, censored, xlabel="median valid prediction time  (Lyapunov times)",
                             title=f"Lorenz: time until error exceeds {THRESHOLD:g}",
                             path=fig_dir / "lorenz_valid_time.png")
    viz.plot_spectrum({"DMD": dmd.eigenvalues, "EDMD": edmd.eigenvalues, "Deep Koopman": model.eigenvalues},
                      title="Lorenz: Koopman eigenvalues (Δt = 0.02)", path=fig_dir / "lorenz_spectrum.png",
                      zoom=(0.9, 1.02, -0.3, 0.3))
    viz.plot_training_history(history, title="Lorenz: Deep Koopman training",
                              path=fig_dir / "lorenz_training.png")
    viz.plot_latent_pca([model.latent(test[: min(len(test), 3000)])],
                        title="Lorenz: latent trajectory (PCA)", path=fig_dir / "lorenz_latent.png")
    viz.plot_valid_time_bars(ladder, xlabel="mean normalised error  ‖x̂ − x‖ / σ",
                             title="Deep Koopman: reconstruction vs. prediction error",
                             path=fig_dir / "lorenz_reconstruction_vs_prediction.png",
                             color=viz.AQUA, fmt="{:.3f}")

    metrics["runtime_s"] = time.time() - t_start
    (out / "lorenz_metrics.json").write_text(json.dumps(metrics, indent=2))

    log("\n[lorenz] results on held-out initial states")
    log(f"{'method':14s} {'mean err':>9s} {'median valid (Lyap)':>20s} {'one-step RMSE':>14s}")
    for name, m in metrics["methods"].items():
        log(f"{name:14s} {m['mean_rollout_error']:9.3f} {m['median_valid_time_lyapunov']:20.2f} {m['one_step_rmse']:14.2e}")
    log("[lorenz] Deep Koopman error ladder:", {k: round(v, 3) for k, v in ladder.items()})
    return metrics
