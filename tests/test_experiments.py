"""End-to-end smoke tests: both experiments (tiny settings) and the command-line entry point."""

import json

import numpy as np

import main
from experiments import lorenz, pendulum

PENDULUM_FIGURES = [
    "pendulum_phase_portraits", "pendulum_time_series", "pendulum_error_curves", "pendulum_valid_time",
    "pendulum_error_vs_amplitude", "pendulum_spectrum", "pendulum_training", "pendulum_latent",
    "pendulum_edmd_ablation",
]
LORENZ_FIGURES = [
    "lorenz_attractor_comparison", "lorenz_time_series", "lorenz_error_curves", "lorenz_valid_time",
    "lorenz_spectrum", "lorenz_training", "lorenz_latent", "lorenz_reconstruction_vs_prediction",
]


def _check_figures(out, names):
    for n in names:
        f = out / "figures" / f"{n}.png"
        assert f.exists() and f.stat().st_size > 1000, f"missing or empty figure {n}"


def test_pendulum_experiment_quick(tmp_path):
    m = pendulum.run(out_dir=tmp_path, quick=True, seed=0, verbose=False)
    _check_figures(tmp_path, PENDULUM_FIGURES)
    saved = json.loads((tmp_path / "pendulum_metrics.json").read_text())
    assert saved["methods"].keys() == {"DMD", "EDMD", "Deep Koopman"}
    for name, v in saved["methods"].items():
        assert np.isfinite(v["mean_rollout_error"]) and v["median_valid_time_s"] >= 0
    assert "reconstruction_rmse" in saved["methods"]["Deep Koopman"]
    assert all(0.0 <= v["fraction_never_exceeded"] <= 1.0 for v in saved["methods"].values())
    assert saved["edmd_selected"]["n_features"] > 2
    assert len(saved["amplitude_sweep"]["theta0"]) == len(saved["amplitude_sweep"]["DMD"])
    # DMD cannot represent the amplitude-dependent frequency, EDMD's one-step map is nearly exact
    assert m["methods"]["EDMD"]["one_step_rmse"] < m["methods"]["DMD"]["one_step_rmse"]


def test_lorenz_experiment_quick(tmp_path):
    m = lorenz.run(out_dir=tmp_path, quick=True, seed=0, verbose=False)
    _check_figures(tmp_path, LORENZ_FIGURES)
    saved = json.loads((tmp_path / "lorenz_metrics.json").read_text())
    assert saved["methods"].keys() == {"DMD", "EDMD", "Deep Koopman"}
    assert saved["edmd_selected"]["degree"] in (2, 3)
    assert saved["config"]["n_test_starts"] >= 1
    assert all(0.0 <= v["fraction_never_exceeded"] <= 1.0 for v in saved["methods"].values())
    assert "spectral_radius_before" in saved["deep_koopman_stabilisation"]
    assert "1 step ahead" in saved["deep_koopman_error_ladder"]
    assert m["methods"]["EDMD"]["one_step_rmse"] <= m["methods"]["DMD"]["one_step_rmse"]


def test_cli_runs_selected_system(tmp_path, capsys):
    assert main.main(["--system", "pendulum", "--quick", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "pendulum_metrics.json").exists()
    assert not (tmp_path / "lorenz_metrics.json").exists()
    assert "Done." in capsys.readouterr().out


def test_cli_rejects_unknown_system():
    import pytest

    with pytest.raises(SystemExit):
        main.main(["--system", "nonsense"])
