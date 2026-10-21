import re

import matplotlib.pyplot as plt
import numpy as np
import pytest

from koopman import viz


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _saved(path):
    return path.exists() and path.stat().st_size > 1000


def traj(n=60, d=2, phase=0.0):
    t = np.linspace(0, 6, n)
    cols = [np.sin(t + phase + k) * np.exp(-0.1 * t) for k in range(d)]
    return np.stack(cols, axis=1)


def preds(d=2, n=60):
    return {"DMD": traj(n, d, 0.2), "EDMD": traj(n, d, 0.1), "Deep Koopman": traj(n, d, 0.05)}


def test_palette_constants_are_valid_hex_and_methods_are_distinguishable():
    for c in (viz.SURFACE, viz.INK, viz.BLUE, viz.ORANGE, viz.AQUA, viz.GRID):
        assert re.fullmatch(r"#[0-9a-f]{6}", c)
    styles = [viz.style_for(n) for n in ("Truth", "DMD", "EDMD", "Deep Koopman")]
    assert len({s["color"] for s in styles}) == 4
    # secondary encoding: no two methods share both colour and line style
    assert len({(s["color"], s["ls"]) for s in styles}) == 4


def test_style_for_unknown_series_uses_fallback_colours():
    a, b = viz.style_for("foo", 0), viz.style_for("bar", 1)
    assert a["color"] != b["color"]


def test_phase_portraits(tmp_path):
    p = tmp_path / "pp.png"
    fig = viz.plot_phase_portraits([traj(), traj(phase=1)], {k: [v, v] for k, v in preds().items()},
                                   ["a", "b"], path=p)
    assert _saved(p)
    assert fig is not None


def test_time_series_and_returned_figure_without_path():
    fig = viz.plot_time_series(np.linspace(0, 6, 60), traj(d=3), preds(3), ["x", "y", "z"], title="t")
    assert len(fig.axes) == 3


def test_attractor_comparison(tmp_path):
    p = tmp_path / "att.png"
    truth = traj(80, 3)
    viz.plot_attractor_comparison(truth, preds(3, 80), panel_notes={"DMD": "note"},
                                  background=traj(200, 3), path=p)
    assert _saved(p)


def test_error_curves_and_valid_time_bars(tmp_path):
    t = np.linspace(0, 5, 50)
    rng = np.random.default_rng(0)
    curves = {k: np.abs(rng.normal(size=(6, 50))) * np.exp(0.5 * t) * 1e-3 for k in preds()}
    p1, p2 = tmp_path / "err.png", tmp_path / "bars.png"
    viz.plot_error_curves(t, curves, threshold=0.2, path=p1)
    viz.plot_valid_time_bars({"DMD": 0.3, "EDMD": 1.2, "Deep Koopman": 2.0},
                             censored={"Deep Koopman": True}, path=p2)
    assert _saved(p1) and _saved(p2)


def test_valid_time_bars_with_custom_colour_and_format(tmp_path):
    p = tmp_path / "ladder.png"
    viz.plot_valid_time_bars({"reconstruction": 0.01, "1 step": 0.02}, color=viz.AQUA, fmt="{:.3f}", path=p)
    assert _saved(p)


def test_error_vs_amplitude(tmp_path):
    p = tmp_path / "amp.png"
    amps = np.linspace(0.2, 2.6, 7)
    viz.plot_error_vs_amplitude(amps, {k: amps * s for k, s in zip(preds(), (1.0, 0.5, 0.2))}, path=p)
    assert _saved(p)


def test_spectrum_with_zoom_panel(tmp_path):
    p = tmp_path / "spec_zoom.png"
    ev = {"DMD": 0.99 * np.exp(1j * np.array([0.1, -0.1])), "EDMD": 0.98 + 0.05j * np.arange(-2, 3)}
    fig = viz.plot_spectrum(ev, reference=np.array([0.99 + 0.12j]), zoom=(0.9, 1.02, -0.3, 0.3), path=p)
    assert _saved(p)
    assert len(fig.axes) == 2  # full disc + magnified window


def test_error_curves_skip_the_zero_error_initial_sample():
    t = np.linspace(0, 5, 50)
    c = np.zeros((4, 50))
    c[:, 1:] = 0.05  # exactly zero at t = 0, as for every rollout started from the true state
    fig = viz.plot_error_curves(t, {"DMD": c}, threshold=0.2)
    ax = fig.axes[0]
    line = ax.get_lines()[0]
    assert len(line.get_xdata()) == 49 and line.get_xdata()[0] == pytest.approx(t[1])
    assert np.all(np.isfinite(line.get_ydata())) and np.all(line.get_ydata() > 0)
    assert any("threshold" in txt for txt in ax.get_legend_handles_labels()[1])


def test_valid_time_bars_flag_values_capped_at_the_horizon():
    fig = viz.plot_valid_time_bars({"DMD": 1.1, "EDMD": 12.0}, censored={"EDMD": True})
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert any("≥" in t and "12.00" in t for t in texts)
    assert not any("≥" in t and "1.10" in t for t in texts)


def test_spectrum(tmp_path):
    p = tmp_path / "spec.png"
    ev = {"DMD": 0.9 * np.exp(1j * np.array([0.3, -0.3])), "EDMD": np.random.default_rng(0).uniform(-1, 1, 5) + 0j,
          "Deep Koopman": np.array([0.8 + 0.1j, 0.8 - 0.1j])}
    viz.plot_spectrum(ev, reference=np.array([0.95 + 0.2j, 0.95 - 0.2j]), path=p)
    assert _saved(p)


def test_training_history_with_and_without_validation(tmp_path):
    h = {"iteration": [0, 10, 20], "loss": [1, .5, .1], "rec": [.5, .2, .05], "lin": [.3, .2, .03], "pred": [.2, .1, .02]}
    p1, p2 = tmp_path / "h1.png", tmp_path / "h2.png"
    viz.plot_training_history(h, path=p1)
    viz.plot_training_history({**h, "val_pred": [.3, .2, .05]}, path=p2)
    assert _saved(p1) and _saved(p2)


def test_latent_pca(tmp_path):
    p = tmp_path / "lat.png"
    rng = np.random.default_rng(0)
    viz.plot_latent_pca([rng.normal(size=(50, 6)).cumsum(0) for _ in range(3)], path=p)
    assert _saved(p)


def test_dictionary_ablation(tmp_path):
    p = tmp_path / "abl.png"
    grid = np.random.default_rng(0).uniform(0.05, 0.6, size=(3, 4))
    viz.plot_dictionary_ablation(grid, [1, 2, 3], [1, 2, 3, 4], chosen=(1, 2), path=p)
    assert _saved(p)


def test_saving_closes_the_figure(tmp_path):
    plt.close("all")
    viz.plot_training_history({"iteration": [0, 1], "loss": [1, 0.5], "rec": [1, .5], "lin": [1, .5], "pred": [1, .5]},
                              path=tmp_path / "x.png")
    assert plt.get_fignums() == []


def test_saving_creates_missing_directories(tmp_path):
    p = tmp_path / "a" / "b" / "c.png"
    viz.plot_error_vs_amplitude([0, 1], {"DMD": [1, 2]}, path=p)
    assert _saved(p)
