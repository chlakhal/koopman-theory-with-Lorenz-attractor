"""Plotting helpers.

Design notes
------------
* Ground truth is always neutral dark ink; the three learned models use the
  first three slots of a colour-blind-validated categorical palette
  (blue / orange / aqua) **and** a distinct line style, so identity never relies
  on colour alone.
* One axis per plot (no dual axes); recessive grids; legends on every
  multi-series plot.
* Every function returns the Matplotlib figure and, when ``path`` is given, also
  saves it (PNG) and closes it. Choose the backend yourself (e.g.
  ``matplotlib.use("Agg")`` on a headless machine).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3-D projection)

# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SPINE = "#b9b8b2"

# Categorical slots 1-3 (validated all-pairs for colour-vision deficiency).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
VIOLET, RED = "#6250d6", "#e34948"

STYLE = {
    "Truth": dict(color="#2b2b29", lw=2.4, ls="-", zorder=2),
    "DMD": dict(color=BLUE, lw=1.7, ls="--", zorder=3),
    "EDMD": dict(color=ORANGE, lw=1.7, ls="-.", zorder=3),
    "Deep Koopman": dict(color=AQUA, lw=1.7, ls="-", zorder=4),
}
_FALLBACK = [VIOLET, RED, "#eda100", "#e87ba4"]

# One-hue sequential ramp (light -> dark blue) for magnitudes.
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)


def apply_style() -> None:
    """Set the global Matplotlib defaults used by every figure."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": SPINE,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "text.color": INK,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.size": 10,
            "lines.solid_capstyle": "round",
            "figure.dpi": 100,
        }
    )


def style_for(name: str, index: int = 0) -> dict:
    """Style dictionary for a series name (unknown names get fallback colours)."""
    if name in STYLE:
        return dict(STYLE[name])
    return dict(color=_FALLBACK[index % len(_FALLBACK)], lw=1.7, ls="-", zorder=3)


def _finish(fig, path):
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


def _handles(names):
    out = []
    for i, n in enumerate(names):
        st = style_for(n, i)
        out.append(Line2D([0], [0], color=st["color"], lw=st["lw"], ls=st["ls"], label=n))
    return out


# --------------------------------------------------------------------------- #
# Trajectories
# --------------------------------------------------------------------------- #
def plot_phase_portraits(truths, preds, titles, labels=("θ  (rad)", "ω  (rad/s)"), path=None):
    """Phase-plane comparison, one panel per initial condition.

    ``truths``: list of ``(T, 2)`` arrays; ``preds``: ``{name: [ (T, 2) per panel ]}``.
    """
    apply_style()
    n = len(truths)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.4), squeeze=False)
    for j, ax in enumerate(axes[0]):
        ax.plot(truths[j][:, 0], truths[j][:, 1], **style_for("Truth"), label="Truth")
        for i, (name, series) in enumerate(preds.items()):
            ax.plot(series[j][:, 0], series[j][:, 1], **style_for(name, i))
        ax.plot(*truths[j][0], "o", ms=6, color=INK, mec=SURFACE, mew=1.5, zorder=6)
        ax.set_title(titles[j])
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1] if j == 0 else "")
        ax.set_aspect("equal", adjustable="datalim")
    fig.legend(
        handles=_handles(["Truth", *preds]) + [Line2D([0], [0], marker="o", color=INK, ls="", ms=6, label="start")],
        loc="lower center", ncol=len(preds) + 2, bbox_to_anchor=(0.5, -0.04),
    )
    fig.tight_layout()
    return _finish(fig, path)


def plot_time_series(t, truth, preds, component_labels, xlabel="time", title=None, path=None, figsize=None):
    """One row per state component: truth vs every method, shared time axis."""
    apply_style()
    d = truth.shape[1]
    fig, axes = plt.subplots(d, 1, sharex=True, figsize=figsize or (9, 2.0 * d + 0.9), squeeze=False)
    for c, ax in enumerate(axes[:, 0]):
        ax.plot(t, truth[:, c], **style_for("Truth"))
        for i, (name, series) in enumerate(preds.items()):
            ax.plot(t, series[:, c], **style_for(name, i))
        ax.set_ylabel(component_labels[c])
    axes[-1, 0].set_xlabel(xlabel)
    if title:
        axes[0, 0].set_title(title)
    fig.legend(handles=_handles(["Truth", *preds]), loc="lower center", ncol=len(preds) + 1,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _finish(fig, path)


def plot_attractor_comparison(truth, preds, panel_notes=None, background=None, path=None):
    """2x2 grid of 3-D views: truth and one panel per method (rollout vs truth).

    ``background`` (optional ``(N, 3)``) is drawn faintly in every panel to show the attractor.
    """
    apply_style()
    names = list(preds)
    fig = plt.figure(figsize=(11, 8.4))
    panels = [("Truth", None)] + [(n, preds[n]) for n in names]
    for k, (name, series) in enumerate(panels[:4]):
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        ax.grid(True)
        if background is not None:
            ax.plot(background[:, 0], background[:, 1], background[:, 2], color=GRID, lw=0.4, zorder=1)
        ax.plot(*truth.T, **{**style_for("Truth"), "lw": 1.4 if series is not None else 1.2})
        if series is not None:
            ax.plot(*series.T, **{**style_for(name, k - 1), "lw": 1.3})
        note = f"\n{panel_notes[name]}" if panel_notes and name in panel_notes else ""
        ax.set_title(("Truth" if series is None else f"{name} rollout vs truth") + note, fontsize=10)
        ax.set_xlabel("x"), ax.set_ylabel("y"), ax.set_zlabel("z")
        ax.view_init(elev=22, azim=-58)
        ax.set_box_aspect((1, 1, 0.85), zoom=1.18)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.set_pane_color((0.99, 0.99, 0.98, 1.0))
    fig.tight_layout()
    return _finish(fig, path)


# --------------------------------------------------------------------------- #
# Errors and summaries
# --------------------------------------------------------------------------- #
def plot_error_curves(t, curves, threshold=None, xlabel="time", ylabel="normalised error",
                      title="Rollout error growth", path=None, ylim=None):
    """Median error (line) with inter-quartile band (shading) per method, log-y.

    The first sample (``t[0]``) is left out: every rollout starts from the true state, so the
    error there is zero by construction and would plunge to ``-inf`` on a log axis.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    t = np.asarray(t)[1:]
    for i, (name, c) in enumerate(curves.items()):
        c = np.atleast_2d(c)[:, 1:]
        st = style_for(name, i)
        lo, med, hi = np.percentile(c, [25, 50, 75], axis=0)
        ax.fill_between(t, np.maximum(lo, 1e-12), hi, color=st["color"], alpha=0.14, lw=0)
        ax.plot(t, med, color=st["color"], lw=st["lw"], ls=st["ls"], label=name, zorder=st["zorder"])
    if threshold is not None:
        ax.axhline(threshold, color=INK_SECONDARY, lw=1.1, ls=":", zorder=1,
                   label=f"validity threshold ({threshold:g})")
    ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _finish(fig, path)


def plot_valid_time_bars(values, censored=None, xlabel="valid prediction time",
                         title="How long does each method stay accurate?", path=None,
                         color=None, fmt="{:.2f}"):
    """Horizontal bars (one per entry) with the value printed at the bar end.

    By default each bar takes its method colour; pass ``color`` to force a single colour
    (useful when the bars are not methods but, e.g., error types).
    """
    apply_style()
    names = list(values)
    fig, ax = plt.subplots(figsize=(7.5, 0.9 * len(names) + 1.4))
    y = np.arange(len(names))[::-1]
    for yi, n in zip(y, names):
        st = style_for(n, int(yi))
        v = values[n]
        ax.barh(yi, v, height=0.55, color=color or st["color"], zorder=3)
        is_cap = bool(censored and censored.get(n))
        label = f"  ≥ {fmt.format(v)}  (reached the horizon)" if is_cap else f"  {fmt.format(v)}"
        ax.text(v, yi, label, va="center", color=INK, fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    room = 1.75 if censored and any(censored.values()) else 1.3
    ax.set_xlim(0, max(values.values()) * room + 1e-9)
    fig.tight_layout()
    return _finish(fig, path)


def plot_error_vs_amplitude(amplitudes, errors, ylabel="mean normalised rollout error",
                            xlabel=r"initial angle $\theta_0$  (rad)", title="Error vs. initial amplitude", path=None):
    """Mean rollout error as a function of the initial amplitude, one line per method."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for i, (name, e) in enumerate(errors.items()):
        st = style_for(name, i)
        ax.plot(amplitudes, e, color=st["color"], ls=st["ls"], lw=st["lw"], marker="o", ms=4.5,
                mec=SURFACE, mew=1.0, label=name, zorder=st["zorder"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return _finish(fig, path)


# --------------------------------------------------------------------------- #
# Spectra, training, latent space, ablation
# --------------------------------------------------------------------------- #
def plot_spectrum(eigs, reference=None, reference_label="linearisation at θ = 0",
                  title="Koopman eigenvalues (discrete time)", path=None, zoom=None):
    """Eigenvalues in the complex plane with the unit circle.

    ``zoom=(xmin, xmax, ymin, ymax)`` adds a second panel magnifying that window (axes not
    to scale there), which is needed when several methods place eigenvalues close together,
    typically near ``lambda = 1`` for a small time step.
    """
    apply_style()
    markers = {"DMD": "o", "EDMD": "s", "Deep Koopman": "^"}
    sizes = {"DMD": 1.0, "EDMD": 0.55, "Deep Koopman": 0.8}  # relative marker size
    order = {"EDMD": 3, "Deep Koopman": 4, "DMD": 5}
    ang = np.linspace(0, 2 * np.pi, 400)
    ncols = 2 if zoom else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5.8 * ncols, 5.6), squeeze=False)

    def draw(ax, base):
        ax.plot(np.cos(ang), np.sin(ang), color=SPINE, lw=1.0, zorder=1)
        for i, (name, ev) in enumerate(eigs.items()):
            ev = np.asarray(ev)
            st = style_for(name, i)
            ax.scatter(ev.real, ev.imag, s=base * sizes.get(name, 0.8), marker=markers.get(name, "D"),
                       color=st["color"], edgecolor=SURFACE, linewidth=0.8,
                       label=f"{name} ({len(ev)})", zorder=order.get(name, 3))
        if reference is not None:
            ref = np.asarray(reference)
            ax.scatter(ref.real, ref.imag, s=base * 1.5, marker="x", color=INK, linewidth=1.5,
                       label=reference_label, zorder=9)
        ax.axhline(0, color=SPINE, lw=0.6)
        ax.axvline(0, color=SPINE, lw=0.6)
        ax.set_xlabel("Re λ")

    draw(axes[0, 0], 60)
    ax = axes[0, 0]
    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.15), ax.set_ylim(-1.15, 1.15)
    ax.set_ylabel("Im λ")
    ax.set_title(title)
    if zoom:
        x0, x1, y0, y1 = zoom
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=INK_SECONDARY, lw=0.9, ls="--"))
        draw(axes[0, 1], 90)
        axes[0, 1].set_xlim(x0, x1), axes[0, 1].set_ylim(y0, y1)
        axes[0, 1].set_title("zoom (axes not to scale)")
    ax.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    return _finish(fig, path)


def plot_training_history(history, title="Deep Koopman training", path=None):
    """Loss components versus iteration (log scale)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 4.4))
    it = history["iteration"]
    series = [("rec", "reconstruction", BLUE, "-"), ("lin", "latent linearity", ORANGE, "--"),
              ("pred", "multi-step prediction", AQUA, "-")]
    for key, label, color, ls in series:
        ax.plot(it, history[key], color=color, ls=ls, lw=1.7, label=label)
    if "val_pred" in history:
        ax.plot(it, history["val_pred"], color=INK, ls=":", lw=1.9, label="validation prediction")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("mean squared error (normalised units)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return _finish(fig, path)


def plot_latent_pca(latents, n_components=2, title="Learned latent space (PCA projection)", path=None):
    """Latent trajectories projected onto their top principal components.

    ``latents``: list of ``(T, L)`` arrays. Colour encodes time (light -> dark blue).
    """
    apply_style()
    allz = np.concatenate(latents)
    mean = allz.mean(axis=0)
    _, _, Vt = np.linalg.svd(allz - mean, full_matrices=False)
    P = Vt[:n_components].T
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    for z in latents:
        p = (z - mean) @ P
        pts = np.stack([p[:, 0], p[:, 1]], axis=1)
        ax.plot(pts[:, 0], pts[:, 1], color="#9ec5f4", lw=0.8, zorder=2)
        ax.scatter(pts[:, 0], pts[:, 1], c=np.linspace(0, 1, len(pts)), cmap=SEQ_BLUE, s=4, zorder=3)
    ax.set_xlabel("PC 1"), ax.set_ylabel("PC 2")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    sm = plt.cm.ScalarMappable(cmap=SEQ_BLUE, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("time (start → end)")
    cb.outline.set_edgecolor(SPINE)
    fig.tight_layout()
    return _finish(fig, path)


def _relative_luminance(rgb):
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _best_text_color(background_rgb, light="#f0efec", dark=INK):
    """Whichever of ``light`` / ``dark`` has the higher WCAG contrast on the background."""
    from matplotlib.colors import to_rgb

    lb = _relative_luminance(background_rgb)

    def contrast(fg):
        lf = _relative_luminance(to_rgb(fg))
        hi, lo = max(lb, lf), min(lb, lf)
        return (hi + 0.05) / (lo + 0.05)

    return light if contrast(light) > contrast(dark) else dark


def plot_dictionary_ablation(grid, row_values, col_values, chosen=None,
                             row_label=r"trigonometric harmonics  $N_h$", col_label=r"velocity degree  $P$",
                             title="EDMD dictionary ablation (validation rollout error)", path=None):
    """Heat map of a scalar (lower is better) over two dictionary hyper-parameters."""
    apply_style()
    grid = np.asarray(grid, float)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    im = ax.imshow(grid, cmap=SEQ_BLUE, aspect="auto", vmin=0.0, vmax=np.nanmax(grid))
    ax.set_xticks(range(len(col_values)))
    ax.set_xticklabels(col_values)
    ax.set_yticks(range(len(row_values)))
    ax.set_yticklabels(row_values)
    ax.grid(False)
    norm = plt.Normalize(0.0, np.nanmax(grid))
    for (r, c), v in np.ndenumerate(grid):
        ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=9,
                color=_best_text_color(SEQ_BLUE(norm(v))[:3]),
                fontweight="semibold" if chosen == (r, c) else "normal")
    if chosen is not None:
        r, c = chosen
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, ec=ORANGE, lw=2.6))
    ax.set_xlabel(col_label), ax.set_ylabel(row_label)
    ax.set_title(title)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("mean normalised error")
    cb.outline.set_edgecolor(SPINE)
    fig.tight_layout()
    return _finish(fig, path)
