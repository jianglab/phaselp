"""Simple matplotlib views of the filter, its effect on maps and the augmentation.

Every function returns a ``matplotlib.figure.Figure``; save it with
``fig.savefig(path)``. Colours follow the manuscript figures.
"""

from __future__ import annotations

import warnings
from typing import Mapping, Sequence

import numpy as np

from .augment import SIGMA_MAX, ramp_sigma
from .compare import local_rms
from .envelope import KAPPA, coherence, envelope, sigma_from_coherence
from .filters import amplitude_only_filter, hard_cutoff_filter, ramped_phase_filter

GRAY, BLUE, RED, PURPLE, TEAL = "#62666b", "#2277b5", "#cb4545", "#8c59b5", "#168f8a"
_CYCLE = (PURPLE, BLUE, GRAY, RED, TEAL, "#d08a2e")


def _figure(rows: int, cols: int, width: float, height: float):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(rows, cols, figsize=(width, height), squeeze=False, layout="constrained")
    return fig, axes


def _section(volume: np.ndarray, axis: int, index: int | None) -> np.ndarray:
    volume = np.asarray(volume)
    if volume.ndim == 2:
        return volume
    index = volume.shape[axis] // 2 if index is None else index
    return np.take(volume, index, axis=axis)


def _limits(image: np.ndarray, low: float = 0.5, high: float = 99.5) -> tuple[float, float]:
    lo, hi = np.percentile(image, [low, high])
    return float(lo), float(hi if hi > lo else lo + 1.0)


def plot_sections(
    maps: Sequence[np.ndarray],
    titles: Sequence[str] | None = None,
    *,
    axis: int = 0,
    index: int | None = None,
    size: float = 2.6,
):
    """Central sections of several maps side by side (contrast per panel, for display only)."""
    fig, axes = _figure(1, len(maps), size * len(maps), size + 0.3)
    for k, (ax, volume) in enumerate(zip(axes[0], maps)):
        image = _section(volume, axis, index)
        lo, hi = _limits(image)
        ax.imshow(image, cmap="gray", vmin=lo, vmax=hi, origin="lower")
        ax.set_xticks([])
        ax.set_yticks([])
        if titles is not None:
            ax.set_title(titles[k], fontsize=10)
    return fig


def plot_fsc(
    curves: Mapping[str, np.ndarray],
    *,
    target_res: float | None = None,
    title: str | None = None,
    ax=None,
):
    """Plot FSC curves against frequency.

    ``curves`` maps labels to arrays and must contain ``"frequency"``. A 2-D
    array (seeds x shells) is drawn as mean +/- SD. An ``"expected"`` entry is
    drawn as a pale gray band, as in Figure 2. The dict returned by
    :meth:`phaselp.Comparison.fsc_curves` can be passed directly.
    """
    if ax is None:
        fig, axes = _figure(1, 1, 4.4, 3.2)
        ax = axes[0, 0]
    else:
        fig = ax.figure
    frequency = np.asarray(curves["frequency"])
    if "expected" in curves:
        ax.plot(frequency, curves["expected"], color="0.82", lw=6, solid_capstyle="butt",
                label="expected, c(s)", zorder=0)
    names = {"ramped_phase": "ramped phase", "amplitude_only": "amplitude only",
             "experimental": "experimental"}
    colours = iter(_CYCLE)
    for key, values in curves.items():
        if key in ("frequency", "expected"):
            continue
        values = np.asarray(values)
        colour = next(colours)
        label = names.get(key, key)
        if values.ndim == 2:
            with warnings.catch_warnings():  # the DC shell is NaN in every row
                warnings.simplefilter("ignore", RuntimeWarning)
                mean, sd = np.nanmean(values, axis=0), np.nanstd(values, axis=0)
            ax.fill_between(frequency, mean - sd, mean + sd, color=colour, alpha=0.25, lw=0)
            ax.plot(frequency, mean, color=colour, lw=1.6, label=f"{label} (n={len(values)})")
        else:
            ax.plot(frequency, values, color=colour, lw=1.6, label=label)
    if target_res is not None:
        ax.axvline(1.0 / target_res, color="0.3", ls="--", lw=0.9)
    ax.axhline(0.5, color="0.6", ls=":", lw=0.8)
    ax.set(xlabel="Spatial frequency (1/Å)", ylabel="FSC", ylim=(-0.1, 1.05), xlim=(0, frequency[-1]))
    ax.legend(frameon=False, fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    return fig


def plot_filter_profile(
    target_res: float,
    parent_res: float | None = None,
    *,
    pixel_size: float = 1.0,
    kappa: float = KAPPA,
    unit: str = "1/Å",
):
    """The filter design: envelopes and coherence c(s) (equations 5, 7) and offset width sigma(s) (equation 6)."""
    s = np.linspace(0.0, 0.5 / pixel_size, 600)
    c = coherence(s, target_res, parent_res, kappa)
    fig, axes = _figure(1, 2, 8.0, 3.0)
    ax = axes[0, 0]
    ax.plot(s, envelope(s, target_res, kappa), color=GRAY, ls="--", lw=1.2, label=f"a_T (R = {target_res:g})")
    if parent_res:
        ax.plot(s, envelope(s, parent_res, kappa), color=BLUE, ls="--", lw=1.2, label=f"a_H (R = {parent_res:g})")
    ax.plot(s, c, color=PURPLE, lw=2, label="c(s) = T(s)")
    ax.axvline(1.0 / target_res, color="0.4", ls=":", lw=0.9)
    ax.axhline(0.5, color="0.7", ls=":", lw=0.8)
    ax.set(xlabel=f"s ({unit})", ylabel="coherence", ylim=(-0.02, 1.05))
    ax.legend(frameon=False, fontsize=8)
    ax = axes[0, 1]
    ax.plot(s, sigma_from_coherence(c), color=PURPLE, lw=2)
    ax.axvline(1.0 / target_res, color="0.4", ls=":", lw=0.9)
    ax.set(xlabel=f"s ({unit})", ylabel="offset SD σ(s) (rad)")
    return fig


def plot_differences(
    differences: Mapping[str, np.ndarray],
    *,
    sigma: float,
    voxel_size: float = 1.0,
    axis: int = 0,
    index: int | None = None,
    mask: np.ndarray | None = None,
):
    """Signed difference and local rms sections, one row per comparison (Figure 3 D, E, H, I).

    Rows share colour scales, so a larger, localized difference stands out
    against a same-state control.
    """
    sections = {k: _section(d, axis, index) for k, d in differences.items()}
    rms = {k: _section(local_rms(d, sigma, voxel_size), axis, index) for k, d in differences.items()}
    signed = max(float(np.percentile(np.abs(v), 99.5)) for v in sections.values())
    top = max(float(np.percentile(v, 99.5)) for v in rms.values())
    fig, axes = _figure(len(differences), 2, 5.6, 2.7 * len(differences))
    for row, key in enumerate(differences):
        a = axes[row, 0].imshow(sections[key], cmap="RdBu_r", vmin=-signed, vmax=signed, origin="lower")
        b = axes[row, 1].imshow(rms[key], cmap="magma", vmin=0, vmax=top, origin="lower")
        axes[row, 0].set_ylabel(key)
        if mask is not None:
            for ax in axes[row]:
                ax.contour(_section(mask, axis, index), levels=[0.5], colors="0.5", linewidths=0.6)
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_title("difference", fontsize=10)
    axes[0, 1].set_title(f"local rms (σ = {sigma:g} Å)", fontsize=10)
    fig.colorbar(a, ax=axes[:, 0], shrink=0.7)
    fig.colorbar(b, ax=axes[:, 1], shrink=0.7)
    return fig


def plot_augmentation(
    x: np.ndarray,
    augmented: np.ndarray,
    s0: float,
    s1: float,
    *,
    sigma_max: float = SIGMA_MAX,
    count: int = 4,
):
    """Original and augmented examples (1-D traces or 2-D images) and the sigma(s) ramp."""
    x, augmented = np.atleast_2d(x), np.atleast_2d(augmented)
    images = x.ndim == 3
    count = min(count, len(x))
    s = np.linspace(0.0, 0.5, 400)
    if images:
        fig, axes = _figure(2, count + 1, 2.0 * (count + 1), 4.2)
        for k in range(count):
            for row, data in enumerate((x, augmented)):
                lo, hi = _limits(data[k])
                axes[row, k].imshow(data[k], cmap="gray", vmin=lo, vmax=hi)
                axes[row, k].set_xticks([])
                axes[row, k].set_yticks([])
        axes[0, 0].set_ylabel("original")
        axes[1, 0].set_ylabel("augmented")
        ramp_ax = axes[0, count]
        axes[1, count].axis("off")
    else:
        fig, axes = _figure(1, 2, 8.5, 3.0)
        ax = axes[0, 0]
        offset = 0.0
        step = 1.2 * float(np.nanmax(np.abs(x[:count]))) * 2
        for k in range(count):
            ax.plot(x[k] + offset, color=GRAY, lw=1.4)
            ax.plot(augmented[k] + offset, color=TEAL, lw=1.0)
            offset -= step
        ax.set(xlabel="sample", yticks=[])
        ax.plot([], [], color=GRAY, label="original")
        ax.plot([], [], color=TEAL, label="augmented")
        ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
        ramp_ax = axes[0, 1]
    ramp_ax.plot(s, ramp_sigma(s, s0, s1, sigma_max), color=TEAL, lw=2)
    ramp_ax.set(xlabel="s (cycles/sample)", ylabel="σ(s) (rad)", xlim=(0, 0.5))
    return fig


def plot_signal_filters(
    signal: np.ndarray,
    target_res: float,
    *,
    parent_res: float | None = None,
    seed: int | None = 0,
):
    """Amplitude-only, hard-cutoff and ramped-phase filtering of a 1-D signal (Figure 1B)."""
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1:
        raise ValueError("signal must be 1-D")
    rows = [
        ("original", signal, GRAY),
        ("amplitude only", amplitude_only_filter(signal, 1.0, target_res, parent_res), BLUE),
        ("hard cutoff", hard_cutoff_filter(signal, 1.0, target_res), RED),
        ("ramped phase", ramped_phase_filter(signal, 1.0, target_res, parent_res, seed=seed), PURPLE),
    ]
    frequency = np.fft.rfftfreq(signal.size)
    fig, axes = _figure(len(rows), 2, 9.0, 1.6 * len(rows) + 0.6)
    lo, hi = float(signal.min()), float(signal.max())
    pad = 0.15 * (hi - lo or 1.0)
    for row, (name, y, colour) in enumerate(rows):
        axes[row, 0].plot(y, color=colour, lw=1.2)
        axes[row, 0].set_ylim(lo - pad, hi + pad)
        axes[row, 0].set_ylabel(name, color=colour)
        magnitude = np.abs(np.fft.rfft(y)) / signal.size
        axes[row, 1].semilogy(frequency, np.where(magnitude > 1e-10, magnitude, np.nan), color=colour, lw=1.2)
        axes[row, 1].axvline(1.0 / target_res, color="0.5", ls=":", lw=0.9)
        axes[row, 1].set_xlim(0.0, 0.5)
        if row < len(rows) - 1:
            axes[row, 0].tick_params(labelbottom=False)
            axes[row, 1].tick_params(labelbottom=False)
    axes[-1, 0].set_xlabel("position (samples)")
    axes[-1, 1].set_xlabel("s (cycles/sample)")
    axes[0, 1].set_title("Fourier magnitude", fontsize=10)
    return fig
