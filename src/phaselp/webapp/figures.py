"""Quality-control figures for each web-app step (matplotlib, no pyplot state)."""

from __future__ import annotations

import warnings

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

MEASURED, EXPECTED, ACCENT, SECOND = "#0072B2", "#8a8a8a", "#D55E00", "#009E73"
FREQ = "spatial frequency (1/Å)"


def _figure(width: float, height: float, rows: int = 1, cols: int = 1):
    fig = Figure(figsize=(width, height), layout="constrained")
    FigureCanvasAgg(fig)
    return fig, fig.subplots(rows, cols, squeeze=False)


def _section(volume: np.ndarray) -> np.ndarray:
    """Central section along the first axis, contrast limits at the 1st and 99.5th percentiles."""
    image = np.take(volume, volume.shape[0] // 2, axis=0)
    low, high = np.percentile(image, [1.0, 99.5])
    return np.clip((image - low) / (high - low), 0, 1) if high > low else np.zeros_like(image)


def _show(ax, volume: np.ndarray, title: str) -> None:
    ax.imshow(_section(volume), cmap="gray", vmin=0, vmax=1, origin="lower", interpolation="bilinear")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def _fsc_axis(ax, target_res: float | None, ylabel: str = "FSC") -> None:
    for level in (0.5, 0.143):
        ax.axhline(level, color="0.65", ls=":", lw=0.8)
    if target_res:
        ax.axvline(1.0 / target_res, color="0.35", ls="--", lw=0.9)
    ax.set(xlabel=FREQ, ylabel=ylabel, ylim=(-0.15, 1.05))
    ax.grid(alpha=0.15)


def _band(ax, frequency, curves, color, label):
    curves = np.atleast_2d(curves)
    with warnings.catch_warnings():  # the DC shell is NaN in every curve
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(curves, axis=0)
        sd = np.nanstd(curves, axis=0, ddof=1) if len(curves) > 1 else None
    ax.plot(frequency, mean, color=color, lw=1.8, label=label + (f" (mean of {len(curves)})" if len(curves) > 1 else ""))
    if sd is not None:
        ax.fill_between(frequency, mean - sd, mean + sd, color=color, alpha=0.2, lw=0)


def registration(low, before, after, frequency, fsc_before, fsc_after, align_res, fit):
    fig, axes = _figure(13.5, 3.6, 1, 4)
    axes = axes[0]
    _show(axes[0], low, "reference (working grid)")
    _show(axes[1], before, "parent, resampled")
    _show(axes[2], after, "parent, registered")
    ax = axes[3]
    ax.plot(frequency, fsc_before, color=EXPECTED, lw=1.5, label="before")
    ax.plot(frequency, fsc_after, color=MEASURED, lw=1.8, label="after")
    _fsc_axis(ax, align_res, "FSC with reference")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    if fit is None:
        text = "registration skipped"
    else:
        text = (f"NCC {fit.score_before:.3f} → {fit.score_after:.3f}\nscale {fit.scale:.4f}\n"
                f"rotation {fit.rotation_degrees:.2f}°\ntranslation "
                f"{np.round(fit.translation_voxels, 2).tolist()} vox")
    ax.text(0.03, 0.03, text, transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
    fig.suptitle("Step 1 · resample and register (Section 5.1)", fontsize=11)
    return fig


def sections(volume, title):
    fig, axes = _figure(9.0, 3.2, 1, 3)
    for axis, ax in enumerate(axes[0]):
        ax.imshow(_section(np.moveaxis(volume, axis, 0)), cmap="gray", origin="lower")
        ax.set_title(f"central {'zyx'[axis]} section", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=11)
    return fig


def filtering(frequency, fsc_parent_filtered, expected, target_res, mode_label, expected_label):
    fig, axes = _figure(5.6, 4.0)
    ax = axes[0, 0]
    _band(ax, frequency, fsc_parent_filtered, MEASURED, "FSC(parent, filtered)")
    ax.plot(frequency, expected, color=ACCENT, lw=1.5, ls="--", label=expected_label, zorder=3)
    _fsc_axis(ax, target_res)
    ax.set_xlim(0, float(np.nanmax(frequency)))
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title(f"{mode_label} · target {target_res:g} Å", fontsize=10)
    return fig


def amplitude(frequency, before, reference, after, gain, target_res):
    fig, axes = _figure(11.0, 4.0, 1, 2)
    ax, ax2 = axes[0]
    keep = frequency > 0   # DC is not matched (unit gain) and is ~0 for standardized maps
    frequency, before, reference, after, gain = (np.asarray(a)[keep] for a in (frequency, before, reference, after, gain))
    ax.semilogy(frequency, before, color=MEASURED, lw=1.6, label="filtered, before")
    ax.semilogy(frequency, reference, color=ACCENT, lw=1.6, label="reference")
    ax.semilogy(frequency, after, color=SECOND, lw=1.3, ls="--", label="filtered, matched")
    ax2.plot(frequency, gain, color="#555555", lw=1.6)
    ax2.axhline(1.0, color="0.6", ls=":", lw=0.9)
    for a, label in ((ax, "shell-mean |F|"), (ax2, "shell gain (capped at 5)")):
        a.axvline(1.0 / target_res, color="0.35", ls="--", lw=0.9)
        a.set(xlabel=FREQ, ylabel=label)
        a.grid(alpha=0.15)
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Step 3 · radial amplitude matching (Section 5.2)", fontsize=11)
    return fig


def comparison(frequency, fsc_parent_low, expected_parent_low, fsc_final_low, predicted_final_low,
               stage_label, target_res, difference, rms, sigma):
    fig, axes = _figure(11.0, 7.8, 2, 2)
    ax = axes[0, 0]
    ax.plot(frequency, fsc_parent_low, color=MEASURED, lw=1.8, label="measured")
    ax.plot(frequency, expected_parent_low, color="0.25", lw=1.4, ls="--", label="expected, eq. 8", zorder=3)
    _fsc_axis(ax, target_res)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("parent vs reference", fontsize=9)
    ax = axes[0, 1]
    _band(ax, frequency, fsc_final_low, SECOND, "measured")
    ax.plot(frequency, predicted_final_low, color="0.25", lw=1.4, ls="--", label="expected (Section S2)", zorder=3)
    _fsc_axis(ax, target_res)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title(f"{stage_label} vs reference", fontsize=9)
    section = np.take(difference, difference.shape[0] // 2, axis=0)
    limit = float(np.percentile(np.abs(difference), 99.5)) or 1.0
    image = axes[1, 0].imshow(section, cmap="RdBu_r", vmin=-limit, vmax=limit, origin="lower")
    fig.colorbar(image, ax=axes[1, 0], shrink=0.8)
    axes[1, 0].set_title(f"difference, {stage_label} − reference (one map)", fontsize=9)
    image = axes[1, 1].imshow(np.take(rms, rms.shape[0] // 2, axis=0), cmap="magma", vmin=0, origin="lower")
    fig.colorbar(image, ax=axes[1, 1], shrink=0.8)
    axes[1, 1].set_title(f"local rms of the difference (σ = {sigma:g} Å)", fontsize=9)
    for ax in axes[1]:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Step 4 · comparison (Sections 3.1, 3.2, 5.3)", fontsize=11)
    return fig


def summary(low, parent, final, stage_label, frequency, fsc_parent_low, fsc_final_low, expected_c,
            target_res, rms, sigma):
    fig, axes = _figure(13.5, 7.6, 2, 3)
    _show(axes[0, 0], low, "reference")
    _show(axes[0, 1], parent, "parent, registered")
    _show(axes[0, 2], final, f"{stage_label} (one map)")
    ax = axes[1, 0]
    ax.plot(frequency, fsc_parent_low, color=MEASURED, lw=1.8, label="parent vs reference")
    _band(ax, frequency, fsc_final_low, SECOND, f"{stage_label} vs reference")
    if expected_c is not None:
        ax.plot(frequency, expected_c, color=ACCENT, lw=1.3, ls="--", label="filter c(s)")
    _fsc_axis(ax, target_res)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    signed = np.take(final - low, low.shape[0] // 2, axis=0)
    limit = float(np.percentile(np.abs(signed), 99.5)) or 1.0
    axes[1, 1].imshow(signed, cmap="RdBu_r", vmin=-limit, vmax=limit, origin="lower")
    axes[1, 1].set_title("difference (one map)", fontsize=9)
    axes[1, 2].imshow(np.take(rms, rms.shape[0] // 2, axis=0), cmap="magma", vmin=0, origin="lower")
    axes[1, 2].set_title(f"local rms (σ = {sigma:g} Å)", fontsize=9)
    for ax in (axes[1, 1], axes[1, 2]):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("phaselp comparison summary", fontsize=12)
    return fig
