"""Low-pass filters of equation (2) for signals, images and density maps.

``ramped_phase_filter`` is the method of the manuscript: amplitude response
``T = c`` plus wrapped-normal phase offsets whose coherence ``c(s)`` follows
equation (5), or equation (7) when the parent resolution is given.

The three comparison filters used in the manuscript are also provided:
``amplitude_only_filter`` (same ``T``, no offsets; the control of Figure 2),
``hard_cutoff_filter`` (band limitation; Figure 1) and
``binary_phase_filter`` (offsets uniform on the circle beyond ``1/R`` only,
as in high-resolution noise substitution; Section 2.2).

All filters leave the DC coefficient unchanged and keep the output real.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .envelope import COHERENCE_FLOOR, KAPPA, coherence, sigma_from_coherence
from .fourier import (
    apply_operator,
    make_antisymmetric,
    normal_offsets,
    radial_frequency,
    resolve_ndim,
    rfft_shape,
)


class FilterProfile(NamedTuple):
    """The filter on the rFFT grid: frequency ``s``, coherence ``c``, gain ``T``, width ``sigma``."""

    frequency: np.ndarray
    coherence: np.ndarray
    gain: np.ndarray
    sigma: np.ndarray


def filter_profile(
    shape: tuple[int, ...],
    pixel_size: float,
    target_res: float,
    parent_res: float | None = None,
    *,
    kappa: float = KAPPA,
    coherence_curve: tuple[np.ndarray, np.ndarray] | None = None,
) -> FilterProfile:
    """Coherence, amplitude response and offset width on the rFFT grid of ``shape``.

    ``coherence_curve`` replaces the model envelope by a measured profile,
    given as ``(frequency, coherence)`` 1-D arrays (Section 2.2); it is
    interpolated onto the grid and taken as zero beyond its last frequency.
    """
    if not np.isfinite(pixel_size) or pixel_size <= 0:
        raise ValueError("pixel_size must be finite and positive")
    if target_res < 2.0 * pixel_size:
        raise ValueError(
            f"target_res {target_res} is finer than the Nyquist resolution {2.0 * pixel_size}"
        )
    s = radial_frequency(shape, pixel_size)
    if coherence_curve is None:
        c = np.asarray(coherence(s, target_res, parent_res, kappa), dtype=float)
    else:
        freq = np.asarray(coherence_curve[0], dtype=float)
        values = np.clip(np.asarray(coherence_curve[1], dtype=float), 0.0, 1.0)
        if freq.ndim != 1 or freq.shape != values.shape or freq.size < 2:
            raise ValueError("coherence_curve must be two 1-D arrays of equal length >= 2")
        c = np.interp(s, freq, values, left=float(values[0]), right=0.0)
    c = np.array(c, dtype=float)
    c.flat[0] = 1.0  # DC is excluded from filtering
    gain = np.where(c < COHERENCE_FLOOR, 0.0, c)
    sigma = sigma_from_coherence(c)
    return FilterProfile(s, c, gain, sigma)


def _rng(seed) -> np.random.Generator:
    return np.random.default_rng(seed)


def ramped_phase_filter(
    x: np.ndarray,
    pixel_size: float,
    target_res: float,
    parent_res: float | None = None,
    *,
    kappa: float = KAPPA,
    seed: int | np.random.Generator | None = None,
    coherence_curve: tuple[np.ndarray, np.ndarray] | None = None,
    ndim: int | None = None,
) -> np.ndarray:
    """Low-pass filter with ramped phase randomization (equations 2, 5-7).

    Parameters
    ----------
    x
        Real signal, image or map. The last ``ndim`` axes are filtered
        (default: all); leading axes are a batch, each with its own offsets.
    pixel_size
        Sample spacing (angstrom per voxel for maps, 1 for signals in samples).
    target_res
        Target resolution ``R_T``, in the unit of ``pixel_size``.
    parent_res
        Resolution ``R_H`` of the parent. When given, equation (7) corrects
        for the parent's own loss of correlation; otherwise the parent is
        treated as ideal.
    kappa
        SSNR scale of equation (5) (default 100).
    seed
        Seed or ``numpy.random.Generator`` for the phase offsets. Each call
        returns one realization; use the filtered map as it is, not an average
        of several (averaging restores the parent's phases).
    coherence_curve
        Optional measured ``(frequency, coherence)`` profile used instead of
        the model envelope.
    """
    x = np.asarray(x, dtype=float)
    ndim = resolve_ndim(x, ndim)
    shape = x.shape[-ndim:]
    profile = filter_profile(
        shape, pixel_size, target_res, parent_res, kappa=kappa, coherence_curve=coherence_curve
    )
    offsets = normal_offsets(profile.sigma, shape, _rng(seed), batch=x.shape[:-ndim])
    return apply_operator(x, profile.gain, offsets, ndim)


def amplitude_only_filter(
    x: np.ndarray,
    pixel_size: float,
    target_res: float,
    parent_res: float | None = None,
    *,
    kappa: float = KAPPA,
    coherence_curve: tuple[np.ndarray, np.ndarray] | None = None,
    ndim: int | None = None,
) -> np.ndarray:
    """Amplitude response ``T = c`` of :func:`ramped_phase_filter`, without phase offsets.

    Every retained phase of the parent is kept, so the FSC with the parent
    stays at 1 (Section 2.1). This is the amplitude-only control of Figure 2.
    """
    x = np.asarray(x, dtype=float)
    ndim = resolve_ndim(x, ndim)
    profile = filter_profile(
        x.shape[-ndim:], pixel_size, target_res, parent_res,
        kappa=kappa, coherence_curve=coherence_curve,
    )
    return apply_operator(x, profile.gain, None, ndim)


def hard_cutoff_filter(
    x: np.ndarray,
    pixel_size: float,
    target_res: float,
    *,
    ndim: int | None = None,
) -> np.ndarray:
    """Band limitation: keep coefficients with ``s <= 1/R``, remove the rest.

    Removes amplitude and phase beyond the cutoff but rings in real space
    (Figure 1).
    """
    x = np.asarray(x, dtype=float)
    ndim = resolve_ndim(x, ndim)
    if target_res < 2.0 * pixel_size:
        raise ValueError("target_res is finer than the Nyquist resolution")
    s = radial_frequency(x.shape[-ndim:], pixel_size)
    return apply_operator(x, (s <= 1.0 / target_res).astype(float), None, ndim)


def binary_phase_filter(
    x: np.ndarray,
    pixel_size: float,
    target_res: float,
    parent_res: float | None = None,
    *,
    kappa: float = KAPPA,
    seed: int | np.random.Generator | None = None,
    ndim: int | None = None,
) -> np.ndarray:
    """Binary phase randomization, for comparison (Section 2.2).

    Uses the amplitude response of equation (5) (or 7) but keeps every phase
    up to ``1/R`` and draws offsets uniformly on the circle beyond it, as in
    high-resolution noise substitution (Chen et al., 2013). The expected
    filtered signal then drops abruptly at ``1/R`` and rings.
    """
    x = np.asarray(x, dtype=float)
    ndim = resolve_ndim(x, ndim)
    shape = x.shape[-ndim:]
    profile = filter_profile(shape, pixel_size, target_res, parent_res, kappa=kappa)
    full = x.shape[:-ndim] + rfft_shape(shape)
    offsets = _rng(seed).uniform(-np.pi, np.pi, full)
    offsets *= profile.frequency >= 1.0 / target_res
    offsets = make_antisymmetric(np.ascontiguousarray(offsets), shape)
    return apply_operator(x, profile.gain, offsets, ndim)
