"""Fourier shell correlation (equation 1) and radial amplitude matching (Section 5.2).

Shells are one reciprocal-sampling step wide, ``1 / (N * pixel_size)`` with
``N`` the largest box dimension, and run to the Nyquist frequency. Sums use
the real-input FFT with each coefficient weighted by the number of
full-spectrum coefficients it stands for, which gives the same result as a
full FFT at half the memory.
"""

from __future__ import annotations

import numpy as np

from .fourier import radial_frequency


def _shells(shape: tuple[int, ...], pixel_size: float):
    if not np.isfinite(pixel_size) or pixel_size <= 0:
        raise ValueError("pixel_size must be finite and positive")
    step = 1.0 / (max(shape) * pixel_size)
    count = max(shape) // 2 + 1
    labels = np.rint(radial_frequency(shape, pixel_size) / step).astype(np.int64)
    weight = np.full(shape[-1] // 2 + 1, 2.0)
    weight[0] = 1.0
    if shape[-1] % 2 == 0:
        weight[-1] = 1.0
    weight = np.broadcast_to(weight, labels.shape)
    inside = labels < count
    return labels[inside], weight[inside], inside, count, np.arange(count) * step


def _shell_sum(labels, values, count):
    return np.bincount(labels, weights=values, minlength=count)[:count]


def _check_pair(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"arrays must have the same shape, got {x.shape} and {y.shape}")
    return x, y


def fsc(x: np.ndarray, y: np.ndarray, pixel_size: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """FSC between two same-shaped arrays (equation 1), without masks.

    Returns ``(frequency, fsc)``. The DC shell is excluded (NaN), and so is
    any shell where either map has no power.
    """
    x, y = _check_pair(x, y)
    labels, weight, inside, count, frequency = _shells(x.shape, pixel_size)
    fx = np.fft.rfftn(x)[inside]
    fy = np.fft.rfftn(y)[inside]
    cross = _shell_sum(labels, weight * np.real(fx * np.conj(fy)), count)
    power_x = _shell_sum(labels, weight * np.abs(fx) ** 2, count)
    power_y = _shell_sum(labels, weight * np.abs(fy) ** 2, count)
    denominator = np.sqrt(power_x * power_y)
    values = np.full(count, np.nan)
    good = denominator > 0
    values[good] = cross[good] / denominator[good]
    values[0] = np.nan
    return frequency, values


def radial_amplitude(x: np.ndarray, pixel_size: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Shell-mean Fourier magnitude ``<|F|>`` of an array; returns ``(frequency, amplitude)``."""
    x = np.asarray(x, dtype=float)
    labels, weight, inside, count, frequency = _shells(x.shape, pixel_size)
    magnitude = np.abs(np.fft.rfftn(x)[inside])
    totals = _shell_sum(labels, weight * magnitude, count)
    counts = _shell_sum(labels, weight, count)
    amplitude = np.full(count, np.nan)
    amplitude[counts > 0] = totals[counts > 0] / counts[counts > 0]
    return frequency, amplitude


def match_radial_amplitude(
    moving: np.ndarray,
    reference: np.ndarray,
    pixel_size: float = 1.0,
    max_gain: float = 5.0,
) -> np.ndarray:
    """Rescale each Fourier shell of ``moving`` to the shell-mean magnitude of ``reference``.

    The gain of each shell is the ratio of shell-mean magnitudes, capped at
    ``max_gain`` (5 in the manuscript), with unit gain at DC and beyond the
    last complete shell. Phases are unchanged, so the FSC with any other map
    is unchanged too.
    """
    moving, reference = _check_pair(moving, reference)
    _, moving_profile = radial_amplitude(moving, pixel_size)
    _, reference_profile = radial_amplitude(reference, pixel_size)
    gain = np.ones_like(moving_profile)
    good = np.isfinite(moving_profile) & (moving_profile > 0)
    gain[good] = reference_profile[good] / moving_profile[good]
    gain = np.clip(gain, 0.0, max_gain)
    gain[0] = 1.0
    labels, _, inside, count, _ = _shells(moving.shape, pixel_size)
    gain_grid = np.ones(inside.shape)
    gain_grid[inside] = gain[labels]
    return np.fft.irfftn(np.fft.rfftn(moving) * gain_grid, s=moving.shape, axes=tuple(range(moving.ndim)))
