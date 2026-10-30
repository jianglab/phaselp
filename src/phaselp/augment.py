"""Ramped phase augmentation for training data (manuscript Section 5.4).

Equation (2) with unit amplitude response, ``T = 1``. Each augmented copy
keeps the power spectrum and the phases below the start frequency ``s0`` of
the original example, and receives a new high-frequency pattern. The offset
width rises smoothly from zero at ``s0`` to ``sigma_max`` at the end
frequency ``s1``:

    sigma(s) = sigma_max * t^2 * (3 - 2 t),   t = clip((s - s0) / (s1 - s0), 0, 1)

Frequencies are normalized, in cycles per sample (time series) or cycles per
pixel (images); each axis uses its own length.
"""

from __future__ import annotations

import numpy as np

from .fourier import apply_operator, normal_offsets, radial_frequency, resolve_ndim

#: Maximum offset width used in all augmentation experiments ("2.60 rad").
SIGMA_MAX = 2.6043009757995605


def ramp_sigma(s, s0, s1: float, sigma_max: float = SIGMA_MAX):
    """Offset width ``sigma(s)`` of the smoothstep ramp from ``s0`` to ``s1``.

    ``s0`` may be an array that broadcasts against ``s`` (one start frequency
    per example).
    """
    s0 = np.asarray(s0, dtype=float)
    if np.any(s0 < 0) or np.any(s0 >= s1):
        raise ValueError("start frequency s0 must satisfy 0 <= s0 < s1")
    t = np.clip((np.asarray(s, dtype=float) - s0) / (s1 - s0), 0.0, 1.0)
    return sigma_max * t * t * (3.0 - 2.0 * t)


def phase_augment(
    x: np.ndarray,
    s0,
    s1: float,
    *,
    sigma_max: float = SIGMA_MAX,
    ndim: int | None = None,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Return one ramped-phase copy of each example in ``x``.

    Parameters
    ----------
    x
        Real array whose last ``ndim`` axes are the signal (1 for time series,
        2 for images, 3 for volumes; default: all axes, a single example).
        Leading axes are a batch; every example gets its own offsets.
    s0, s1
        Start and end frequency of the ramp, in cycles per sample. ``s0`` is a
        scalar or an array with the batch shape (one start per example).
    sigma_max
        Offset width at and beyond ``s1``, in radians (default 2.60).
    seed
        Seed or ``numpy.random.Generator``.

    Fourier magnitudes, the DC term and all self-conjugate coefficients are
    preserved exactly.
    """
    x = np.asarray(x, dtype=float)
    ndim = resolve_ndim(x, ndim)
    shape, batch = x.shape[-ndim:], x.shape[:-ndim]
    s = radial_frequency(shape, 1.0)
    s0 = np.asarray(s0, dtype=float)
    if s0.ndim:
        if s0.shape != batch:
            raise ValueError(f"s0 must be a scalar or have the batch shape {batch}")
        s0 = s0.reshape(batch + (1,) * ndim)
    sigma = ramp_sigma(s, s0, s1, sigma_max)
    offsets = normal_offsets(sigma, shape, np.random.default_rng(seed), batch=batch)
    return apply_operator(x, 1.0, offsets, ndim)


class PhaseAugment:
    """Training-time ramped phase augmentation, as used in the experiments.

    Each call augments every example independently with probability ``p``
    (0.5 in the manuscript), drawing a new start frequency uniformly within
    ``+/- jitter`` (10%) of the nominal ``s0`` and new offsets. Apply it to
    training inputs only; test inputs are not augmented.

    Example
    -------
    >>> augment = PhaseAugment(s0=1/28, s1=0.25, ndim=1, seed=0)   # UCI traces
    >>> batch_out = augment(batch)                                  # (n, 60)
    """

    def __init__(
        self,
        s0: float,
        s1: float,
        *,
        sigma_max: float = SIGMA_MAX,
        p: float = 0.5,
        jitter: float = 0.1,
        ndim: int | None = None,
        seed: int | np.random.Generator | None = None,
    ):
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must lie in [0, 1]")
        if not 0.0 <= jitter < 1.0:
            raise ValueError("jitter must lie in [0, 1)")
        if not 0.0 <= s0 * (1.0 + jitter) < s1:
            raise ValueError("every sampled start frequency must lie in [0, s1)")
        if not np.isfinite(sigma_max) or sigma_max < 0:
            raise ValueError("sigma_max must be finite and nonnegative")
        self.s0, self.s1, self.sigma_max = float(s0), float(s1), float(sigma_max)
        self.p, self.jitter, self.ndim = float(p), float(jitter), ndim
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        ndim = resolve_ndim(x, self.ndim)
        if x.ndim == ndim:  # a single example
            return self._batch(x[None], ndim)[0]
        return self._batch(x, ndim)

    def _batch(self, x: np.ndarray, ndim: int) -> np.ndarray:
        batch = x.shape[:-ndim]
        events = self.rng.random(batch) < self.p
        starts = self.s0 * (1.0 + self.jitter * (2.0 * self.rng.random(batch) - 1.0))
        out = x.copy()
        if np.any(events):
            out[events] = phase_augment(
                x[events], starts[events], self.s1,
                sigma_max=self.sigma_max, ndim=ndim, seed=self.rng,
            )
        return out

    def __repr__(self) -> str:
        return (
            f"PhaseAugment(s0={self.s0:g}, s1={self.s1:g}, sigma_max={self.sigma_max:g}, "
            f"p={self.p:g}, jitter={self.jitter:g}, ndim={self.ndim})"
        )
