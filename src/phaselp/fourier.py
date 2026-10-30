"""The Fourier operator of equation (2), shared by the filters and the augmentation.

    F_H'(k) = T(s) exp(i delta(k)) F_H(k),     s = |k|

The operator works on the real-input FFT (``rfftn``) over the last ``ndim``
axes of an array, so it applies to time series (``ndim=1``), images
(``ndim=2``) and density maps (``ndim=3``); any leading axes are a batch.
Phase offsets are drawn once per independent conjugate pair and copied with
opposite sign to the partner, so the output stays real. Self-conjugate
coefficients, including DC, are left unperturbed.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


def radial_frequency(shape: tuple[int, ...], spacing: float = 1.0) -> np.ndarray:
    """Frequency magnitude ``s = |k|`` on the ``rfftn`` grid of a real array.

    ``spacing`` is the sample spacing: the voxel size in angstrom for maps
    (``s`` in 1/angstrom), or 1 for normalized frequency in cycles per sample
    or pixel.
    """
    shape = tuple(int(n) for n in shape)
    if not shape or min(shape) < 1:
        raise ValueError("shape must contain positive sizes")
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be finite and positive")
    axes = [np.fft.fftfreq(n, d=spacing) for n in shape[:-1]]
    axes.append(np.fft.rfftfreq(shape[-1], d=spacing))
    squared = 0.0
    for grid in np.meshgrid(*axes, indexing="ij", sparse=True):
        squared = squared + grid**2
    return np.sqrt(np.broadcast_to(squared, rfft_shape(shape)))


def rfft_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Shape of ``np.fft.rfftn`` output for a real array of ``shape``."""
    return tuple(shape[:-1]) + (shape[-1] // 2 + 1,)


@lru_cache(maxsize=32)
def _pairing(shape: tuple[int, ...]):
    """Conjugate pairing inside the rFFT planes that hold both k and -k.

    Only the last-axis planes 0 and (for even length) n/2 contain both
    members of a conjugate pair; elsewhere the partner lies in the omitted
    half-space. Returns, for the flattened leading axes, the canonical
    indices, their partners and the self-conjugate indices, plus the
    boundary columns.
    """
    lead = shape[:-1]
    count = int(np.prod(lead)) if lead else 1
    index = np.arange(count)
    if lead:
        coords = np.unravel_index(index, lead)
        partner = np.ravel_multi_index(
            tuple((-c) % n for c, n in zip(coords, lead)), lead
        )
    else:
        partner = index
    canonical = index[index < partner]
    columns = (0, shape[-1] // 2) if shape[-1] % 2 == 0 else (0,)
    return canonical, partner[canonical], index[index == partner], columns


def make_antisymmetric(offsets: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Make rFFT phase offsets odd under k -> -k, in place, and return them.

    ``offsets`` has shape ``batch + rfft_shape(shape)``. Each partner receives
    the negated offset of its canonical coefficient; self-conjugate
    coefficients receive zero.
    """
    shape = tuple(int(n) for n in shape)
    if not offsets.flags.c_contiguous:
        raise ValueError("offsets must be C-contiguous")
    canonical, partner, self_conjugate, columns = _pairing(shape)
    batch = offsets.shape[: offsets.ndim - len(shape)]
    lead_count = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
    flat = offsets.reshape(batch + (lead_count, shape[-1] // 2 + 1))  # a view
    for column in columns:
        plane = flat[..., column]
        plane[..., partner] = -plane[..., canonical]
        plane[..., self_conjugate] = 0.0
    return offsets


def normal_offsets(
    sigma: np.ndarray,
    shape: tuple[int, ...],
    rng: np.random.Generator,
    batch: tuple[int, ...] = (),
) -> np.ndarray:
    """Draw ``delta ~ N(0, sigma^2)`` per conjugate pair (equation 6).

    ``sigma`` broadcasts against ``batch + rfft_shape(shape)``.
    """
    full = tuple(batch) + rfft_shape(shape)
    offsets = rng.standard_normal(full) * np.broadcast_to(sigma, full)
    return make_antisymmetric(np.ascontiguousarray(offsets), shape)


def apply_operator(
    x: np.ndarray,
    gain: np.ndarray | float,
    offsets: np.ndarray | None,
    ndim: int,
) -> np.ndarray:
    """Return ``irfftn(T * exp(i delta) * rfftn(x))`` over the last ``ndim`` axes."""
    axes = tuple(range(-ndim, 0))
    shape = x.shape[-ndim:]
    spectrum = np.fft.rfftn(x, axes=axes)
    spectrum *= gain
    if offsets is not None:
        spectrum *= np.exp(1j * offsets)
    return np.fft.irfftn(spectrum, s=shape, axes=axes)


def resolve_ndim(x: np.ndarray, ndim: int | None) -> int:
    """Number of trailing signal axes (default: all axes)."""
    ndim = x.ndim if ndim is None else int(ndim)
    if ndim < 1 or ndim > 3 or ndim > x.ndim:
        raise ValueError(f"ndim must be 1, 2 or 3 and at most x.ndim; got {ndim}")
    if min(x.shape[-ndim:]) < 2:
        raise ValueError("each signal axis needs at least two samples")
    return ndim
