"""Cross-resolution map comparison (manuscript Sections 3.1, 3.2 and 5.1-5.3).

``compare_maps`` brings a higher-resolution parent to the state of a
lower-resolution experimental map in three steps: registration onto the
experimental grid, ramped phase filtering to the target resolution
(equation 7) and radial amplitude matching to the experimental map. It also
returns the amplitude-only control. The functions below it compute the
difference maps and the statistics reported in Section 3.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy import ndimage

from .envelope import KAPPA, coherence
from .filters import amplitude_only_filter, ramped_phase_filter
from .fsc import fsc, match_radial_amplitude
from .maps import resample_to_grid, standardize
from .registration import Registration, register


@dataclass
class Comparison:
    """Maps produced by :func:`compare_maps`, all on the experimental grid."""

    parent: np.ndarray
    experimental: np.ndarray
    filtered: list[np.ndarray]
    amplitude_only: np.ndarray
    voxel_size: float
    target_res: float
    parent_res: float | None
    seeds: tuple[int, ...]
    kappa: float = KAPPA
    registration: Registration | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def fsc_curves(self) -> dict[str, np.ndarray]:
        """FSCs with the registered parent, as in the last column of Figure 2.

        Keys: ``frequency``; ``ramped_phase`` (seeds x shells);
        ``amplitude_only``; ``experimental`` (parent vs experimental map);
        ``expected``, the coherence ``c(s)`` of equation (7), which is the
        expected FSC of each ramped-phase map (equation 4).
        """
        frequency, experimental = fsc(self.parent, self.experimental, self.voxel_size)
        ramped = np.vstack([fsc(self.parent, f, self.voxel_size)[1] for f in self.filtered])
        _, amplitude = fsc(self.parent, self.amplitude_only, self.voxel_size)
        expected = coherence(frequency, self.target_res, self.parent_res, self.kappa)
        return {
            "frequency": frequency,
            "ramped_phase": ramped,
            "amplitude_only": amplitude,
            "experimental": experimental,
            "expected": np.asarray(expected),
        }

    def differences(self) -> list[np.ndarray]:
        """Filtered minus experimental map, one per seed (Section 5.3)."""
        return [difference_map(f, self.experimental) for f in self.filtered]


def compare_maps(
    parent: np.ndarray,
    experimental: np.ndarray,
    *,
    parent_voxel_size: float,
    voxel_size: float,
    target_res: float,
    parent_res: float | None = None,
    seeds: Sequence[int] = (0,),
    kappa: float = KAPPA,
    register_maps: bool = True,
    translation_bound: float = 20.0,
    register_options: dict[str, Any] | None = None,
    match_amplitude: bool = True,
    max_gain: float = 5.0,
    standardize_inputs: bool = True,
    antialias: bool = True,
    order: int = 1,
) -> Comparison:
    """Filter a high-resolution parent to the resolution of an experimental map.

    Parameters
    ----------
    parent, experimental
        3-D maps (z, y, x). The experimental map defines the output grid.
    parent_voxel_size, voxel_size
        Voxel sizes of the parent and of the experimental map, in angstrom.
    target_res
        Target resolution ``R_T`` (the experimental map's resolution).
    parent_res
        Parent resolution ``R_H`` for equation (7); ``None`` treats the
        parent as ideal.
    seeds
        One ramped-phase map is made per seed. Use each map on its own;
        average statistics over seeds, never maps.
    register_maps
        Fit the similarity transform (Section 5.1). Set to ``False`` when the
        parent is already aligned to the experimental map.
    translation_bound
        Translation bound in voxels of the experimental grid (20; 24 was used
        for the 70S ribosome).
    register_options
        Further keyword arguments for :func:`phaselp.register`.
    match_amplitude, max_gain
        Radial amplitude matching to the experimental map (Section 5.2).
    standardize_inputs
        Standardize both maps to zero mean and unit variance (Section 5.3).
    antialias
        Low-pass the parent at the output Nyquist frequency before downsampling
        it onto the experimental grid. ``False`` reproduces the plain linear
        resampling used for the manuscript's 70S comparison.
    """
    parent = np.asarray(parent, dtype=float)
    experimental = np.asarray(experimental, dtype=float)
    if parent.ndim != 3 or experimental.ndim != 3:
        raise ValueError("compare_maps expects 3-D maps")
    if len(seeds) < 1:
        raise ValueError("at least one seed is required")
    if standardize_inputs:
        experimental = standardize(experimental)
    if parent.shape != experimental.shape or not np.isclose(parent_voxel_size, voxel_size):
        parent = resample_to_grid(parent, parent_voxel_size, experimental.shape, voxel_size, order, antialias)
    if standardize_inputs:
        parent = standardize(parent)

    registration = None
    if register_maps:
        options = {"max_frequency": 1.0 / target_res, "translation_bound": translation_bound,
                   "order": order, **(register_options or {})}
        parent, registration = register(experimental, parent, voxel_size, **options)

    def finish(x: np.ndarray) -> np.ndarray:
        if match_amplitude:
            return match_radial_amplitude(x, experimental, voxel_size, max_gain)
        return x

    filtered = [
        finish(ramped_phase_filter(parent, voxel_size, target_res, parent_res, kappa=kappa, seed=int(s)))
        for s in seeds
    ]
    amplitude_only = finish(amplitude_only_filter(parent, voxel_size, target_res, parent_res, kappa=kappa))
    return Comparison(
        parent=parent, experimental=experimental, filtered=filtered,
        amplitude_only=amplitude_only, voxel_size=float(voxel_size),
        target_res=float(target_res), parent_res=parent_res,
        seeds=tuple(int(s) for s in seeds), kappa=float(kappa), registration=registration,
        options={"match_amplitude": match_amplitude, "max_gain": max_gain,
                 "standardize_inputs": standardize_inputs, "register_maps": register_maps},
    )


# --------------------------------------------------------------------------
# Difference maps and residual statistics (Section 5.3)
# --------------------------------------------------------------------------


def difference_map(filtered: np.ndarray, experimental: np.ndarray) -> np.ndarray:
    """Signed difference, filtered minus experimental map."""
    filtered = np.asarray(filtered, dtype=float)
    experimental = np.asarray(experimental, dtype=float)
    if filtered.shape != experimental.shape:
        raise ValueError("maps must have the same shape")
    return filtered - experimental


def local_rms(difference: np.ndarray, sigma: float, voxel_size: float = 1.0) -> np.ndarray:
    """Local rms of a difference map: ``sqrt(G_sigma * d^2)``.

    ``sigma`` is the standard deviation of the Gaussian kernel in angstrom
    (the target resolution, 13.2 angstrom, in Figure 3).
    """
    d = np.asarray(difference, dtype=float)
    return np.sqrt(np.maximum(ndimage.gaussian_filter(d * d, sigma / voxel_size), 0.0))


def union_mask(
    *maps: np.ndarray,
    threshold_sigma: float = 0.25,
    soft_sigma: float = 2.0,
    smooth: float = 1.0,
) -> np.ndarray:
    """Soft mask covering all ``maps`` (the shared mask of Section 5.3).

    Each map is standardized and smoothed by ``smooth`` voxels; their
    voxelwise maximum is thresholded at ``mean + threshold_sigma * std``,
    holes are filled, and the edge is softened by a Gaussian of
    ``soft_sigma`` voxels. Values lie in [0, 1]; statistics use mask > 0.5.
    """
    if not maps:
        raise ValueError("give at least one map")
    source = np.maximum.reduce([ndimage.gaussian_filter(standardize(m), smooth) for m in maps])
    hard = ndimage.binary_fill_holes(source > source.mean() + threshold_sigma * source.std())
    soft = ndimage.gaussian_filter(hard.astype(float), soft_sigma)
    if soft.max() > 0:
        soft /= soft.max()
    return np.clip(soft, 0.0, 1.0)


def residual_variance(difference: np.ndarray, mask: np.ndarray | None = None, threshold: float = 0.5) -> float:
    """Variance of a difference map inside ``mask > threshold`` (everywhere without a mask)."""
    d = np.asarray(difference, dtype=float)
    if mask is None:
        return float(d.var())
    mask = np.asarray(mask)
    if mask.shape != d.shape:
        raise ValueError("mask must have the same shape as the difference map")
    return float(d[mask > threshold].var())


def variance_ratio(
    difference: np.ndarray,
    control_difference: np.ndarray,
    mask: np.ndarray | None = None,
    threshold: float = 0.5,
) -> float:
    """Residual variance of a comparison relative to a same-state control (Section 3.2)."""
    return residual_variance(difference, mask, threshold) / residual_variance(control_difference, mask, threshold)

