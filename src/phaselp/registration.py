"""Seven-parameter similarity registration of a parent map onto a target grid (Section 5.1).

The forward transform maps a moving-map coordinate ``x`` to a fixed-map
coordinate ``y = s R (x - c) + c + t`` (scale ``s``, rotation ``R``,
translation ``t``, box centre ``c``; array order z, y, x). It is fitted by
L-BFGS-B to the normalized cross-correlation of band-limited copies
downsampled to at most ``max_dimension`` voxels, starting from the best of 24
principal-axis orientations (or the identity, if that scores higher), and then
applied once to the full-resolution map.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy import ndimage, optimize
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class Registration:
    """Fitted transform on the full grid and the NCC before and after fitting.

    ``parameters`` = ``[ln s, rotation vector (z, y, x) in rad, translation (z, y, x) in voxels]``.
    Scores are band-limited NCC values on the optimization grid.
    """

    parameters: np.ndarray
    score_before: float
    score_after: float
    success: bool
    message: str

    @property
    def scale(self) -> float:
        return float(np.exp(self.parameters[0]))

    @property
    def rotation_matrix(self) -> np.ndarray:
        return Rotation.from_rotvec(self.parameters[1:4]).as_matrix()

    @property
    def rotation_degrees(self) -> float:
        return float(np.degrees(np.linalg.norm(self.parameters[1:4])))

    @property
    def translation_voxels(self) -> np.ndarray:
        return self.parameters[4:7].copy()


def apply_similarity(
    moving: np.ndarray,
    parameters: np.ndarray,
    output_shape: tuple[int, int, int] | None = None,
    order: int = 1,
) -> np.ndarray:
    """Resample ``moving`` under the forward similarity transform ``parameters``."""
    p = np.asarray(parameters, dtype=float)
    if p.shape != (7,) or not np.all(np.isfinite(p)):
        raise ValueError("parameters must be a finite seven-element vector")
    output_shape = tuple(moving.shape) if output_shape is None else tuple(output_shape)
    inverse = Rotation.from_rotvec(p[1:4]).as_matrix().T / np.exp(p[0])
    centre_in = (np.asarray(moving.shape, dtype=float) - 1.0) / 2.0
    centre_out = (np.asarray(output_shape, dtype=float) - 1.0) / 2.0
    return ndimage.affine_transform(
        moving, inverse, offset=centre_in - inverse @ (centre_out + p[4:7]),
        output_shape=output_shape, order=order, mode="constant", cval=0.0,
        prefilter=order > 1,
    )


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    da, db = a - a.mean(), b - b.mean()
    denominator = float(np.sqrt(np.sum(da * da) * np.sum(db * db)))
    return float(np.sum(da * db)) / denominator if denominator > 0 else -1.0


def _band_limit(x: np.ndarray, voxel_size: float, max_frequency: float | None) -> np.ndarray:
    """Low-pass an optimization copy with a raised-cosine edge from 0.85 to 1 x max_frequency."""
    if max_frequency is None or not 0.0 < max_frequency < 0.5 / voxel_size:
        return x
    axes = [np.fft.fftfreq(n, d=voxel_size) for n in x.shape]
    zz, yy, xx = np.meshgrid(*axes, indexing="ij", sparse=True)
    radius = np.sqrt(zz**2 + yy**2 + xx**2)
    start = 0.85 * max_frequency
    weight = np.ones(x.shape, dtype=np.float32)
    weight[radius >= max_frequency] = 0.0
    edge = (radius > start) & (radius < max_frequency)
    weight[edge] = 0.5 * (1.0 + np.cos(np.pi * (radius[edge] - start) / (max_frequency - start)))
    return np.fft.ifftn(np.fft.fftn(x) * weight).real


def _phase_correlation_shift(reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """Integer translation of ``moving`` onto ``reference`` by phase correlation.

    Only coefficients that carry signal are whitened. In band-limited copies
    the stopband holds rounding noise, and normalizing it to unit magnitude
    would let that noise, which differs between machines, pick the peak.
    """
    product = np.fft.fftn(reference) * np.conj(np.fft.fftn(moving))
    magnitude = np.abs(product)
    signal = magnitude > 1e-8 * magnitude.max()
    product = np.where(signal, product / np.where(signal, magnitude, 1.0), 0.0)
    peak = np.asarray(np.unravel_index(np.argmax(np.abs(np.fft.ifftn(product))), reference.shape))
    shape = np.asarray(reference.shape)
    return np.where(peak > shape // 2, peak - shape, peak).astype(float)


def _principal_axes(volume: np.ndarray) -> np.ndarray:
    weights = np.maximum(volume - np.percentile(volume, 70.0), 0.0)
    if weights.sum() <= 0:
        weights = volume - volume.min()
    coords = np.indices(volume.shape, dtype=float).reshape(3, -1)
    w = weights.ravel()
    total = float(np.sum(w, dtype=np.float64))
    centred = coords - (np.einsum("in,n->i", coords, w, optimize=False) / total)[:, None]
    covariance = np.einsum("in,jn,n->ij", centred, centred, w, optimize=False) / total
    values, vectors = np.linalg.eigh(covariance)
    return vectors[:, np.argsort(values)[::-1]]


def principal_axis_rotations(fixed: np.ndarray, moving: np.ndarray) -> list[np.ndarray]:
    """The 24 proper rotations that map the principal axes of ``moving`` onto those of ``fixed``."""
    axes_fixed, axes_moving = _principal_axes(fixed), _principal_axes(moving)
    rotations = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            rotation = axes_fixed @ np.eye(3)[:, permutation] @ np.diag(signs) @ axes_moving.T
            if np.linalg.det(rotation) > 0:
                rotations.append(rotation)
    return rotations


def _fit(fixed, moving, *, scale_bounds, rotation_bound, translation_bound, maxiter, order,
         orientation_search="global"):
    def score(p):
        return _ncc(fixed, apply_similarity(moving, p, fixed.shape, order))

    # The identity (with a phase-correlation shift) is added to the 24 principal-axis
    # starts, so maps with nearly degenerate principal axes cannot be pulled into a
    # worse basin than the one they started in.
    identity = np.zeros(7)
    identity[4:7] = np.clip(_phase_correlation_shift(fixed, moving), -translation_bound, translation_bound)
    starts = [identity]
    rotations = principal_axis_rotations(fixed, moving) if orientation_search == "global" else []
    for rotation in rotations:
        p = np.zeros(7)
        p[1:4] = Rotation.from_matrix(rotation).as_rotvec()
        rotated = apply_similarity(moving, p, fixed.shape, order=1)
        p[4:7] = np.clip(_phase_correlation_shift(fixed, rotated), -translation_bound, translation_bound)
        starts.append(p)
    initial = max(starts, key=score)
    angle = np.deg2rad(rotation_bound)
    bounds = [(np.log(scale_bounds[0]), np.log(scale_bounds[1]))]
    bounds += [(v - angle, v + angle) for v in initial[1:4]]
    bounds += [(-translation_bound, translation_bound)] * 3
    initial = np.clip(initial, [b[0] for b in bounds], [b[1] for b in bounds])
    result = optimize.minimize(
        lambda p: -score(p), initial, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-6, "maxls": 30, "eps": 1e-3},
    )
    solution, best, message = np.asarray(result.x, dtype=float), -float(result.fun), str(result.message)
    start_score = score(initial)
    if start_score > best:
        solution, best, message = initial, start_score, message + "; kept the starting orientation"
    return solution, score(np.zeros(7)), best, bool(result.success), message


def register(
    fixed: np.ndarray,
    moving: np.ndarray,
    voxel_size: float,
    *,
    max_frequency: float | None = None,
    max_dimension: int = 96,
    scale_bounds: tuple[float, float] = (0.9, 1.1),
    rotation_bound: float = 20.0,
    translation_bound: float = 20.0,
    maxiter: int = 250,
    order: int = 1,
    orientation_search: str = "global",
) -> tuple[np.ndarray, Registration]:
    """Register ``moving`` onto ``fixed`` (both on the same grid); return ``(aligned, fit)``.

    Parameters
    ----------
    max_frequency
        Band limit of the NCC in 1/angstrom; use ``1 / target_res``.
    max_dimension
        Largest dimension of the downsampled optimization copies (96).
    scale_bounds, rotation_bound, translation_bound
        Scale range (0.9-1.1), rotation radius in degrees around the best
        principal-axis start (20) and translation bound in full-grid voxels (20).
    orientation_search
        ``"global"`` (default, Section 5.1) starts from the best-scoring of the
        24 principal-axis orientations and the identity. ``"local"`` starts from
        the identity only, for maps that are already roughly aligned.
    """
    if orientation_search not in ("global", "local"):
        raise ValueError("orientation_search must be 'global' or 'local'")
    fixed = np.asarray(fixed, dtype=float)
    moving = np.asarray(moving, dtype=float)
    if fixed.ndim != 3 or fixed.shape != moving.shape:
        raise ValueError("fixed and moving must be same-shaped 3-D maps; resample first")
    if fixed.std() == 0 or moving.std() == 0:
        raise ValueError("maps must have nonzero variance")
    if not 0 < scale_bounds[0] <= 1 <= scale_bounds[1]:
        raise ValueError("scale_bounds must be positive and contain 1")
    factor = min(1.0, max_dimension / max(fixed.shape))
    small_fixed, small_moving = fixed, moving
    if factor < 1.0:
        small_fixed = ndimage.zoom(fixed, factor, order=order)
        small_moving = ndimage.zoom(moving, factor, order=order)
    small_voxel = voxel_size / factor
    small_fixed = _band_limit(small_fixed, small_voxel, max_frequency)
    small_moving = _band_limit(small_moving, small_voxel, max_frequency)
    parameters, before, after, success, message = _fit(
        small_fixed, small_moving, scale_bounds=scale_bounds, rotation_bound=rotation_bound,
        translation_bound=translation_bound * factor, maxiter=maxiter, order=order,
        orientation_search=orientation_search,
    )
    parameters = parameters.copy()
    parameters[4:7] /= factor
    aligned = apply_similarity(moving, parameters, fixed.shape, order)
    return aligned, Registration(parameters, before, after, success, message)
