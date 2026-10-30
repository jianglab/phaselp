"""MRC input and output, standardization and resampling onto a common grid."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage


@dataclass
class Map:
    """A density map in array order (z, y, x) with isotropic voxel size in angstrom."""

    data: np.ndarray
    voxel_size: float
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)


def read_map(path: str | Path) -> Map:
    """Read an MRC/CCP4 map with standard axis order and isotropic voxels."""
    import mrcfile

    with mrcfile.open(Path(path), permissive=True) as mrc:
        data = np.asarray(mrc.data, dtype=np.float32).copy()
        axes = (int(mrc.header.mapc), int(mrc.header.mapr), int(mrc.header.maps))
        voxel = np.array([mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z], dtype=float)
        origin = tuple(float(v) for v in (mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z))
    if data.ndim != 3:
        raise ValueError(f"{path}: expected a 3-D map, got shape {data.shape}")
    if axes != (1, 2, 3):
        raise ValueError(f"{path}: nonstandard axis order mapc/mapr/maps={axes}; transpose it first")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{path}: map contains nonfinite values")
    if not (np.all(voxel > 0) and np.allclose(voxel, voxel[0], rtol=1e-4)):
        raise ValueError(f"{path}: voxel size must be positive and isotropic, got {voxel.tolist()}")
    return Map(data, float(voxel[0]), origin)


def write_map(
    path: str | Path,
    data: np.ndarray,
    voxel_size: float,
    origin: tuple[float, float, float] | None = None,
) -> Path:
    """Write a 3-D array (z, y, x) as a float32 MRC map."""
    import mrcfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(np.asarray(data, dtype=np.float32))
        mrc.voxel_size = float(voxel_size)
        if origin is not None:
            mrc.header.origin = tuple(float(v) for v in origin)
        mrc.update_header_stats()
    return path


def standardize(x: np.ndarray) -> np.ndarray:
    """Global zero mean and unit variance (Section 5.3)."""
    x = np.asarray(x, dtype=float)
    std = float(x.std())
    if std == 0:
        raise ValueError("cannot standardize a constant array")
    return (x - x.mean()) / std


def resample_to_grid(
    moving: np.ndarray,
    moving_voxel_size: float,
    fixed_shape: tuple[int, int, int],
    fixed_voxel_size: float,
    order: int = 1,
    antialias: bool = True,
) -> np.ndarray:
    """Resample ``moving`` onto a grid of ``fixed_shape`` and voxel size, keeping box centres aligned.

    When the output voxels are coarser, ``antialias`` first removes every
    Fourier coefficient beyond the output Nyquist frequency. Without it,
    interpolation folds the parent's fine detail back into lower frequencies.
    On the bundled demo this halves FSC(parent, reference) at the target for
    a 3.4-fold downsampling. ``antialias=False`` reproduces the plain linear
    resampling used for the 70S ribosome comparison in the manuscript.
    """
    moving = np.asarray(moving, dtype=float)
    if moving.ndim != 3 or len(fixed_shape) != 3:
        raise ValueError("resampling expects 3-D maps")
    if moving_voxel_size <= 0 or fixed_voxel_size <= 0:
        raise ValueError("voxel sizes must be positive")
    scale = fixed_voxel_size / moving_voxel_size
    if antialias and scale > 1.0 + 1e-6:
        from .fourier import radial_frequency

        spectrum = np.fft.rfftn(moving)
        spectrum[radial_frequency(moving.shape, moving_voxel_size) > 0.5 / fixed_voxel_size] = 0.0
        moving = np.fft.irfftn(spectrum, s=moving.shape, axes=tuple(range(moving.ndim)))
        del spectrum
    matrix = np.eye(3) * scale
    centre_in = (np.asarray(moving.shape, dtype=float) - 1.0) / 2.0
    centre_out = (np.asarray(fixed_shape, dtype=float) - 1.0) / 2.0
    return ndimage.affine_transform(
        moving, matrix, offset=centre_in - matrix @ centre_out,
        output_shape=tuple(int(n) for n in fixed_shape),
        order=order, mode="constant", cval=0.0, prefilter=order > 1,
    )
