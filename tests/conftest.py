import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")


def make_phantom(n: int = 40, seed: int = 0) -> np.ndarray:
    """An asymmetric set of Gaussian blobs, so principal axes are well defined."""
    rng = np.random.default_rng(seed)
    grid = np.indices((n, n, n), dtype=float) - (n - 1) / 2
    volume = np.zeros((n, n, n))
    centres = [(-6, 2, 0), (5, -3, 4), (0, 7, -5), (3, 3, 8), (-4, -6, -3)]
    for k, centre in enumerate(centres):
        r2 = sum((g - c) ** 2 for g, c in zip(grid, centre))
        volume += (1.0 + 0.4 * k) * np.exp(-r2 / (2 * (2.0 + 0.5 * k) ** 2))
    return volume + 0.01 * rng.standard_normal(volume.shape)


@pytest.fixture
def phantom():
    return make_phantom()
