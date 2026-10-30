import numpy as np
import pytest

import phaselp as p


def test_ramp_sigma_shape():
    s = np.array([0.0, 0.1, 0.15, 0.2, 0.3])
    sigma = p.ramp_sigma(s, 0.1, 0.2, 2.0)
    np.testing.assert_allclose(sigma, [0.0, 0.0, 1.0, 2.0, 2.0])
    assert p.SIGMA_MAX == pytest.approx(2.60, abs=0.01)


@pytest.mark.parametrize("shape,ndim", [((8, 60), 1), ((5, 28, 28), 2), ((2, 3, 27, 30), 2), ((2, 16, 16, 16), 3)])
def test_power_spectrum_and_dc_preserved(shape, ndim):
    x = np.random.default_rng(0).standard_normal(shape)
    y = p.phase_augment(x, 0.05, 0.3, ndim=ndim, seed=1)
    axes = tuple(range(-ndim, 0))
    np.testing.assert_allclose(np.abs(np.fft.fftn(y, axes=axes)), np.abs(np.fft.fftn(x, axes=axes)), atol=1e-9)
    np.testing.assert_allclose(y.mean(axis=axes), x.mean(axis=axes), atol=1e-12)
    assert not np.allclose(x, y)


def test_low_frequencies_kept():
    x = np.random.default_rng(1).standard_normal((4, 64, 64))
    y = p.phase_augment(x, 0.1, 0.3, ndim=2, seed=2)
    fx, fy = np.fft.rfft2(x), np.fft.rfft2(y)
    s = p.fourier.radial_frequency((64, 64))
    np.testing.assert_allclose(fy[:, s <= 0.1], fx[:, s <= 0.1], atol=1e-9)


def test_offsets_follow_the_ramp():
    # E[cos delta] = exp(-sigma^2 / 2) in each band of the ramp
    x = np.random.default_rng(2).standard_normal((2000, 128))
    y = p.phase_augment(x, 0.05, 0.4, ndim=1, seed=3)
    delta = np.angle(np.fft.rfft(y) / np.fft.rfft(x))
    s = np.fft.rfftfreq(128)
    for lo, hi in [(0.1, 0.15), (0.2, 0.25), (0.3, 0.35)]:
        band = (s > lo) & (s < hi)
        expected = p.coherence_from_sigma(p.ramp_sigma(s[band], 0.05, 0.4)).mean()
        assert np.cos(delta[:, band]).mean() == pytest.approx(expected, abs=0.03)


def test_per_example_start_frequency():
    x = np.random.default_rng(3).standard_normal((2, 128))
    y = p.phase_augment(x, np.array([0.1, 0.3]), 0.45, ndim=1, seed=0)
    s = np.fft.rfftfreq(128)
    keep = (s > 0.12) & (s <= 0.3)
    np.testing.assert_allclose(np.fft.rfft(y[1])[keep], np.fft.rfft(x[1])[keep], atol=1e-9)
    assert not np.allclose(np.fft.rfft(y[0])[keep], np.fft.rfft(x[0])[keep])


def test_phase_augment_class():
    x = np.random.default_rng(4).standard_normal((2000, 60))
    augment = p.PhaseAugment(1 / 28, 0.25, ndim=1, seed=0)
    y = augment(x)
    changed = ~np.all(np.isclose(x, y), axis=1)
    assert 0.45 < changed.mean() < 0.55  # p = 0.5
    np.testing.assert_allclose(np.abs(np.fft.rfft(y)), np.abs(np.fft.rfft(x)), atol=1e-9)
    single = augment(x[0])  # one example
    assert single.shape == (60,)
    images = p.PhaseAugment(0.16, 0.28, ndim=2, p=1.0, seed=1)(np.random.default_rng(5).standard_normal((3, 28, 28)))
    assert images.shape == (3, 28, 28)


def test_phase_augment_validation():
    with pytest.raises(ValueError):
        p.PhaseAugment(0.3, 0.3)
    with pytest.raises(ValueError):
        p.phase_augment(np.zeros((2, 10)), np.array([0.1, 0.2, 0.3]), 0.4, ndim=1)
