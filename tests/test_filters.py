import numpy as np
import pytest

import phaselp as p

SHAPES = [(256,), (60,), (31, 32), (32, 32, 32), (17, 20, 23)]


@pytest.mark.parametrize("shape", SHAPES)
def test_ramped_phase_keeps_amplitude_only_magnitudes(shape):
    # each realization has exactly the Fourier magnitudes T|F| of the amplitude-only filter;
    # on the full FFT this also verifies conjugate symmetry (a real output)
    x = np.random.default_rng(0).standard_normal(shape)
    ramped = p.ramped_phase_filter(x, 1.0, 6.0, seed=1)
    amplitude = p.amplitude_only_filter(x, 1.0, 6.0)
    np.testing.assert_allclose(np.abs(np.fft.fftn(ramped)), np.abs(np.fft.fftn(amplitude)), atol=1e-9)
    assert ramped.shape == shape and ramped.dtype == np.float64


@pytest.mark.parametrize("shape", SHAPES)
def test_dc_is_unchanged(shape):
    x = np.random.default_rng(1).standard_normal(shape) + 3.0
    for y in (p.ramped_phase_filter(x, 1.0, 6.0, seed=0), p.amplitude_only_filter(x, 1.0, 6.0),
              p.binary_phase_filter(x, 1.0, 6.0, seed=0), p.hard_cutoff_filter(x, 1.0, 6.0)):
        assert y.mean() == pytest.approx(x.mean(), abs=1e-12)


def test_amplitude_only_keeps_phases():
    # every retained phase is the parent's, so the FSC with the parent stays near 1 (Section 2.1);
    # it is exactly 1 only where T is constant within a shell
    x = np.random.default_rng(2).standard_normal((64, 64, 64))
    y = p.amplitude_only_filter(x, 1.0, 6.0, 2.5)
    profile = p.filter_profile(x.shape, 1.0, 6.0, 2.5)
    kept = profile.gain > 1e-3
    fx, fy = np.fft.rfftn(x), np.fft.rfftn(y)
    np.testing.assert_allclose(np.angle(fy[kept] * np.conj(fx[kept])), 0.0, atol=1e-9)
    frequency, values = p.fsc(x, y)
    band = (frequency > 0) & (p.coherence(frequency, 6.0, 2.5) > 0.05)
    assert values[band].min() > 0.97


def test_expected_fsc_equals_coherence():
    # equation (4): E[FSC(H, H')] = c(s); average over seeds on a large random map
    x = np.random.default_rng(3).standard_normal((64, 64, 64))
    curves = np.array([p.fsc(x, p.ramped_phase_filter(x, 1.0, 6.0, 2.5, seed=s))[1] for s in range(6)])
    frequency = np.arange(curves.shape[1]) / 64
    expected = p.coherence(frequency, 6.0, 2.5)
    band = (frequency > 0.05) & (expected > 0.05)
    assert np.max(np.abs(curves.mean(0)[band] - expected[band])) < 0.03


def test_seed_reproducible_and_batch_independent():
    z = np.random.default_rng(4).standard_normal((48, 48))
    x = np.stack([z, z, z])
    a = p.ramped_phase_filter(x, 1.0, 5.0, seed=9, ndim=2)
    np.testing.assert_array_equal(a, p.ramped_phase_filter(x, 1.0, 5.0, seed=9, ndim=2))
    assert not np.allclose(a[0], a[1])  # each example in a batch gets its own offsets


def test_stopband_removed():
    x = np.random.default_rng(5).standard_normal(512)
    y = p.ramped_phase_filter(x, 1.0, 10.0, seed=0)
    profile = p.filter_profile((512,), 1.0, 10.0)
    removed = profile.coherence < p.COHERENCE_FLOOR
    assert removed.any()
    assert np.all(np.abs(np.fft.rfft(y))[removed] < 1e-9)


def test_hard_cutoff():
    x = np.random.default_rng(6).standard_normal(200)
    y = p.hard_cutoff_filter(x, 1.0, 5.0)
    s = np.fft.rfftfreq(200)
    np.testing.assert_allclose(np.fft.rfft(y)[s <= 0.2], np.fft.rfft(x)[s <= 0.2], atol=1e-10)
    assert np.all(np.abs(np.fft.rfft(y)[s > 0.2]) < 1e-10)


def test_binary_keeps_phases_below_cutoff():
    x = np.random.default_rng(7).standard_normal(400)
    y = p.binary_phase_filter(x, 1.0, 8.0, seed=1)
    s = np.fft.rfftfreq(400)
    below = (s < 1 / 8) & (s > 0)
    np.testing.assert_allclose(np.angle(np.fft.rfft(y)[below]), np.angle(np.fft.rfft(x)[below]), atol=1e-9)


def test_coherence_curve_override():
    x = np.random.default_rng(8).standard_normal((32, 32, 32))
    curve = (np.array([0.0, 0.2, 0.5]), np.array([1.0, 0.5, 0.0]))
    profile = p.filter_profile(x.shape, 1.0, 4.0, coherence_curve=curve)
    np.testing.assert_allclose(profile.gain.flat[1:20], np.interp(profile.frequency.flat[1:20], *curve))


def test_rejects_target_beyond_nyquist():
    with pytest.raises(ValueError):
        p.ramped_phase_filter(np.zeros(64), 2.0, 3.0)
