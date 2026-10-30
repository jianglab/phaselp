import numpy as np
import pytest

import phaselp as p


def test_fsc_identity_and_dc():
    x = np.random.default_rng(0).standard_normal((32, 32, 32))
    frequency, values = p.fsc(x, x, 2.0)
    assert np.isnan(values[0])  # DC excluded
    np.testing.assert_allclose(values[1:], 1.0)
    assert frequency[-1] == pytest.approx(0.25)  # Nyquist of 2 A voxels
    assert len(frequency) == 17


def test_fsc_scale_invariant_and_independent_noise():
    rng = np.random.default_rng(1)
    x, n = rng.standard_normal((48, 48, 48)), rng.standard_normal((48, 48, 48))
    _, a = p.fsc(x, 3.0 * x + 2.0)
    np.testing.assert_allclose(a[1:], 1.0)
    _, b = p.fsc(x, n)
    assert np.abs(b[3:]).max() < 0.15


def test_fsc_matches_full_fft_definition():
    rng = np.random.default_rng(2)
    x, y = rng.standard_normal((20, 22, 25)), rng.standard_normal((20, 22, 25))
    y = x + y
    frequency, values = p.fsc(x, y, 1.0)
    fx, fy = np.fft.fftn(x), np.fft.fftn(y)
    grid = np.sqrt(sum(g**2 for g in np.meshgrid(*[np.fft.fftfreq(n) for n in x.shape], indexing="ij")))
    labels = np.rint(grid * 25).astype(int)
    for shell in (1, 5, 12):
        m = labels == shell
        direct = np.real(np.sum(fx[m] * np.conj(fy[m]))) / np.sqrt(np.sum(np.abs(fx[m]) ** 2) * np.sum(np.abs(fy[m]) ** 2))
        assert values[shell] == pytest.approx(direct, abs=1e-12)


def test_radial_amplitude_matching():
    rng = np.random.default_rng(3)
    reference = p.amplitude_only_filter(rng.standard_normal((40, 40, 40)), 1.0, 4.0)
    moving = rng.standard_normal((40, 40, 40))
    matched = p.match_radial_amplitude(moving, reference, 1.0)
    _, target = p.radial_amplitude(reference)
    _, got = p.radial_amplitude(matched)
    _, before = p.radial_amplitude(moving)
    ok = (target / before) < 5.0
    ok[0] = False
    np.testing.assert_allclose(got[ok], target[ok], rtol=1e-9)
    assert matched.mean() == pytest.approx(moving.mean())  # unit DC gain
    # phases unchanged, so the FSC with the input is 1 wherever the gain is nonzero
    _, values = p.fsc(moving, matched)
    assert np.nanmin(values[ok]) > 0.999999
