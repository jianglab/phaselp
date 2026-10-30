import numpy as np
import pytest

import phaselp as p


def _corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def test_mrc_round_trip(tmp_path, phantom):
    path = p.write_map(tmp_path / "x.mrc", phantom, 1.5, origin=(1.0, 2.0, 3.0))
    m = p.read_map(path)
    assert m.voxel_size == pytest.approx(1.5)
    assert m.origin == pytest.approx((1.0, 2.0, 3.0))
    np.testing.assert_allclose(m.data, phantom.astype(np.float32))


def test_standardize():
    x = p.standardize(np.arange(10.0) * 3 + 7)
    assert x.mean() == pytest.approx(0.0) and x.std() == pytest.approx(1.0)


def test_resample_keeps_centre(phantom):
    finer = p.resample_to_grid(phantom, 1.0, (80, 80, 80), 0.5)
    back = p.resample_to_grid(finer, 0.5, phantom.shape, 1.0)
    assert _corr(back, phantom) > 0.999


def test_register_recovers_similarity(phantom):
    truth = np.array([np.log(1.04), 0.05, -0.12, 0.08, 1.5, -2.0, 1.0])
    moving = p.apply_similarity(phantom, truth)
    aligned, fit = p.register(phantom, moving, 1.0, max_dimension=40, translation_bound=6)
    assert fit.score_after > fit.score_before
    assert fit.score_after > 0.99
    assert _corr(aligned, phantom) > 0.99
    # the fit undoes the applied scale
    assert fit.scale * 1.04 == pytest.approx(1.0, abs=0.01)


def test_register_from_arbitrary_orientation(phantom):
    # a 90 degree rotation lies outside the 20 degree local radius; the principal-axis start finds it
    moving = np.rot90(phantom, 1, axes=(0, 2)).copy()
    aligned, fit = p.register(phantom, moving, 1.0, max_dimension=40)
    assert _corr(aligned, phantom) > 0.99
    assert fit.rotation_degrees == pytest.approx(90.0, abs=2.0)


def test_register_downsampled_grid(phantom):
    moving = p.apply_similarity(phantom, np.array([0.0, 0.0, 0.1, 0.0, 2.0, 0.0, -1.0]))
    aligned, fit = p.register(phantom, moving, 1.0, max_dimension=24, max_frequency=0.2, translation_bound=6)
    assert _corr(aligned, phantom) > 0.98


def test_phase_correlation_ignores_stopband(phantom):
    # band-limited copies: the stopband holds only rounding noise, which must not pick the peak
    from phaselp.registration import _band_limit, _phase_correlation_shift

    fixed = _band_limit(phantom, 1.0, 0.12)
    moving = _band_limit(np.roll(phantom, (3, -2, 5), axis=(0, 1, 2)), 1.0, 0.12)
    np.testing.assert_array_equal(_phase_correlation_shift(fixed, moving), [-3, 2, -5])


def test_local_orientation_search(phantom):
    moving = p.apply_similarity(phantom, np.array([0.0, 0.0, 0.08, 0.0, 1.0, 0.0, 0.0]))
    aligned, fit = p.register(phantom, moving, 1.0, max_dimension=40, translation_bound=6,
                              orientation_search="local")
    assert _corr(aligned, phantom) > 0.99 and fit.rotation_degrees < 10
    with pytest.raises(ValueError):
        p.register(phantom, moving, 1.0, orientation_search="auto")
