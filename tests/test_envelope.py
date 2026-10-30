import numpy as np
import pytest

import phaselp as p


@pytest.mark.parametrize("resolution", [3.0, 8.0, 13.2])
def test_envelope_crosses_half_at_target(resolution):
    # equation (5): c(1/R) = 0.5, SSNR = 1/3 there
    assert p.envelope(1.0 / resolution, resolution) == pytest.approx(0.5, abs=1e-12)
    assert p.ssnr(1.0 / resolution, resolution) == pytest.approx(1.0 / 3.0, abs=1e-12)
    assert p.envelope(0.0, resolution) == pytest.approx(np.sqrt(100 / 101))


def test_envelope_decreases():
    s = np.linspace(0, 0.5, 200)
    assert np.all(np.diff(p.envelope(s, 8.0)) <= 0)


def test_coherence_increment_rule():
    s = np.linspace(0.0, 0.3, 300)
    c = p.coherence(s, 8.0, 3.0)
    ratio = p.envelope(s, 8.0) / p.envelope(s, 3.0)
    np.testing.assert_allclose(c, np.minimum(1.0, ratio))
    assert np.all(c <= 1.0)
    # an ideal parent reduces equation (7) to equation (5)
    np.testing.assert_allclose(p.coherence(s, 8.0), p.envelope(s, 8.0))
    # parent correction raises c at the target: the filter only adds the remaining loss
    assert p.coherence(1 / 8, 8.0, 6.0) > p.coherence(1 / 8, 8.0) == pytest.approx(0.5)


def test_coherence_rejects_coarser_parent():
    with pytest.raises(ValueError):
        p.coherence(0.1, 8.0, 10.0)


def test_sigma_coherence_inverse():
    c = np.linspace(0.01, 1.0, 50)
    np.testing.assert_allclose(p.coherence_from_sigma(p.sigma_from_coherence(c)), c, rtol=1e-12)
    assert p.sigma_from_coherence(1.0) == 0.0 and not np.signbit(p.sigma_from_coherence(1.0))


def test_halfmap_relations():
    R = 8.0
    # half-map FSC at 1/R is 1/7 ~ 0.143, and the full map then correlates with the ideal map at 0.5
    rho = p.halfmap_envelope(1 / R, R) ** 2
    assert rho == pytest.approx(1 / 7)
    assert p.full_map_envelope(rho) == pytest.approx(0.5)
    # equation (8): two half maps agree at a_half^2
    assert p.expected_fsc_independent(np.sqrt(rho), np.sqrt(rho)) == pytest.approx(rho)


def test_mean_of_realizations_limit():
    c = 0.5
    assert p.expected_fsc_mean_of_realizations(c, 1) == pytest.approx(c)
    assert p.expected_fsc_mean_of_realizations(c, 10**8) == pytest.approx(1.0, abs=1e-6)
    assert p.expected_fsc_filtered_vs_other(0.534, 0.649) == pytest.approx(0.3466, abs=1e-4)
