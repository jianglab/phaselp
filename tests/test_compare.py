import numpy as np
import pytest

import phaselp as p


@pytest.fixture
def pair(phantom):
    rng = np.random.default_rng(1)
    experimental = p.ramped_phase_filter(phantom, 1.0, 8.0, seed=99) + 0.02 * rng.standard_normal(phantom.shape)
    return phantom, experimental


def test_compare_maps_no_registration(pair):
    parent, experimental = pair
    result = p.compare_maps(parent, experimental, parent_voxel_size=1.0, voxel_size=1.0,
                            target_res=8.0, parent_res=2.5, seeds=(0, 1, 2), register_maps=False)
    assert len(result.filtered) == 3 and result.registration is None
    assert result.experimental.std() == pytest.approx(1.0)
    curves = result.fsc_curves()
    assert set(curves) == {"frequency", "ramped_phase", "amplitude_only", "experimental", "expected"}
    assert curves["ramped_phase"].shape == (3, len(curves["frequency"]))
    k = int(np.argmin(np.abs(curves["frequency"] - 1 / 8)))
    assert curves["amplitude_only"][1:k + 1].min() > 0.95  # amplitude-only keeps the parent's phases
    assert curves["ramped_phase"][:, k].mean() < 0.8
    assert curves["expected"][k] == pytest.approx(p.coherence(curves["frequency"][k], 8.0, 2.5))
    # amplitude matched: shell means follow the experimental map
    _, a = p.radial_amplitude(result.filtered[0])
    _, b = p.radial_amplitude(result.experimental)
    np.testing.assert_allclose(a[1:k], b[1:k], rtol=1e-6)


def test_compare_maps_resamples_and_registers(pair):
    parent, experimental = pair
    finer = p.resample_to_grid(parent, 1.0, (80, 80, 80), 0.5)
    shifted = p.apply_similarity(finer, np.array([0.0, 0.0, 0.0, 0.1, 2.0, 0.0, 0.0]))
    result = p.compare_maps(shifted, experimental, parent_voxel_size=0.5, voxel_size=1.0,
                            target_res=8.0, seeds=(0,), translation_bound=6,
                            register_options={"max_dimension": 40})
    assert result.parent.shape == experimental.shape
    assert result.registration.score_after > result.registration.score_before
    assert np.corrcoef(result.parent.ravel(), parent.ravel())[0, 1] > 0.98


def test_difference_statistics(pair):
    parent, experimental = pair
    d = p.difference_map(parent, experimental)
    np.testing.assert_allclose(d, parent - experimental)
    rms = p.local_rms(d, sigma=4.0, voxel_size=2.0)
    assert rms.shape == d.shape and rms.min() >= 0
    mask = p.union_mask(parent, experimental)
    assert 0.0 <= mask.min() and mask.max() == pytest.approx(1.0)
    assert 0 < (mask > 0.5).mean() < 1
    assert p.variance_ratio(2 * d, d, mask) == pytest.approx(4.0)
    assert p.residual_variance(d, mask) == pytest.approx(d[mask > 0.5].var())
