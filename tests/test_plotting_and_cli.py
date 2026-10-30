import json

import numpy as np
from matplotlib.figure import Figure

import phaselp as p
from phaselp import plotting
from phaselp.cli import main


def test_plots(phantom):
    rng = np.random.default_rng(0)
    experimental = p.ramped_phase_filter(phantom, 1.0, 8.0, seed=5)
    result = p.compare_maps(phantom, experimental, parent_voxel_size=1.0, voxel_size=1.0,
                            target_res=8.0, parent_res=2.5, seeds=(0, 1), register_maps=False)
    figures = [
        plotting.plot_sections([result.parent, result.experimental, result.filtered[0]], ["a", "b", "c"]),
        plotting.plot_fsc(result.fsc_curves(), target_res=8.0),
        plotting.plot_filter_profile(8.0, 2.5, pixel_size=1.0),
        plotting.plot_differences({"control": result.differences()[0], "other": result.differences()[1]},
                                  sigma=8.0, mask=p.union_mask(experimental)),
        plotting.plot_signal_filters(np.sin(np.arange(300) / 2.0) * (np.arange(300) > 150), 5.0),
    ]
    traces = rng.standard_normal((4, 60))
    figures.append(plotting.plot_augmentation(traces, p.phase_augment(traces, 1 / 28, 0.25, ndim=1, seed=0), 1 / 28, 0.25))
    images = rng.standard_normal((3, 28, 28))
    figures.append(plotting.plot_augmentation(images, p.phase_augment(images, 0.16, 0.28, ndim=2, seed=0), 0.16, 0.28))
    assert all(isinstance(f, Figure) for f in figures)


def test_cli(tmp_path, phantom):
    parent = tmp_path / "parent.mrc"
    low = tmp_path / "low.mrc"
    p.write_map(parent, phantom, 1.0)
    p.write_map(low, p.ramped_phase_filter(phantom, 1.0, 8.0, seed=3), 1.0)

    assert main(["filter", str(parent), str(tmp_path / "f.mrc"), "--target-res", "8", "--parent-res", "2.5"]) == 0
    assert p.read_map(tmp_path / "f.mrc").data.shape == phantom.shape
    assert main(["fsc", str(parent), str(low), "--out", str(tmp_path / "fsc.csv")]) == 0
    assert (tmp_path / "fsc.csv").read_text().startswith("frequency_invA,fsc")

    out = tmp_path / "cmp"
    assert main(["compare", str(parent), str(low), str(out), "--target-res", "8",
                 "--no-register", "--seeds", "2", "--plots"]) == 0
    for name in ["parent_registered.mrc", "amplitude_only.mrc", "ramped_phase_seed0.mrc", "ramped_phase_seed1.mrc",
                 "difference_seed0.mrc", "local_rms_seed0.mrc", "fsc_with_parent.csv", "comparison.json",
                 "sections.png", "fsc_with_parent.png", "difference.png"]:
        assert (out / name).exists(), name
    assert json.loads((out / "comparison.json").read_text())["seeds"] == [0, 1]
