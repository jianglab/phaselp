"""The browser app's Python session, run natively through the same dispatch the page uses."""

import asyncio
import io
import json
import zipfile

import numpy as np
import pytest

import phaselp as p
from phaselp.webapp import session as app
from phaselp.webapp import sources
from phaselp.webapp.store import mrc_bytes, read_mrc_bytes

from conftest import make_phantom


def call(method, payload=None, raw=None):
    return json.loads(asyncio.run(app.dispatch_json(method, json.dumps(payload or {}), raw)))


def upload(volume, voxel, name):
    return {"type": "upload", "filename": call("upload", {"name": name}, mrc_bytes(volume, voxel))["filename"]}


@pytest.fixture
def maps():
    parent = make_phantom(48)                                             # 1.0 A voxels
    rotated = np.rot90(parent, 1, axes=(0, 2)).copy()
    reference = p.resample_to_grid(rotated, 1.0, (32, 32, 32), 1.5)
    reference = p.ramped_phase_filter(reference, 1.5, 6.0, seed=5)
    return parent, reference


def load(maps, **options):
    parent, reference = maps
    payload = {"high": upload(parent, 1.0, "high.mrc"), "low": upload(reference, 1.5, "low.mrc"),
               "max_dimension": 32, "align_res": 6.0, **options}
    return call("input", payload)


def test_full_workflow(maps):
    loaded = load(maps)
    assert loaded["has_reference"] and loaded["low"]["shape"] == [32, 32, 32]
    reg = call("register")
    assert reg["metrics"]["registered"] and reg["metrics"]["ncc_after"] > 0.9
    assert reg["metrics"]["rotation_deg"] == pytest.approx(90, abs=3)
    assert reg["metrics"]["working_grid_zyx"] == [32, 32, 32]
    filt = call("filter", {"mode": "ramped", "target_res": 6.0, "parent_res": 2.0, "seeds": 3})
    m = filt["metrics"]
    assert m["n_seeds"] == 3 and abs(m["fsc_parent_filtered_at_target"] - m["expected_c_at_target"]) < 0.1
    chart = filt["fsc_chart"]
    assert chart["mode"] == "model" and len(chart["envelope"]["freq"]) == 17
    env = np.interp(1 / 6.0, chart["envelope"]["freq"], chart["envelope"]["value"])
    assert env == pytest.approx(float(p.coherence(1 / 6.0, 6.0, 2.0)), abs=0.05)    # eq. 7 is what is drawn
    assert "qc_filter.png" in filt["plot_data"]
    call("amplitude")
    cmp = call("compare")
    m = cmp["metrics"]
    for key in ("fsc_parent_reference_at_target", "expected_parent_reference_at_target",
                "fsc_filtered_reference_at_target", "expected_filtered_reference_at_target",
                "residual_variance_in_mask", "local_rms_p95"):
        assert key in m
    assert m["expected_parent_reference_at_target"] == pytest.approx(
        float(p.envelope(1 / 6.0, 6.0) * p.envelope(1 / 6.0, 2.0)), abs=0.02)     # eq. 8 on the nearest shell
    assert {"phaselp_results.zip", "results/fsc.csv", "results/difference_seed0.mrc"} <= set(cmp["_assets"])
    with zipfile.ZipFile(io.BytesIO(app.file_bytes("phaselp_results.zip"))) as archive:
        names = set(archive.namelist())
    assert {"parent_registered.mrc", "filtered_seed2.mrc", "local_rms_seed0.mrc", "fsc.csv",
            "parameters.json", "qc_compare.png", "qc_summary.png"} <= names


def test_difference_uses_one_realization(maps):
    load(maps, skip_register=True)
    call("register")
    call("filter", {"mode": "ramped", "target_res": 6.0, "seeds": 2})
    call("amplitude")
    call("compare")
    s = app.SESSION
    stored = read_mrc_bytes(app.file_bytes("results/difference_seed0.mrc")).data
    np.testing.assert_allclose(stored, (s.matched[0] - s.low).astype(np.float32), atol=1e-6)


@pytest.mark.parametrize("mode", ["amplitude", "cutoff"])
def test_other_modes_and_direct_compare(maps, mode):
    load(maps, skip_register=True)
    call("register")
    filt = call("filter", {"mode": mode, "target_res": 6.0})
    assert "n_seeds" not in filt["metrics"]
    cmp = call("compare")
    assert cmp["metrics"]["fsc_filtered_reference_at_target"] == pytest.approx(
        cmp["metrics"]["expected_filtered_reference_at_target"], abs=0.1)
    call("register")                                   # rerun clears filtered maps and results
    assert not app.SESSION.filtered and "phaselp_results.zip" not in app.SESSION.store.files
    direct = call("compare")
    assert direct["metrics"]["compared"] == "registered parent, unfiltered"


def test_trace_profile(maps):
    load(maps, skip_register=True)
    call("register")
    filt = call("filter", {"mode": "ramped", "target_res": 6.0, "seeds": 1, "trace_profile": True})
    assert filt["fsc_chart"]["mode"] == "traced"
    assert filt["metrics"]["expected_c_at_target"] == pytest.approx(filt["metrics"]["fsc_parent_reference_at_target"], abs=1e-3)


def test_no_reference_and_resampling(maps):
    parent, _ = maps
    call("input", {"high": upload(parent, 1.0, "h.mrc"), "low": {"type": "none"}})
    filt = call("filter", {"mode": "ramped", "target_res": 5.0, "seeds": 2})       # Step 1 is optional
    assert filt["reference_viewer"] == "asset:input_high.mrc"
    reg = call("register", {"resample_dimension": 24})
    assert reg["metrics"]["working_grid_zyx"] == [24, 24, 24] and reg["metrics"]["working_voxel_A"] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="Nyquist"):
        call("filter", {"mode": "ramped", "target_res": 3.0})


@pytest.mark.parametrize("payload, message", [
    ({"mode": "aplp"}, "Choose"), ({"target_res": 10, "parent_res": 12}, "finer"),
    ({"seeds": 64}, "Seed count"), ({"kappa": 0.1}, "κ"),
])
def test_filter_validation(maps, payload, message):
    load(maps, skip_register=True)
    call("register")
    with pytest.raises(ValueError, match=message):
        call("filter", {"target_res": 6.0, **payload})


def test_register_needs_inputs_and_grid_limits(maps):
    load(maps)
    with pytest.raises(ValueError, match="Resample grid"):
        call("register", {"resample_dimension": 8})
    with pytest.raises(ValueError, match="Step 1"):
        call("compare")


def test_mrc_bytes_nonstandard_axes():
    data = np.random.default_rng(0).standard_normal((6, 7, 8)).astype(np.float32)
    raw = bytearray(mrc_bytes(data.transpose(2, 1, 0).copy(), 1.5))    # stored as (x, y, z) sections
    raw[64:76] = np.array([3, 2, 1], dtype="<i4").tobytes()              # mapc=z, mapr=y, maps=x
    m = read_mrc_bytes(bytes(raw))
    np.testing.assert_array_equal(m.data, data)
    assert m.voxel_size == pytest.approx(1.5)


def test_pdb_simulation():
    pdb = "\n".join(f"ATOM  {i:5d}  CA  ALA A{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
                    for i, (x, y, z) in enumerate([(0, 0, 0), (3.8, 0, 0), (3.8, 3.8, 0), (0, 3.8, 3.8)], 1))
    atoms = sources.parse_pdb(pdb)
    density = sources.simulate_density(atoms, resolution=3.0, voxel_size=1.0)
    assert density.shape == sources.simulated_shape(atoms, 3.0, 1.0)
    assert density.sum() == pytest.approx(atoms.weights.sum(), rel=1e-3)
    assert sources.emdb_map_urls("EMD-1080")[0].endswith("EMD-1080/map/emd_1080.map.gz")


def test_antialiased_resampling_keeps_low_frequencies():
    fine = make_phantom(64) + np.random.default_rng(1).standard_normal((64, 64, 64))   # strong fine detail
    reference = p.resample_to_grid(fine, 1.0, (20, 20, 20), 3.2)
    plain = p.resample_to_grid(fine, 1.0, (20, 20, 20), 3.2, antialias=False)
    smooth = p.resample_to_grid(p.hard_cutoff_filter(fine, 1.0, 6.4), 1.0, (20, 20, 20), 3.2, antialias=False)
    f, a = p.fsc(reference, smooth, 3.2)
    _, b = p.fsc(plain, smooth, 3.2)
    band = (f > 0) & (f < 0.12)
    assert a[band].min() > 0.99 > b[band].min()      # folding of fine detail corrupts plain downsampling


def test_session_import_leaves_matplotlib_for_later():
    # the browser loads matplotlib after the input maps; importing the session must not need it
    import os
    import subprocess
    import sys
    code = "import sys, phaselp.webapp.session; assert 'matplotlib' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})


def test_bundled_demo_matches_manifest():
    import hashlib
    from pathlib import Path

    site = Path(__file__).resolve().parents[1] / "apps"
    if not (site / "demo" / "manifest.json").exists():
        pytest.skip("the browser app (apps/) is only in the repository, not in the source package")
    manifest = json.loads((site / "demo" / "manifest.json").read_text())
    app.configure(demo_dir=str(site / "demo"))
    try:
        loaded = call("input", {"high": {"type": "demo"}, "low": {"type": "demo"}})
    finally:
        app.configure(demo_dir="")
    for role in ("high", "low"):
        assert loaded[role]["shape"] == manifest[role]["shape"]
        assert loaded[role]["voxel"] == pytest.approx(manifest[role]["voxel_A"], abs=1e-3)
    data = read_mrc_bytes((site / "demo" / "high.mrc.gz").read_bytes()).data
    assert hashlib.sha256(np.ascontiguousarray(data, "<f4").tobytes()).hexdigest() == manifest["high"]["sha256_voxel_data_float32"]
