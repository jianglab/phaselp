"""Session state and the four steps of the phaselp browser app.

The page sends each request to :func:`dispatch_json` through a Pyodide web
worker. The function also runs in ordinary Python, where the tests call it.
Every step returns plain dicts holding metrics, chart data and the names of
figures and maps kept in the session :class:`~phaselp.webapp.store.Store`.
All numerical work is done by phaselp itself:

1. resample to a working grid (anti-aliased) and register the parent (Section 5.1);
2. filter: ramped phase (equations 2, 5-7), amplitude only, or a hard cutoff;
3. radial amplitude matching to the reference (Section 5.2);
4. compare: FSCs with their expected values (equations 4, 8 and Section S2), the
   difference of one filtered map with the reference, its local rms and its
   residual variance inside a soft mask (Section 5.3).
"""

from __future__ import annotations

import asyncio
import gc
import json
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage

from ..compare import local_rms, residual_variance, union_mask
from ..envelope import COHERENCE_FLOOR, KAPPA, coherence, envelope
from ..filters import amplitude_only_filter, hard_cutoff_filter, ramped_phase_filter
from ..fsc import fsc, match_radial_amplitude, radial_amplitude
from ..maps import resample_to_grid, standardize
from ..registration import register
from . import sources
from .store import Store, read_mrc_bytes

VIEWER_MAX_DIM = 160        # 3-D viewer copies are downsampled for display only
MAX_VOXELS = 256 ** 3       # browser memory budget per input map
MODES = {"ramped": "Ramped phase", "amplitude": "Amplitude only", "cutoff": "Hard cutoff"}
STAGES = ("register", "filter", "amplitude", "compare")
CONFIG: dict[str, Any] = {"site_url": "", "demo_dir": None}


def configure(site_url: str | None = None, demo_dir: str | None = None) -> None:
    """Where bundled demo maps live: the site URL (browser) or a local folder."""
    if site_url is not None:
        CONFIG["site_url"] = site_url
    if demo_dir is not None:
        CONFIG["demo_dir"] = demo_dir


def _figures():
    """The figure module, imported on first use: the browser loads matplotlib after the input maps."""
    from . import figures

    return figures


def _number(payload: dict, key: str, default: float, label: str, low: float = 0.0,
            high: float = float("inf"), inclusive: bool = False) -> float:
    value = payload.get(key, default)
    value = default if value in (None, "") else float(value)
    ok = np.isfinite(value) and (value >= low if inclusive else value > low) and value <= high
    if not ok:
        raise ValueError(f"{label} must be {'at least' if inclusive else 'greater than'} {low:g}"
                         + (f" and at most {high:g}." if np.isfinite(high) else "."))
    return value


def _shells(shape: tuple[int, ...], voxel: float) -> np.ndarray:
    n = max(shape)
    return np.arange(n // 2 + 1) / (n * voxel)


def _nanmean(curves: np.ndarray, axis: int = 0, sd: bool = False) -> np.ndarray:
    """Mean (or SD) over seeds; the DC shell is NaN in every curve."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanstd(curves, axis=axis) if sd else np.nanmean(curves, axis=axis)


def _curve(frequency: np.ndarray, values: np.ndarray) -> dict[str, list]:
    ok = np.isfinite(values) & (frequency > 0)
    return {"freq": [float(x) for x in frequency[ok]], "fsc": [float(x) for x in values[ok]]}


def _at(frequency: np.ndarray, values: np.ndarray, resolution: float) -> float:
    return float(np.asarray(values)[int(np.argmin(np.abs(frequency - 1.0 / resolution)))])


@dataclass
class Session:
    store: Store = field(default_factory=Store)
    # inputs
    high: np.ndarray | None = None
    high_vox: float = 1.0
    low_input: np.ndarray | None = None
    low_input_vox: float = 1.0
    has_reference: bool = True
    options: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    reference_viewer: str | None = None
    # working grid and Step 1
    low: np.ndarray | None = None
    work_vox: float = 1.0
    parent_on_grid: np.ndarray | None = None
    parent: np.ndarray | None = None
    registration: Any = None
    # Step 2
    mode: str = "ramped"
    target_res: float = 10.0
    parent_res: float | None = None
    kappa: float = KAPPA
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    traced: tuple[np.ndarray, np.ndarray] | None = None
    filtered: list[np.ndarray] = field(default_factory=list)
    # Step 3
    matched: list[np.ndarray] = field(default_factory=list)

    # -- helpers -----------------------------------------------------------
    def viewer(self, name: str, volume: np.ndarray, voxel: float) -> str:
        """Store a display copy (at most VIEWER_MAX_DIM per axis) for the 3-D viewer."""
        array = np.asarray(volume, dtype=np.float32)
        factor = min(1.0, VIEWER_MAX_DIM / max(array.shape))
        if factor < 1.0:
            array = ndimage.zoom(array, factor, order=1)
            voxel = voxel / factor
        return "asset:" + self.store.mrc(f"{name}.mrc", array, voxel)

    @property
    def frequency(self) -> np.ndarray:
        return _shells(self.parent.shape, self.work_vox)

    def coherence_profile(self, frequency: np.ndarray) -> np.ndarray:
        """The c(s) the current filter applies, on the given frequencies."""
        if self.traced is not None:
            return np.interp(frequency, *self.traced, left=float(self.traced[1][0]), right=0.0)
        return np.asarray(coherence(frequency, self.target_res, self.parent_res, self.kappa))

    def invalidate(self, stage: str) -> None:
        index = STAGES.index(stage)
        if index == 0:
            self.parent = self.parent_on_grid = self.registration = None
            if not self.has_reference:
                self.work_vox = self.high_vox
        if index <= 1:
            self.filtered, self.traced = [], None
        if index <= 2:
            self.matched = []
        prefixes = tuple(f"step{i + 1}" for i in range(index, 4)) + tuple(f"qc_{s}.png" for s in STAGES[index:])
        self.store.drop(lambda n: n.startswith(prefixes) or n.startswith("results/")
                        or n in ("qc_summary.png", "phaselp_results.zip"))
        gc.collect()

    # -- inputs ------------------------------------------------------------
    def load_inputs(self, prepared: dict[str, tuple], payload: dict[str, Any]) -> dict[str, Any]:
        orientation = payload.get("orientation_search", "global")
        if orientation not in ("global", "local"):
            raise ValueError("Orientation search must be global or local.")
        self.options = {
            "orientation_search": orientation,
            "translation_bound": _number(payload, "translation_bound", 20.0, "Translation bound", 0, inclusive=True),
            "max_dimension": int(_number(payload, "max_dimension", 96, "Alignment search grid", 16, 256, inclusive=True)),
            "align_res": _number(payload, "align_res", 10.0, "Alignment resolution"),
            "skip_register": bool(payload.get("skip_register", False)),
            "standardize": bool(payload.get("standardize", True)),
        }
        self.target_res = self.options["align_res"]   # provisional, until a filter target is chosen
        high, self.high_vox, high_meta = prepared["high"]
        self.high = standardize(high) if self.options["standardize"] else np.asarray(high, float)
        self.has_reference = "low" in prepared
        low_meta = None
        if self.has_reference:
            low, self.low_input_vox, low_meta = prepared["low"]
            self.low_input = standardize(low) if self.options["standardize"] else np.asarray(low, float)
            self.work_vox = self.low_input_vox
        else:
            self.low_input, self.work_vox = None, self.high_vox
        self.meta = {"high": high_meta, "low": low_meta}
        high_view = self.viewer("input_high", self.high, self.high_vox)
        self.reference_viewer = (self.viewer("input_low", self.low_input, self.low_input_vox)
                                 if self.has_reference else None)
        return {"high": high_meta, "low": low_meta, "high_viewer": high_view,
                "low_viewer": self.reference_viewer, "has_reference": self.has_reference}

    # -- Step 1 ------------------------------------------------------------
    def step_register(self, payload: dict[str, Any]) -> dict[str, Any]:
        size = _number(payload, "resample_dimension", 0, "Resample grid", 0, 256, inclusive=True)
        if size != int(size) or (size and size < 16):
            raise ValueError("Resample grid must be a whole number from 16 to 256, or 0 to keep the input grid.")
        size = int(size)
        source, voxel = (self.low_input, self.low_input_vox) if self.has_reference else (self.high, self.high_vox)
        if source is None:
            raise ValueError("Load the input maps first.")
        factor = size / max(source.shape) if size else 1.0
        shape = tuple(max(2, int(round(n * factor))) for n in source.shape)
        self.work_vox = voxel / factor
        working = source if shape == source.shape else resample_to_grid(source, voxel, shape, self.work_vox)
        store = self.store
        if self.has_reference:
            self.low = working
            self.parent_on_grid = resample_to_grid(self.high, self.high_vox, shape, self.work_vox)
            if self.options["skip_register"]:
                self.parent, self.registration = self.parent_on_grid, None
            else:
                self.parent, self.registration = register(
                    self.low, self.parent_on_grid, self.work_vox,
                    max_frequency=1.0 / self.options["align_res"],
                    max_dimension=self.options["max_dimension"],
                    translation_bound=self.options["translation_bound"],
                    orientation_search=self.options["orientation_search"])
            f = self.frequency
            fig = _figures().registration(self.low, self.parent_on_grid, self.parent, f,
                                       fsc(self.parent_on_grid, self.low, self.work_vox)[1],
                                       fsc(self.parent, self.low, self.work_vox)[1],
                                       self.options["align_res"], self.registration)
            self.reference_viewer = self.viewer("step1_reference_view", self.low, self.work_vox)
            label, reference = "registered parent", {}
        else:
            self.parent_on_grid = self.parent = working
            self.registration = None
            fig = _figures().sections(working, "Step 1 · resampled parent (no reference)")
            label = "resampled parent"
            reference = {"reference_viewer": "asset:input_high.mrc", "reference_label": "original high map"}
        store.figure("qc_register.png", fig, dpi=150)
        store.mrc("step1_parent.mrc", self.parent, self.work_vox)
        metrics: dict[str, Any] = {"registered": self.registration is not None}
        fit = self.registration
        if fit is not None:
            metrics.update({"ncc_before": round(fit.score_before, 4), "ncc_after": round(fit.score_after, 4),
                            "scale": round(fit.scale, 5), "rotation_deg": round(fit.rotation_degrees, 2),
                            "translation_zyx_vox": [round(float(t), 2) for t in fit.translation_voxels]})
        metrics.update({"working_grid_zyx": list(shape), "working_voxel_A": round(self.work_vox, 6)})
        return {"metrics": metrics, "plot": "asset:qc_register.png",
                "download": {"label": f"{label} (MRC)", "url": "asset:step1_parent.mrc?dl=1"},
                "viewer": self.viewer("step1_view", self.parent, self.work_vox), "viewer_label": label,
                "working_grid": {"shape": list(shape), "voxel": self.work_vox, "requested_dimension": size},
                **reference}

    # -- Step 2 ------------------------------------------------------------
    def step_filter(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.parent is None:
            if self.has_reference:
                raise ValueError("Run Step 1 (resample and register) first.")
            self.parent = self.parent_on_grid = self.high      # no reference: filter the input grid
            self.work_vox = self.high_vox
        mode = payload.get("mode", "ramped")
        if mode not in MODES:
            raise ValueError("Choose Ramped phase, Amplitude only or Hard cutoff.")
        vox = self.work_vox
        target = _number(payload, "target_res", 10.0, "Filter resolution")
        if target < 2 * vox:
            raise ValueError(f"Filter resolution must be at least the Nyquist limit ({2 * vox:.3g} Å).")
        parent_res = _number(payload, "parent_res", 0.0, "Parent resolution", 0, inclusive=True) or None
        if parent_res is not None and parent_res >= target:
            raise ValueError("Parent resolution must be finer (smaller) than the filter resolution.")
        kappa = _number(payload, "kappa", KAPPA, "κ", 1.0 / 3.0)
        count = int(_number(payload, "seeds", 4, "Seed count", 1, 32, inclusive=True)) if mode == "ramped" else 1
        if self.parent.size * 8 * (2 * count + 16) > 1_500_000_000:
            raise ValueError("This grid and seed count exceed the browser memory budget. "
                             "Use fewer seeds or a smaller grid in Step 1.")
        self.mode, self.target_res, self.parent_res, self.kappa = mode, target, parent_res, kappa
        self.seeds = list(range(count))
        f = self.frequency
        parent_low = fsc(self.parent, self.low, vox)[1] if self.has_reference else None
        self.traced = None
        if payload.get("trace_profile") and self.has_reference and mode == "ramped":
            ok = np.isfinite(parent_low) & (f > 0)
            self.traced = (f[ok], np.clip(parent_low[ok], 0.0, 1.0))
        if mode == "ramped":
            self.filtered = [ramped_phase_filter(self.parent, vox, target, parent_res, kappa=kappa, seed=s,
                                                 coherence_curve=self.traced) for s in self.seeds]
            c = self.coherence_profile(f)
            expected, expected_label = c, ("traced c(s)" if self.traced else "expected c(s), eq. 7")
        elif mode == "amplitude":
            self.filtered = [amplitude_only_filter(self.parent, vox, target, parent_res, kappa=kappa)]
            c = self.coherence_profile(f)
            expected, expected_label = np.where(c >= COHERENCE_FLOOR, 1.0, np.nan), "expected FSC = 1 (phases kept)"
        else:
            self.filtered = [hard_cutoff_filter(self.parent, vox, target)]
            c = (f <= 1.0 / target).astype(float)
            expected, expected_label = np.where(c > 0, 1.0, np.nan), "kept band, s ≤ 1/R"
        curves = np.vstack([fsc(self.parent, m, vox)[1] for m in self.filtered])
        self.store.figure("qc_filter.png", _figures().filtering(f, curves, expected, target, MODES[mode], expected_label), dpi=150)
        self.store.mrc("step2_filtered.mrc", self.filtered[0], vox)
        mean = _nanmean(curves, axis=0)
        chart = {"has_reference": self.has_reference, "target_res": target, "kappa": kappa,
                 "mode": "traced" if self.traced else ("cutoff" if mode == "cutoff" else "model"),
                 "envelope": {"freq": [float(x) for x in f], "value": [float(x) for x in c]}}
        if self.has_reference:
            filtered_low = _nanmean(np.vstack([fsc(m, self.low, vox)[1] for m in self.filtered]), axis=0)
            chart.update(parent_low=_curve(f, parent_low), filtered_low=_curve(f, filtered_low))
        else:
            chart.update(self_filtered=_curve(f, mean))
        metrics: dict[str, Any] = {"mode": MODES[mode], "filter_res_A": round(target, 3)}
        if mode == "ramped":
            metrics["n_seeds"] = count
        if mode != "cutoff":
            metrics.update({"kappa": kappa, "parent_res_A": parent_res or 0.0})
        metrics["fsc_parent_filtered_at_target"] = round(_at(f, mean, target), 4)
        if mode == "ramped":
            metrics["expected_c_at_target"] = round(_at(f, c, target), 4)
        if self.has_reference:
            metrics["fsc_parent_reference_at_target"] = round(_at(f, parent_low, target), 4)
        result = {"metrics": metrics, "plot": "asset:qc_filter.png", "fsc_chart": chart,
                  "download": {"label": f"filtered map, {MODES[mode].lower()} (MRC)",
                               "url": "asset:step2_filtered.mrc?dl=1"},
                  "viewer": self.viewer("step2_view", self.filtered[0], vox),
                  "viewer_label": f"{MODES[mode].lower()}" + (" (seed 0)" if mode == "ramped" else "")}
        if not self.has_reference:
            result.update(reference_viewer="asset:input_high.mrc", reference_label="original high map")
        return result

    # -- Step 3 ------------------------------------------------------------
    def step_amplitude(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.has_reference:
            raise ValueError("Amplitude matching needs a reference map.")
        if not self.filtered:
            raise ValueError("Run the filter step first.")
        vox = self.work_vox
        self.matched = [match_radial_amplitude(m, self.low, vox, 5.0) for m in self.filtered]
        f, before = radial_amplitude(self.filtered[0], vox)
        _, reference = radial_amplitude(self.low, vox)
        _, after = radial_amplitude(self.matched[0], vox)
        gain = np.ones_like(before)
        ok = np.isfinite(before) & (before > 0)
        gain[ok] = np.clip(reference[ok] / before[ok], 0, 5)
        gain[0] = 1.0
        self.store.figure("qc_amplitude.png", _figures().amplitude(f, before, reference, after, gain, self.target_res), dpi=150)
        self.store.mrc("step3_matched.mrc", self.matched[0], vox)
        return {"metrics": {"n_maps": len(self.matched), "max_gain": 5.0,
                            "note": "each Fourier shell rescaled to the reference's shell-mean amplitude"},
                "plot": "asset:qc_amplitude.png",
                "download": {"label": "amplitude-matched map, first seed (MRC)", "url": "asset:step3_matched.mrc?dl=1"},
                "viewer": self.viewer("step3_view", self.matched[0], vox), "viewer_label": "amplitude-matched map"}

    # -- Step 4 ------------------------------------------------------------
    def step_compare(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.has_reference:
            raise ValueError("Comparison needs a reference map.")
        if self.parent is None:
            raise ValueError("Run Step 1 (resample and register) first.")
        if self.matched:
            finals, stage = self.matched, "filtered"
        elif self.filtered:
            finals, stage = self.filtered, "filtered"
        else:
            finals, stage = [self.parent], "parent"
        vox, target, store = self.work_vox, self.target_res, self.store
        f, parent_low = fsc(self.parent, self.low, vox)
        expected_parent_low = envelope(f, target, self.kappa) * (
            envelope(f, self.parent_res, self.kappa) if self.parent_res else 1.0)       # eq. 8
        final_low = np.vstack([fsc(m, self.low, vox)[1] for m in finals])
        c = None
        if stage == "parent" or self.mode == "amplitude":
            factor = np.ones_like(f)          # all retained phases kept: the FSC is unchanged
        elif self.mode == "cutoff":
            factor = (f <= 1.0 / target).astype(float)
        else:
            factor = c = self.coherence_profile(f)                                      # Section S2
        predicted = factor * parent_low
        difference = finals[0] - self.low                                              # one map (Section 5.3)
        rms = local_rms(difference, target, vox)
        mask = union_mask(self.low)
        variances = [residual_variance(m - self.low, mask) for m in finals]
        label = "filtered" if stage == "filtered" else "parent"
        store.figure("qc_compare.png", _figures().comparison(
            f, parent_low, expected_parent_low, final_low, predicted, label, target, difference, rms, target), dpi=140)
        store.figure("qc_summary.png", _figures().summary(
            self.low, self.parent, finals[0], label, f, parent_low, final_low, c, target, rms, target), dpi=130)
        # results bundle
        names = [store.mrc("results/parent_registered.mrc", self.parent, vox),
                 store.mrc("results/reference_working_grid.mrc", self.low, vox)]
        if stage == "filtered":
            names += [store.mrc(f"results/filtered_seed{s}.mrc", m, vox) for s, m in zip(self.seeds, finals)]
        names += [store.mrc("results/difference_seed0.mrc", difference, vox),
                  store.mrc("results/local_rms_seed0.mrc", rms, vox)]
        names.append(store.csv("results/fsc.csv", {
            "frequency_invA": f, "fsc_parent_reference": parent_low,
            "expected_parent_reference_eq8": expected_parent_low,
            "fsc_final_reference_mean": _nanmean(final_low, axis=0),
            "fsc_final_reference_sd": _nanmean(final_low, sd=True),
            "expected_final_reference": predicted,
            "filter_coherence_c": c if c is not None else np.full_like(f, np.nan)}))
        record = {"compared": stage, "mode": MODES[self.mode] if stage == "filtered" else None,
                  "target_res_A": target, "parent_res_A": self.parent_res, "kappa": self.kappa,
                  "seeds": self.seeds if stage == "filtered" else [], "voxel_A": vox,
                  "grid_zyx": list(self.low.shape), "options": self.options, "inputs": self.meta,
                  "residual_variance_in_mask": variances}
        if self.registration is not None:
            r = self.registration
            record["registration"] = {"scale": r.scale, "rotation_deg": r.rotation_degrees,
                                      "translation_zyx_vox": r.translation_voxels.tolist(),
                                      "parameters": r.parameters.tolist(), "ncc_before": r.score_before,
                                      "ncc_after": r.score_after}
        names.append(store.json("results/parameters.json", json.loads(json.dumps(record, default=float))))
        names += [n for n in ("qc_register.png", "qc_filter.png", "qc_amplitude.png", "qc_compare.png",
                              "qc_summary.png") if n in store.files]
        store.zip("phaselp_results.zip", names)
        region = mask > 0.5
        metrics: dict[str, Any] = {
            "compared": (f"filtered maps ({MODES[self.mode].lower()}, n={len(finals)})" if stage == "filtered"
                         else "registered parent, unfiltered"),
            "fsc_parent_reference_at_target": round(_at(f, parent_low, target), 4),
            "expected_parent_reference_at_target": round(_at(f, expected_parent_low, target), 4),
        }
        if stage == "filtered":
            metrics["fsc_filtered_reference_at_target"] = round(_at(f, _nanmean(final_low, axis=0), target), 4)
            metrics["expected_filtered_reference_at_target"] = round(_at(f, predicted, target), 4)
        metrics.update({
            "residual_variance_in_mask": round(float(np.mean(variances)), 4),
            "local_rms_p95": round(float(np.percentile(rms[region], 95)), 4),
            "max_local_rms": round(float(rms[region].max()), 4),
        })
        downloads = [{"label": "FSC table (CSV)", "url": "asset:results/fsc.csv?dl=1"},
                     {"label": "difference map, one map (MRC)", "url": "asset:results/difference_seed0.mrc?dl=1"},
                     {"label": "local rms of the difference (MRC)", "url": "asset:results/local_rms_seed0.mrc?dl=1"}]
        downloads.append({"label": "all results (ZIP)", "url": "asset:phaselp_results.zip?dl=1"})
        return {"metrics": metrics, "plot": "asset:qc_compare.png", "summary_plot": "asset:qc_summary.png",
                "downloads": downloads, "viewer": self.viewer("step4_view", rms, vox),
                "viewer_label": f"local rms of {label} − reference (σ = {target:g} Å)"}


# --------------------------------------------------------------------------
# Inputs and dispatch (shared by the browser worker and native Python)
# --------------------------------------------------------------------------

SESSION = Session()
UPLOADS: dict[str, tuple[bytes, str]] = {}
_UPLOAD_ID = 0


async def fetch_bytes(url: str, timeout: float = 90.0) -> bytes:
    """Download with the browser's fetch under Pyodide, or urllib in ordinary Python."""
    try:
        from pyodide.http import pyfetch
    except ImportError:
        import urllib.request

        def get() -> bytes:
            request = urllib.request.Request(url, headers={"User-Agent": "phaselp"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()

        return await asyncio.get_running_loop().run_in_executor(None, get)

    async def get_async() -> bytes:
        response = await pyfetch(url)
        if not response.ok:
            raise ValueError(f"HTTP {response.status}")
        return await response.bytes()

    return await asyncio.wait_for(get_async(), timeout)


async def prepare_source(spec: dict[str, Any], role: str, emdb_cap: int) -> tuple[np.ndarray, float, dict]:
    kind, info = spec.get("type"), {}
    if kind == "demo":
        name = f"{role}.mrc.gz"
        if CONFIG["demo_dir"]:
            with open(f"{CONFIG['demo_dir']}/{name}", "rb") as handle:
                raw = handle.read()
        else:
            raw = await fetch_bytes(CONFIG["site_url"] + "demo/" + name)
        source = f"Bundled EMD-{8750 if role == 'high' else 1080} demo"
        if role == "high":
            info["note"] = ("Resampled (anti-aliased) from the deposited 240³ grid at 1.23 Å onto the 2.7 Å "
                            "voxel size of EMD-1080, as Step 1 would, to shorten the download.")
    elif kind == "upload":
        if spec.get("filename") not in UPLOADS:
            raise ValueError("Select the map file again; the previous upload has been cleared.")
        raw, original = UPLOADS[spec["filename"]]
        source = f"upload:{original}"
    elif kind == "emd":
        code, failures = sources.emdb_code(spec.get("value", "")), []
        for url in sources.emdb_map_urls(code):
            try:
                raw = await fetch_bytes(url)
                break
            except Exception as error:   # try the next mirror
                failures.append(str(error))
        else:
            raise ValueError(f"EMD-{code} could not be downloaded from either EMDB mirror. The browser may "
                             "block the request (CORS), the entry may be unavailable or the network may be "
                             "offline. Download its .map.gz from emdb.org and choose Upload MRC/MAP. "
                             + " / ".join(failures))
        source = f"EMD-{code}"
        try:
            entry = json.loads(await fetch_bytes(sources.EMDB_API.format(code=code), timeout=10))
            info["title"] = entry.get("admin", {}).get("title")
        except Exception:
            pass   # metadata is optional
    elif kind == "pdb":
        code = sources.pdb_code(spec.get("value", ""))
        resolution = _number(spec, "sim_res", 3.0, "Simulation resolution")
        voxel = _number(spec, "sim_apix", 1.5, "Simulation voxel size")
        failures = []
        for fmt in ("pdb", "cif"):
            try:
                text = (await fetch_bytes(sources.RCSB_URL.format(code=code, fmt=fmt))).decode("latin-1")
                atoms = sources.parse_cif(text) if fmt == "cif" else sources.parse_pdb(text)
                break
            except Exception as error:
                failures.append(str(error))
        else:
            raise ValueError(f"Could not load PDB {code} as PDB or mmCIF. Check the accession and network "
                             "connection, or upload a simulated MRC/MAP instead. " + " / ".join(failures))
        if np.prod(sources.simulated_shape(atoms, resolution, voxel)) > MAX_VOXELS:
            raise ValueError("This simulation exceeds the browser memory budget. Increase the simulation voxel size.")
        density = sources.simulate_density(atoms, resolution, voxel)
        return np.asarray(density, float), voxel, {
            "source": f"PDB {code} (simulated {resolution:g} Å)", "shape": list(density.shape),
            "voxel": round(voxel, 3), "n_atoms": len(atoms.coords), "format": fmt}
    else:
        raise ValueError(f"Unknown input type: {kind}")
    m = read_mrc_bytes(raw)
    data, voxel = m.data, m.voxel_size
    if kind == "emd" and emdb_cap and max(data.shape) > emdb_cap:
        factor = emdb_cap / max(data.shape)
        shape = tuple(max(2, int(round(n * factor))) for n in data.shape)
        info["original_shape"] = list(data.shape)
        data = resample_to_grid(data, voxel, shape, voxel / factor).astype(np.float32)
        voxel = voxel / factor
        info["note"] = (f"EMDB input resampled (anti-aliased) to a {emdb_cap}-voxel maximum dimension "
                        "for browser memory. Step 1 selects the working grid.")
    if data.size > MAX_VOXELS:
        raise ValueError("This map exceeds the browser memory budget (256³ voxels). Upload a downsampled map.")
    return np.asarray(data, float), float(voxel), {"source": source, "shape": list(data.shape),
                                                    "voxel": round(float(voxel), 3), **info}


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def file_bytes(name: str) -> bytes:
    return SESSION.store.files[name]


async def dispatch_json(method: str, payload_json: str, incoming=None) -> str:
    """Run one request from the page and return its JSON result."""
    global SESSION, _UPLOAD_ID
    payload = json.loads(payload_json or "{}")
    if method == "upload":
        _UPLOAD_ID += 1
        key = f"upload-{_UPLOAD_ID}"
        UPLOADS[key] = (bytes(incoming.to_py()) if hasattr(incoming, "to_py") else bytes(incoming), payload["name"])
        return json.dumps({"filename": key})
    if method == "reset":
        SESSION = Session()
        UPLOADS.clear()
        gc.collect()
        return '{"ok": true}'
    if method == "input":
        SESSION = Session()
        cap = int(_number(payload, "emdb_max_dimension", 160, "EMDB grid cap", 0, 256, inclusive=True))
        if cap and cap < 32:
            raise ValueError("EMDB grid cap must be 32–256, or 0 to keep the original grid.")
        try:
            prepared = {"high": await prepare_source(payload["high"], "high", cap)}
            if (payload.get("low") or {}).get("type") not in (None, "", "none"):
                prepared["low"] = await prepare_source(payload["low"], "low", cap)
            result = SESSION.load_inputs(prepared, payload)
        finally:
            UPLOADS.clear()
    elif method in STAGES:
        SESSION.invalidate(method)   # clear this and later stages even if validation fails
        result = getattr(SESSION, f"step_{method}")(payload)
        if SESSION.reference_viewer:
            result.setdefault("reference_viewer", SESSION.reference_viewer)
            result.setdefault("reference_label", "reference map")
    else:
        raise ValueError(f"Unknown operation: {method}")
    assets: set[str] = set()

    def collect(value):
        if isinstance(value, str) and value.startswith("asset:"):
            assets.add(value[6:].split("?")[0])
        elif isinstance(value, dict):
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(result)
    result["plot_data"] = {n: SESSION.store.plots[n] for n in assets if n in SESSION.store.plots}
    result["_assets"] = sorted(assets)
    gc.collect()
    return json.dumps(_clean(result), allow_nan=False)
