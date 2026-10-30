# phaselp

Low-pass filtering with **ramped phase randomization** for cryo-EM map
comparison and data augmentation, as described in Dang & Jiang, *Low-pass
filtering revisited: ramped phase randomization for cryo-EM map comparison
and data augmentation*.

**Browser app:** https://jianglab.github.io/phaselp/ (runs entirely in your browser; no installation)

A conventional low-pass filter attenuates Fourier amplitudes but keeps every
phase wherever the amplitude response is above zero, so fine detail of the
parent survives beyond the target resolution. `phaselp` adds random phase
offsets whose spread ramps up smoothly with frequency (equation 2):

    F_H'(k) = T(s) · exp(i δ(k)) · F_H(k),   δ ~ N(0, σ²(s)),   E[cos δ] = c(s)

The filtered map then carries the parent's information only to the extent a
map at the target resolution would. This limits model bias in initial models
and templates, and lets maps of different resolution be compared on equal
terms.

## Install

```bash
pip install phaselp                                        # from PyPI
pip install git+https://github.com/jianglab/phaselp.git    # latest version on GitHub
```

Python 3.10 or newer. numpy, scipy, mrcfile and matplotlib are installed with
it, and so is the `phaselp` command-line tool.

## What is in the package

phaselp has three parts: a Python library, a command-line tool and the
browser app.

| Module | What it covers |
|---|---|
| `filters` | The ramped phase filter and the three comparison filters from the paper |
| `envelope` | c(s), the SSNR envelope and the expected-FSC formulas (eq 3–8, S1, S2) |
| `augment` | Ramped phase augmentation of training data (§5.4) |
| `compare` | The full map-comparison workflow and residual statistics (§3.1–3.2, §5.3) |
| `registration` | 7-parameter similarity registration (§5.1) |
| `fsc` | FSC and radial amplitude matching (§5.2) |
| `maps` | MRC read/write, standardization, anti-aliased resampling |
| `plotting` | Figures for each step |
| `fourier` | Internal helpers (frequency grids, conjugate-paired phase offsets) |
| `webapp` | The Python side of the browser app |

## Usage

```python
import phaselp as p
```

### 1. Filter a map, image or signal

This is the core method:

```python
m = p.read_map("parent.mrc")
filtered = p.ramped_phase_filter(m.data, m.voxel_size, target_res=13.2, parent_res=3.0, seed=0)
p.write_map("filtered.mrc", filtered, m.voxel_size, m.origin)
```

`parent_res` applies the correction of equation (7), `c = min(1, a_T / a_H)`;
leave it out to treat the parent as ideal (equation 5). The same call filters
1-D signals and 2-D images (`pixel_size=1`, resolutions in samples).

Each call returns one realization, so use the filtered maps one at a time.
Averaging realizations converges to the amplitude filter and restores the
parent's phases (Supplementary S1).

### 2. Compare a high-resolution map with a lower-resolution one

This is the workflow of Sections 3.1–3.2. The parent is registered, filtered
to the target resolution and amplitude-matched to the experimental map, and
the amplitude-only control is made alongside:

```python
parent, low = p.read_map("parent.mrc"), p.read_map("experimental.mrc")
c = p.compare_maps(parent.data, low.data, parent_voxel_size=parent.voxel_size,
                   voxel_size=low.voxel_size, target_res=13.2, parent_res=3.0,
                   seeds=range(8))
curves = c.fsc_curves()     # FSC with the parent: ramped phase, amplitude only, experimental, expected c(s)
diffs = c.differences()     # filtered − experimental, one per seed
```

To compare a rotated state with a same-state control inside a shared mask
(Section 3.2):

```python
mask = p.union_mask(low_a, low_b)
ratio = p.variance_ratio(diff_b, diff_a, mask)              # variance inside mask > 0.5
rms = p.local_rms(diff_b, sigma=13.2, voxel_size=c.voxel_size)
```

### 3. Augment training data

```python
copies = p.phase_augment(batch, s0=0.04, s1=0.20, ndim=2, seed=0)   # one fixed copy of each example
augment = p.PhaseAugment(s0=0.04, s1=0.20, ndim=2)                 # random (p = 0.5, s0 ± 10 %), for training loops
batch = augment(batch)                                              # (n, h, w); training inputs only
```

Frequencies are in cycles per sample (or pixel). The offset width rises by a
smoothstep from 0 at `s0` to `sigma_max` = 2.60 rad at `s1`. Fourier
magnitudes, the DC term and all phases below `s0` are preserved exactly. The
paper used these settings:

| Dataset | `s0` | `s1` |
|---|---|---|
| CryoBench particles | 0.04 | 0.20 |
| UCI Synthetic Control | 1/28 | 0.25 |
| Noisy MNIST | 0.16 | 0.28 |

### Supporting functions

- **Map I/O:** `read_map`, `write_map`
- **Measurement:** `fsc`, `match_radial_amplitude`
- **Registration:** `register`, `resample_to_grid`
- **Section 3.2 statistics:** `union_mask`, `local_rms`, `variance_ratio`
- **Other filters:** `amplitude_only_filter`, `hard_cutoff_filter`, `binary_phase_filter`
- **Envelope and expected-FSC formulas:** `coherence`, `envelope`, `sigma_from_coherence`, plus the `expected_fsc_*` functions

### Command line

```bash
phaselp filter  parent.mrc filtered.mrc --target-res 13.2 --parent-res 3.0 --seed 0
phaselp filter  parent.mrc aolp.mrc --target-res 13.2 --parent-res 3.0 --amplitude-only
phaselp compare parent.mrc experimental.mrc results/ --target-res 13.2 --parent-res 3.0 --seeds 8 --plots
phaselp fsc     a.mrc b.mrc --out fsc.csv --plot fsc.png
```

`phaselp compare` writes the files below into the output folder:
- one ramped-phase map per seed and the amplitude-only control, both
  amplitude-matched to the experimental map;
- the difference map and its local rms for the first filtered map (seed 0);
- the FSCs with the parent (`fsc_with_parent.csv`, with the expected `c(s)`);
- the registration record.

### Plots

```python
from phaselp import plotting
plotting.plot_filter_profile(13.2, 3.0, pixel_size=2.93)         # c(s), a_T, a_H and σ(s)
plotting.plot_sections([c.parent, c.experimental, c.amplitude_only, c.filtered[0]])
plotting.plot_fsc(c.fsc_curves(), target_res=13.2)               # Figure 2, last column
plotting.plot_differences({"control": diff_a, "rotated": diff_b}, sigma=13.2, voxel_size=2.93, mask=mask)
plotting.plot_signal_filters(signal, target_res=5)               # Figure 1B
plotting.plot_augmentation(x, augment(x), s0, s1)                # Figure 5A, E
```

## Browser app

https://jianglab.github.io/phaselp/ runs the comparison workflow step by step,
with 3-D isosurfaces, live sections, figures, metrics and downloads after
each step:

1. **Resample and register.** The parent is resampled onto the reference grid
   and registered there (Section 5.1). Optionally, choose a coarser working grid.
2. **Filter.** Choose ramped phase (equations 2, 5–7), amplitude only or hard
   cutoff. Two options are available:
   - *Construct to match* sets the target where the parent–reference FSC
     crosses 0.5.
   - *Trace profile* uses the measured parent–reference FSC as c(s)
     (Section 2.2).
3. **Match amplitudes.** Radial amplitude matching to the reference (Section 5.2).
4. **Compare.**
   - FSC(parent, reference) against equation 8, and FSC(filtered, reference)
     against c·FSC (Section S2).
   - The difference of one filtered map with the reference and its local rms
     (σ = target resolution), as in Figure 3.
   - The residual variance inside a soft mask of the reference. Its ratio to
     a same-state control is the statistic of Section 3.2.

Possible inputs:
- the bundled EMD-8750 / EMD-1080 demo, which loads on startup;
- uploaded MRC/MAP files;
- EMDB accessions, fetched in the browser where CORS allows;
- PDB entries, whose model density is simulated.

Without a reference, the app filters the high map alone. Everything runs in a
web worker with [Pyodide](https://pyodide.org) 0.29.3, using this same
library, and maps and results never leave the browser.

The first visit downloads Pyodide, NumPy, SciPy and NGL (about 32 MB, then
cached), plus 5 MB of demo maps. Registering the demo with the manuscript
settings (96-voxel search grid) takes about a minute. Inputs are limited to
256³ voxels, and filtering checks the memory needed for the chosen seed count.

**Demo maps.** EMD-1080 is bundled as deposited (100³ at 2.7 Å). EMD-8750
(240³ at 1.23 Å, 52 MB) is bundled resampled onto a 108³ grid at 2.7 Å
(4.7 MB), with the anti-aliased resampling that Step 1 applies. The script is
`scripts/make_demo.py`, and provenance is in `apps/demo/manifest.json`. The
reference grid is an exact sub-grid of the resampled map, so Step 1 gets the
same parent and the same results as from the full map. Those are a rotation
of 77.3°, NCC 0.924 and FSC(parent, reference) 0.679 at 11.5 Å.

## Where each part of the manuscript lives

| Manuscript | Function |
|---|---|
| Eq. 1, FSC (DC excluded, unmasked) | `fsc` |
| Eq. 2, the operator | `ramped_phase_filter`, `phase_augment` |
| Eq. 3, 6, coherence and offset width | `sigma_from_coherence`, `coherence_from_sigma` |
| Eq. 5, envelope (κ = 100) | `envelope`, `ssnr`, `coherence` |
| Eq. 7, parent correction | `coherence(s, target_res, parent_res)` |
| Eq. 4, 8; S1, S2 expected FSCs | `expected_fsc_independent`, `full_map_envelope`, `halfmap_envelope`, `expected_fsc_mean_of_realizations`, `expected_fsc_filtered_vs_other` |
| Amplitude-only, hard cutoff, binary (Fig. 1, 2; §2.2) | `amplitude_only_filter`, `hard_cutoff_filter`, `binary_phase_filter` |
| §5.1 registration | `register`, `resample_to_grid` |
| §5.2 phase and amplitude operations | `filter_profile`, `match_radial_amplitude` |
| §5.3 FSC and residual statistics | `standardize`, `difference_map`, `local_rms`, `union_mask`, `variance_ratio` |
| §5.4 augmentation | `ramp_sigma`, `phase_augment`, `PhaseAugment` |
| Workflow of §3.1–3.2 | `compare_maps` |
| Browser application (§5.5) | `apps/`, `phaselp.webapp` |

## Notes

* **Units.** For maps, `pixel_size` and resolutions are in Å and `s` in 1/Å.
  For signals, use `pixel_size=1` and resolutions in samples.
* **Real output.** Offsets are drawn per conjugate pair. Self-conjugate
  coefficients, including DC, are never perturbed, and coefficients with
  `c < 1e-6` are set to zero.
* **Registration.** It starts from the best-scoring of the 24 principal-axis
  orientations. The identity is scored as one more start, so maps whose
  principal axes are nearly degenerate cannot end in a worse fit than the one
  they began with. Each start's translation comes from phase correlation,
  which whitens only Fourier coefficients that carry signal. Whitening the
  empty stopband of the band-limited copies would amplify rounding noise and
  make the chosen start depend on the machine. For maps that are already
  aligned, use `register_maps=False` (CLI: `--no-register`).
* **Resampling.** `resample_to_grid` low-passes at the output Nyquist
  frequency before downsampling (`antialias=True`). The manuscript's 70S
  comparison used plain linear resampling of EMD-25420 from 0.83 to 3.0 Å.
  With anti-aliasing, the §3.2 numbers move slightly (125³ grid, published
  transforms):
  - FSC(parent, L): 0.649 → 0.661 and 0.368 → 0.371;
  - variance ratio: 2.47 → 2.50.

  Pass `antialias=False` to reproduce the published runs exactly.
* **Reproducibility.**
  - For the same seed, `ramped_phase_filter`, `amplitude_only_filter` and
    `binary_phase_filter` give maps bit-identical to the operator used for the
    manuscript figures.
  - `compare_maps` reproduces the Section 3.2 statistics from the Figure 3
    maps exactly (per-seed variance ratios 2.45–2.48).
  - `register` recovers the published 70S ribosome transforms (rotation to
    0.01°, scale to 2 × 10⁻⁵).

## Development

```bash
git clone https://github.com/jianglab/phaselp.git && cd phaselp
pip install -e ".[dev]" && pytest
```

* **Browser app (`apps/`).** It is a static site. `.github/workflows/pages.yml`
  publishes it to GitHub Pages on every push to `main` that changes it. The
  one-time setup is Settings → Pages → Source: **GitHub Actions**.
* **Local preview.** Run `python -m http.server 8000` in the repository root
  and open `http://localhost:8000/apps/`. Opening `apps/index.html` directly
  (`file://`) cannot work, because browsers block its scripts, and the page
  says so.
* **After changing the package.** Run `python scripts/build_site.py` to rebuild
  the copy the app loads (`apps/vendor/phaselp-<version>.whl`). Then run
  `node scripts/browser_check.cjs` for an end-to-end check in Chromium
  (Playwright; see the script header).
* **Releasing a new version.**
  1. Raise `version` in `pyproject.toml`, and the wheel name in `apps/worker.js`.
  2. Run `python scripts/build_site.py` and push.
  3. Create a GitHub release tagged `v<version>`.

  `.github/workflows/publish.yml` then tests the package and uploads it to PyPI.

### Changes from the earlier mapmatch browser app

* Defaults now follow Section 5.1: global orientation search, a 96-voxel
  search grid, and the reference grid as the working grid (previously 64
  voxels).
* Downsampling is anti-aliased. The old 64-voxel default with plain linear
  downsampling halved FSC(parent, reference) at 11.5 Å on the demo (0.35
  instead of 0.67).
* The local-rms map comes from a single filtered map. The old map averaged
  the seeds, which restores the parent's phases (Supplementary S1).
* The FSC chart draws the c(s) the filter applied. The old chart drew an
  outdated SNR/(1+SNR) envelope.
* DC is left unfiltered, "Cutoff" is the hard cutoff of Figure 1, and the
  expected parent–reference FSC uses equation 8.
* The real-versus-null "difference test" is removed. It scores even the
  manuscript's same-state 70S control at 13.4 (expected ≈ 1 for agreement),
  because two filtered maps of one parent are nearly identical below 1/R
  while any independent map is not.

## Citation

If you use phaselp, please cite: L. Dang and W. Jiang, *Low-pass filtering
revisited: ramped phase randomization for cryo-EM map comparison and data
augmentation*.

## License

MIT
