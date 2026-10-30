# phaselp

Low-pass filtering with ramped phase randomization for cryo-EM and more:
filter signals, images and maps to a target resolution without carrying over
high-resolution phase detail, and augment training data by randomizing
high-frequency phases.

**Browser app:** https://jianglab.github.io/phaselp/

## Install

```bash
pip install git+https://github.com/jianglab/phaselp.git
```

Requires Python 3.10 or newer. This also installs numpy, scipy, mrcfile and
matplotlib, and the `phaselp` command-line tool.

To work on the code and run the tests:

```bash
git clone https://github.com/jianglab/phaselp.git && cd phaselp
pip install -e ".[dev]" && pytest
```

## Usage

```python
import phaselp as p
```

### Filter a map

```python
m = p.read_map("parent.mrc")
filtered = p.ramped_phase_filter(m.data, m.voxel_size, target_res=13.2, parent_res=3.0, seed=0)
p.write_map("filtered.mrc", filtered, m.voxel_size, m.origin)
```

- `target_res` is the resolution to filter to and `parent_res` the resolution
  of the input map (optional), both in Å.
- Each call returns one random realization. Use a different `seed` for
  another, and don't average realizations.
- 1-D signals and 2-D images work the same way, with `pixel_size=1` and
  resolutions in samples or pixels.

### Compare two maps at a common resolution

```python
parent, low = p.read_map("parent.mrc"), p.read_map("experimental.mrc")
c = p.compare_maps(parent.data, low.data, parent_voxel_size=parent.voxel_size,
                   voxel_size=low.voxel_size, target_res=13.2, parent_res=3.0, seeds=range(4))
curves = c.fsc_curves()     # FSC curves with the parent
diffs = c.differences()     # filtered − experimental map, one per seed
```

`compare_maps` registers the parent to the experimental map, filters it to
`target_res` and matches its radial amplitude to the experimental map.

### Augment training data

```python
augment = p.PhaseAugment(s0=0.04, s1=0.20, ndim=2)   # frequencies in cycles per pixel
batch = augment(batch)                                # batch: array of shape (n, h, w)

copies = p.phase_augment(batch, s0=0.04, s1=0.20, ndim=2, seed=0)   # one fixed copy of each
```

Phases are kept below `s0` and randomized increasingly up to `s1`.

### Other functions

- `p.fsc(a, b, pixel_size)`: Fourier shell correlation of two maps
- `p.register(fixed, moving, voxel_size)`: align two maps on the same grid
- `p.resample_to_grid(map, voxel_size, shape, new_voxel_size)`: resample a map onto another grid
- `p.amplitude_only_filter`, `p.hard_cutoff_filter`: conventional low-pass filters, for comparison
- `phaselp.plotting`: quick figures (`plot_filter_profile`, `plot_fsc`, `plot_sections`)

### Command line

```bash
phaselp filter  parent.mrc filtered.mrc --target-res 13.2 --parent-res 3.0
phaselp compare parent.mrc experimental.mrc results/ --target-res 13.2 --seeds 4 --plots
phaselp fsc     a.mrc b.mrc --out fsc.csv --plot fsc.png
```

Run `phaselp <command> --help` for all options.

## Browser app

https://jianglab.github.io/phaselp/ filters and compares maps in the browser,
with no installation, and loads a demo on startup. To run it locally, run
`python -m http.server 8000` in the repository root and open
http://localhost:8000/apps/.

## Citation

L. Dang and W. Jiang, *Low-pass filtering revisited: ramped phase
randomization for cryo-EM map comparison and data augmentation*.

## License

MIT
