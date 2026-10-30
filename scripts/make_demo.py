"""Make the browser app's demo parent, apps/demo/high.mrc.gz, from the deposited EMD-8750 map.

    python scripts/make_demo.py emd_8750.map.gz          # from https://www.ebi.ac.uk/emdb/EMD-8750

The 240³ map at 1.23 Å is resampled onto a 108³ grid at the voxel size of the
bundled EMD-1080 reference (2.7 Å), with the anti-aliased resampling that
Step 1 of the app applies. The boxes stay centred, so Step 1 later takes the
100³ reference grid as an exact sub-grid of this one and gets the same parent as
it would from the full map. This cuts the download from 52 MB to 4.7 MB.
"""

import gzip
import hashlib
import sys
from pathlib import Path

import numpy as np

from phaselp import resample_to_grid
from phaselp.webapp.store import mrc_bytes, read_mrc_bytes

DEMO = Path(__file__).resolve().parents[1] / "apps" / "demo"
EDGE = 108          # the largest even edge whose samples all lie inside the deposited box

parent = read_mrc_bytes(Path(sys.argv[1]).read_bytes())
voxel = read_mrc_bytes((DEMO / "low.mrc.gz").read_bytes()).voxel_size
data = resample_to_grid(parent.data.astype(np.float64), parent.voxel_size, (EDGE,) * 3, voxel).astype(np.float32)
(DEMO / "high.mrc.gz").write_bytes(gzip.compress(mrc_bytes(data, voxel), 9, mtime=0))
# the MRC header label records a creation time, so the manifest hashes the voxel data
print(f"apps/demo/high.mrc.gz: {EDGE}³ at {voxel:.3f} Å, voxel-data sha256 "
      f"{hashlib.sha256(data.astype('<f4').tobytes()).hexdigest()}")
