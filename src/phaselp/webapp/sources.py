"""Map sources for the web app: EMDB accessions and densities simulated from PDB models.

Downloading is left to the caller (the browser's fetch in the web app), so
this module only builds URLs and turns coordinate text into a density map.
The simulated density is a simple model surrogate (atoms weighted by atomic
number, trilinear splatting and one Gaussian blur whose FWHM is the requested
resolution), not a scattering calculation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

EMDB_MIRRORS = (
    "https://ftp.ebi.ac.uk/pub/databases/emdb/structures",
    "https://files.wwpdb.org/pub/emdb/structures",
)
RCSB_URL = "https://files.rcsb.org/download/{code}.{fmt}"
EMDB_API = "https://www.ebi.ac.uk/emdb/api/entry/EMD-{code}"


def emdb_code(value: str | int) -> str:
    """Numeric part of an EMDB accession such as ``EMD-1080`` or ``1080``."""
    match = re.fullmatch(r"(?:EMD[-_ ]?)?(\d+)", str(value).strip(), re.I)
    if not match:
        raise ValueError("Enter an EMDB accession such as EMD-1080 or 1080.")
    return str(int(match[1]))


def emdb_map_urls(value: str | int) -> list[str]:
    """Map URLs for an EMDB entry on the EBI and wwPDB mirrors."""
    code = emdb_code(value)
    return [f"{base}/EMD-{code}/map/emd_{code}.map.gz" for base in EMDB_MIRRORS]


def pdb_code(value: str) -> str:
    code = str(value).strip().upper()
    if not re.fullmatch(r"[A-Za-z0-9]{4}|PDB_[A-Za-z0-9]{8}", code, re.I):
        raise ValueError("Enter a valid PDB accession, for example 1CRN or 4V9D.")
    return code


_ATOMIC_NUMBER = {"H": 1, "C": 6, "N": 7, "O": 8, "NA": 11, "MG": 12, "P": 15, "S": 16,
                  "CL": 17, "K": 19, "CA": 20, "MN": 25, "FE": 26, "CO": 27, "NI": 28,
                  "CU": 29, "ZN": 30, "SE": 34}


@dataclass
class Atoms:
    coords: np.ndarray   # (n, 3) x, y, z in angstrom
    weights: np.ndarray  # (n,) atomic numbers


def _weight(symbol: str) -> float:
    return float(_ATOMIC_NUMBER.get(symbol.strip().upper(), 6))


def parse_pdb(text: str) -> Atoms:
    coords, weights = [], []
    for line in text.splitlines():
        if not line.startswith(("ATOM", "HETATM")) or len(line) < 54:
            continue
        try:
            coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        except ValueError:
            continue
        element = line[76:78].strip() if len(line) >= 78 else ""
        weights.append(_weight(element or line[12:16].strip()[:1]))
    if not coords:
        raise ValueError("no ATOM/HETATM records found in the PDB file")
    return Atoms(np.asarray(coords, float), np.asarray(weights, float))


def parse_cif(text: str) -> Atoms:
    """Minimal mmCIF ``_atom_site`` reader (Cartesian coordinates and element)."""
    lines, coords, weights, i = text.splitlines(), [], [], 0
    while i < len(lines):
        if lines[i].strip() != "loop_":
            i += 1
            continue
        headers, j = [], i + 1
        while j < len(lines) and lines[j].strip().startswith("_"):
            headers.append(lines[j].strip())
            j += 1
        if any(h.startswith("_atom_site.") for h in headers):
            col = {h: k for k, h in enumerate(headers)}
            ix, iy, iz = (col.get(f"_atom_site.Cartn_{a}") for a in "xyz")
            ie = col.get("_atom_site.type_symbol")
            while j < len(lines):
                row = lines[j].split()
                if not row or lines[j].strip() in ("loop_", "#") or lines[j].startswith("_"):
                    break
                if len(row) >= len(headers) and ix is not None:
                    try:
                        coords.append((float(row[ix]), float(row[iy]), float(row[iz])))
                        weights.append(_weight(row[ie]) if ie is not None else 6.0)
                    except (ValueError, IndexError):
                        pass
                j += 1
        i = j
    if not coords:
        raise ValueError("no _atom_site records found in the mmCIF file")
    return Atoms(np.asarray(coords, float), np.asarray(weights, float))


def simulated_shape(atoms: Atoms, resolution: float, voxel_size: float, pad_sigma: float = 3.0) -> tuple:
    sigma = resolution / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    margin = pad_sigma * sigma + 2.0 * voxel_size
    dims = np.ceil((np.ptp(atoms.coords, axis=0) + 2 * margin) / voxel_size).astype(int) + 1
    return tuple(int(d) for d in dims[::-1])


def simulate_density(atoms: Atoms, resolution: float = 3.0, voxel_size: float = 1.5,
                     pad_sigma: float = 3.0) -> np.ndarray:
    """Density (z, y, x) on a grid of ``voxel_size``, blurred to Gaussian FWHM ``resolution``."""
    if resolution <= 0 or voxel_size <= 0:
        raise ValueError("resolution and voxel size must be positive")
    sigma = resolution / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    margin = pad_sigma * sigma + 2.0 * voxel_size
    lower = atoms.coords.min(axis=0) - margin
    nx, ny, nz = (int(d) for d in np.ceil((atoms.coords.max(axis=0) + margin - lower) / voxel_size) + 1)
    grid = np.zeros((nz, ny, nx))
    position = (atoms.coords - lower) / voxel_size
    base = np.floor(position).astype(int)
    frac = position - base
    for corner in range(8):
        cz, cy, cx = (corner >> 2) & 1, (corner >> 1) & 1, corner & 1
        ix, iy, iz = base[:, 0] + cx, base[:, 1] + cy, base[:, 2] + cz
        w = (atoms.weights * np.where(cx, frac[:, 0], 1 - frac[:, 0])
             * np.where(cy, frac[:, 1], 1 - frac[:, 1]) * np.where(cz, frac[:, 2], 1 - frac[:, 2]))
        inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
        np.add.at(grid, (iz[inside], iy[inside], ix[inside]), w[inside])
    return ndimage.gaussian_filter(grid, sigma / voxel_size, mode="constant").astype(np.float32)
