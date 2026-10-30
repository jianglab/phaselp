"""In-memory outputs of a web-app session: MRC maps, PNG figures, tables and ZIPs.

Nothing is written to disk, so the same code runs in the browser (Pyodide)
and in ordinary Python. Figures also record their plotted data, which the
page uses for hover readouts.
"""

from __future__ import annotations

import gzip
import io
import json
import zipfile
from typing import Any

import numpy as np

from ..maps import Map


def read_mrc_bytes(raw: bytes) -> Map:
    """Read an MRC/CCP4 map (optionally gzip-compressed) from bytes.

    Nonstandard axis orders are transposed to (z, y, x); voxels must be
    isotropic to within 1%.
    """
    from mrcfile.mrcinterpreter import MrcInterpreter

    raw = bytes(raw)
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    if len(raw) < 1024:
        raise ValueError("This file is too short to be an MRC/MAP volume.")
    with MrcInterpreter(io.BytesIO(raw), permissive=True) as mrc:
        if mrc.data is None or mrc.data.ndim != 3 or min(mrc.data.shape) < 2:
            raise ValueError("Choose a three-dimensional MRC/MAP density map.")
        data = np.asarray(mrc.data, dtype=np.float32)
        labels = [int(mrc.header.maps), int(mrc.header.mapr), int(mrc.header.mapc)]
        voxel = np.array([mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z], dtype=float)
        origin = (float(mrc.header.origin.x), float(mrc.header.origin.y), float(mrc.header.origin.z))
    if sorted(labels) == [1, 2, 3] and labels != [3, 2, 1]:
        data = data.transpose([labels.index(3), labels.index(2), labels.index(1)])
    if not (np.all(np.isfinite(voxel)) and np.all(voxel > 0)):
        raise ValueError("The map header must give a positive voxel size in Å.")
    if np.ptp(voxel) > 0.01 * voxel.mean():
        raise ValueError(f"Voxels must be isotropic; this map has {voxel.round(4).tolist()} Å.")
    if not np.all(np.isfinite(data)):
        raise ValueError("The map contains NaN or infinite density values.")
    return Map(np.ascontiguousarray(data), float(voxel.mean()), origin)


def mrc_bytes(data: np.ndarray, voxel_size: float) -> bytes:
    """Serialize a 3-D array (z, y, x) as a float32 MRC file."""
    from mrcfile.mrcinterpreter import MrcInterpreter

    buffer = io.BytesIO()
    mrc = MrcInterpreter()
    mrc._create_default_attributes()
    mrc._iostream = buffer
    mrc.set_data(np.asarray(data, dtype=np.float32))
    mrc.voxel_size = float(voxel_size)
    mrc.update_header_stats()
    mrc.flush()
    out = buffer.getvalue()
    mrc.close()
    return out


def _finite(values) -> list:
    return [float(v) if np.isfinite(v) else None for v in np.asarray(values, dtype=float).ravel()]


class Store:
    """Named byte outputs (``files``) and the plotted data of each figure (``plots``)."""

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.plots: dict[str, list] = {}

    def put(self, name: str, data: bytes) -> str:
        self.files[name] = bytes(data)
        return name

    def mrc(self, name: str, data: np.ndarray, voxel_size: float) -> str:
        return self.put(name, mrc_bytes(data, voxel_size))

    def csv(self, name: str, columns: dict[str, np.ndarray]) -> str:
        text = io.StringIO()
        np.savetxt(text, np.column_stack([np.asarray(v, float) for v in columns.values()]),
                   delimiter=",", header=",".join(columns), comments="", fmt="%.6g")
        return self.put(name, text.getvalue().encode())

    def json(self, name: str, value: Any) -> str:
        return self.put(name, (json.dumps(value, indent=2) + "\n").encode())

    def figure(self, name: str, fig, dpi: int = 150) -> str:
        """Save a matplotlib figure as PNG and keep its line and image data for hover readouts."""
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=dpi)
        axes = []
        for ax in fig.axes:
            item = {"bounds": list(ax.get_position().bounds), "title": ax.get_title(),
                    "xlabel": ax.get_xlabel(), "ylabel": ax.get_ylabel(),
                    "xlim": list(ax.get_xlim()), "ylim": list(ax.get_ylim()),
                    "xscale": ax.get_xscale(), "yscale": ax.get_yscale(), "lines": [], "images": []}
            for line in ax.lines:
                x, y = line.get_data()
                if len(x) == len(y) and len(x) > 0:
                    item["lines"].append({"label": line.get_label(), "x": _finite(x), "y": _finite(y)})
            for image in ax.images:
                array = np.asarray(image.get_array())
                if array.ndim == 2:
                    item["images"].append({"shape": list(array.shape), "values": _finite(array),
                                           "extent": list(image.get_extent()), "origin": image.origin})
            axes.append(item)
        self.plots[name] = axes
        return self.put(name, buffer.getvalue())

    def zip(self, name: str, members: list[str]) -> str:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for member in members:
                if member in self.files:
                    archive.writestr(member.split("/")[-1], self.files[member])
        return self.put(name, buffer.getvalue())

    def drop(self, predicate) -> None:
        for name in [n for n in self.files if predicate(n)]:
            self.files.pop(name, None)
            self.plots.pop(name, None)
