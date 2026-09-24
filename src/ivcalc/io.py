"""Reading what the pipeline wrote: planes, ledgers, binaries, metadata.

Thirteen scripts were importing `resolve_from_metadata` from `run_roi_suite2p`,
a command-line script kept alive only because it held that one function. The
readers live here instead, so no script is a library for another.
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

IVWIB_REPO = Path(os.environ.get(
    "IVWIB_REPO", "/media/tshino/DATA/Projects/in_vivo_water_imaging_brain"))

# Files Suite2p may write its image summaries into, newest naming first. In
# 1.1 the mean image moved out of ops.npy into reg_outputs.npy, so a reader
# that only knows ops.npy silently finds no background to draw on.
_OPS_FILES = ("reg_outputs.npy", "detect_outputs.npy", "ops.npy", "db.npy")


@dataclass
class Plane:
    """One Suite2p plane, with only what the analysis actually uses."""
    path: Path
    F: np.ndarray
    Fneu: np.ndarray
    stat: np.ndarray
    iscell: np.ndarray
    ops: dict = field(default_factory=dict)
    spks: np.ndarray | None = None

    @property
    def n_roi(self) -> int:
        return self.F.shape[0]

    @property
    def n_frames(self) -> int:
        return self.F.shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        for k in ("meanImg", "max_proj", "Vcorr"):
            if self.ops.get(k) is not None:
                return tuple(np.asarray(self.ops[k]).shape[:2])
        return (int(self.ops.get("Ly", 128)), int(self.ops.get("Lx", 128)))

    @property
    def crop_offset(self) -> tuple[int, int]:
        """Where the images sit inside the full field, if they were cropped."""
        ly, lx = int(self.ops.get("Ly", 0)), int(self.ops.get("Lx", 0))
        if ly and self.shape != (ly, lx) and self.ops.get("yrange") is not None:
            return int(self.ops["yrange"][0]), int(self.ops["xrange"][0])
        return (0, 0)

    def corrected(self, neucoeff: float) -> np.ndarray:
        return (self.F - neucoeff * self.Fneu).astype(np.float32)


def load_plane(path, *, all_roi: bool = False, need_spks: bool = False) -> Plane:
    """Read a plane directory, optionally keeping every detected ROI.

    `all_roi` matters more than it looks: the built-in classifier was trained on
    other indicators at other pixel sizes, and on this data it rejects most of
    what it finds. Whichever choice an analysis makes, every figure drawn from
    it has to make the same one or the ROI numbers stop matching.
    """
    p = Path(path).expanduser().resolve()
    F = np.load(p / "F.npy").astype(np.float32)
    Fneu = np.load(p / "Fneu.npy").astype(np.float32)
    stat = np.load(p / "stat.npy", allow_pickle=True)
    iscell = np.load(p / "iscell.npy")
    spks = None
    if (p / "spks.npy").exists():
        spks = np.load(p / "spks.npy").astype(np.float32)
    elif need_spks:
        raise FileNotFoundError(f"no spks.npy in {p}")

    ops: dict = {}
    for cand in _OPS_FILES:
        f = p / cand
        if f.exists():
            ops.update(np.load(f, allow_pickle=True).item())

    keep = np.ones(F.shape[0], bool) if all_roi else iscell[:, 0].astype(bool)
    if keep.size != F.shape[0]:
        keep = np.ones(F.shape[0], bool)
    return Plane(path=p, F=F[keep], Fneu=Fneu[keep],
                 stat=stat[keep] if len(stat) == keep.size else stat,
                 iscell=iscell[keep] if len(iscell) == keep.size else iscell,
                 ops=ops, spks=None if spks is None else spks[keep])


def load_ledger(path, n_frames: int | None = None):
    """Acquisition boundaries as (name, start, stop) with an exclusive stop.

    Returns a single whole-recording segment when there is no ledger, so a
    caller written for a series still works on one acquisition.
    """
    p = Path(path)
    if not p.exists():
        if n_frames is None:
            raise FileNotFoundError(f"no frame ledger at {p}")
        return [("all", 0, n_frames)]
    segs = []
    with open(p) as fh:
        for r in csv.DictReader(fh):
            segs.append((r["source_file"], int(r["frame_start"]),
                         int(r["frame_end"]) + 1))
    if n_frames is not None:
        covered = sum(b - a for _, a, b in segs)
        if covered != n_frames:
            raise ValueError(
                f"{p.name} covers {covered} frames but the traces have "
                f"{n_frames}. The ledger and the traces come from different "
                "runs, so the acquisitions cannot be separated.")
    return segs


def find_ledger(plane_path, explicit=None) -> Path | None:
    """The ledger that belongs to a plane, wherever the writer put it."""
    if explicit:
        return Path(explicit)
    p = Path(plane_path)
    for cand in (p.parent.parent / "frame_ledger.csv",
                 p.parent.parent.parent / "frame_ledger.csv"):
        if cand.exists():
            return cand
    return None


def read_metadata(dataset) -> dict:
    """Frame rate, pixel size and channel count from the dataset's metadata.

    Reads the file the metadata generator writes rather than re-parsing the
    ScanImage headers, so the analysis and the record agree by construction.
    """
    ds = Path(dataset).expanduser().resolve()
    meta = ds / "raw" / "metadata.yaml" if ds.is_dir() else Path(dataset)
    if not meta.exists():
        raise FileNotFoundError(f"no metadata at {meta}")
    import yaml
    with open(meta) as fh:
        d = yaml.safe_load(fh) or {}
    acq = (d.get("imaging", {}) or {}).get("acquisition", {}) or {}
    out = {"fs_hz": acq.get("frame_rate_hz"),
           "pixel_size_um": acq.get("pixel_size_um"),
           "n_channels": acq.get("n_channels"),
           "zoom": acq.get("zoom")}
    if out["n_channels"] is not None:
        out["n_channels"] = int(round(float(out["n_channels"])))
    return out


def resolve_fs(dataset=None, fs=None, *, quiet: bool = False) -> float:
    """Frame rate from the argument if given, otherwise from the metadata."""
    if fs is not None:
        return float(fs)
    if dataset is None:
        raise ValueError("pass --fs or --dataset")
    info = read_metadata(dataset)
    if info.get("fs_hz") is None:
        raise ValueError(f"no frame rate in the metadata for {dataset}")
    if not quiet:
        print(f"fs {info['fs_hz']:.4g} Hz from metadata", file=sys.stderr)
    return float(info["fs_hz"])


def open_binary(plane, ops: dict | None = None, chan2: bool = False):
    """Memory-map Suite2p's registered binary.

    This is the movie the analysis actually saw: registered, cropped to the
    valid region, and de-interleaved to the functional channel.
    """
    p = Path(plane)
    ops = ops or {}
    if not ops:
        for cand in _OPS_FILES:
            f = p / cand
            if f.exists():
                ops.update(np.load(f, allow_pickle=True).item())
    name = "data_chan2.bin" if chan2 else "data.bin"
    path = p / name
    if not path.exists():
        key = "reg_file_chan2" if chan2 else "reg_file"
        if ops.get(key) and Path(ops[key]).exists():
            path = Path(ops[key])
    if not path.exists():
        raise FileNotFoundError(
            f"no registered binary at {p / name}. Suite2p deletes it when "
            "delete_bin is set; re-run with delete_bin False.")
    ly, lx = int(ops["Ly"]), int(ops["Lx"])
    n = path.stat().st_size // (ly * lx * 2)
    return np.memmap(path, dtype=np.int16, mode="r", shape=(n, ly, lx)), ops


def write_rows(path, rows) -> None:
    """Write a list of dicts as CSV, taking the header from the first row."""
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
