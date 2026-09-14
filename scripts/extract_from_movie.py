#!/usr/bin/env python
"""Re-extract traces from a denoised movie using an existing ROI set.

A denoiser can only be compared against the untouched data if both are read
through the same ROIs. Detecting ROIs separately on each version would change
the pixels being summed, and any difference in the activity rate afterwards
would be a mixture of the denoising and the redetection, with no way to tell
which contributed what.

So: ROIs are detected once, on the untouched registered movie, and this script
applies those same masks to whatever movie it is given. The output is written in
Suite2p's own layout, so every downstream script reads it without modification
and without knowing that the traces came from somewhere else.

Two modes.

    --export    write the registered binary out as a TIFF, which is what the
                denoisers take as input
    --movie     read a denoised movie and write F.npy and Fneu.npy for it

Example
-------
    # give the denoiser something to work on
    python extract_from_movie.py --s2p-dir <plane> --export work/registered.tif

    # after denoising, read it back through the same ROIs
    python extract_from_movie.py --s2p-dir <plane> \
        --movie work/registered_denoised.tif --out work/s2p_deepcad/suite2p/plane0
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def load_plane(plane: Path):
    d: dict = {}
    for cand in ("reg_outputs.npy", "detect_outputs.npy", "ops.npy", "db.npy"):
        f = plane / cand
        if f.exists():
            d.update(np.load(f, allow_pickle=True).item())
    stat = np.load(plane / "stat.npy", allow_pickle=True)
    iscell = np.load(plane / "iscell.npy")
    return d, stat, iscell


def open_binary(plane: Path, ops: dict, chan2=False):
    name = "data_chan2.bin" if chan2 else "data.bin"
    path = plane / name
    if not path.exists():
        key = "reg_file_chan2" if chan2 else "reg_file"
        if ops.get(key) and Path(ops[key]).exists():
            path = Path(ops[key])
    if not path.exists():
        raise FileNotFoundError(
            f"no registered binary at {plane / name}; re-run Suite2p with "
            "delete_bin False")
    ly, lx = int(ops["Ly"]), int(ops["Lx"])
    n = path.stat().st_size // (ly * lx * 2)
    return np.memmap(path, dtype=np.int16, mode="r", shape=(n, ly, lx))


def neuropil_masks(stat, shape, inner=2, min_pixels=350, max_radius=30):
    """Annulus around each ROI, excluding every ROI's pixels.

    Matches what Suite2p extracts into Fneu: a ring that starts a few pixels
    outside the cell and grows until it holds enough pixels, with all somatic
    pixels removed so the ring never contains another cell.
    """
    ly, lx = shape
    occupied = np.zeros(shape, bool)
    for s in stat:
        occupied[s["ypix"], s["xpix"]] = True
    yy, xx = np.mgrid[0:ly, 0:lx]
    masks = []
    for s in stat:
        cy = float(np.mean(s["ypix"]))
        cx = float(np.mean(s["xpix"]))
        r2 = (yy - cy) ** 2 + (xx - cx) ** 2
        own = np.zeros(shape, bool)
        own[s["ypix"], s["xpix"]] = True
        r_in = np.sqrt(own.sum() / np.pi) + inner
        for r_out in range(int(r_in) + 2, max_radius):
            ring = (r2 > r_in ** 2) & (r2 <= r_out ** 2) & ~occupied
            if ring.sum() >= min_pixels:
                break
        masks.append(ring)
    return masks


def extract(mov, stat, npil, batch=500):
    """Weighted somatic sums and unweighted neuropil means, as Suite2p does."""
    n_t, ly, lx = mov.shape
    n_roi = len(stat)
    F = np.zeros((n_roi, n_t), np.float32)
    Fneu = np.zeros((n_roi, n_t), np.float32)
    idx = [np.ravel_multi_index((s["ypix"], s["xpix"]), (ly, lx)) for s in stat]
    lam = [np.asarray(s["lam"], np.float32) if "lam" in s
           else np.ones(len(s["xpix"]), np.float32) for s in stat]
    lam = [w / max(w.sum(), 1e-9) for w in lam]
    nidx = [np.flatnonzero(m.ravel()) for m in npil]
    for a in range(0, n_t, batch):
        b = min(a + batch, n_t)
        chunk = np.asarray(mov[a:b], np.float32).reshape(b - a, -1)
        for i in range(n_roi):
            F[i, a:b] = chunk[:, idx[i]] @ lam[i]
            if nidx[i].size:
                Fneu[i, a:b] = chunk[:, nidx[i]].mean(axis=1)
        print(f"  {b / n_t * 100:3.0f}%", end="\r", flush=True)
    print(" " * 12, end="\r")
    return F, Fneu


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="re-extract traces from a denoised movie, same ROIs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--s2p-dir", type=Path, required=True,
                   help="plane0 of the detection run that defines the ROIs")
    p.add_argument("--export", type=Path, default=None,
                   help="write the registered binary out as a TIFF and stop")
    p.add_argument("--movie", type=Path, default=None,
                   help="denoised movie to read through the existing ROIs")
    p.add_argument("--out", type=Path, default=None,
                   help="output plane directory (created, Suite2p layout)")
    p.add_argument("--crop", action="store_true",
                   help="export only the registration's valid region")
    p.add_argument("--frames", type=int, nargs=2, default=None, metavar=("A", "B"))
    p.add_argument("--inner-neuropil-radius", type=int, default=2)
    p.add_argument("--min-neuropil-pixels", type=int, default=350)
    p.add_argument("--dtype", default="int16", choices=["int16", "float32"])
    args = p.parse_args(argv)

    plane = args.s2p_dir.expanduser().resolve()
    ops, stat, iscell = load_plane(plane)
    ly, lx = int(ops["Ly"]), int(ops["Lx"])

    # --- export -------------------------------------------------------------
    if args.export:
        import tifffile
        mov = open_binary(plane, ops)
        a, b = args.frames if args.frames else (0, mov.shape[0])
        sl = (slice(a, b),)
        if args.crop and ops.get("yrange") is not None:
            (y0, y1) = map(int, ops["yrange"])
            (x0, x1) = map(int, ops["xrange"])
            sl = sl + (slice(y0, y1), slice(x0, x1))
            print(f"cropping to the valid region y {y0}:{y1} x {x0}:{x1}")
        out = args.export.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        arr = np.asarray(mov[sl])
        tifffile.imwrite(out, arr.astype(args.dtype))
        print(f"wrote {out}  {arr.shape} {args.dtype} "
              f"({out.stat().st_size / 1e9:.2f} GB)")
        print("\nNOTE: a denoiser that assumes neighbouring frames share the "
              "signal and not\n  the noise needs the movie to be registered "
              "first, which this one is. Feed\n  it this file, then read the "
              "result back with --movie.")
        return 0

    if not args.movie or not args.out:
        p.error("pass --export, or both --movie and --out")

    # --- re-extract ---------------------------------------------------------
    import tifffile
    src = args.movie.expanduser().resolve()
    mov = tifffile.imread(str(src))
    if mov.ndim != 3:
        print(f"ERROR: {src.name} has shape {mov.shape}", file=sys.stderr)
        return 2
    print(f"movie: {src.name}  {mov.shape}  {mov.dtype}")

    offset = (0, 0)
    if mov.shape[1:] != (ly, lx):
        if ops.get("yrange") is not None:
            (y0, y1) = map(int, ops["yrange"])
            (x0, x1) = map(int, ops["xrange"])
            if mov.shape[1:] == (y1 - y0, x1 - x0):
                offset = (y0, x0)
                print(f"  movie is the cropped valid region; ROI coordinates "
                      f"shifted by ({y0}, {x0})")
            else:
                print(f"ERROR: movie is {mov.shape[1:]}, the ROIs live in "
                      f"({ly}, {lx}) and the valid region is "
                      f"({y1 - y0}, {x1 - x0})", file=sys.stderr)
                return 2
        else:
            print(f"ERROR: movie is {mov.shape[1:]} but the ROIs live in "
                  f"({ly}, {lx})", file=sys.stderr)
            return 2

    shape = mov.shape[1:]
    stat2 = []
    for s in stat:
        d = {k: (np.asarray(v).copy() if isinstance(v, np.ndarray) else v)
             for k, v in s.items()}
        d["ypix"] = np.asarray(s["ypix"]) - offset[0]
        d["xpix"] = np.asarray(s["xpix"]) - offset[1]
        keep = ((d["ypix"] >= 0) & (d["ypix"] < shape[0])
                & (d["xpix"] >= 0) & (d["xpix"] < shape[1]))
        d["ypix"], d["xpix"] = d["ypix"][keep], d["xpix"][keep]
        if "lam" in d:
            d["lam"] = np.asarray(d["lam"])[keep]
        stat2.append(d)
    dropped = sum(1 for d in stat2 if d["xpix"].size == 0)
    if dropped:
        print(f"WARNING: {dropped} ROI(s) fall outside the movie", file=sys.stderr)

    print("building neuropil rings ...")
    npil = neuropil_masks(stat2, shape, args.inner_neuropil_radius,
                          args.min_neuropil_pixels)
    print(f"extracting {len(stat2)} ROIs from {mov.shape[0]} frames ...")
    F, Fneu = extract(mov, stat2, npil)

    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "F.npy", F)
    np.save(out / "Fneu.npy", Fneu)
    np.save(out / "stat.npy", np.array(stat2, dtype=object))
    np.save(out / "iscell.npy", iscell)
    # carry the images so the QC and figure scripts still find a background
    reg = {k: ops[k] for k in ("meanImg", "meanImgE", "refImg", "Vcorr",
                               "max_proj", "yrange", "xrange", "bidiphase")
           if k in ops}
    reg.update({"Ly": shape[0], "Lx": shape[1], "nframes": int(mov.shape[0])})
    if offset != (0, 0):
        for k in ("meanImg", "meanImgE", "refImg"):
            if k in reg and np.asarray(reg[k]).shape == (ly, lx):
                reg[k] = np.asarray(reg[k])[offset[0]:offset[0] + shape[0],
                                            offset[1]:offset[1] + shape[1]]
        reg["yrange"] = [0, shape[0]]
        reg["xrange"] = [0, shape[1]]
    np.save(out / "reg_outputs.npy", np.array(reg, dtype=object))

    src_ledger = plane.parent.parent / "frame_ledger.csv"
    if src_ledger.exists():
        shutil.copy2(src_ledger, out.parent.parent / "frame_ledger.csv")
        print(f"copied {src_ledger.name} so the acquisitions stay separable")

    with open(out / "extraction_record.json", "w") as fh:
        json.dump({"source_movie": str(src), "roi_source": str(plane),
                   "n_roi": len(stat2), "n_frames": int(mov.shape[0]),
                   "offset_yx": list(offset),
                   "note": "traces re-extracted with the ROI set from "
                           "roi_source, so this version and the untouched one "
                           "are read through identical masks"},
                  fh, indent=2)
    print(f"\nwrote F.npy {F.shape} and Fneu.npy to {out}")
    print(f"  mean F {F.mean():.1f}   mean Fneu {Fneu.mean():.1f}")
    print("\nrun the usual analysis against this directory:")
    print(f"  python run_event_auc.py --s2p-dir {out} --dataset <dataset> \\")
    print(f"      --ledger {out.parent.parent / 'frame_ledger.csv'} "
          "--all-roi --neucoeff 0 --out <out>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
