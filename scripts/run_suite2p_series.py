#!/usr/bin/env python
"""Register and segment a series of ScanImage acquisitions in one Suite2p run.

Suite2p concatenates every file it is given into a single binary and registers
the whole thing against one reference image, then detects ROIs once on the
result. That is what makes a series comparable: one reference means one
coordinate frame, and one detection pass means one ROI set, so a trace can be
split by acquisition afterwards and the same cell is the same row throughout.
Running Suite2p once per acquisition would produce a different ROI set each
time and the per-acquisition values could not be compared.

Two points that are easy to get wrong:

  file_list  -- the db key that restricts processing to specific files is
                `file_list`. `tiff_list` is not a key; passing it is silently
                ignored and Suite2p processes every tiff in data_path, which
                looks like a working run but concatenates unrelated files.
                The frame-count check at the end catches this.

  nchannels  -- ScanImage writes channels interleaved in one file, and Suite2p
                de-interleaves them itself given `nchannels` and
                `functional_chan` (1-based). Splitting channels beforehand is
                unnecessary.

Example
-------
    python run_suite2p_series.py \
        --dataset <dataset_dir> \
        --pattern 'sub-sk28_ses-01_cond-*_run-*.tif' \
        --functional-chan 1 --torch-device cuda
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import inspect
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

IVWIB_REPO = Path(os.environ.get(
    "IVWIB_REPO", "/media/tshino/DATA/Projects/in_vivo_water_imaging_brain"))
_nd = IVWIB_REPO / "src" / "metadata"
if _nd.is_dir():
    sys.path.insert(0, str(_nd))
try:
    from naming import FILE_RE
except ImportError:
    FILE_RE = None


# Acquisitions that belong before the others, whatever their run numbers say.
# A baseline recorded first is the reference every later acquisition is compared
# against, so it has to sit at the start of the series; the ordering cannot be
# left to the file names alone.
_COND_ORDER = {"baseline": 0, "pre": 0, "ctrl": 0, "control": 0}


def order_key(p: Path):
    """Sort by condition, then run number, then acquisition number.

    Sorting by the trailing acquisition number alone reorders a series whenever
    a run was restarted: a run-01 retaken as _00003 lands after run-05's
    _00001, and the time course silently comes out shuffled. The run number is
    what carries the order, and the condition comes before it so a baseline
    stays first.
    """
    name = p.name
    cond = re.search(r"_cond-([^_]+)", name)
    run = re.search(r"_run-(\d+)", name)
    idx = None
    if FILE_RE is not None:
        m = FILE_RE.match(name)
        if m:
            try:
                idx = int(m.group("idx"))
            except (IndexError, TypeError, ValueError):
                idx = None
    if idx is None:
        nums = re.findall(r"(\d+)", p.stem)
        idx = int(nums[-1]) if nums else 10**9
    c = cond.group(1) if cond else ""
    return (_COND_ORDER.get(c, 1), c, int(run.group(1)) if run else 10**9,
            idx, name)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="one Suite2p run over a series of acquisitions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--pattern", default="*.tif", help="glob inside raw/")
    p.add_argument("--cond", default=None,
                   help="comma-separated conditions to keep, in this order "
                        "(e.g. 'baseline,wi'). Overrides --exclude-cond")
    p.add_argument("--exclude-cond", default="qc",
                   help="comma-separated conditions to drop. Alignment and "
                        "focus checks live under the same subject and match "
                        "the same glob, but they are short, often a different "
                        "frame size, and are not part of the series")
    p.add_argument("--min-frames", type=int, default=100,
                   help="drop acquisitions shorter than this. An aborted "
                        "recording of a few frames has no usable baseline and "
                        "would still become its own acquisition in the ledger")
    p.add_argument("--save-path", type=Path, default=None)
    p.add_argument("--fs", type=float, default=None)
    p.add_argument("--pixel-size-um", type=float, default=None)
    p.add_argument("--nchannels", type=int, default=None,
                   help="channels interleaved in each file (default: from metadata.yaml)")
    p.add_argument("--functional-chan", type=int, default=1,
                   help="1-based channel used for detection")
    p.add_argument("--soma-um", type=float, default=15.0)
    p.add_argument("--diameter", type=int, default=None)
    p.add_argument("--algorithm", default="cellpose",
                   choices=["cellpose", "sparsery", "sourcery"])
    p.add_argument("--cellpose-img", default="meanImg",
                   choices=["meanImg", "max_proj", "max_proj / meanImg"])
    p.add_argument("--cellprob-threshold", type=float, default=None)
    p.add_argument("--spatial-scale", type=int, default=0, choices=[0, 1, 2, 3, 4])
    p.add_argument("--threshold-scaling", type=float, default=1.0)
    p.add_argument("--denoise", action="store_true",
                   help="PCA-denoise the binned movie before detection "
                        "(sparsery only). This affects which ROIs are found and "
                        "nothing else: the extracted traces are still the raw "
                        "weighted pixel sums, so it does not improve event "
                        "detection downstream")
    p.add_argument("--denoise-block-size", type=int, nargs=2, default=None,
                   metavar=("Y", "X"))
    p.add_argument("--nbins", type=int, default=None,
                   help="max binned frames used for detection")
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--nonrigid", action="store_true",
                   help="piecewise-rigid registration (refused on a small field)")
    p.add_argument("--nonrigid-min-size", type=int, default=256)
    p.add_argument("--maxregshift", type=float, default=0.1)
    p.add_argument("--smooth-sigma-time", type=float, default=1.0,
                   help="temporal smoothing for SHIFT ESTIMATION only; the saved data "
                        "is not smoothed. Helps at low SNR")
    p.add_argument("--two-step", action="store_true", default=True)
    p.add_argument("--do-bidiphase", action="store_true", default=True)
    p.add_argument("--torch-device", default="cuda")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    ds = args.dataset.expanduser().resolve()
    raw = ds / "raw"
    save_path = (args.save_path or ds / "work" / "s2p_series").expanduser().resolve()

    files = sorted(raw.glob(args.pattern), key=order_key)
    if not files:
        print(f"ERROR: no files matched {args.pattern!r} under {raw}",
              file=sys.stderr)
        return 2

    # --- keep only the conditions that make up the series -------------------
    # A QC or alignment recording sits in the same folder under the same
    # subject and matches the same glob, but it is not part of the series:
    # including it puts a different field, and often a different frame size,
    # into the shared registration reference.
    def cond_of(f):
        m = re.search(r"_cond-([^_]+)", f.name)
        return m.group(1) if m else ""

    dropped = []
    if args.cond:
        want = [c.strip() for c in args.cond.split(",") if c.strip()]
        rank = {c: i for i, c in enumerate(want)}
        keep = [f for f in files if cond_of(f) in rank]
        dropped += [(f, "condition not in --cond") for f in files
                    if cond_of(f) not in rank]
        files = sorted(keep, key=lambda f: (rank[cond_of(f)], order_key(f)))
    elif args.exclude_cond:
        drop = {c.strip() for c in args.exclude_cond.split(",") if c.strip()}
        dropped += [(f, "excluded condition") for f in files
                    if cond_of(f) in drop]
        files = [f for f in files if cond_of(f) not in drop]
    if not files:
        print("ERROR: every file was excluded by --cond / --exclude-cond",
              file=sys.stderr)
        return 2

    # --- acquisition parameters ---------------------------------------------
    fs, px, nch = args.fs, args.pixel_size_um, args.nchannels
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    meta_path = raw / "metadata.yaml"
    if meta_path.exists():
        try:
            from run_roi_suite2p import resolve_from_metadata
            info = resolve_from_metadata(meta_path)
            fs = fs if fs is not None else info.get("fs_hz")
            px = px if px is not None else info.get("pixel_size_um")
            if nch is None and info.get("n_channels"):
                nch = int(round(float(info["n_channels"])))
            print(f"metadata: fs={fs}  pixel_size_um={px}  n_channels={nch}")
        except Exception as e:  # noqa: BLE001
            print(f"metadata unreadable: {e}", file=sys.stderr)
    if fs is None:
        print("ERROR: frame rate unresolved; pass --fs", file=sys.stderr)
        return 2
    nch = nch or 1
    if not 1 <= args.functional_chan <= nch:
        print(f"ERROR: --functional-chan {args.functional_chan} outside 1..{nch}",
              file=sys.stderr)
        return 2

    # --- inputs and expected frame count ------------------------------------
    import tifffile

    if args.min_frames > 0:
        keep, short = [], []
        for f in files:
            with tifffile.TiffFile(f) as tf:
                pages = len(tf.pages)
            (keep if pages // max(nch, 1) >= args.min_frames
             else short).append(f)
        dropped += [(f, f"fewer than {args.min_frames} frames") for f in short]
        files = keep
        if not files:
            print(f"ERROR: every file is shorter than --min-frames "
                  f"{args.min_frames}", file=sys.stderr)
            return 2

    if dropped:
        print(f"\nexcluded {len(dropped)} file(s):")
        for f, why in sorted(dropped, key=lambda x: x[0].name):
            print(f"  {f.name:54s} {why}")

    ledger, total, shapes = [], 0, set()
    conds = [re.search(r"_cond-([^_]+)", f.name) for f in files]
    conds = [c.group(1) if c else "?" for c in conds]
    if len(set(conds)) > 1:
        print("\nconditions in order: " + " -> ".join(dict.fromkeys(conds)))
    print(f"\ninputs from {raw}")
    for f in files:
        with tifffile.TiffFile(f) as tf:
            pages = len(tf.pages)
            shapes.add(tf.pages[0].shape[:2])
        if pages % nch:
            print(f"ERROR: {f.name}: {pages} pages not divisible by nchannels={nch}",
                  file=sys.stderr)
            return 2
        n = pages // nch
        ledger.append({"source_file": f.name, "frame_start": total,
                       "frame_end": total + n - 1, "n_frames": n})
        print(f"  {f.name:52s} {n:6d} frames  [{total} .. {total + n - 1}]")
        total += n
    if len(shapes) > 1:
        print(f"\nERROR: the selected files are not all the same size: "
              f"{shapes}.\n  One registration reference cannot span two frame "
              "sizes. Narrow the\n  selection with --cond, or widen "
              "--exclude-cond.", file=sys.stderr)
        return 2
    ny, nx = shapes.pop()
    print(f"  total {total} frames = {total / fs / 60:.1f} min at {fs:.4g} Hz, "
          f"{ny} x {nx} px")

    nonrigid = args.nonrigid
    if nonrigid and min(ny, nx) < args.nonrigid_min_size and not args.force:
        print(f"\nERROR: refusing non-rigid registration on a {ny}x{nx} field. The "
              "default\n  block size leaves about one block per dimension, and at low "
              "SNR the\n  per-block shift estimate fits noise. Use rigid, or --force.",
              file=sys.stderr)
        return 2

    diameter = args.diameter
    if diameter is None and px:
        diameter = int(round(args.soma_um / px))
    if px:
        print(f"\nscale: {px:.3g} um/px -> FOV {nx * px:.0f} x {ny * px:.0f} um; "
              f"a {args.soma_um:g} um soma spans {args.soma_um / px:.1f} px")

    if args.dry_run:
        print(f"\n--dry-run: would write to {save_path}")
        print(f"  registration: {'non-rigid' if nonrigid else 'rigid'}, one reference "
              f"across all {len(files)} acquisition(s)")
        print(f"  detection: {args.algorithm}, diameter={diameter}")
        return 0

    # --- settings ------------------------------------------------------------
    import suite2p
    params = inspect.signature(suite2p.run_s2p).parameters
    if "settings" not in params:
        print("ERROR: this script targets the suite2p 1.1 API", file=sys.stderr)
        return 2
    settings = copy.deepcopy(params["settings"].default)

    settings["fs"] = float(fs)
    settings["tau"] = args.tau
    settings["torch_device"] = args.torch_device
    if diameter:
        settings["diameter"] = [float(diameter), float(diameter)]

    run = settings.setdefault("run", {})
    run["do_registration"] = 1
    run["do_regmetrics"] = total >= 1500   # registration metrics need >=1500 frames
    run["do_detection"] = True
    run["do_deconvolution"] = False        # not part of this pipeline

    reg = settings.setdefault("registration", {})
    reg["nonrigid"] = nonrigid
    reg["maxregshift"] = args.maxregshift
    reg["smooth_sigma_time"] = args.smooth_sigma_time
    reg["two_step_registration"] = args.two_step
    reg["do_bidiphase"] = args.do_bidiphase

    det = settings.setdefault("detection", {})
    det["algorithm"] = args.algorithm
    det["threshold_scaling"] = args.threshold_scaling
    if args.denoise:
        if args.algorithm != "sparsery":
            print("NOTE: denoise applies to sparsery only; ignored for "
                  f"{args.algorithm}", file=sys.stderr)
        det["denoise"] = True
        if args.denoise_block_size:
            det["block_size"] = tuple(args.denoise_block_size)
    if args.nbins:
        det["nbins"] = args.nbins
    if args.algorithm == "cellpose":
        cp = det.setdefault("cellpose_settings", {})
        cp["img"] = args.cellpose_img
        if args.cellprob_threshold is not None:
            cp["cellprob_threshold"] = args.cellprob_threshold
    else:
        det.setdefault("sparsery_settings", {})["spatial_scale"] = args.spatial_scale

    io_ = settings.setdefault("io", {})
    io_["delete_bin"] = False
    io_["move_bin"] = False

    save_path.mkdir(parents=True, exist_ok=True)

    # Stage the chosen files in a directory of their own, as symlinks.
    #
    # `file_list` is the documented way to restrict which files Suite2p reads,
    # and it is not reliable here: a run that selected 24000 frames came back
    # with 28173, having also read the QC recordings that share the folder. A
    # directory containing nothing else cannot be over-read whatever the key
    # does, and the frame-count check at the end still verifies it.
    stage = save_path / "input"
    if stage.exists():
        for old_link in stage.iterdir():
            old_link.unlink()
    stage.mkdir(parents=True, exist_ok=True)
    staged = []
    for f in files:
        link = stage / f.name
        link.symlink_to(f.resolve())
        staged.append(link)
    print(f"staged {len(staged)} file(s) in {stage}")

    db = dict(suite2p.default_db())
    db.update(
        data_path=[str(stage)],
        file_list=[f.name for f in staged],  # belt and braces; the directory
                                             # is what actually guarantees it
        save_path0=str(save_path),
        nplanes=1,
        nchannels=nch,
        functional_chan=args.functional_chan,
        keep_movie_raw=args.two_step,
    )

    print(f"\nrunning Suite2p on {args.torch_device}")
    print(f"  {len(files)} file(s), one reference, {'non-rigid' if nonrigid else 'rigid'}")
    print(f"  nchannels={nch}  functional_chan={args.functional_chan}")
    print(f"  detection={args.algorithm}  diameter={settings.get('diameter')}"
          + ("  denoise=True" if det.get("denoise") else ""))
    suite2p.run_s2p(db=db, settings=settings)

    # --- outputs -------------------------------------------------------------
    plane = save_path / "suite2p" / "plane0"
    if not plane.exists():
        cands = sorted(save_path.glob("**/plane0"))
        if cands:
            plane = cands[0]
    F = np.load(plane / "F.npy")
    stat = np.load(plane / "stat.npy", allow_pickle=True)
    iscell = np.load(plane / "iscell.npy")
    print(f"\noutputs in {plane}")
    print(f"detected {len(stat)} ROIs; classifier accepts {int(iscell[:, 0].sum())}")
    print(f"F shape {F.shape}")

    if F.shape[1] != total:
        print(f"\nERROR: {F.shape[1]} timepoints but the selected inputs hold "
              f"{total} frames.\n  Suite2p read {F.shape[1] - total} frames "
              "that were not selected, despite the staged\n  input directory. "
              f"Check what is inside {stage}.", file=sys.stderr)
        return 2

    # registration quality, per acquisition
    regp = plane / "reg_outputs.npy"
    reg_out = np.load(regp, allow_pickle=True).item() if regp.exists() else {}
    if not reg_out and (plane / "ops.npy").exists():
        reg_out = np.load(plane / "ops.npy", allow_pickle=True).item()

    xo, yo = reg_out.get("xoff"), reg_out.get("yoff")
    corr = reg_out.get("corrXY")
    if xo is not None:
        print("\nregistration per acquisition "
              "(shift magnitude and correlation with the shared reference):")
        for row in ledger:
            a, b = row["frame_start"], row["frame_end"] + 1
            mag = float(np.max(np.hypot(np.asarray(xo)[a:b], np.asarray(yo)[a:b])))
            c = float(np.mean(np.asarray(corr)[a:b])) if corr is not None else float("nan")
            row["max_shift_px"] = round(mag, 2)
            row["mean_corrXY"] = round(c, 4)
            print(f"  {row['source_file'][:46]:46s} max |shift| {mag:5.2f} px   "
                  f"corrXY {c:.4f}")
        cs = [r["mean_corrXY"] for r in ledger if "mean_corrXY" in r]
        if cs and (max(cs) - min(cs)) > 0.05:
            print("\n  corrXY differs noticeably between acquisitions: the field moved\n"
                  "  between runs, or drifted axially. Two-dimensional registration\n"
                  "  cannot correct axial movement, so treat a declining corrXY as a\n"
                  "  candidate explanation for any decline in fluorescence.")
    if reg_out.get("bidiphase"):
        print(f"\nbidirectional phase offset: {reg_out['bidiphase']} px")
    if reg_out.get("xrange") is not None:
        print(f"valid region: x {list(reg_out['xrange'])}  y {list(reg_out['yrange'])}")

    with open(save_path / "frame_ledger.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ledger[0].keys()))
        w.writeheader()
        w.writerows(ledger)

    with open(save_path / "series_run_record.json", "w") as fh:
        json.dump({
            "date_run": dt.date.today().isoformat(),
            "dataset": str(ds), "inputs": [f.name for f in files],
            "n_frames": total, "frame_shape_px": [int(ny), int(nx)],
            "fs_hz": float(fs), "pixel_size_um": float(px) if px else None,
            "nchannels": nch, "functional_chan": args.functional_chan,
            "excluded": [{"file": f.name, "reason": w} for f, w in dropped],
            "n_roi_detected": int(len(stat)),
            "n_roi_iscell": int(iscell[:, 0].sum()),
            "ledger": ledger,
            "settings": json.loads(json.dumps(settings, default=str)),
            "python": sys.version.split()[0],
            "note": "single registration reference and single detection pass across "
                    "all acquisitions; deconvolution disabled.",
        }, fh, indent=2)
    print(f"\nwrote {save_path / 'frame_ledger.csv'} and series_run_record.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
