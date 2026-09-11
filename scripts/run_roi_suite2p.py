#!/usr/bin/env python
"""Suite2p cell detection on motion-corrected movies.

Registration is assumed already done (CaImAn), so Suite2p runs with
``do_registration=0`` and performs detection, extraction and neuropil
extraction only.

Acquisition parameters are resolved from ``raw/metadata.yaml`` rather than
passed on the command line. ``generate_metadata.py`` already fills
``frame_rate_hz`` and ``pixel_size_um`` for the dataset's main time series, and
``objective_calibration.yaml`` is the calibration ledger behind the pixel size;
re-entering those values by hand would create a second source for numbers that
are already settled.

Filename ordering uses ``naming.FILE_RE`` from the analysis repo, so the
ScanImage sequence number grammar is not re-implemented here.

Deconvolution is not part of this pipeline. Suite2p always writes ``spks.npy``;
it is left on disk and ignored. ``neucoeff`` is stored but not baked in --
``F`` and ``Fneu`` are saved separately, so any neuropil coefficient can be
applied downstream without re-running detection.

Example
-------
    python run_roi_suite2p.py \
        --dataset /media/tshino/DATA/workspace/2026_aqp4ko_water_intoxication/2photon/20260804_sub-sk50_ses-01 \
        --dry-run
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import yaml

# naming.py lives in src/metadata/, outside the importable ivwib package, so
# `pip install -e` does not expose it. generate_metadata.py solves this the same
# way; do not copy the regexes -- naming.py is the single source for the
# filename grammar.
IVWIB_REPO = Path(os.environ.get(
    "IVWIB_REPO",
    "/media/tshino/DATA/Projects/in_vivo_water_imaging_brain",
))
_naming_dir = IVWIB_REPO / "src" / "metadata"
if _naming_dir.is_dir():
    sys.path.insert(0, str(_naming_dir))

try:
    from naming import FILE_RE
except ImportError:
    FILE_RE = None


# ---------------------------------------------------------------------------
# dataset resolution
# ---------------------------------------------------------------------------


def _walk(node, key):
    """Yield every value stored under ``key`` anywhere in a nested structure."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key and v is not None:
                yield v
            else:
                yield from _walk(v, key)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v, key)


def resolve_from_metadata(meta_path: Path) -> dict:
    """Pull frame rate, pixel size and channel count out of metadata.yaml.

    Dataset-level ``imaging.acquisition`` is preferred, since generate_metadata
    fills it from the acquisition with the most frames -- the main time series.
    Otherwise the most common value found anywhere in the file is used, and the
    spread is reported so a disagreement is visible rather than silently
    averaged away.
    """
    with open(meta_path) as fh:
        meta = yaml.safe_load(fh) or {}

    out: dict = {"source": str(meta_path)}
    top = (meta.get("imaging") or {}).get("acquisition") or {}

    for field, key in [
        ("fs_hz", "frame_rate_hz"),
        ("pixel_size_um", "pixel_size_um"),
        ("n_channels", "n_channels"),
    ]:
        vals = [v for v in _walk(meta, key) if isinstance(v, (int, float))]
        chosen = top.get(key)
        if chosen is None and vals:
            uniq, counts = np.unique(np.round(vals, 6), return_counts=True)
            chosen = float(uniq[int(np.argmax(counts))])
        out[field] = chosen
        if vals:
            out[field + "_range"] = [float(min(vals)), float(max(vals))]

    obj = [v for v in _walk(meta, "objective") if isinstance(v, str)]
    out["objective"] = obj[0] if obj else None
    return out


def order_key(path: Path):
    """Sort key for motion-corrected files.

    The names produced by the motion-correction step do not follow the raw
    ScanImage grammar, so the acquisition number is taken from whichever
    trailing integer is present; ``naming.FILE_RE`` is used when the raw
    grammar does survive in the name.
    """
    if FILE_RE is not None:
        m = FILE_RE.match(path.name)
        if m:
            return (int(m.group("idx")), path.name)
    nums = re.findall(r"(\d+)", path.stem)
    return (int(nums[-1]) if nums else 10**9, path.name)


# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Suite2p detection on motion-corrected movies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", type=Path, required=True,
                   help="workspace dataset dir containing raw/ and results/")
    p.add_argument("--pattern", default="*_MC.tif",
                   help="glob inside results/ selecting the corrected movies")
    p.add_argument("--save-path", type=Path, default=None,
                   help="Suite2p output root (default: <dataset>/work/s2p)")
    p.add_argument("--fs", type=float, default=None,
                   help="override the frame rate from metadata.yaml")
    p.add_argument("--pixel-size-um", type=float, default=None,
                   help="override the pixel size from metadata.yaml")
    p.add_argument("--soma-um", type=float, default=15.0,
                   help="expected soma diameter in um, used to set --diameter")
    p.add_argument("--diameter", type=int, default=None,
                   help="cell diameter in px; default derives it from the pixel size")
    p.add_argument("--anatomical-only", type=int, default=2,
                   help="0=functional detection, 2=Cellpose on the mean image")
    p.add_argument("--tau", type=float, default=0.5,
                   help="stored for provenance; deconvolution is not used")
    p.add_argument("--threshold-scaling", type=float, default=1.0)
    p.add_argument("--spatial-scale", type=int, default=0, choices=[0, 1, 2, 3, 4],
                   help="functional detection scale: 0=auto, 1=6px, 2=12px, 3=24px, 4=48px. "
                        "Auto estimation fails on small or low-contrast somata and falls back "
                        "to 1, which finds fragments rather than cell bodies")
    p.add_argument("--npix-norm-min", type=float, default=None,
                   help="reject ROIs whose normalised pixel count is below this; raises the "
                        "floor on ROI size so neuropil fragments are dropped")
    p.add_argument("--npix-norm-max", type=float, default=None)
    p.add_argument("--max-overlap", type=float, default=None,
                   help="fraction of overlapping pixels above which an ROI is discarded")
    p.add_argument("--active-percentile", type=float, default=None,
                   help="functional detection: percentile of activity used to seed ROIs")
    p.add_argument("--highpass-neuropil", type=int, default=None)
    p.add_argument("--no-soma-crop", dest="soma_crop", action="store_false",
                   help="keep processes attached to each ROI instead of cropping to the soma")
    p.add_argument("--cellpose-img", default="meanImg",
                   choices=["meanImg", "max_proj", "max_proj / meanImg"],
                   help="image Cellpose segments. meanImg is activity-independent; max_proj "
                        "shows active somata far more clearly but weights cells by activity")
    p.add_argument("--cellprob-threshold", type=float, default=None,
                   help="Cellpose acceptance threshold (default 0.0). Lower values keep "
                        "dimmer or less certain masks, so visibly bright somata that were "
                        "skipped are usually recovered here first")
    p.add_argument("--flow-threshold", type=float, default=None,
                   help="Cellpose flow-error tolerance (default 0.4). Higher values keep "
                        "masks whose shape is less canonical")
    p.add_argument("--highpass-spatial", type=int, default=None,
                   help="spatial high-pass applied before Cellpose; helps when uneven "
                        "background brightness hides cells in part of the field")
    p.add_argument("--cellpose-model", default=None,
                   help="Cellpose model name (default cpsam)")
    p.add_argument("--high-pass", type=int, default=100)
    p.add_argument("--inner-neuropil-radius", type=int, default=2)
    p.add_argument("--min-neuropil-pixels", type=int, default=350)
    p.add_argument("--torch-device", default="cpu",
                   help="cpu or cuda; must match the installed torch build")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve everything and print the plan without running Suite2p")
    args = p.parse_args(argv)

    ds = args.dataset.expanduser().resolve()
    raw_dir, res_dir = ds / "raw", ds / "results"
    save_path = (args.save_path or ds / "work" / "s2p").expanduser().resolve()

    # --- acquisition parameters -------------------------------------------
    meta_path = raw_dir / "metadata.yaml"
    info: dict = {}
    if meta_path.exists():
        info = resolve_from_metadata(meta_path)
        print(f"metadata: {meta_path}")
        for k in ("fs_hz", "pixel_size_um", "n_channels", "objective"):
            rng = info.get(k + "_range")
            extra = ""
            if rng and abs(rng[1] - rng[0]) > 1e-9:
                extra = f"   (values in file span {rng[0]} .. {rng[1]})"
            print(f"  {k:16s} {info.get(k)}{extra}")
    else:
        print(f"no metadata.yaml at {meta_path}", file=sys.stderr)

    fs = args.fs if args.fs is not None else info.get("fs_hz")
    px = args.pixel_size_um if args.pixel_size_um is not None else info.get("pixel_size_um")
    if fs is None:
        print("\nERROR: frame rate unresolved. Run generate_metadata.py on this "
              "dataset, or pass --fs.", file=sys.stderr)
        return 2
    if px is None:
        print("\nWARNING: pixel size unresolved; --diameter cannot be derived.",
              file=sys.stderr)

    # --- inputs -------------------------------------------------------------
    files = sorted(res_dir.glob(args.pattern), key=order_key)
    if not files:
        print(f"\nERROR: no files matched {args.pattern!r} under {res_dir}",
              file=sys.stderr)
        return 2

    import tifffile

    counts, shapes = [], set()
    for f in files:
        with tifffile.TiffFile(f) as tf:
            counts.append(len(tf.pages))
            shapes.add(tf.pages[0].shape[:2])
    total = int(sum(counts))
    if len(shapes) > 1:
        print(f"\nERROR: inconsistent frame shapes across inputs: {shapes}",
              file=sys.stderr)
        return 2
    ny, nx = shapes.pop()

    print(f"\ninputs from {res_dir}")
    running, ledger = 0, []
    for f, n in zip(files, counts):
        ledger.append({"source_file": f.name, "frame_start": running,
                       "frame_end": running + n - 1, "n_frames": n})
        print(f"  {f.name:52s} {n:6d} frames  [{running} .. {running + n - 1}]")
        running += n
    print(f"  total {total} frames = {total / fs / 60:.1f} min at {fs:.4g} Hz")
    print(f"  frame {ny} x {nx} px")

    if len(files) > 1:
        print("\nNOTE: several input files. If motion correction was run per file,\n"
              "  each had its own template and inter-file drift is not corrected.\n"
              "  The motion-correction notebook also writes a fixed output name per\n"
              "  channel, so running it repeatedly overwrites earlier results --\n"
              "  check that these files are distinct.")

    # --- diameter -----------------------------------------------------------
    diameter = args.diameter
    if diameter is None and px:
        diameter = int(round(args.soma_um / px))
    if px:
        fov_um = nx * px
        print(f"\nscale: {px:.3f} um/px  ->  FOV {fov_um:.0f} x {ny * px:.0f} um")
        print(f"  a {args.soma_um:g} um soma spans about {args.soma_um / px:.1f} px")
        if args.soma_um / px < 8 and args.anatomical_only:
            print("\n  Somata below roughly 8 px are hard for anatomical segmentation\n"
                  "  to separate, and neuropil contamination rises as the soma\n"
                  "  approaches the PSF. Consider increasing zoom on the next\n"
                  "  acquisition, or --anatomical-only 0 to fall back to functional\n"
                  "  detection here, accepting that it weights cells by activity.")

    # --- ops ----------------------------------------------------------------
    if args.dry_run:
        print("\n--dry-run: stopping before Suite2p.")
        print(f"  would write to {save_path}")
        print(f"  fs={fs}  diameter={diameter}  anatomical_only={args.anatomical_only}")
        print(f"  torch_device={args.torch_device}")
        return 0

    import suite2p

    # Suite2p 1.1.0 replaced the flat `ops` dict with a nested `settings` dict.
    # Start from the shipped defaults so unspecified keys keep upstream values,
    # then override only what this pipeline actually decides.
    import copy
    import inspect

    # run_s2p(db={}, settings={...}, server={}) -- take the settings default by
    # name rather than by position, so a signature change fails loudly here
    # instead of silently handing Suite2p an empty dict.
    _params = inspect.signature(suite2p.run_s2p).parameters
    if "settings" not in _params:
        print("ERROR: suite2p.run_s2p has no 'settings' parameter; this script "
              "targets the 1.1.0 API.", file=sys.stderr)
        return 2
    _default_settings = _params["settings"].default
    if not isinstance(_default_settings, dict) or "detection" not in _default_settings:
        print("ERROR: unexpected default settings from suite2p.run_s2p.", file=sys.stderr)
        return 2
    settings = copy.deepcopy(_default_settings)

    def sub(key):
        settings.setdefault(key, {})
        return settings[key]

    settings["fs"] = float(fs)
    settings["tau"] = args.tau
    if diameter:
        settings["diameter"] = [float(diameter), float(diameter)]
    settings["torch_device"] = args.torch_device

    run = sub("run")
    run["do_registration"] = 0        # caiman already did this
    run["do_regmetrics"] = False      # registration metrics need a registration step
    run["do_detection"] = True
    run["do_deconvolution"] = False   # deconvolution is not part of this pipeline

    det = sub("detection")
    det["algorithm"] = "cellpose" if args.anatomical_only else "sparsery"
    det["threshold_scaling"] = args.threshold_scaling
    det["highpass_time"] = args.high_pass
    det["soma_crop"] = args.soma_crop
    if args.npix_norm_min is not None:
        det["npix_norm_min"] = args.npix_norm_min
    if args.npix_norm_max is not None:
        det["npix_norm_max"] = args.npix_norm_max
    if args.max_overlap is not None:
        det["max_overlap"] = args.max_overlap

    sp = det.setdefault("sparsery_settings", {})
    sp["spatial_scale"] = args.spatial_scale
    if args.active_percentile is not None:
        sp["active_percentile"] = args.active_percentile
    if args.highpass_neuropil is not None:
        sp["highpass_neuropil"] = args.highpass_neuropil
    if args.anatomical_only:
        # anatomical detection on the mean image: activity-independent, so a
        # condition with lower activity does not silently yield fewer ROIs.
        cp_set = det.setdefault("cellpose_settings", {})
        cp_set["img"] = args.cellpose_img
        if args.cellprob_threshold is not None:
            cp_set["cellprob_threshold"] = args.cellprob_threshold
        if args.flow_threshold is not None:
            cp_set["flow_threshold"] = args.flow_threshold
        if args.highpass_spatial is not None:
            cp_set["highpass_spatial"] = args.highpass_spatial
        if args.cellpose_model:
            cp_set["cellpose_model"] = args.cellpose_model

    ext = sub("extraction")
    ext["neuropil_extract"] = True
    ext["neuropil_coefficient"] = 0.7   # stored only; F and Fneu are saved separately
    ext["inner_neuropil_radius"] = args.inner_neuropil_radius
    ext["min_neuropil_pixels"] = args.min_neuropil_pixels
    ext["allow_overlap"] = False

    io_ = sub("io")
    io_["delete_bin"] = False
    io_["move_bin"] = False

    save_path.mkdir(parents=True, exist_ok=True)

    # Stage the selected movies in a directory of their own and point Suite2p at
    # that. Relying on `tiff_list` to filter within results/ is fragile: if the
    # key is ignored, Suite2p silently globs every TIFF in the directory and
    # concatenates the other channels onto the end of the recording, which looks
    # like a working run and produces traces made of channel-boundary steps.
    stage = save_path / "input"
    if stage.exists():
        for old_link in stage.iterdir():
            old_link.unlink()
    stage.mkdir(parents=True, exist_ok=True)
    for f in files:
        link = stage / f.name
        try:
            link.symlink_to(f.resolve())
        except OSError:
            import shutil
            shutil.copy2(f, link)

    staged = sorted(stage.glob("*.tif"))
    if len(staged) != len(files):
        print(f"ERROR: staged {len(staged)} files but selected {len(files)}",
              file=sys.stderr)
        return 2
    print(f"\nstaged {len(staged)} file(s) in {stage}")

    db = dict(suite2p.default_db())
    db.update(
        data_path=[str(stage)],
        file_list=[f.name for f in staged],  # `tiff_list` is not a db key
        save_path0=str(save_path),
        nplanes=1,
        nchannels=1,
    )

    print(f"\nrunning Suite2p {getattr(suite2p, '__version__', settings.get('version', '?'))} "
          f"on {args.torch_device} ...")
    print(f"  detection algorithm : {det['algorithm']}")
    print(f"  diameter            : {settings.get('diameter')}")
    if det["algorithm"] == "sparsery":
        print(f"  spatial_scale       : {sp['spatial_scale']}"
              f"{' (auto)' if sp['spatial_scale'] == 0 else ''}")
    else:
        _c = det["cellpose_settings"]
        print(f"  cellpose img        : {_c['img']}")
        print(f"  cellprob/flow       : {_c.get('cellprob_threshold')} / {_c.get('flow_threshold')}")
    print(f"  deconvolution       : {run['do_deconvolution']}")
    suite2p.run_s2p(db=db, settings=settings)

    plane = save_path / "suite2p" / "plane0"
    if not plane.exists():
        cands = sorted(save_path.glob("**/plane0"))
        if cands:
            plane = cands[0]
    print(f"\noutputs in {plane}")
    stat = np.load(plane / "stat.npy", allow_pickle=True)
    iscell = np.load(plane / "iscell.npy")
    F = np.load(plane / "F.npy")
    print(f"\ndetected {len(stat)} ROIs; classifier accepts {int(iscell[:, 0].sum())}")
    print(f"F shape {F.shape}")
    if F.shape[1] != total:
        print(f"\nERROR: Suite2p produced {F.shape[1]} timepoints but the selected\n"
              f"  input has {total} frames. Extra frames mean other files were read\n"
              f"  and concatenated -- most likely the other channels. The traces are\n"
              f"  not usable; check {stage} and the --pattern.", file=sys.stderr)
        return 2

    # Report ROI size against the expected soma size. Detection that returns
    # ROIs far smaller than a cell body has found fragments of neuropil or
    # processes, not somata, and no amount of threshold tuning fixes that --
    # the detection scale is what needs changing.
    npix = np.array([len(x["xpix"]) for x in stat])
    diam_eq = 2 * np.sqrt(npix / np.pi)
    acc = iscell[:, 0].astype(bool)
    if acc.any() and px:
        expect = args.soma_um / px
        med = float(np.median(diam_eq[acc]))
        print(f"\nROI size: median equivalent diameter {med:.1f} px "
              f"({med * px:.1f} um); a {args.soma_um:g} um soma is {expect:.1f} px")
        print(f"  npix: median {int(np.median(npix[acc]))}, "
              f"a soma of that diameter is about {int(np.pi * (expect / 2) ** 2)} px")
        if med < 0.6 * expect:
            print("  These ROIs are well under cell-body size. Try a larger\n"
                  "  --spatial-scale, or --npix-norm-min to drop fragments.")

    with open(save_path / "frame_ledger.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ledger[0].keys()))
        w.writeheader()
        w.writerows(ledger)

    record = {
        "date_run": dt.date.today().isoformat(),
        "dataset": str(ds),
        "inputs": [f.name for f in files],
        "n_frames": total,
        "frame_shape_px": [int(ny), int(nx)],
        "fs_hz": float(fs),
        "pixel_size_um": float(px) if px else None,
        "metadata_source": info.get("source"),
        "objective": info.get("objective"),
        "suite2p_version": getattr(suite2p, "__version__", settings.get("version", "unknown")),
        "torch_device": args.torch_device,
        "settings": json.loads(json.dumps(settings, default=str)),
        "python": sys.version.split()[0],
        "diameter_px": int(diameter or 0),
        "anatomical_only": args.anatomical_only,
        "n_roi_detected": int(len(stat)),
        "n_roi_iscell": int(iscell[:, 0].sum()),
        "median_roi_diameter_px": float(np.median(diam_eq[acc])) if acc.any() else None,
        "median_roi_npix": int(np.median(npix[acc])) if acc.any() else None,
        "note": "registration by caiman; suite2p do_registration=0. "
                "deconvolution unused (spks.npy ignored). "
                "neucoeff not baked in: F and Fneu saved separately.",
    }
    with open(save_path / "roi_run_record.json", "w") as fh:
        json.dump(record, fh, indent=2)
    print(f"wrote {save_path / 'roi_run_record.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
