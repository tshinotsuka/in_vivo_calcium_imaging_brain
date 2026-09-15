#!/usr/bin/env python
"""Suite2p's own deconvolution, applied to traces that already exist.

Runs the same two steps the pipeline runs when deconvolution is enabled --
`dcnv.preprocess` to remove a running baseline, then `dcnv.oasis` to fit a
non-negative spike train under an autoregressive kernel -- but on traces that
have already been extracted, so the ROI set does not change and the result is
comparable row by row with the untouched version and with CaImAn's.

Two things this does not give you.

It is not a spike count. The output is in arbitrary units, and its absolute
scale depends on the kernel and the noise estimate; published comparisons
against simultaneous electrophysiology find that inferred rates track relative
changes far better than they recover absolute ones. Use it to compare
conditions, not to state a firing rate.

It is not an independent check on CaImAn. Both implement OASIS, so running
both compares two parameter choices for one algorithm rather than two
algorithms. The comparison is still worth making -- it shows how much the
answer moves with the implementation -- but it is not corroboration.

The kernel is set from the indicator rather than fitted, because a decay
constant fitted to low signal-to-noise data follows the noise: published fits
of these constants come out several times longer than the indicator's measured
decay and behave as free parameters rather than physical ones.

Example
-------
    python run_suite2p_spikes.py --s2p-dir <plane> --dataset <dataset> \
        --neucoeff 0 --tau 0.27 --out <dataset>/work/s2p_spks
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def call_compatibly(fn, **kwargs):
    """Call fn with whichever of these keyword names its signature accepts."""
    sig = inspect.signature(fn)
    ok = {k: v for k, v in kwargs.items() if k in sig.parameters}
    missing = [k for k in sig.parameters
               if sig.parameters[k].default is inspect.Parameter.empty
               and k not in ok and k != "self"]
    if missing:
        raise TypeError(f"{fn.__name__} needs {missing}, which this script does "
                        "not supply; the suite2p API has changed")
    return fn(**ok)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Suite2p deconvolution on existing traces",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--s2p-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True,
                   help="output root; suite2p/plane0 is created inside")
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--ledger", type=Path, default=None)
    p.add_argument("--fs", type=float, default=None)
    p.add_argument("--tau", type=float, default=0.27,
                   help="indicator decay in seconds (jGCaMP8s about 0.27)")
    p.add_argument("--neucoeff", type=float, default=0.0)
    p.add_argument("--baseline", default="maximin",
                   choices=["maximin", "constant", "constant_prctile"])
    p.add_argument("--win-baseline", type=float, default=60.0,
                   help="baseline window in seconds")
    p.add_argument("--sig-baseline", type=float, default=10.0,
                   help="gaussian filter width in frames, before the running "
                        "minimum")
    p.add_argument("--prctile-baseline", type=float, default=8.0)
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--all-roi", action="store_true")
    args = p.parse_args(argv)

    try:
        from suite2p.extraction import dcnv
    except ImportError:
        print("ERROR: suite2p is required.", file=sys.stderr)
        return 2

    plane = args.s2p_dir.expanduser().resolve()
    F = np.load(plane / "F.npy").astype(np.float32)
    Fneu = np.load(plane / "Fneu.npy").astype(np.float32)
    iscell = np.load(plane / "iscell.npy")
    stat = np.load(plane / "stat.npy", allow_pickle=True)
    ops = {}
    for cand in ("reg_outputs.npy", "detect_outputs.npy", "ops.npy", "db.npy"):
        f = plane / cand
        if f.exists():
            ops.update(np.load(f, allow_pickle=True).item())

    keep = np.ones(F.shape[0], bool) if args.all_roi else iscell[:, 0].astype(bool)
    F, Fneu, stat = F[keep], Fneu[keep], stat[keep]
    n_roi, n_t = F.shape

    fs = args.fs
    if fs is None and args.dataset:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        try:
            from run_roi_suite2p import resolve_from_metadata
            fs = resolve_from_metadata(
                args.dataset.expanduser().resolve() / "raw" / "metadata.yaml"
            ).get("fs_hz")
        except Exception as e:  # noqa: BLE001
            print(f"metadata unreadable: {e}", file=sys.stderr)
    if fs is None:
        print("ERROR: pass --fs or --dataset", file=sys.stderr)
        return 2

    print(f"{n_roi} ROIs x {n_t} frames at {fs:.4g} Hz")
    print(f"kernel from the indicator: tau {args.tau:g} s, so one time step "
          f"decays by {np.exp(-1.0 / (args.tau * fs)):.4f}")

    Fc = (F - args.neucoeff * Fneu).astype(np.float32)
    try:
        Fp = call_compatibly(
            dcnv.preprocess, F=Fc, baseline=args.baseline,
            win_baseline=args.win_baseline, sig_baseline=args.sig_baseline,
            fs=fs, prctile_baseline=args.prctile_baseline)
        spks = call_compatibly(
            dcnv.oasis, F=Fp, batch_size=args.batch_size, tau=args.tau, fs=fs)
    except TypeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    spks = np.asarray(spks, np.float32)
    print(f"spks {spks.shape}, {float((spks > 0).mean()) * 100:.1f}% of frames "
          "non-zero")

    # --- write as a Suite2p plane -------------------------------------------
    out_root = args.out.expanduser().resolve()
    out_plane = out_root / "suite2p" / "plane0"
    out_plane.mkdir(parents=True, exist_ok=True)
    np.save(out_plane / "spks.npy", spks)
    np.save(out_plane / "F.npy", F)
    np.save(out_plane / "Fneu.npy", Fneu)
    np.save(out_plane / "stat.npy", stat)
    np.save(out_plane / "iscell.npy", iscell[keep])
    np.save(out_plane / "Fpreprocessed.npy", np.asarray(Fp, np.float32))
    reg = {k: ops[k] for k in ("meanImg", "meanImgE", "refImg", "Vcorr",
                               "max_proj", "yrange", "xrange", "Ly", "Lx")
           if k in ops}
    reg["nframes"] = int(n_t)
    np.save(out_plane / "reg_outputs.npy", np.array(reg, dtype=object))

    ledger = args.ledger or (plane.parent.parent / "frame_ledger.csv")
    segs = []
    if Path(ledger).exists():
        shutil.copy2(ledger, out_root / "frame_ledger.csv")
        with open(ledger) as fh:
            for r in csv.DictReader(fh):
                segs.append((r["source_file"], int(r["frame_start"]),
                             int(r["frame_end"]) + 1))
    else:
        segs = [("all", 0, n_t)]

    # --- per ROI and per acquisition, in the shared column format -----------
    with open(out_root / "auc_per_roi_per_run.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["roi", "run", "file", "spike_rate_per_min",
                    "auc_per_min_dff", "n_nonzero_frames", "duration_min"])
        for k, (name, a, b) in enumerate(segs, start=1):
            dur = (b - a) / fs / 60.0
            for i in range(n_roi):
                rate = float(spks[i, a:b].sum()) / dur
                w.writerow([i + 1, k, name, round(rate, 4), round(rate, 4),
                            int((spks[i, a:b] > 0).sum()), round(dur, 3)])

    # --- the non-zero frames, written as events -----------------------------
    # OASIS returns a sparse train, so each non-zero frame is already one
    # inferred event with an amplitude. Writing them in the same form the
    # event-based analysis uses lets the change classification resample them
    # exactly as it resamples detected transients, without a separate code
    # path. They carry no duration, so that column is left at one frame.
    nz = np.nonzero(spks)
    with open(out_root / "events.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["roi", "onset_frame", "offset_frame", "t_onset_s",
                    "duration_s", "peak_dff_pct", "area_pct_s"])
        for i, t in zip(*nz):
            v = float(spks[i, t])
            w.writerow([int(i) + 1, int(t), int(t) + 1, round(t / fs, 3),
                        round(1.0 / fs, 4), round(v, 4), round(v, 4)])
    print(f"wrote events.csv: {nz[0].size} non-zero frames as events "
          f"({nz[0].size / n_roi:.0f} per ROI)")

    print(f"\n{'run':>3} {'file':38s} {'rate':>10} {'s.e.m.':>8}")
    rows = []
    for k, (name, a, b) in enumerate(segs, start=1):
        dur = (b - a) / fs / 60.0
        per_roi = spks[:, a:b].sum(axis=1) / dur
        rows.append({"run": k, "file": name,
                     "rate_mean": round(float(per_roi.mean()), 4),
                     "rate_sem": round(float(per_roi.std(ddof=1)
                                             / np.sqrt(n_roi)), 4)})
        print(f"{k:>3} {name[:38]:38s} {per_roi.mean():10.3f} "
              f"{per_roi.std(ddof=1) / np.sqrt(n_roi):8.3f}")
    with open(out_root / "spike_rate_per_run.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(out_root / "spikes_record.json", "w") as fh:
        json.dump({"s2p_dir": str(plane), "n_roi": n_roi, "n_frames": n_t,
                   "fs_hz": fs, "tau_s": args.tau, "neucoeff": args.neucoeff,
                   "baseline": args.baseline,
                   "win_baseline_s": args.win_baseline,
                   "sig_baseline": args.sig_baseline,
                   "prctile_baseline": args.prctile_baseline,
                   "per_run": rows,
                   "note": "suite2p dcnv.preprocess then dcnv.oasis on "
                           "previously extracted traces. Arbitrary units, not "
                           "spike counts; both this and CaImAn's foopsi "
                           "implement OASIS, so the two are not independent."},
                  fh, indent=2)
    print(f"\nwrote {out_plane} and auc_per_roi_per_run.csv")
    print("\nplot it against the other variants:")
    print(f"  python fig_auc_lines.py --value spike_rate_per_min --csv "
          f"{out_root / 'auc_per_roi_per_run.csv'} --out <out>")
    print(f"  python fig_change_pie.py --auc-dir {out_root} "
          "--value spike_rate_per_min \\")
    print(f"      --ledger {out_root / 'frame_ledger.csv'} --to-run all "
          "--out <out>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
