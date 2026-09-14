#!/usr/bin/env python
"""Apply CaImAn's temporal model to traces that Suite2p already extracted.

Keeps the ROIs fixed and changes only what happens to the time courses, so the
result is comparable with the untouched version row by row.

What this actually does. constrained_foopsi fits each trace as a non-negative
spike train convolved with an autoregressive kernel, plus a baseline and noise.
It returns the fitted calcium trace and the spike train that generated it. The
denoised trace and the deconvolution are therefore the same estimate seen from
two sides: there is no setting that denoises without also assuming the spike
model, and p=0 removes the kernel and with it the denoising. So the choice here
is not whether to deconvolve but whether to accept the model.

What the model assumes, and where it bites. The kernel is a single decay
constant shared by the whole trace. If that constant is estimated from the data
it lands wherever the noise pushes it, which at low signal-to-noise is not the
indicator's decay; published fits of these decay constants come out several
times longer than the indicator's measured value and behave as free parameters
rather than physical ones. --g fixes it from the indicator instead, which is
the more defensible choice when the point is to compare conditions rather than
to fit each trace as well as possible.

Three outputs, written as a Suite2p plane so every downstream script reads them
unchanged:

    denoised   the fitted calcium trace, in place of F
    spikes     the inferred rate, saved separately; this is not a spike count,
               and absolute rates from it are not reliable
    residual   what the model could not explain, which is where to look if the
               fit is doing something unexpected

Example
-------
    python run_caiman_traces.py --s2p-dir <plane> --neucoeff 0 \
        --fs 13.2908 --tau 0.27 --p 1 --out <dataset>/work/s2p_foopsi
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def rolling_baseline(F, win, pct=50.0):
    n = F.shape[1]
    win = max(int(win), 3)
    step = max(win // 4, 1)
    centres = np.arange(0, n, step)
    vals = np.empty((F.shape[0], centres.size), np.float32)
    for j, c in enumerate(centres):
        a, b = max(0, c - win // 2), min(n, c + win // 2 + 1)
        vals[:, j] = np.percentile(F[:, a:b], pct, axis=1)
    out = np.empty_like(F, np.float32)
    for i in range(F.shape[0]):
        out[i] = np.interp(np.arange(n), centres, vals[i])
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="CaImAn temporal model on Suite2p ROIs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--s2p-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True,
                   help="output root; suite2p/plane0 is created inside")
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--ledger", type=Path, default=None)
    p.add_argument("--fs", type=float, default=None)
    p.add_argument("--tau", type=float, default=0.27,
                   help="indicator decay used to fix the kernel")
    p.add_argument("--p", type=int, default=1, choices=[1, 2],
                   help="autoregressive order. There is no p=0 here: without "
                        "the kernel there is nothing to denoise with")
    p.add_argument("--g", default="fixed", choices=["fixed", "estimate"],
                   help="'fixed' derives the kernel from --tau; 'estimate' fits "
                        "it per trace, which at low signal-to-noise follows the "
                        "noise rather than the indicator")
    p.add_argument("--method", default="oasis",
                   choices=["oasis", "cvxpy", "cvx"])
    p.add_argument("--neucoeff", type=float, default=0.0)
    p.add_argument("--baseline-window-s", type=float, default=15.0)
    p.add_argument("--all-roi", action="store_true")
    p.add_argument("--noise-range", type=float, nargs=2, default=[0.25, 0.5],
                   metavar=("LO", "HI"),
                   help="frequency band, as a fraction of Nyquist, used to "
                        "estimate the noise level")
    args = p.parse_args(argv)

    try:
        from caiman.source_extraction.cnmf.deconvolution import constrained_foopsi
    except ImportError:
        print("ERROR: caiman is required. Use the caiman environment.",
              file=sys.stderr)
        return 2

    plane = args.s2p_dir.expanduser().resolve()
    F = np.load(plane / "F.npy").astype(np.float64)
    Fneu = np.load(plane / "Fneu.npy").astype(np.float64)
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

    Fc = F - args.neucoeff * Fneu
    F0 = rolling_baseline(Fc.astype(np.float32),
                          int(round(args.baseline_window_s * fs)), 50.0)
    dff = (Fc - F0) / np.maximum(F0, 1.0) * 100.0

    g_fixed = None
    if args.g == "fixed":
        # one time step of decay: exp(-dt / tau)
        g1 = float(np.exp(-1.0 / (args.tau * fs)))
        g_fixed = (g1,) if args.p == 1 else (g1 * 2, -g1 ** 2)
        print(f"kernel fixed from the indicator: tau {args.tau:g} s at "
              f"{fs:.4g} Hz gives g = "
              + ", ".join(f"{x:.4f}" for x in g_fixed))
    else:
        print("kernel estimated per trace")

    print(f"deconvolving {n_roi} ROIs x {n_t} frames "
          f"(p={args.p}, method={args.method}, neucoeff={args.neucoeff:g}) ...")

    C = np.zeros_like(dff)
    S = np.zeros_like(dff)
    rows = []
    failed = 0
    for i in range(n_roi):
        try:
            c, bl, c1, g, sn, sp, lam = constrained_foopsi(
                dff[i].astype(np.float64), g=g_fixed, p=args.p,
                method_deconvolution=args.method,
                noise_range=list(args.noise_range))
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ROI {i + 1}: {type(e).__name__}: {e}", file=sys.stderr)
            C[i] = dff[i]
            rows.append({"roi": i + 1, "ok": 0, "baseline": np.nan,
                         "noise_sd": np.nan, "g1": np.nan,
                         "spike_sum": 0.0, "resid_sd": np.nan})
            continue
        C[i], S[i] = c, sp
        rows.append({"roi": i + 1, "ok": 1, "baseline": float(bl),
                     "noise_sd": float(sn),
                     "g1": float(np.atleast_1d(g)[0]),
                     "spike_sum": float(sp.sum()),
                     "resid_sd": float(np.std(dff[i] - c))})
        if n_roi >= 10 and i % max(n_roi // 10, 1) == 0:
            print(f"  {i / n_roi * 100:3.0f}%", end="\r", flush=True)
    print(" " * 12, end="\r")
    if failed:
        print(f"{failed} ROI(s) failed to fit and were passed through unchanged",
              file=sys.stderr)

    ok = np.array([r["ok"] == 1 for r in rows])
    if ok.any():
        sd_raw = np.median(np.abs(np.diff(dff[ok], axis=1))) / (np.sqrt(2) * 0.6745)
        sd_den = np.median(np.abs(np.diff(C[ok], axis=1))) / (np.sqrt(2) * 0.6745)
        print(f"\nframe-to-frame spread: raw {sd_raw:.2f} -> denoised "
              f"{sd_den:.2f} %dF/F ({sd_den / max(sd_raw, 1e-9) * 100:.0f}%)")
        gs = np.array([r["g1"] for r in rows if r["ok"]])
        if args.g == "estimate" and np.isfinite(gs).any():
            tau_fit = -1.0 / (fs * np.log(np.clip(gs, 1e-6, 0.999999)))
            print(f"fitted decay: median {np.median(tau_fit):.2f} s against an "
                  f"indicator decay of {args.tau:g} s "
                  f"({np.median(tau_fit) / args.tau:.1f}x)")
            if np.median(tau_fit) > 3 * args.tau:
                print("  the fitted kernel is several times slower than the "
                      "indicator, so it is\n  following something other than "
                      "the indicator's decay; --g fixed avoids this")
        corr = np.median([np.corrcoef(dff[i], C[i])[0, 1]
                          for i in range(n_roi) if ok[i]])
        print(f"denoised trace correlates with the raw one at {corr:.3f}")

    # --- write as a Suite2p plane ------------------------------------------
    out_plane = args.out.expanduser().resolve() / "suite2p" / "plane0"
    out_plane.mkdir(parents=True, exist_ok=True)
    # F is reconstructed so that a downstream dF/F recovers the denoised trace:
    # the same baseline goes back in, so nothing downstream has to be told that
    # these traces came from a model.
    np.save(out_plane / "F.npy", (F0 * (1.0 + C / 100.0)).astype(np.float32))
    np.save(out_plane / "Fneu.npy", np.zeros_like(F, np.float32))
    np.save(out_plane / "stat.npy", stat)
    np.save(out_plane / "iscell.npy", iscell[keep])
    np.save(out_plane / "spks.npy", S.astype(np.float32))
    np.save(out_plane / "dff_denoised.npy", C.astype(np.float32))
    np.save(out_plane / "dff_raw.npy", dff.astype(np.float32))
    reg = {k: ops[k] for k in ("meanImg", "meanImgE", "refImg", "Vcorr",
                               "max_proj", "yrange", "xrange", "Ly", "Lx")
           if k in ops}
    reg["nframes"] = int(n_t)
    np.save(out_plane / "reg_outputs.npy", np.array(reg, dtype=object))

    ledger = args.ledger or (plane.parent.parent / "frame_ledger.csv")
    segs = []
    if Path(ledger).exists():
        import shutil
        shutil.copy2(ledger, args.out.expanduser().resolve() / "frame_ledger.csv")
        with open(ledger) as fh:
            for r in csv.DictReader(fh):
                segs.append((r["source_file"], int(r["frame_start"]),
                             int(r["frame_end"]) + 1))

    # --- per ROI and per acquisition ---------------------------------------
    # The denoised trace is already a model fit, so running the event detector
    # over it would set its threshold from a noise level the model removed.
    # Integrating the fitted trace directly gives the same quantity in the same
    # units as the event-based AUC, without a second thresholding step.
    if segs:
        out_root = args.out.expanduser().resolve()
        with open(out_root / "auc_per_roi_per_run.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["roi", "run", "file", "auc_per_min_dff",
                        "auc_per_min_df", "spike_rate_per_min", "n_events",
                        "duration_min", "raw_F", "raw_F0", "resid_sd"])
            for k, (name, a, b) in enumerate(segs, start=1):
                dur_min = (b - a) / fs / 60.0
                for i in range(n_roi):
                    auc_dff = float(C[i, a:b].sum()) / fs / dur_min
                    auc_df = float((C[i, a:b] / 100.0
                                    * F0[i, a:b]).sum()) / fs / dur_min
                    rate = float(S[i, a:b].sum()) / dur_min
                    n_ev = int((S[i, a:b] > 0).sum())
                    w.writerow([i + 1, k, name, round(auc_dff, 4),
                                round(auc_df, 4), round(rate, 4), n_ev,
                                round(dur_min, 3),
                                round(float(F[i, a:b].mean()), 2),
                                round(float(np.median(F0[i, a:b])), 2),
                                round(float(np.std(dff[i, a:b] - C[i, a:b])), 3)])
        print(f"\nwrote auc_per_roi_per_run.csv "
              f"({n_roi} ROIs x {len(segs)} acquisitions)")

    if segs:
        print(f"\n{'run':>3} {'file':38s} {'inferred rate':>14} {'s.e.m.':>8}")
        srows = []
        for k, (name, a, b) in enumerate(segs, start=1):
            per_roi = S[:, a:b].sum(axis=1) / ((b - a) / fs / 60.0)
            srows.append({"run": k, "file": name,
                          "rate_mean": round(float(per_roi.mean()), 4),
                          "rate_sem": round(float(per_roi.std(ddof=1)
                                                  / np.sqrt(n_roi)), 4)})
            print(f"{k:>3} {name[:38]:38s} {per_roi.mean():14.3f} "
                  f"{per_roi.std(ddof=1) / np.sqrt(n_roi):8.3f}")
        with open(args.out.expanduser().resolve() / "spike_rate_per_run.csv",
                  "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(srows[0].keys()))
            w.writeheader()
            w.writerows(srows)
        print("\nThe inferred rate is an estimate in arbitrary units, not a "
              "spike count; use it\nfor relative comparisons only.")

    with open(args.out.expanduser().resolve() / "foopsi_record.json", "w") as fh:
        json.dump({"s2p_dir": str(plane), "n_roi": n_roi, "n_frames": n_t,
                   "fs_hz": fs, "p": args.p, "g_mode": args.g,
                   "tau_s": args.tau, "method": args.method,
                   "neucoeff": args.neucoeff, "n_failed": failed,
                   "per_roi": rows,
                   "note": "denoised traces and inferred rates from "
                           "constrained_foopsi, on ROIs detected by Suite2p. "
                           "The denoising and the deconvolution are the same "
                           "estimate; the transient shape comes from the model."},
                  fh, indent=2, default=str)
    print(f"\nwrote {out_plane}")
    print("\nplot it alongside the other variants:")
    print(f"  python fig_auc_lines.py --csv "
          f"{args.out.expanduser().resolve() / 'auc_per_roi_per_run.csv'} "
          "--out <out>")
    print("\nDo not put these traces through the event detector. It sets its "
          "threshold from\nthe frame-to-frame spread, which a deconvolved "
          "trace does not have: the fit is\nzero for most frames, the spread "
          "reads as zero, and every frame becomes an event.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
