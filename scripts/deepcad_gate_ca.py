#!/usr/bin/env python
"""Decide whether a denoiser may be used for calcium event AUC, before touching real data.

The same discipline as the t50 gate in docs/deepcad_runbook.md, aimed at the
failure modes that matter for an activity rate rather than for an arrival time.
A denoiser that is merely "good" is not enough: it has to be unbiased in the
specific ways this measurement depends on.

Three failure modes, each with its own synthetic ground truth.

  1. Amplitude bias. Denoising can shrink transients toward the baseline or
     sharpen them. AUC is an integral of amplitude, so a uniform shrinkage of
     10% is a 10% error in every number reported.

  2. Brightness-dependent bias. THE critical one for a long recording. If
     performance depends on signal-to-noise, and the preparation dims over the
     session, then the denoiser produces an apparent decline in activity from a
     recording where activity never changed. This experiment measures exactly
     that kind of decline, so a denoiser with this bias would fabricate the
     result. The synthetic movie therefore holds activity constant while the
     brightness falls by a set amount, and asks whether the measured AUC stays
     flat.

  3. Event fabrication and destruction. A self-supervised denoiser trained on
     the noise itself can produce smooth excursions where there was nothing.
     The gate counts events in cells that were given none.

Usage mirrors the t50 gate: make the synthetic movies, denoise them with
whatever tool is being evaluated, then evaluate.

    python deepcad_gate_ca.py --make-synth work/gate_ca.tif
    # denoise externally, e.g. scripts/deepcad_run.py --input work/gate_ca.tif
    python deepcad_gate_ca.py --eval work/gate_ca.tif \
        --denoised work/gate_ca_denoised.tif
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from run_event_auc import (auc_per_min, detect_events, matched_filter,
                               percentile_filter)
except ImportError:  # pragma: no cover
    print("ERROR: run_event_auc.py must be importable (same directory)",
          file=sys.stderr)
    raise


# ---------------------------------------------------------------------------
# synthesis
# ---------------------------------------------------------------------------


def make_synth(path: Path, *, n_t=3600, ny=128, nx=128, fs=13.2908,
               n_cell=24, n_epoch=6, tau_s=0.27, soma_px=4.0,
               bleach=0.75, snr_lo=1.5, snr_hi=6.0, n_events=18,
               n_silent=6, seed=0):
    """A movie whose activity is constant while its brightness falls.

    Cells span a range of signal-to-noise so a brightness-dependent bias shows
    up as a gradient, and several cells are given no events at all so that
    fabricated events have somewhere to be counted. Event times and amplitudes
    are identical in every epoch: any change the analysis reports across epochs
    is an artefact, by construction.
    """
    rng = np.random.default_rng(seed)
    per = n_t // n_epoch
    kern = np.exp(-np.arange(int(8 * tau_s * fs)) / (tau_s * fs))
    kern[0] = 0.0
    kern[:2] = np.linspace(0, 1, 2)

    yy, xx = np.mgrid[0:ny, 0:nx]
    centres, masks = [], []
    for i in range(n_cell):
        while True:
            cy, cx = rng.integers(10, ny - 10), rng.integers(10, nx - 10)
            if all((cy - c[0]) ** 2 + (cx - c[1]) ** 2 > (3 * soma_px) ** 2
                   for c in centres):
                break
        centres.append((cy, cx))
        masks.append(np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2)
                              / (2 * soma_px ** 2))))

    # per-cell signal-to-noise, log-spaced across the requested range
    snr = np.geomspace(snr_lo, snr_hi, n_cell)
    order = rng.permutation(n_cell)
    snr = snr[order]

    # events: the SAME pattern in every epoch, for every cell
    silent = set(rng.choice(n_cell, n_silent, replace=False).tolist())
    base_times = {}
    for i in range(n_cell):
        if i in silent:
            base_times[i] = np.array([], int)
        else:
            base_times[i] = np.sort(rng.choice(
                np.arange(int(2 * tau_s * fs), per - len(kern)),
                n_events, replace=False))

    sigma = 30.0
    F0 = 800.0
    bright = np.repeat(np.linspace(1.0, bleach, n_epoch), per)[:n_t]

    clean = np.zeros((n_t, ny, nx), np.float32)
    traces = np.zeros((n_cell, n_t), np.float32)
    for i in range(n_cell):
        ev = np.zeros(n_t)
        for e in range(n_epoch):
            for t0 in base_times[i]:
                ev[e * per + t0] = 1.0
        tr = np.convolve(ev, kern, mode="full")[:n_t]
        if tr.max() > 0:
            tr = tr / tr.max()
        # amplitude set so that the PEAK dF/F is snr * (noise sd / F0)
        amp = snr[i] * sigma
        traces[i] = tr * amp
        clean += (masks[i][None] * traces[i][:, None, None]).astype(np.float32)

    clean = (clean + F0) * bright[:, None, None]
    noisy = clean + rng.normal(0, sigma, clean.shape).astype(np.float32)

    import tifffile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.clip(noisy, 0, 32767).astype(np.int16))

    gt = {
        "fs": fs, "n_t": n_t, "n_epoch": n_epoch, "per": per, "tau_s": tau_s,
        "n_cell": n_cell, "sigma": sigma, "F0": F0, "bleach": bleach,
        "centres": np.array(centres), "masks": np.array(masks, np.float32),
        "snr": snr, "silent": np.array(sorted(silent)),
        "traces_clean": traces,
    }
    np.savez_compressed(path.with_suffix(".gt.npz"), **gt)
    print(f"wrote {path}  ({n_t} frames, {ny}x{nx}, {n_cell} cells, "
          f"{len(silent)} of them silent)")
    print(f"  activity is IDENTICAL in all {n_epoch} epochs; brightness falls "
          f"{(1 - bleach) * 100:.0f}% across them")
    print(f"  per-cell peak dF/F ranges {snr.min() * sigma / F0 * 100:.0f}% to "
          f"{snr.max() * sigma / F0 * 100:.0f}%")
    print(f"wrote {path.with_suffix('.gt.npz')}")
    return path


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def extract(mov, masks):
    """Weighted pixel sums, the same quantity Suite2p writes to F.npy."""
    n_t = mov.shape[0]
    flat = mov.reshape(n_t, -1)
    w = masks.reshape(masks.shape[0], -1)
    w = w / w.sum(axis=1, keepdims=True)
    return (w @ flat.T).astype(np.float32)


def auc_matrix(F, gt, *, smooth, fp_method, baseline_window_s=15.0,
               onset_sd=3.0, min_dur_s=0.5):
    fs, per, n_epoch = float(gt["fs"]), int(gt["per"]), int(gt["n_epoch"])
    F0 = np.empty_like(F)
    win = int(round(baseline_window_s * fs))
    for e in range(n_epoch):
        a, b = e * per, (e + 1) * per
        F0[:, a:b] = percentile_filter(F[:, a:b], win, 50.0)
    d = (F - F0) / np.maximum(F0, 1.0) * 100.0
    if smooth == "matched":
        d = matched_filter(d, float(gt["tau_s"]) * fs)
    ev, _, st = detect_events(d, fs, on_k=onset_sd, min_dur_s=min_dur_s,
                              fp_method=fp_method)
    mat = np.stack([auc_per_min(ev, F.shape[0], e * per, (e + 1) * per, fs)
                    for e in range(n_epoch)], axis=1)
    return mat, ev, st


def evaluate(raw_path: Path, denoised_path: Path | None, *, smooth, fp_method,
             tol_epoch_pct=10.0, tol_snr_corr=0.4, tol_fabricate=1):
    import tifffile

    gt = np.load(Path(raw_path).with_suffix(".gt.npz"))
    masks = gt["masks"]
    snr = gt["snr"]
    silent = set(gt["silent"].tolist())
    n_epoch = int(gt["n_epoch"])

    out = {}
    for label, path in (("raw", raw_path), ("denoised", denoised_path)):
        if path is None:
            continue
        mov = tifffile.imread(str(path)).astype(np.float32)
        if mov.shape[0] != int(gt["n_t"]):
            print(f"ERROR: {label} has {mov.shape[0]} frames, ground truth has "
                  f"{int(gt['n_t'])}", file=sys.stderr)
            return None
        F = extract(mov, masks)
        mat, ev, st = auc_matrix(F, gt, smooth=smooth, fp_method=fp_method)
        out[label] = {"auc": mat, "events": len(ev), "stats": st}

    act = np.array([i for i in range(len(snr)) if i not in silent])
    print(f"\nsettings: smooth={smooth}  fp={fp_method}")
    print(f"{'':10s} {'events':>7} {'AUC ep1':>9} {'AUC ep%d' % n_epoch:>9} "
          f"{'trend':>9} {'corr(bias,SNR)':>15} {'silent ROIs w/ events':>22}")

    rows = {}
    for label, r in out.items():
        mat = r["auc"]
        e1 = float(mat[act, 0].mean())
        eN = float(mat[act, -1].mean())
        trend = (eN - e1) / max(abs(e1), 1e-9) * 100
        # per-cell bias against the cell's own first epoch, versus its SNR
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = mat[act] / np.maximum(mat[act, 0:1], 1e-9)
        last = rel[:, -1]
        ok = np.isfinite(last) & (mat[act, 0] > 0)
        corr = (float(np.corrcoef(np.log(snr[act][ok]), last[ok])[0, 1])
                if ok.sum() > 3 else float("nan"))
        n_fab = int(sum(1 for i in silent if r["auc"][i].sum() > 0))
        rows[label] = dict(events=r["events"], e1=e1, eN=eN, trend=trend,
                           corr=corr, fabricated=n_fab)
        print(f"{label:10s} {r['events']:7d} {e1:9.2f} {eN:9.2f} "
              f"{trend:+8.1f}% {corr:15.3f} {n_fab:22d}")

    if "denoised" not in rows:
        print("\n(no --denoised given; the raw row is the reference)")
        return rows

    d, rw = rows["denoised"], rows["raw"]
    print("\n--- gate ---")
    checks = []
    # If the denoiser has flattened almost everything, the trend and the
    # brightness-correlation both read as clean because there is nothing left
    # to be biased. Say so rather than letting a collapse pass two checks.
    collapsed = d["e1"] < 0.2 * rw["e1"]
    if collapsed:
        print("  NOTE: the denoised activity rate is under a fifth of the raw "
              "one. The trend\n        and correlation checks below are "
              "uninformative when the signal is gone.")
    checks.append((
        "activity retained",
        not collapsed,
        f"epoch 1 rate {d['e1']:.2f} vs {rw['e1']:.2f} raw"))
    checks.append((
        "no fabricated activity",
        d["fabricated"] <= tol_fabricate,
        f"{d['fabricated']} of {len(silent)} silent cells gained events "
        f"(raw: {rw['fabricated']})"))
    checks.append((
        "activity flat across epochs",
        abs(d["trend"]) <= tol_epoch_pct,
        f"{d['trend']:+.1f}% from first to last epoch, where the truth is 0% "
        f"(raw: {rw['trend']:+.1f}%)"))
    checks.append((
        "bias independent of brightness",
        not np.isfinite(d["corr"]) or abs(d["corr"]) <= tol_snr_corr,
        f"corr(per-cell change, log SNR) = {d['corr']:.3f} "
        f"(raw: {rw['corr']:.3f})"))
    checks.append((
        "sensitivity not reduced",
        d["events"] >= rw["events"],
        f"{d['events']} events vs {rw['events']} raw"))
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:32s} {detail}")
    verdict = all(ok for _, ok, _ in checks)
    print(f"\n  quantitative gate: {'PASS' if verdict else 'FAIL'}")
    if not verdict:
        print("  A failure here means the denoiser may still be used for "
              "representative images,\n  but not for the numbers: it would "
              "move the activity rate on its own.")
    else:
        print("  Usable as a cross-check. The headline numbers should still "
              "come from the\n  untransformed traces, with the denoised "
              "version shown to agree.")
    rows["checks"] = [{"name": n, "pass": bool(o), "detail": t}
                      for n, o, t in checks]
    rows["verdict"] = verdict
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="quantitative gate for a denoiser, for calcium event AUC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--make-synth", type=Path, default=None)
    p.add_argument("--eval", dest="eval_path", type=Path, default=None)
    p.add_argument("--denoised", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None, help="write the verdict as JSON")
    p.add_argument("--frames", type=int, default=3600)
    p.add_argument("--cells", type=int, default=24)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--bleach", type=float, default=0.75,
                   help="brightness at the end, as a fraction of the start")
    p.add_argument("--snr-range", type=float, nargs=2, default=[1.5, 6.0],
                   metavar=("LO", "HI"), help="peak transient in units of noise sd")
    p.add_argument("--smooth", choices=["none", "matched"], default="matched")
    p.add_argument("--fp-method", choices=["bin", "cumulative"], default="cumulative")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if args.make_synth:
        make_synth(args.make_synth, n_t=args.frames, n_cell=args.cells,
                   n_epoch=args.epochs, bleach=args.bleach,
                   snr_lo=args.snr_range[0], snr_hi=args.snr_range[1],
                   seed=args.seed)
        return 0

    if args.eval_path:
        res = evaluate(args.eval_path, args.denoised, smooth=args.smooth,
                       fp_method=args.fp_method)
        if res is None:
            return 2
        if args.out:
            ser = {k: (v if not isinstance(v, dict) else
                       {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                        for kk, vv in v.items() if kk != "stats"})
                   for k, v in res.items()}
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w") as fh:
                json.dump(ser, fh, indent=2, default=str)
            print(f"wrote {args.out}")
        return 0 if res.get("verdict", True) else 1

    p.error("pass --make-synth or --eval")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
