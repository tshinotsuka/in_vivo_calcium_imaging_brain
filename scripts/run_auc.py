#!/usr/bin/env python
"""AUC of dF/F across a series of acquisitions, from one Suite2p detection.

Reads the traces produced by a single detection run spanning every acquisition,
splits them at the acquisition boundaries recorded in frame_ledger.csv, and
reports the integrated dF/F per window, per run and per ROI.

Two baselines are reported side by side because they answer different questions
and neither is correct alone:

  rolling  -- a percentile baseline in a sliding window, computed WITHIN each
              acquisition. Multiplicative changes that are slow compared with
              the window cancel, so bleaching, focus drift and indicator
              dilution do not masquerade as an activity change. Genuinely slow
              activity changes cancel too.
  fixed    -- one baseline per ROI taken from the first acquisition and applied
              to all of them. Slow activity changes survive, but so do
              bleaching and drift.

A decline that appears under both is an activity change; one that appears only
under the fixed baseline is not yet distinguishable from photobleaching, and the
raw F trend reported alongside is what separates them.

AUC is threshold-free: it integrates dF/F without deciding what counts as an
event, so it does not inherit a threshold's sensitivity to the noise level.
Units are percent-seconds per second of recording, i.e. mean dF/F in percent.

The FOV mean is the primary unit. Treating each ROI as an independent
observation would understate the uncertainty, since neurons in one field share
a state, a focal plane and an animal.

Example
-------
    python run_auc.py \
        --s2p-dir <dataset>/work/s2p/suite2p/plane0 \
        --dataset <dataset> \
        --out <dataset>/work/auc
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------


# Standard-normal quantiles for the percentiles a baseline is usually taken at.
_Z = {1.0: -2.3263, 5.0: -1.6449, 8.0: -1.4051, 10.0: -1.2816,
      15.0: -1.0364, 20.0: -0.8416, 25.0: -0.6745, 50.0: 0.0}


def _z_of(pct: float) -> float:
    try:
        from scipy.stats import norm
        return float(norm.ppf(pct / 100.0))
    except ImportError:
        k = min(_Z, key=lambda x: abs(x - pct))
        return _Z[k]


def robust_sd(F: np.ndarray) -> np.ndarray:
    """Noise scale per ROI from the median absolute frame-to-frame difference.

    Transients inflate a plain standard deviation, so an active trace would
    appear noisier and receive a larger bias correction than it should.
    """
    d = np.abs(np.diff(F, axis=1))
    return np.median(d, axis=1) / (np.sqrt(2) * 0.6745)


def rolling_baseline(F: np.ndarray, win: int, pct: float = 10.0,
                     correct_bias: bool = True) -> np.ndarray:
    """Percentile baseline in a sliding window, corrected for percentile bias.

    A low percentile of a noisy trace lands about |z| * sigma BELOW the true
    resting level, so dF/F acquires a constant positive offset of |z| * sigma /
    F0. That offset is not activity, and because F0 shrinks with photobleaching
    while sigma does not shrink as fast, the offset GROWS over a long recording
    and can hide, or invent, a change in activity across acquisitions. Adding
    |z| * sigma back removes it.
    """
    n = F.shape[1]
    win = max(int(win), 3)
    step = max(win // 4, 1)
    centres = np.arange(0, n, step)
    vals = np.empty((F.shape[0], centres.size), dtype=np.float32)
    for j, c in enumerate(centres):
        a, b = max(0, c - win // 2), min(n, c + win // 2 + 1)
        vals[:, j] = np.percentile(F[:, a:b], pct, axis=1)
    out = np.empty_like(F, dtype=np.float32)
    for i in range(F.shape[0]):
        out[i] = np.interp(np.arange(n), centres, vals[i])
    if correct_bias:
        out = out + (abs(_z_of(pct)) * robust_sd(F))[:, None]
    return out


def fixed_baseline(F: np.ndarray, a: int, b: int, pct: float = 10.0,
                   correct_bias: bool = True) -> np.ndarray:
    """One baseline per ROI from frames [a, b), with the same bias correction."""
    v = np.percentile(F[:, a:b], pct, axis=1).astype(np.float32)
    if correct_bias:
        v = v + abs(_z_of(pct)) * robust_sd(F[:, a:b])
    return v[:, None]


def dff(F: np.ndarray, F0: np.ndarray, floor: float = 1.0) -> np.ndarray:
    return (F - F0) / np.maximum(F0, floor)


def nu_of(d_pct: np.ndarray, fs: float) -> np.ndarray:
    return np.median(np.abs(np.diff(d_pct, axis=1)), axis=1) / np.sqrt(fs)


def auc_of(d_pct: np.ndarray, fs: float) -> np.ndarray:
    """Integrated dF/F per second of recording, per ROI. Percent-seconds / second."""
    return d_pct.sum(axis=1) / fs / (d_pct.shape[1] / fs)


def segments(ledger_rows, n_frames):
    segs = []
    for r in ledger_rows:
        a, b = int(r["frame_start"]), int(r["frame_end"]) + 1
        segs.append((r["source_file"], a, min(b, n_frames)))
    return segs


# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="AUC of dF/F across acquisitions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--s2p-dir", type=Path, required=True)
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--ledger", type=Path, default=None,
                   help="frame_ledger.csv (default: two levels above --s2p-dir)")
    p.add_argument("--timing", type=Path, default=None,
                   help="acquisition_timing.csv from the motion-correction step")
    p.add_argument("--fs", type=float, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--window-s", type=float, default=60.0,
                   help="window length for the time-resolved AUC")
    p.add_argument("--baseline-window-s", type=float, default=45.0)
    p.add_argument("--baseline-percentile", type=float, default=10.0)
    p.add_argument("--neucoeff", type=float, default=0.7)
    p.add_argument("--neucoeff-sweep", type=float, nargs="+", default=[0.0, 0.5, 0.7])
    p.add_argument("--all-roi", action="store_true")
    args = p.parse_args(argv)

    s2p = args.s2p_dir.expanduser().resolve()
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    F = np.load(s2p / "F.npy")
    Fneu = np.load(s2p / "Fneu.npy")
    iscell = np.load(s2p / "iscell.npy")
    keep = np.ones(F.shape[0], bool) if args.all_roi else iscell[:, 0].astype(bool)
    F, Fneu = F[keep].astype(np.float32), Fneu[keep].astype(np.float32)
    n_roi, n_frames = F.shape

    fs = args.fs
    if fs is None and args.dataset:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        try:
            from run_roi_suite2p import resolve_from_metadata  # noqa: F401
            fs = resolve_from_metadata(
                args.dataset.expanduser().resolve() / "raw" / "metadata.yaml").get("fs_hz")
        except Exception as e:  # noqa: BLE001
            print(f"metadata unreadable: {e}", file=sys.stderr)
    if fs is None:
        print("ERROR: frame rate unresolved; pass --fs or --dataset", file=sys.stderr)
        return 2

    ledger_path = args.ledger or (s2p.parent.parent / "frame_ledger.csv")
    if not ledger_path.exists():
        print(f"ERROR: no frame ledger at {ledger_path}. Detection must be run once\n"
              "  over all acquisitions together so the ROI set is shared; a ledger\n"
              "  is what records where each acquisition sits in the trace.",
              file=sys.stderr)
        return 2
    with open(ledger_path) as fh:
        rows = list(csv.DictReader(fh))
    segs = segments(rows, n_frames)
    covered = sum(b - a for _, a, b in segs)
    if covered != n_frames:
        print(f"ERROR: ledger covers {covered} frames but traces have {n_frames}",
              file=sys.stderr)
        return 2

    timing = {}
    tpath = args.timing or (s2p.parent.parent.parent / "acquisition_timing.csv")
    if Path(tpath).exists():
        with open(tpath) as fh:
            for r in csv.DictReader(fh):
                timing[r["file"]] = r

    print(f"ROIs {n_roi}   frames {n_frames}   {n_frames / fs / 60:.1f} min at {fs:.4g} Hz")
    print(f"acquisitions: {len(segs)}")

    win_b = int(round(args.baseline_window_s * fs))
    win_w = int(round(args.window_s * fs))

    # --- baselines ----------------------------------------------------------
    # Both are computed within each acquisition. A rolling window that spanned a
    # boundary would smear any step in brightness caused by refocusing or a
    # changed laser setting across both runs.
    Fc = F - args.neucoeff * Fneu
    roll = np.empty_like(Fc)
    for _, a, b in segs:
        roll[:, a:b] = rolling_baseline(Fc[:, a:b], win_b, args.baseline_percentile)
    d_roll = dff(Fc, roll) * 100.0

    a0, b0 = segs[0][1], segs[0][2]
    fixed = fixed_baseline(Fc, a0, b0, args.baseline_percentile)
    d_fix = dff(Fc, fixed) * 100.0

    # --- per run ------------------------------------------------------------
    per_run, per_roi_rows, per_window = [], [], []
    for k, (name, a, b) in enumerate(segs, start=1):
        seg_roll, seg_fix = d_roll[:, a:b], d_fix[:, a:b]
        auc_r, auc_f = auc_of(seg_roll, fs), auc_of(seg_fix, fs)
        nu = nu_of(seg_roll, fs)
        rawF = float(F[:, a:b].mean())
        t = timing.get(name, {})
        row = {
            "run": k, "file": name,
            "frame_start": a, "frame_end": b - 1,
            "duration_s": round((b - a) / fs, 1),
            "t_start_min": (round(float(t["t_start_s"]) / 60, 2)
                            if t.get("t_start_s") not in (None, "", "None") else None),
            "gap_before_s": (round(float(t["gap_before_s"]), 1)
                             if t.get("gap_before_s") not in (None, "", "None") else None),
            "auc_rolling_mean": round(float(auc_r.mean()), 3),
            "auc_rolling_sem": round(float(auc_r.std(ddof=1) / np.sqrt(n_roi)), 3),
            "auc_fixed_mean": round(float(auc_f.mean()), 3),
            "auc_fixed_sem": round(float(auc_f.std(ddof=1) / np.sqrt(n_roi)), 3),
            "nu_median": round(float(np.median(nu)), 2),
            "raw_F_mean": round(rawF, 1),
        }
        per_run.append(row)
        for i in range(n_roi):
            per_roi_rows.append({"run": k, "file": name, "roi": i,
                                 "auc_rolling": round(float(auc_r[i]), 3),
                                 "auc_fixed": round(float(auc_f[i]), 3),
                                 "nu": round(float(nu[i]), 2)})
        edges = np.arange(a, b + 1, win_w)
        if edges[-1] != b:
            edges = np.append(edges, b)
        for w0, w1 in zip(edges[:-1], edges[1:]):
            if w1 - w0 < win_w // 2:
                continue
            per_window.append({
                "run": k, "file": name,
                "t_center_min": round(((w0 + w1) / 2) / fs / 60, 2),
                "auc_rolling": round(float(auc_of(d_roll[:, w0:w1], fs).mean()), 3),
                "auc_fixed": round(float(auc_of(d_fix[:, w0:w1], fs).mean()), 3),
            })

    print("\nrun  file                                       AUC(roll)   AUC(fixed)   nu    rawF")
    for r in per_run:
        print(f"{r['run']:3d}  {r['file'][:42]:42s} "
              f"{r['auc_rolling_mean']:8.2f}   {r['auc_fixed_mean']:9.2f}  "
              f"{r['nu_median']:5.2f}  {r['raw_F_mean']:7.1f}")

    first, last = per_run[0], per_run[-1]
    for lab, key in [("rolling", "auc_rolling_mean"), ("fixed", "auc_fixed_mean")]:
        d_abs = last[key] - first[key]
        # a relative change is only meaningful while the quantity stays away
        # from zero; a fixed baseline drives dF/F through zero when the trace
        # bleaches below its reference, and the ratio then explodes
        rel = (f"  ({d_abs / abs(first[key]) * 100:+.1f}%)"
               if abs(first[key]) > 0.5 else "")
        print(f"run1 -> run{len(per_run)} ({lab} baseline): "
              f"{first[key]:.2f} -> {last[key]:.2f}, {d_abs:+.2f} pp{rel}")
    dF = (last["raw_F_mean"] - first["raw_F_mean"]) / max(first["raw_F_mean"], 1e-9) * 100
    print(f"raw F over the series: {dF:+.1f}%")

    # --- neuropil sweep ------------------------------------------------------
    sweep = {}
    for r in args.neucoeff_sweep:
        Fr = F - r * Fneu
        rb = np.empty_like(Fr)
        for _, a, b in segs:
            rb[:, a:b] = rolling_baseline(Fr[:, a:b], win_b, args.baseline_percentile)
        d = dff(Fr, rb) * 100.0
        sweep[str(r)] = [round(float(auc_of(d[:, a:b], fs).mean()), 3) for _, a, b in segs]
    print("\nneuropil coefficient sweep (AUC per run, rolling baseline):")
    for r, vals in sweep.items():
        print(f"  r = {r:<4s} {['%.2f' % v for v in vals]}")

    # --- figure --------------------------------------------------------------
    try:
        import figstyle_tshino as FS
        FS.set_style()
    except Exception:  # noqa: BLE001
        plt.rcParams.update({"font.family": "sans-serif",
                             "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"]})
    plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 9})

    runs = [r["run"] for r in per_run]
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))

    a0_ = ax[0][0]
    a0_.errorbar(runs, [r["auc_rolling_mean"] for r in per_run],
                 yerr=[r["auc_rolling_sem"] for r in per_run], marker="o", capsize=3,
                 label="rolling F0")
    a0_.errorbar(runs, [r["auc_fixed_mean"] for r in per_run],
                 yerr=[r["auc_fixed_sem"] for r in per_run], marker="s", capsize=3,
                 label="fixed F0 (run 1)")
    a0_.set_xlabel("acquisition")
    a0_.set_ylabel(r"mean $\Delta$F/F (%)")
    a0_.set_title("AUC per acquisition (mean $\\pm$ s.e.m. over ROIs)")
    a0_.legend(fontsize=8)

    a1_ = ax[0][1]
    tc = [w["t_center_min"] for w in per_window]
    a1_.plot(tc, [w["auc_rolling"] for w in per_window], marker="o", ms=3, lw=0.8,
             label="rolling F0")
    a1_.plot(tc, [w["auc_fixed"] for w in per_window], marker="s", ms=3, lw=0.8,
             label="fixed F0")
    for _, a, b in segs[1:]:
        a1_.axvline(a / fs / 60, color="0.6", ls=":", lw=0.8)
    a1_.set_xlabel("time in concatenated recording (min)")
    a1_.set_ylabel(r"mean $\Delta$F/F (%)")
    a1_.set_title(f"{args.window_s:g} s windows; dotted = acquisition boundary")
    a1_.legend(fontsize=8)

    a2_ = ax[1][0]
    for r, vals in sweep.items():
        a2_.plot(runs, vals, marker="o", label=f"r = {r}")
    a2_.set_xlabel("acquisition")
    a2_.set_ylabel(r"mean $\Delta$F/F (%)")
    a2_.set_title("neuropil coefficient sweep")
    a2_.legend(fontsize=8)

    a3_ = ax[1][1]
    a3_.plot(runs, [r["raw_F_mean"] for r in per_run], marker="o", color="C3",
             label="raw F")
    a3_.set_xlabel("acquisition")
    a3_.set_ylabel("raw F (a.u.)", color="C3")
    a3_.tick_params(axis="y", labelcolor="C3")
    tw = a3_.twinx()
    tw.plot(runs, [r["nu_median"] for r in per_run], marker="s", color="C0")
    tw.set_ylabel(r"median $\nu$ (%$\cdot$Hz$^{-1/2}$)", color="C0")
    tw.tick_params(axis="y", labelcolor="C0")
    a3_.set_title("brightness and noise across the series")

    fig.tight_layout()
    fig.savefig(out / "auc_summary.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "auc_summary.pdf", bbox_inches="tight")
    plt.close(fig)

    # --- outputs -------------------------------------------------------------
    def write(name, rows):
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write("auc_per_run.csv", per_run)
    write("auc_per_roi.csv", per_roi_rows)
    write("auc_per_window.csv", per_window)
    with open(out / "auc_summary.json", "w") as fh:
        json.dump({"s2p_dir": str(s2p), "fs_hz": fs, "n_roi": n_roi,
                   "n_frames": n_frames, "neucoeff": args.neucoeff,
                   "window_s": args.window_s,
                   "baseline_window_s": args.baseline_window_s,
                   "per_run": per_run, "neucoeff_sweep": sweep,
                   "note": "AUC is mean dF/F in percent, threshold-free. FOV mean is "
                           "the primary unit; per-ROI values are a distribution, not "
                           "independent observations."}, fh, indent=2)
    print(f"\nwrote 3 CSVs, auc_summary.json and auc_summary.png/.pdf to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
