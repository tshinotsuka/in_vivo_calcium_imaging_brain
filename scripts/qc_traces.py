#!/usr/bin/env python
"""QC and trace-level diagnostics from Suite2p output.

Runs the checks that must pass before any activity claim is made, and produces
the pilot measurements that determine the design of the main experiment.

Panels
------
1. ROI map over the mean image, plus ROI count.
2. nu, the standardised noise level: median of the absolute frame-to-frame
   difference of dF/F, divided by sqrt(fs). Reported per ROI as a distribution,
   and split at the injection frame if one is given. A shift in nu between
   conditions moves every threshold-based measure on its own.
3. Raw F and Fneu, population mean over time. Bleaching, axial drift and
   swelling all appear here. A step at the injection frame is the thing to look
   for.
4. Fneu / F ratio over time -- an activity-independent proxy for tissue
   swelling that costs nothing, since Suite2p already writes both.
5. Mean image correlation across input-file boundaries. If motion correction
   was run per file, each file has its own template and inter-file drift is not
   corrected; this quantifies what is left.
6. Neuropil coefficient sweep: dF/F AUC computed for several values of r in
   F - r * Fneu. Under anaesthesia the population is strongly synchronised, so
   the neuropil trace resembles the somatic one and over-subtraction removes
   real signal. The result must not depend on r.
7. Time-resolved AUC in fixed windows, with two baselines:
     - rolling F0 (dilution- and drift-robust; [G] cancels)
     - fixed pre-injection F0 (retains slow changes, but also retains dilution,
       bleaching and drift)
   The two are reported side by side; their divergence is itself informative.

Nothing here uses spks.npy. Deconvolution is not part of the pipeline.

Example
-------
    python qc_traces.py \
        --s2p-dir <dataset>/work/s2p/suite2p/plane0 \
        --fs 13.0107 \
        --ledger <dataset>/work/s2p/frame_ledger.csv \
        --injection-frame 7800 \
        --out <dataset>/results/qc
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------


def rolling_baseline(F: np.ndarray, win: int, pct: float = 10.0) -> np.ndarray:
    """Percentile baseline in a sliding window, evaluated on a coarse grid.

    A rolling F0 tracks slow multiplicative changes such as indicator dilution
    from cell swelling, so it cancels them. It also removes genuinely slow
    activity changes -- which is why the fixed baseline is reported alongside.
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
    return out


def fixed_baseline(F: np.ndarray, upto: int, pct: float = 10.0) -> np.ndarray:
    """Single baseline value per ROI from the frames before ``upto``."""
    upto = max(int(upto), 1)
    return np.percentile(F[:, :upto], pct, axis=1)[:, None].astype(np.float32)


def dff(F: np.ndarray, F0: np.ndarray, floor: float = 1.0) -> np.ndarray:
    return (F - F0) / np.maximum(F0, floor)


def nu_of(dff_pct: np.ndarray, fs: float) -> np.ndarray:
    """Standardised noise, in %/sqrt(Hz). Input must be dF/F in percent."""
    return np.median(np.abs(np.diff(dff_pct, axis=1)), axis=1) / np.sqrt(fs)


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------


def panel_roi_map(ops, stat, iscell, out: Path):
    img = ops.get("meanImg")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    if img is not None:
        vmin, vmax = np.percentile(img, [1, 99.5])
        for ax in axes:
            ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    keep = iscell[:, 0].astype(bool)
    for s, k in zip(stat, keep):
        if not k:
            continue
        axes[1].plot(s["xpix"], s["ypix"], ".", ms=0.25, alpha=0.5)
    axes[0].set_title("mean image")
    axes[1].set_title(f"accepted ROIs (n={int(keep.sum())} of {len(stat)})")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "qc1_roi_map.png", dpi=150)
    plt.close(fig)


def panel_nu(nu: np.ndarray, nu_split, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(nu, bins=40, color="0.35")
    axes[0].set_xlabel(r"$\nu$  (%$\cdot$Hz$^{-1/2}$)")
    axes[0].set_ylabel("ROIs")
    axes[0].set_title(
        rf"noise level: median $\nu$ = {np.median(nu):.2f}"
    )
    if nu_split is not None:
        pre, post = nu_split
        axes[1].scatter(pre, post, s=6, alpha=0.5)
        lim = [0, float(np.nanpercentile(np.concatenate([pre, post]), 99)) * 1.1]
        axes[1].plot(lim, lim, "r--", lw=1)
        axes[1].set_xlim(lim)
        axes[1].set_ylim(lim)
        axes[1].set_xlabel(r"$\nu$ pre")
        axes[1].set_ylabel(r"$\nu$ post")
        axes[1].set_title(
            f"median pre {np.median(pre):.2f} -> post {np.median(post):.2f}"
        )
    else:
        axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(out / "qc2_noise_level.png", dpi=150)
    plt.close(fig)


def panel_raw_trends(F, Fneu, fs, inj, boundaries, out: Path):
    t = np.arange(F.shape[1]) / fs / 60.0
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(t, F.mean(0), lw=0.7, color="C0")
    axes[0].set_ylabel("raw F (population mean)")
    axes[1].plot(t, Fneu.mean(0), lw=0.7, color="C1")
    axes[1].set_ylabel("Fneu (population mean)")

    ratio = Fneu.mean(0) / np.maximum(F.mean(0), 1.0)
    axes[2].plot(t, ratio, lw=0.7, color="C2")
    axes[2].set_ylabel("Fneu / F")
    axes[2].set_xlabel("time (min)")

    for ax in axes:
        for b in boundaries:
            ax.axvline(b / fs / 60.0, color="0.6", ls=":", lw=0.8)
        if inj is not None:
            ax.axvline(inj / fs / 60.0, color="r", ls="--", lw=1.2)
    axes[0].set_title("raw fluorescence trends; red = injection, dotted = file boundary")
    fig.tight_layout()
    fig.savefig(out / "qc3_raw_trends.png", dpi=150)
    plt.close(fig)


def panel_neucoeff_sweep(F, Fneu, fs, inj, coeffs, win_frames, out: Path):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    summary = {}
    for r in coeffs:
        Fc = F - r * Fneu
        F0 = rolling_baseline(Fc, win_frames)
        d = dff(Fc, F0) * 100.0
        centres, auc = windowed_auc(d, fs, win_frames)
        ax.plot(centres / 60.0, auc, marker="o", ms=3, label=f"r = {r:g}")
        summary[str(r)] = [float(v) for v in auc]
    if inj is not None:
        ax.axvline(inj / fs / 60.0, color="r", ls="--", lw=1.2)
    ax.set_xlabel("time (min)")
    ax.set_ylabel(r"mean $\Delta$F/F AUC per window (%$\cdot$s)")
    ax.set_title("neuropil coefficient sweep (rolling baseline)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "qc4_neucoeff_sweep.png", dpi=150)
    plt.close(fig)
    return summary


def windowed_auc(d_pct: np.ndarray, fs: float, win_frames: int):
    """Mean dF/F integrated over consecutive windows. No thresholding."""
    n = d_pct.shape[1]
    edges = np.arange(0, n + 1, win_frames)
    if edges[-1] != n:
        edges = np.append(edges, n)
    centres, auc = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < win_frames // 2:
            continue
        centres.append((a + b) / 2 / fs)
        auc.append(float(np.mean(d_pct[:, a:b].sum(axis=1) / fs)))
    return np.asarray(centres), np.asarray(auc)


def panel_baseline_comparison(F, Fneu, fs, inj, r, win_frames, out: Path):
    Fc = F - r * Fneu
    d_roll = dff(Fc, rolling_baseline(Fc, win_frames)) * 100.0
    c_roll, a_roll = windowed_auc(d_roll, fs, win_frames)

    result = {"rolling": [float(v) for v in a_roll],
              "centres_s": [float(v) for v in c_roll]}

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(c_roll / 60.0, a_roll, marker="o", ms=3, label="rolling F0 (primary)")

    if inj is not None and inj > win_frames:
        d_fix = dff(Fc, fixed_baseline(Fc, inj)) * 100.0
        c_fix, a_fix = windowed_auc(d_fix, fs, win_frames)
        ax.plot(c_fix / 60.0, a_fix, marker="s", ms=3, label="fixed pre-injection F0")
        ax.axvline(inj / fs / 60.0, color="r", ls="--", lw=1.2)
        result["fixed"] = [float(v) for v in a_fix]

    ax.set_xlabel("time (min)")
    ax.set_ylabel(r"mean $\Delta$F/F AUC per window (%$\cdot$s)")
    ax.set_title(f"time-resolved AUC, two baselines (r = {r:g})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "qc5_time_resolved_auc.png", dpi=150)
    plt.close(fig)
    return result


def panel_boundary_check(ops_bin, boundaries, out: Path):
    """Mean image correlation across file boundaries, from the Suite2p binary."""
    if not boundaries or ops_bin is None:
        return None
    path, (ny, nx), nframes = ops_bin
    mm = np.memmap(path, dtype=np.int16, mode="r", shape=(nframes, ny, nx))
    span = 200
    rows = []
    for b in boundaries:
        a0, a1 = max(0, b - span), b
        b0, b1 = b, min(nframes, b + span)
        if a1 - a0 < 10 or b1 - b0 < 10:
            continue
        m_pre = np.asarray(mm[a0:a1]).mean(0).ravel()
        m_post = np.asarray(mm[b0:b1]).mean(0).ravel()
        r = float(np.corrcoef(m_pre, m_post)[0, 1])
        rows.append({"boundary_frame": int(b), "mean_image_corr": r})
    del mm
    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(rows)), [x["mean_image_corr"] for x in rows], color="0.35")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([str(x["boundary_frame"]) for x in rows], rotation=45)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("mean image correlation")
    ax.set_xlabel("boundary frame")
    ax.set_title("across-file alignment (low values = uncorrected inter-file drift)")
    fig.tight_layout()
    fig.savefig(out / "qc6_file_boundary.png", dpi=150)
    plt.close(fig)
    return rows


# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="QC and trace diagnostics from Suite2p output",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--s2p-dir", type=Path, required=True, help="suite2p/plane0 directory")
    p.add_argument("--fs", type=float, required=True)
    p.add_argument("--out", type=Path, required=True, help="output directory for QC figures")
    p.add_argument("--ledger", type=Path, default=None, help="frame_ledger.csv from the ROI step")
    p.add_argument("--injection-frame", type=int, default=None, help="frame index at which the drug was given")
    p.add_argument("--baseline-window-s", type=float, default=45.0, help="rolling baseline window in seconds")
    p.add_argument("--auc-window-s", type=float, default=60.0, help="window length for time-resolved AUC")
    p.add_argument("--baseline-percentile", type=float, default=10.0)
    p.add_argument("--neucoeff", type=float, default=0.7, help="r used for the main panels")
    p.add_argument("--neucoeff-sweep", type=float, nargs="+", default=[0.0, 0.5, 0.7])
    p.add_argument("--all-roi", action="store_true", help="use every ROI instead of only iscell-accepted ones")
    args = p.parse_args(argv)

    s2p = args.s2p_dir.expanduser().resolve()
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    F = np.load(s2p / "F.npy")
    Fneu = np.load(s2p / "Fneu.npy")
    stat = np.load(s2p / "stat.npy", allow_pickle=True)
    iscell = np.load(s2p / "iscell.npy")
    ops = np.load(s2p / "ops.npy", allow_pickle=True).item()

    keep = np.ones(F.shape[0], bool) if args.all_roi else iscell[:, 0].astype(bool)
    F, Fneu = F[keep].astype(np.float32), Fneu[keep].astype(np.float32)
    n_roi, n_frames = F.shape
    print(f"ROIs: {n_roi} ({'all' if args.all_roi else 'iscell-accepted'})  frames: {n_frames}")
    print(f"duration: {n_frames / args.fs / 60:.1f} min at {args.fs} Hz")

    boundaries = []
    if args.ledger and args.ledger.exists():
        with open(args.ledger) as fh:
            rows = list(csv.DictReader(fh))
        boundaries = [int(r["frame_start"]) for r in rows[1:]]
        print(f"file boundaries at frames: {boundaries}")

    inj = args.injection_frame
    win_base = int(round(args.baseline_window_s * args.fs))
    win_auc = int(round(args.auc_window_s * args.fs))
    print(f"rolling baseline window: {win_base} frames ({args.baseline_window_s}s)")
    print(f"AUC window: {win_auc} frames ({args.auc_window_s}s)")

    # --- panel 1 -------------------------------------------------------------
    panel_roi_map(ops, stat[keep], iscell[keep], out)

    # --- panel 2: nu ---------------------------------------------------------
    Fc = F - args.neucoeff * Fneu
    d = dff(Fc, rolling_baseline(Fc, win_base, args.baseline_percentile)) * 100.0
    nu = nu_of(d, args.fs)
    nu_split = None
    if inj is not None and 0 < inj < n_frames - 1:
        nu_split = (nu_of(d[:, :inj], args.fs), nu_of(d[:, inj:], args.fs))
    panel_nu(nu, nu_split, out)
    print(f"nu: median {np.median(nu):.2f}  IQR {np.percentile(nu,25):.2f}-{np.percentile(nu,75):.2f} %/sqrt(Hz)")
    if nu_split is not None:
        print(f"nu pre {np.median(nu_split[0]):.2f} -> post {np.median(nu_split[1]):.2f}")

    # --- panel 3, 4 ----------------------------------------------------------
    panel_raw_trends(F, Fneu, args.fs, inj, boundaries, out)

    # --- panel 5: neucoeff sweep --------------------------------------------
    sweep = panel_neucoeff_sweep(
        F, Fneu, args.fs, inj, args.neucoeff_sweep, win_auc, out
    )

    # --- panel 6: baselines --------------------------------------------------
    auc = panel_baseline_comparison(
        F, Fneu, args.fs, inj, args.neucoeff, win_auc, out
    )

    # --- panel 7: file boundaries -------------------------------------------
    ops_bin = None
    reg = ops.get("reg_file")
    if reg and Path(reg).exists():
        ops_bin = (reg, (ops["Ly"], ops["Lx"]), int(ops["nframes"]))
    bnd = panel_boundary_check(ops_bin, boundaries, out)
    if bnd:
        for row in bnd:
            print(f"boundary {row['boundary_frame']}: mean image r = {row['mean_image_corr']:.4f}")

    # --- record --------------------------------------------------------------
    summary = {
        "s2p_dir": str(s2p),
        "n_roi_used": int(n_roi),
        "n_frames": int(n_frames),
        "fs_hz": args.fs,
        "injection_frame": inj,
        "file_boundaries": boundaries,
        "nu": {
            "median": float(np.median(nu)),
            "q25": float(np.percentile(nu, 25)),
            "q75": float(np.percentile(nu, 75)),
            "pre_median": float(np.median(nu_split[0])) if nu_split else None,
            "post_median": float(np.median(nu_split[1])) if nu_split else None,
        },
        "windows": {
            "baseline_window_frames": win_base,
            "auc_window_frames": win_auc,
            "baseline_percentile": args.baseline_percentile,
        },
        "neucoeff_sweep_auc": sweep,
        "time_resolved_auc": auc,
        "file_boundary_corr": bnd,
        "note": "deconvolution not used. AUC is threshold-free. "
                "rolling F0 cancels multiplicative changes such as indicator "
                "dilution; fixed F0 retains slow activity but also retains "
                "dilution, bleaching and axial drift.",
    }
    with open(out / "qc_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {out / 'qc_summary.json'} and 6 figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
