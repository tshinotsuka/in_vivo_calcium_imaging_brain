#!/usr/bin/env python
"""Activity rate as AUC/min from statistically significant calcium transients.

Implements the event-based activity rate used in Magnus et al. 2019 (Science
aav5282, Fig. 5D) and the hippocampal imaging work it draws on: the cumulative
area under the dF/F trace of statistically significant events, divided by the
duration of the epoch.

Why events rather than the whole trace. Integrating everything counts noise as
activity, and the noise floor is not constant across a long recording -- it
grows as the preparation dims. Restricting the integral to events that pass a
false-positive test makes the measure insensitive to that drift, at the cost of
depending on a detection threshold.

How significance is decided. Deflections caused by noise or by motion along the
axial direction occur about equally often upward and downward, so the
downward-going deflections are a null distribution measured from the same trace
rather than assumed. Events are binned by amplitude and duration; in each bin
the ratio of negative to positive events estimates the false-positive rate, and
only bins below 5% are kept. Detection then runs a second time with the first
pass's events masked out, so that large transients do not inflate the standard
deviation that sets the threshold.

Outputs, per acquisition: an ROI map coloured by that acquisition's AUC/min, the
traces with every counted event shaded, and a bar of AUC/min per ROI. Across
acquisitions: one line per ROI plus the mean, and a matrix of every ROI-by-run
value. Every number is written to CSV.

Example
-------
    python run_event_auc.py \
        --s2p-dir <dataset>/work/s2p_series/suite2p/plane0 \
        --dataset <dataset> \
        --ledger <dataset>/work/s2p_series/frame_ledger.csv \
        --all-roi --out <dataset>/work/event_auc
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# traces
# ---------------------------------------------------------------------------


def percentile_filter(F: np.ndarray, win: int, pct: float = 50.0) -> np.ndarray:
    """Running percentile baseline, evaluated on a coarse grid and interpolated.

    The 50th percentile in a window of roughly 15 s is the baseline definition
    used in the source method. A median is not biased downward by noise the way
    a low percentile is, so no bias correction is needed here.
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


def exp_smooth(x: np.ndarray, tau_frames: float) -> np.ndarray:
    """Causal exponential smoothing, applied along time."""
    if tau_frames <= 0:
        return x.astype(np.float32)
    a = float(np.exp(-1.0 / tau_frames))
    out = np.empty_like(x, dtype=np.float32)
    out[:, 0] = x[:, 0]
    for t in range(1, x.shape[1]):
        out[:, t] = a * out[:, t - 1] + (1 - a) * x[:, t]
    return out


def robust_sd(x: np.ndarray) -> np.ndarray:
    d = np.abs(np.diff(x, axis=1))
    return np.median(d, axis=1) / (np.sqrt(2) * 0.6745)


# ---------------------------------------------------------------------------
# event detection
# ---------------------------------------------------------------------------


@dataclass
class Event:
    roi: int
    onset: int
    offset: int
    amplitude: float      # peak dF/F, percent
    duration_s: float
    area: float           # integral of dF/F over the event, percent-seconds


def _scan(trace: np.ndarray, sd: float, on_k: float, off_k: float, sign: int):
    """Find excursions that cross on_k*sd and end when they fall below off_k*sd."""
    x = trace * sign
    hi, lo = on_k * sd, off_k * sd
    above_hi = x > hi
    spans, i, n = [], 0, x.size
    while i < n:
        if not above_hi[i]:
            i += 1
            continue
        a = i
        while a > 0 and x[a - 1] > lo:
            a -= 1
        b = i
        while b < n - 1 and x[b + 1] > lo:
            b += 1
        spans.append((a, b + 1))
        i = b + 1
    # several crossings can occur inside one excursion; merge them
    merged: list[list[int]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def detect_events(d_pct: np.ndarray, fs: float, *, on_k: float = 3.0,
                  off_k: float = 0.5, min_dur_s: float = 0.5,
                  fp_max: float = 0.05, amp_bin_sd: float = 0.5,
                  dur_bin_s: float = 0.25, n_passes: int = 2):
    """Significant transients per ROI, with a false-positive rate below fp_max.

    Returns (events, kept_bins, stats). Negative-going excursions supply the
    null: in each amplitude-by-duration bin the ratio of negative to positive
    counts is the false-positive rate, and bins above fp_max are discarded
    wholesale rather than being thresholded per event.
    """
    n_roi, n_t = d_pct.shape
    sd = robust_sd(d_pct)
    mask = np.zeros_like(d_pct, bool)
    pos_all: list[tuple] = []
    neg_all: list[tuple] = []

    for p in range(max(n_passes, 1)):
        if p > 0:
            # recompute the threshold from frames that held no event, so that
            # large transients do not raise the bar against themselves
            sd = np.empty(n_roi, np.float32)
            for i in range(n_roi):
                free = d_pct[i][~mask[i]]
                sd[i] = (robust_sd(free[None, :])[0] if free.size > 10
                         else robust_sd(d_pct[i][None, :])[0])
        pos_all, neg_all = [], []
        mask = np.zeros_like(d_pct, bool)
        for i in range(n_roi):
            s = float(max(sd[i], 1e-9))
            base = float(np.median(d_pct[i]))
            tr = d_pct[i] - base
            # amplitude is kept in units of this ROI's own sd so the bins are
            # comparable across ROIs; area is already in percent-seconds and is
            # NOT rescaled, since it is the quantity the activity rate sums
            for a, b in _scan(tr, s, on_k, off_k, +1):
                pos_all.append((i, a, b, float(tr[a:b].max()) / s,
                                (b - a) / fs, float(tr[a:b].sum()) / fs))
                mask[i, a:b] = True
            for a, b in _scan(tr, s, on_k, off_k, -1):
                neg_all.append((i, a, b, float(-tr[a:b].min()) / s,
                                (b - a) / fs, float(-tr[a:b].sum()) / fs))
                mask[i, a:b] = True

    def binned(evs):
        out = {}
        for e in evs:
            if e[4] < min_dur_s:
                continue
            key = (int(e[3] / amp_bin_sd), int(e[4] / dur_bin_s))
            out.setdefault(key, []).append(e)
        return out

    pb, nb = binned(pos_all), binned(neg_all)
    kept_bins, rates = set(), {}
    for key, evs in pb.items():
        fp = len(nb.get(key, [])) / max(len(evs), 1)
        rates[key] = fp
        if fp < fp_max:
            kept_bins.add(key)

    events = [Event(roi=e[0], onset=e[1], offset=e[2],
                    amplitude=e[3] * float(sd[e[0]]),   # sd units -> percent
                    duration_s=e[4], area=e[5])          # already percent-seconds
              for key in kept_bins for e in pb[key]]
    events.sort(key=lambda e: (e.roi, e.onset))

    stats = {
        "n_positive_raw": len(pos_all), "n_negative_raw": len(neg_all),
        "n_bins": len(pb), "n_bins_kept": len(kept_bins),
        "n_events_kept": len(events),
        "median_sd_pct": float(np.median(sd)),
        "false_positive_rates": {f"{k[0]}_{k[1]}": round(v, 4)
                                 for k, v in sorted(rates.items())},
    }
    return events, kept_bins, stats


def auc_per_min(events, n_roi: int, a: int, b: int, fs: float) -> np.ndarray:
    """Cumulative event area within [a, b), per minute of recording, per ROI."""
    dur_min = (b - a) / fs / 60.0
    out = np.zeros(n_roi, float)
    for e in events:
        if e.onset >= a and e.offset <= b:
            out[e.roi] += e.area
    return out / max(dur_min, 1e-9)


# ---------------------------------------------------------------------------


def roi_edge(entry, shape):
    m = np.zeros(shape, bool)
    m[entry["ypix"], entry["xpix"]] = True
    inner = (np.roll(m, 1, 0) & np.roll(m, -1, 0)
             & np.roll(m, 1, 1) & np.roll(m, -1, 1))
    return m & ~inner


def apply_style():
    try:
        import figstyle_tshino as FS
        FS.set_style()
        return FS
    except Exception:  # noqa: BLE001
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"]})
        return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="event-based AUC/min per ROI per acquisition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--s2p-dir", type=Path, required=True)
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--ledger", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fs", type=float, default=None)
    p.add_argument("--pixel-size-um", type=float, default=None)
    p.add_argument("--neucoeff", type=float, default=0.7)
    p.add_argument("--baseline-window-s", type=float, default=15.0,
                   help="window for the running median baseline")
    p.add_argument("--smooth-tau-s", type=float, default=0.2,
                   help="exponential smoothing of dF/F before detection")
    p.add_argument("--onset-sd", type=float, default=3.0)
    p.add_argument("--offset-sd", type=float, default=0.5)
    p.add_argument("--min-duration-s", type=float, default=0.5)
    p.add_argument("--fp-max", type=float, default=0.05)
    p.add_argument("--all-roi", action="store_true")
    p.add_argument("--max-traces-per-page", type=int, default=25)
    # appearance. Every colour is an argument rather than a constant so a figure
    # can be restyled without touching the analysis, and so the same numbers can
    # be redrawn for a talk and for a manuscript.
    g = p.add_argument_group("appearance")
    g.add_argument("--cmap", default="viridis",
                   help="colormap mapping AUC/min onto ROI colour")
    g.add_argument("--cmap-vmax", type=float, default=None,
                   help="AUC/min at the top of the colormap (default: 99th pct "
                        "across all ROIs and runs, shared by every figure)")
    g.add_argument("--color-mean", default="C3", help="mean and s.e.m.")
    g.add_argument("--color-trace", default="0.25", help="dF/F traces")
    g.add_argument("--color-event", default="C3", help="shaded event area")
    g.add_argument("--color-point", default=None,
                   help="ROI points; default is to colour them by AUC/min")
    g.add_argument("--color-link", default="0.80", help="lines joining one ROI")
    g.add_argument("--color-bg", default="gray", help="mean-image colormap")
    g.add_argument("--event-alpha", type=float, default=0.55)
    g.add_argument("--point-size", type=float, default=26.0)
    g.add_argument("--trace-lw", type=float, default=0.45)
    g.add_argument("--bg-clip", type=float, nargs=2, default=[0.5, 99.8],
                   metavar=("LO_PCT", "HI_PCT"))
    g.add_argument("--scalebar-um", type=float, default=50.0)
    g.add_argument("--dpi", type=int, default=200)
    g.add_argument("--style", default=None,
                   help="matplotlib style file or name applied after "
                        "figstyle_tshino")
    g.add_argument("--rcparams", default=None,
                   help="JSON file of matplotlib rcParams applied last, so any "
                        "choice here can be overridden without editing code")
    args = p.parse_args(argv)

    s2p = args.s2p_dir.expanduser().resolve()
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    F = np.load(s2p / "F.npy")
    Fneu = np.load(s2p / "Fneu.npy")
    stat = np.load(s2p / "stat.npy", allow_pickle=True)
    iscell = np.load(s2p / "iscell.npy")
    imgs = {}
    for cand in ("reg_outputs.npy", "detect_outputs.npy", "ops.npy", "db.npy"):
        f = s2p / cand
        if f.exists():
            d = np.load(f, allow_pickle=True).item()
            for k in ("meanImg", "Vcorr", "max_proj", "Ly", "Lx"):
                if k in d and k not in imgs:
                    imgs[k] = d[k]

    keep = np.ones(F.shape[0], bool) if args.all_roi else iscell[:, 0].astype(bool)
    F, Fneu, stat = F[keep].astype(np.float32), Fneu[keep].astype(np.float32), stat[keep]
    n_roi, n_frames = F.shape

    fs, px = args.fs, args.pixel_size_um
    if (fs is None or px is None) and args.dataset:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        try:
            from run_roi_suite2p import resolve_from_metadata
            info = resolve_from_metadata(
                args.dataset.expanduser().resolve() / "raw" / "metadata.yaml")
            fs = fs if fs is not None else info.get("fs_hz")
            px = px if px is not None else info.get("pixel_size_um")
        except Exception as e:  # noqa: BLE001
            print(f"metadata unreadable: {e}", file=sys.stderr)
    if fs is None:
        print("ERROR: pass --fs or --dataset", file=sys.stderr)
        return 2

    ledger_path = args.ledger or (s2p.parent.parent / "frame_ledger.csv")
    if ledger_path.exists():
        with open(ledger_path) as fh:
            rows = list(csv.DictReader(fh))
        segs = [(r["source_file"], int(r["frame_start"]), int(r["frame_end"]) + 1)
                for r in rows]
    else:
        segs = [("all", 0, n_frames)]
        print(f"no ledger at {ledger_path}; treating the recording as one epoch",
              file=sys.stderr)

    print(f"ROIs {n_roi}   frames {n_frames}   {n_frames / fs / 60:.1f} min "
          f"at {fs:.4g} Hz   {len(segs)} acquisition(s)")

    # --- dF/F, per acquisition so a baseline never spans a boundary ---------
    Fc = F - args.neucoeff * Fneu
    F0 = np.empty_like(Fc)
    win_b = int(round(args.baseline_window_s * fs))
    for _, a, b in segs:
        F0[:, a:b] = percentile_filter(Fc[:, a:b], win_b, 50.0)
    d = (Fc - F0) / np.maximum(F0, 1.0) * 100.0
    d = exp_smooth(d, args.smooth_tau_s * fs)

    events, kept_bins, ev_stats = detect_events(
        d, fs, on_k=args.onset_sd, off_k=args.offset_sd,
        min_dur_s=args.min_duration_s, fp_max=args.fp_max)
    print(f"\nevents: {ev_stats['n_positive_raw']} positive, "
          f"{ev_stats['n_negative_raw']} negative (the null) detected raw")
    print(f"  amplitude x duration bins: {ev_stats['n_bins_kept']} of "
          f"{ev_stats['n_bins']} pass the {args.fp_max:.0%} false-positive test")
    print(f"  {ev_stats['n_events_kept']} events counted; "
          f"median threshold sd = {ev_stats['median_sd_pct']:.2f} %dF/F")

    by_roi_run = np.zeros((n_roi, len(segs)))
    for k, (_, a, b) in enumerate(segs):
        by_roi_run[:, k] = auc_per_min(events, n_roi, a, b, fs)

    silent = (by_roi_run.sum(axis=1) == 0)
    if silent.any():
        print(f"  {int(silent.sum())} ROI(s) had no significant event anywhere; "
              "reported as zero rather than dropped")

    shape = (np.asarray(imgs["meanImg"]).shape if "meanImg" in imgs
             else (int(imgs.get("Ly", 128)), int(imgs.get("Lx", 128))))
    apply_style()
    plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none"})
    if args.style:
        plt.style.use(args.style)
    if args.rcparams:
        with open(args.rcparams) as fh:
            plt.rcParams.update(json.load(fh))

    # --- per-acquisition figures --------------------------------------------
    # one scale for every figure, so colours are comparable across acquisitions
    vmax = args.cmap_vmax or float(np.percentile(by_roi_run, 99)) or 1.0
    cmap_auc = plt.get_cmap(args.cmap)

    def auc_color(v):
        return args.color_point or cmap_auc(min(max(v, 0.0) / vmax, 1.0))
    for k, (name, a, b) in enumerate(segs):
        vals = by_roi_run[:, k]
        t = (np.arange(a, b) - a) / fs

        fig = plt.figure(figsize=(15, max(6.0, 0.32 * n_roi + 3.2)))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 2.3],
                              height_ratios=[max(n_roi * 0.32, 4.5), 1.6],
                              wspace=0.14, hspace=0.22)
        axm = fig.add_subplot(gs[0, 0])
        axt = fig.add_subplot(gs[0, 1])
        axb = fig.add_subplot(gs[1, :])

        # ROI map, coloured by this acquisition's AUC/min
        if "meanImg" in imgs:
            im = np.asarray(imgs["meanImg"], float)
            lo, hi = np.percentile(im, list(args.bg_clip))
            axm.imshow(im, cmap=args.color_bg, vmin=lo, vmax=hi,
                       interpolation="nearest")
        for i in range(n_roi):
            ys, xs = np.nonzero(roi_edge(stat[i], shape))
            axm.plot(xs, ys, ".", ms=1.4, color=auc_color(vals[i]))
            axm.text(float(np.mean(stat[i]["xpix"])) + 3,
                     float(np.mean(stat[i]["ypix"])) - 3, str(i + 1),
                     fontsize=6, fontweight="bold", color=auc_color(vals[i]))
        if px and args.scalebar_um:
            bar = args.scalebar_um / px
            axm.plot([shape[1] * 0.06, shape[1] * 0.06 + bar],
                     [shape[0] * 0.94] * 2, "-", color="w", lw=2.5)
            axm.text(shape[1] * 0.06 + bar / 2, shape[0] * 0.94 - 3,
                     f"{args.scalebar_um:g} um",
                     color="w", ha="center", va="bottom", fontsize=7)
        axm.set_xlim(0, shape[1])
        axm.set_ylim(shape[0], 0)
        axm.axis("off")
        axm.set_title("ROIs coloured by AUC/min", fontsize=9)
        sm = plt.cm.ScalarMappable(cmap=cmap_auc,
                                   norm=plt.Normalize(vmin=0, vmax=vmax))
        cb = fig.colorbar(sm, ax=axm, fraction=0.045, pad=0.02)
        cb.set_label(r"AUC/min (%$\Delta$F/F$\cdot$s / min)", fontsize=7)
        cb.ax.tick_params(labelsize=6)

        # traces with counted events shaded
        span = float(np.percentile(d[:, a:b], 99.5))
        step = max(span * 1.15, 20.0)
        for i in range(n_roi):
            off = -step * i
            axt.plot(t, d[i, a:b] + off, lw=args.trace_lw, color=args.color_trace)
            for e in events:
                if e.roi != i or e.onset < a or e.offset > b:
                    continue
                sl = slice(e.onset - a, e.offset - a)
                axt.fill_between(t[sl], off, d[i, a:b][sl] + off,
                                 color=args.color_event, alpha=args.event_alpha,
                                 lw=0)
        axt.set_yticks([-step * i for i in range(n_roi)])
        axt.set_yticklabels([str(i + 1) for i in range(n_roi)], fontsize=6)
        axt.tick_params(axis="y", length=0)
        axt.set_xlabel("time within acquisition (s)")
        axt.set_xlim(0, t[-1])
        for side in ("top", "right", "left"):
            axt.spines[side].set_visible(False)
        axt.set_title("shaded = area counted toward AUC "
                      "(significant transients only)",
                      fontsize=9, loc="left")
        xb = t[-1] * 1.01
        axt.plot([xb, xb], [0, 50], "-", color="k", lw=1.6, clip_on=False)
        axt.text(xb * 1.004, 25, r" 50 %$\Delta$F/F", va="center", ha="left",
                 fontsize=7, clip_on=False)

        # per-ROI AUC for this acquisition. Each ROI is a point rather than a
        # bar: a bar from zero implies the spread is the interesting quantity
        # when it is the individual values that are, and it hides how many ROIs
        # sit at zero.
        xs = np.arange(1, n_roi + 1)
        axb.scatter(xs, vals, s=args.point_size, zorder=3,
                    color=[auc_color(v) for v in vals],
                    edgecolor="0.25", linewidth=0.4)
        mu = float(vals.mean())
        sem = float(vals.std(ddof=1) / np.sqrt(n_roi)) if n_roi > 1 else 0.0
        axb.axhline(mu, color=args.color_mean, lw=1.4, zorder=2,
                    label=f"mean {mu:.2f} $\\pm$ {sem:.2f} (s.e.m.)")
        axb.axhspan(mu - sem, mu + sem, color=args.color_mean, alpha=0.18,
                    lw=0, zorder=1)
        axb.set_xlabel("ROI")
        axb.set_ylabel(r"AUC/min")
        axb.set_xticks(xs)
        axb.tick_params(axis="x", labelsize=6)
        axb.set_xlim(0.3, n_roi + 0.7)
        axb.set_ylim(bottom=min(0.0, float(vals.min()) * 1.1))
        for side in ("top", "right"):
            axb.spines[side].set_visible(False)
        axb.legend(fontsize=7, frameon=False)

        n_ev = sum(1 for e in events if a <= e.onset and e.offset <= b)
        fig.suptitle(
            f"{name}   frames [{a} .. {b - 1}]   {(b - a) / fs / 60:.1f} min   "
            f"{n_roi} ROIs, {n_ev} significant events   "
            f"mean AUC/min = {vals.mean():.2f}",
            fontsize=10, x=0.01, ha="left")
        stem = out / f"event_auc_run{k + 1:02d}"
        fig.savefig(stem.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"  run {k + 1}: {name}  {n_ev:5d} events  "
              f"mean AUC/min {vals.mean():7.3f} -> {stem.name}.png/.pdf")

    # --- across acquisitions -------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8),
                           gridspec_kw={"width_ratios": [1.2, 1.0]})
    runs = np.arange(1, len(segs) + 1)
    # every ROI as a point, jittered so overlapping values stay countable, with
    # the paired lines faint behind them
    rng_j = np.random.default_rng(0)
    jit = rng_j.uniform(-0.12, 0.12, n_roi)
    for i in range(n_roi):
        ax[0].plot(runs + jit[i], by_roi_run[i], lw=0.5, color=args.color_link,
                   zorder=1)
        ax[0].scatter(runs + jit[i], by_roi_run[i], s=args.point_size * 0.46,
                      color=args.color_point or "0.45", zorder=2, linewidth=0)
    m = by_roi_run.mean(axis=0)
    se = (by_roi_run.std(axis=0, ddof=1) / np.sqrt(n_roi) if n_roi > 1
          else np.zeros(len(segs)))
    ax[0].errorbar(runs, m, yerr=se, color=args.color_mean, lw=2.2, marker="o", ms=6,
                   capsize=4, zorder=3, elinewidth=1.6,
                   label=f"mean $\\pm$ s.e.m. (n = {n_roi} ROIs)")
    ax[0].set_xlabel("acquisition")
    ax[0].set_ylabel(r"AUC/min (%$\Delta$F/F$\cdot$s / min)")
    ax[0].set_xticks(runs)
    ax[0].set_xlim(0.5, len(segs) + 0.5)
    ax[0].set_ylim(bottom=min(0.0, float(by_roi_run.min()) * 1.1))
    for side in ("top", "right"):
        ax[0].spines[side].set_visible(False)
    ax[0].legend(fontsize=8, frameon=False)
    ax[0].set_title("activity rate per ROI", fontsize=10, loc="left")

    order = np.argsort(-by_roi_run.mean(axis=1))
    im = ax[1].imshow(by_roi_run[order], aspect="auto", cmap=args.cmap,
                      vmin=0, vmax=vmax, interpolation="nearest")
    ax[1].set_xticks(np.arange(len(segs)))
    ax[1].set_xticklabels(runs)
    ax[1].set_yticks(np.arange(n_roi))
    ax[1].set_yticklabels([str(i + 1) for i in order], fontsize=6)
    ax[1].set_xlabel("acquisition")
    ax[1].set_ylabel("ROI (sorted by mean)")
    cb = fig.colorbar(im, ax=ax[1], fraction=0.046)
    cb.set_label("AUC/min", fontsize=8)
    ax[1].set_title("every ROI, every acquisition", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out / "event_auc_summary.png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out / "event_auc_summary.pdf", bbox_inches="tight")
    plt.close(fig)

    # --- tables --------------------------------------------------------------
    with open(out / "auc_per_roi_per_run.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["roi", "run", "file", "auc_per_min", "n_events",
                    "duration_min"])
        for k, (name, a, b) in enumerate(segs):
            dur = (b - a) / fs / 60.0
            for i in range(n_roi):
                ne = sum(1 for e in events
                         if e.roi == i and a <= e.onset and e.offset <= b)
                w.writerow([i + 1, k + 1, name, round(by_roi_run[i, k], 4), ne,
                            round(dur, 3)])

    with open(out / "events.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["roi", "onset_frame", "offset_frame", "t_onset_s",
                    "duration_s", "peak_dff_pct", "area_pct_s"])
        for e in events:
            w.writerow([e.roi + 1, e.onset, e.offset, round(e.onset / fs, 3),
                        round(e.duration_s, 3), round(e.amplitude, 3),
                        round(e.area, 4)])

    with open(out / "auc_per_run.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "file", "duration_min", "n_events",
                    "auc_per_min_mean", "auc_per_min_sem"])
        for k, (name, a, b) in enumerate(segs):
            v = by_roi_run[:, k]
            w.writerow([k + 1, name, round((b - a) / fs / 60, 3),
                        sum(1 for e in events if a <= e.onset and e.offset <= b),
                        round(float(v.mean()), 4),
                        round(float(v.std(ddof=1) / np.sqrt(n_roi)), 4)])

    with open(out / "event_auc_summary.json", "w") as fh:
        json.dump({
            "s2p_dir": str(s2p), "fs_hz": fs, "n_roi": n_roi,
            "n_frames": n_frames, "neucoeff": args.neucoeff,
            "detection": {"onset_sd": args.onset_sd, "offset_sd": args.offset_sd,
                          "min_duration_s": args.min_duration_s,
                          "fp_max": args.fp_max,
                          "baseline_window_s": args.baseline_window_s,
                          "smooth_tau_s": args.smooth_tau_s},
            "event_stats": ev_stats,
            "appearance": {k: getattr(args, k) for k in
                           ("cmap", "cmap_vmax", "color_mean", "color_trace",
                            "color_event", "color_point", "color_link",
                            "color_bg", "event_alpha", "point_size", "trace_lw",
                            "bg_clip", "scalebar_um", "dpi", "style",
                            "rcparams")},
            "cmap_vmax_used": round(float(vmax), 4),
            "auc_per_min_mean_by_run": [round(float(v), 4)
                                        for v in by_roi_run.mean(axis=0)],
            "note": "AUC/min is the cumulative area under dF/F of statistically "
                    "significant transients divided by epoch duration, after "
                    "Magnus et al. 2019. Negative-going deflections provide the "
                    "null for the false-positive test.",
        }, fh, indent=2)

    print(f"\nrun   {'file':40s} {'AUC/min':>9} {'s.e.m.':>8}")
    for k, (name, a, b) in enumerate(segs):
        v = by_roi_run[:, k]
        print(f"{k + 1:3d}   {name[:40]:40s} {v.mean():9.3f} "
              f"{v.std(ddof=1) / np.sqrt(n_roi):8.3f}")
    print(f"\nwrote {len(segs)} per-acquisition figures, a summary figure and "
          f"3 CSVs to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
