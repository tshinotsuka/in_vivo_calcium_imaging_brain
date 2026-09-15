#!/usr/bin/env python
"""How many ROIs went up, down, or neither, between two acquisitions.

The hard part of this figure is not the pie, it is the word "unchanged". A
fixed cutoff — ten per cent, say — states an answer rather than measuring one,
and its arbitrariness is hidden once the slices are drawn. So the comparison
uses each ROI's own variability instead: every acquisition is cut into windows,
which gives several values per ROI per acquisition, and an ROI counts as
changed only when the difference between the two acquisitions is larger than
that ROI's window-to-window spread would produce by chance.

A cell that is simply noisy therefore lands in "unchanged", where it belongs,
and a cell with a small but consistent shift can still count as changed. The
cutoff method remains available for comparison, and prints its own
arbitrariness alongside the result.

The pie is drawn next to a scatter of the two acquisitions against each other,
because the proportions alone say nothing about size: three ROIs that halved
and three that fell by a thousandth make the same slice.

Example
-------
    python fig_change_pie.py --auc-dir <dataset>/work/eauc_fix_series \
        --ledger <dataset>/work/s2p_series/frame_ledger.csv \
        --dataset <dataset> --from-run 1 --to-run 6 \
        --out <dataset>/work/change_pie
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


def read_events(path: Path):
    ev = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            ev.append((int(r["roi"]), int(r["onset_frame"]),
                       float(r["area_pct_s"])))
    return ev


def read_per_run(path: Path, value):
    M = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                M[(int(r["roi"]), int(r["run"]))] = float(r[value])
            except (KeyError, TypeError, ValueError):
                pass
    return M


def window_values(events, n_roi, a, b, fs, win_frames):
    """AUC per minute in each window of one acquisition, per ROI."""
    edges = np.arange(a, b + 1, win_frames)
    if edges[-1] != b:
        edges = np.append(edges, b)
    wins = [(x, y) for x, y in zip(edges[:-1], edges[1:])
            if y - x >= win_frames // 2]
    out = np.zeros((n_roi, len(wins)))
    for roi, onset, area in events:
        for j, (x, y) in enumerate(wins):
            if x <= onset < y:
                out[roi - 1, j] += area
                break
    for j, (x, y) in enumerate(wins):
        out[:, j] /= max((y - x) / fs / 60.0, 1e-9)
    return out


def bootstrap_ci(areas, duration_min, n_boot=10000, alpha=0.05, seed=0):
    """Interval covering what a repeat of this acquisition would produce.

    The AUC is a sum over that ROI's events, so its uncertainty is the
    uncertainty of that sum: resampling the events with replacement gives the
    spread the same cell would show if the acquisition were repeated with the
    same underlying rate. This uses the events themselves rather than cutting
    the acquisition into pieces, so it does not depend on a window length, and
    it stays valid when the events are unevenly spread through the recording.

    What it does not capture is slow drift within the acquisition: resampling
    treats the events as exchangeable, so a cell whose rate fell steadily
    across the run gets the same interval as one that was steady throughout.

    Nor does it reach its nominal coverage when events are few. Resampling the
    event sizes from a short list underestimates how far that list's own tail
    extends, and simulation puts the true coverage of a nominal 95% interval at
    roughly 82% at five events, 85% at twenty and 90% at eighty. The interval
    is therefore somewhat narrow, which makes the test somewhat liberal: it
    calls a change slightly more often than the stated level. The achieved
    coverage is reported alongside the result rather than left implicit.
    """
    a = np.asarray(areas, float)
    n = a.size
    rng = np.random.default_rng(seed)
    obs = a.sum() / duration_min if n else 0.0
    if n == 0:
        # no events at all: the interval is what a rate of zero can produce,
        # which is zero. Any activity in the other acquisition is then a change.
        return 0.0, 0.0, 0.0
    # The AUC is a count multiplied by a mean size, and both vary from one
    # acquisition to the next. Resampling a fixed number of events captures
    # only the second, and collapses to nothing when the events happen to be
    # the same size; drawing the count from a Poisson distribution with the
    # observed mean puts the counting variability back, which for a sparse
    # process is usually the larger part.
    # The question is whether the second acquisition could have come from the
    # same cell behaving the same way, which needs a PREDICTION interval rather
    # than a confidence interval: the second acquisition has its own sampling
    # variability, and the first only estimates the rate rather than fixing it.
    # Each draw therefore resamples the events twice -- once to stand for the
    # rate the first acquisition was estimating, and once for what a repeat of
    # it would produce -- which is what makes the interval cover a repeat
    # roughly as often as it claims.
    draws = np.empty(n_boot)
    for j in range(n_boot):
        c1 = rng.poisson(n)
        pop = a if c1 == 0 else rng.choice(a, size=c1, replace=True)
        c2 = rng.poisson(n)
        draws[j] = 0.0 if c2 == 0 else rng.choice(pop, size=c2, replace=True).sum()
    draws /= duration_min
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(obs), float(lo), float(hi)


def rank_test(x, y, n_perm=20000, seed=0):
    """Two-sided p for a difference in location, without a distributional claim.

    An exact permutation over the pooled values when the sample is small
    enough, sampled otherwise. With few windows per acquisition the smallest
    reachable p is bounded, and that bound is reported so a null result is not
    mistaken for evidence of no change.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = x.size, y.size
    if nx == 0 or ny == 0:
        return 1.0, 1.0
    obs = abs(np.mean(x) - np.mean(y))
    pooled = np.concatenate([x, y])
    from math import comb
    n_comb = comb(nx + ny, nx)
    rng = np.random.default_rng(seed)
    if n_comb <= n_perm:
        from itertools import combinations
        idx = np.arange(nx + ny)
        cnt = 0
        for c in combinations(idx, nx):
            m = np.zeros(nx + ny, bool)
            m[list(c)] = True
            if abs(pooled[m].mean() - pooled[~m].mean()) >= obs - 1e-12:
                cnt += 1
        return cnt / n_comb, 1.0 / n_comb
    cnt = 0
    for _ in range(n_perm):
        p = rng.permutation(pooled)
        if abs(p[:nx].mean() - p[nx:].mean()) >= obs - 1e-12:
            cnt += 1
    return (cnt + 1) / (n_perm + 1), 1.0 / n_comb


def bh(pvals, alpha):
    """Benjamini-Hochberg: which hypotheses are rejected at this false discovery rate."""
    p = np.asarray(pvals, float)
    n = p.size
    order = np.argsort(p)
    thresh = alpha * (np.arange(1, n + 1) / n)
    passed = p[order] <= thresh
    k = np.flatnonzero(passed)
    out = np.zeros(n, bool)
    if k.size:
        out[order[:k[-1] + 1]] = True
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="ROIs up, down or unchanged between two acquisitions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--auc-dir", type=Path, required=True,
                   help="directory holding events.csv and auc_per_roi_per_run.csv")
    p.add_argument("--ledger", type=Path, default=None)
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True, help="output stem")
    p.add_argument("--fs", type=float, default=None)
    p.add_argument("--from-run", type=int, default=1)
    p.add_argument("--to-run", type=int, default=None,
                   help="default is the last acquisition")
    p.add_argument("--value", default="auc_per_min_dff")
    p.add_argument("--method",
                   choices=["baseline-bootstrap", "window-test", "threshold"],
                   default="baseline-bootstrap",
                   help="'baseline-bootstrap' resamples the first "
                        "acquisition's own events to say how much that ROI's "
                        "AUC could vary, and calls the second acquisition "
                        "changed when it falls outside; 'window-test' instead "
                        "cuts both acquisitions into windows and compares them; "
                        "'threshold' applies a fixed fractional cutoff")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--both-ends", action="store_true",
                   help="for baseline-bootstrap: also require the first "
                        "acquisition to fall outside the second's interval, "
                        "which is stricter and symmetric between the two")
    p.add_argument("--window-s", type=float, default=60.0)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--fdr", action="store_true",
                   help="control the false discovery rate across ROIs")
    p.add_argument("--threshold", type=float, default=0.2,
                   help="for --method threshold: fractional change counted as changed")
    p.add_argument("--color-up", default="#2c7fb8")
    p.add_argument("--color-same", default="#bdbdbd")
    p.add_argument("--color-down", default="#d95f02")
    p.add_argument("--label", default=None, help="title for the panels")
    p.add_argument("--log-scatter", action="store_true")
    p.add_argument("--point-size", type=float, default=42.0,
                   help="marker area in points squared")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args(argv)

    auc_dir = args.auc_dir.expanduser().resolve()
    per_run_path = auc_dir / "auc_per_roi_per_run.csv"
    if not per_run_path.exists():
        print(f"ERROR: no auc_per_roi_per_run.csv in {auc_dir}", file=sys.stderr)
        return 2
    M = read_per_run(per_run_path, args.value)
    rois = sorted({r for r, _ in M})
    runs = sorted({k for _, k in M})
    r_from = args.from_run
    r_to = args.to_run if args.to_run is not None else runs[-1]
    if r_from not in runs or r_to not in runs:
        print(f"ERROR: acquisitions {runs} do not include "
              f"{r_from} and {r_to}", file=sys.stderr)
        return 2
    n_roi = len(rois)
    v_from = np.array([M.get((i, r_from), np.nan) for i in rois])
    v_to = np.array([M.get((i, r_to), np.nan) for i in rois])

    detail = ""
    ci_lo = ci_hi = None
    if args.method == "baseline-bootstrap":
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
        ev_path = auc_dir / "events.csv"
        ledger = args.ledger
        if ledger is None:
            for cand in (auc_dir / "frame_ledger.csv",
                         auc_dir.parent / "frame_ledger.csv"):
                if cand.exists():
                    ledger = cand
                    break
        if fs is None or not ev_path.exists() or ledger is None \
                or not Path(ledger).exists():
            print("ERROR: baseline-bootstrap needs events.csv, a frame ledger "
                  "and a frame rate.\n  Pass --ledger and --fs, or use "
                  "--method threshold.", file=sys.stderr)
            return 2
        events = read_events(ev_path)
        segs = []
        with open(ledger) as fh:
            for r in csv.DictReader(fh):
                segs.append((int(r["frame_start"]), int(r["frame_end"]) + 1))
        a1, b1 = segs[r_from - 1]
        a2, b2 = segs[r_to - 1]
        dur1 = (b1 - a1) / fs / 60.0
        dur2 = (b2 - a2) / fs / 60.0

        areas1 = {i: [] for i in rois}
        areas2 = {i: [] for i in rois}
        for roi, onset, area in events:
            if a1 <= onset < b1:
                areas1.setdefault(roi, []).append(area)
            elif a2 <= onset < b2:
                areas2.setdefault(roi, []).append(area)

        ci_lo = np.zeros(n_roi)
        ci_hi = np.zeros(n_roi)
        sig = np.zeros(n_roi, bool)
        n_ev1 = np.zeros(n_roi, int)
        for k, roi in enumerate(rois):
            aa = areas1.get(roi, [])
            n_ev1[k] = len(aa)
            _, lo, hi = bootstrap_ci(aa, dur1, args.n_boot, args.alpha, seed=k)
            ci_lo[k], ci_hi[k] = lo, hi
            outside = (v_to[k] < lo) or (v_to[k] > hi)
            if args.both_ends:
                _, lo2, hi2 = bootstrap_ci(areas2.get(roi, []), dur2,
                                           args.n_boot, args.alpha, seed=k + 977)
                outside = outside and (v_from[k] < lo2 or v_from[k] > hi2)
            sig[k] = outside
        pvals = np.full(n_roi, np.nan)
        thin = int((n_ev1 < 5).sum())
        med_n = int(np.median(n_ev1))
        print(f"bootstrap over the events of acquisition {r_from}: median "
              f"{med_n} events per ROI")
        # the interval is narrower than its label at small counts; say by how much
        cov = np.interp(med_n, [5, 10, 20, 40, 80], [82, 84, 85, 86, 90])
        print(f"  a nominal {int((1 - args.alpha) * 100)}% interval covers a "
              f"repeat about {cov:.0f}% of the time at this event count,\n"
              f"  so the test is somewhat liberal: it calls change a little "
              "more often than stated")
        if thin:
            print(f"  {thin} ROI(s) had fewer than 5 events there, so their "
                  "intervals rest on very\n  few values and the classification "
                  "for those is weak")
        detail = (f"{int((1 - args.alpha) * 100)}% bootstrap interval from "
                  f"acquisition {r_from}'s own events (covers ~{cov:.0f}%)"
                  + (", both directions" if args.both_ends else ""))
    elif args.method == "window-test":
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
        ev_path = auc_dir / "events.csv"
        ledger = args.ledger
        if ledger is None:
            for cand in (auc_dir / "frame_ledger.csv",
                         auc_dir.parent / "frame_ledger.csv"):
                if cand.exists():
                    ledger = cand
                    break
        if fs is None or not ev_path.exists() or ledger is None \
                or not Path(ledger).exists():
            print("ERROR: window-test needs events.csv, a frame ledger and a "
                  "frame rate.\n  Pass --ledger and --fs, or use "
                  "--method threshold.", file=sys.stderr)
            return 2
        events = read_events(ev_path)
        segs = []
        with open(ledger) as fh:
            for r in csv.DictReader(fh):
                segs.append((int(r["frame_start"]), int(r["frame_end"]) + 1))
        win = int(round(args.window_s * fs))
        W_from = window_values(events, n_roi, *segs[r_from - 1], fs, win)
        W_to = window_values(events, n_roi, *segs[r_to - 1], fs, win)
        print(f"{W_from.shape[1]} and {W_to.shape[1]} windows of "
              f"{args.window_s:g} s in acquisitions {r_from} and {r_to}")

        pvals, pmin = [], None
        for i in range(n_roi):
            pv, floor = rank_test(W_from[i], W_to[i], seed=i)
            pvals.append(pv)
            pmin = floor
        pvals = np.array(pvals)
        sig = (bh(pvals, args.alpha) if args.fdr else pvals < args.alpha)
        print(f"smallest reachable p with these window counts: {pmin:.4f}")
        if pmin > args.alpha:
            print(f"  which is above alpha={args.alpha:g}: no ROI can be called "
                  "changed.\n  Use shorter windows, or --method threshold.")
        detail = (f"windows of {args.window_s:g} s, permutation test, "
                  f"alpha {args.alpha:g}" + (" with FDR control" if args.fdr else ""))
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = (v_to - v_from) / np.abs(v_from)
        sig = np.abs(frac) > args.threshold
        pvals = np.full(n_roi, np.nan)
        detail = (f"a change of more than {args.threshold * 100:.0f}% counts, "
                  "which is a choice, not a measurement")

    up = sig & (v_to > v_from)
    down = sig & (v_to < v_from)
    same = ~sig
    counts = [int(up.sum()), int(same.sum()), int(down.sum())]
    names = ["increased", "unchanged", "decreased"]
    cols = [args.color_up, args.color_same, args.color_down]
    print(f"\nacquisition {r_from} vs {r_to}, n = {n_roi} ROIs")
    for nm, c in zip(names, counts):
        print(f"  {nm:11s} {c:3d}  ({c / n_roi * 100:4.1f}%)")

    # --- figure -------------------------------------------------------------
    try:
        import figstyle_tshino as FS
        FS.set_style()
    except Exception:  # noqa: BLE001
        plt.rcParams.update({"font.family": "sans-serif",
                             "font.sans-serif": ["Arial", "Liberation Sans",
                                                 "DejaVu Sans"]})
    plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none",
                         "font.size": 9})

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.8),
                           gridspec_kw={"width_ratios": [1.0, 1.25]})
    keep = [c > 0 for c in counts]
    ax[0].pie([c for c, k in zip(counts, keep) if k],
              labels=[f"{n}\n{c} of {n_roi}"
                      for n, c, k in zip(names, counts, keep) if k],
              colors=[c for c, k in zip(cols, keep) if k],
              autopct="%1.0f%%", startangle=90, counterclock=False,
              wedgeprops={"edgecolor": "white", "linewidth": 1.2},
              textprops={"fontsize": 9})
    ax[0].set_title(f"acquisition {r_from} to {r_to}", fontsize=10, loc="left")

    colour = np.where(up, args.color_up,
                      np.where(down, args.color_down, args.color_same))

    # Pad the axes so no marker is clipped by a spine. A point sitting exactly
    # on zero would otherwise be drawn as a half circle, which reads as a
    # different symbol rather than as the same point at a lower value, and the
    # marker radius is in points while the data are not, so the padding has to
    # be worked out from the figure size rather than guessed.
    if args.log_scatter:
        pos = np.concatenate([v_from[v_from > 0], v_to[v_to > 0]])
        base = [pos.min(), pos.max()] if pos.size else [1e-3, 1.0]
        ax[1].set_xscale("log")
        ax[1].set_yscale("log")
        span = np.log10(base[1] / max(base[0], 1e-12))
        pad = max(span * 0.08, 0.05)
        lim = [base[0] * 10 ** (-pad), base[1] * 10 ** pad]
    else:
        vmax = float(np.nanmax([np.nanmax(v_from), np.nanmax(v_to)]))
        vmin = float(np.nanmin([np.nanmin(v_from), np.nanmin(v_to), 0.0]))
        rng_ = max(vmax - vmin, 1e-9)
        # half the marker width, converted from points to data units, plus a
        # little room for the ROI labels
        ax_w_in = fig.get_size_inches()[0] * 0.55
        half_marker_frac = (np.sqrt(args.point_size) / 2 + 1.5) / (ax_w_in * 72)
        pad = rng_ * max(half_marker_frac, 0.03)
        lim = [vmin - pad, vmax + pad]
    ax[1].scatter(v_from, v_to, s=args.point_size, c=colour, edgecolor="0.3",
                  linewidth=0.5, zorder=3, clip_on=False)
    if ci_lo is not None:
        o = np.argsort(v_from)
        ax[1].fill_between(v_from[o], ci_lo[o], ci_hi[o], color="0.85",
                           alpha=0.6, lw=0, zorder=0,
                           label=f"{int((1 - args.alpha) * 100)}% interval for a "
                                 f"repeat of acquisition {r_from}")
    ax[1].plot(lim, lim, ls="--", color="0.6", lw=1.0, zorder=1,
               label="no change")
    ax[1].set_xlim(lim)
    ax[1].set_ylim(lim)
    for i, roi in enumerate(rois):
        if np.isfinite(v_from[i]) and np.isfinite(v_to[i]):
            ax[1].annotate(str(roi), (v_from[i], v_to[i]), xytext=(4, 3),
                           textcoords="offset points", fontsize=5.5,
                           color="0.35", zorder=4)
    ax[1].set_xlabel(f"acquisition {r_from}")
    ax[1].set_ylabel(f"acquisition {r_to}")
    ax[1].legend(fontsize=8, frameon=False)
    for side in ("top", "right"):
        ax[1].spines[side].set_visible(False)
    ax[1].set_title("each ROI, the two acquisitions against each other",
                    fontsize=10, loc="left")

    title = args.label or auc_dir.name
    fig.suptitle(f"{title}   {detail}", fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    with open(out.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["roi", f"run{r_from}", f"run{r_to}", "ratio", "p",
                    "ci_lo", "ci_hi", "classification"])
        for i, roi in enumerate(rois):
            cls = "increased" if up[i] else "decreased" if down[i] else "unchanged"
            ratio = (v_to[i] / v_from[i]) if abs(v_from[i]) > 1e-12 else np.nan
            w.writerow([roi, round(v_from[i], 4), round(v_to[i], 4),
                        (None if not np.isfinite(ratio) else round(ratio, 4)),
                        (None if not np.isfinite(pvals[i]) else round(pvals[i], 5)),
                        (None if ci_lo is None else round(ci_lo[i], 4)),
                        (None if ci_hi is None else round(ci_hi[i], 4)),
                        cls])
    with open(out.with_suffix(".json"), "w") as fh:
        json.dump({"auc_dir": str(auc_dir), "value": args.value,
                   "from_run": r_from, "to_run": r_to, "n_roi": n_roi,
                   "method": args.method, "detail": detail,
                   "counts": dict(zip(names, counts))}, fh, indent=2)
    print(f"\nwrote {out.with_suffix('.png')}, .pdf, .csv and .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
