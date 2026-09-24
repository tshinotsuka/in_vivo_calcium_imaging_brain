#!/usr/bin/env python
"""Compare populations detected independently in each acquisition.

The paired analysis follows the same cells across a recording, which requires
the registration to hold and the plane not to drift axially. When it does
drift, the same ROI is no longer the same cell and the pairing is a fiction.
This is the alternative: detect cells separately in each acquisition and
compare whatever was found, as two samples rather than as one sample measured
twice.

What that buys, and what it costs. It survives drift, because nothing has to
correspond between acquisitions. It gives up the pairing, so between-cell
variability is no longer removed and the comparison needs more cells to see
the same effect. And it acquires a bias the paired analysis does not have:
detection depends on how visible a cell is, so if the preparation dims, the
dim cells stop being detected and the survivors are the bright ones. Activity
would appear to rise for no reason other than which cells were counted.

So the detection count is reported as prominently as the activity, and the
brightness of the detected cells alongside it. If the count falls and the
median brightness of the survivors rises, the comparison is measuring
detectability and the figure says so.

Example
-------
    python fig_unpaired.py \
        --csv <...>/indep/baseline/eauc/auc_per_roi_per_run.csv \
              <...>/indep/wi_run-05/eauc/auc_per_roi_per_run.csv \
        --labels baseline "WI run 5" \
        --out <dataset>/work/fig_unpaired
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


def read_group(path: Path, value: str, brightness: str = "raw_F0"):
    """Per-ROI values from one acquisition's table."""
    vals, bright = [], []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                vals.append(float(r[value]))
            except (KeyError, TypeError, ValueError):
                continue
            try:
                bright.append(float(r[brightness]))
            except (KeyError, TypeError, ValueError):
                bright.append(np.nan)
    return np.array(vals, float), np.array(bright, float)


def permutation_p(x, y, n_perm=20000, seed=0):
    """Two-sided p for a difference in medians, without a distributional claim.

    A rank test would do, but the medians are what the figure shows, so the
    statistic tested is the one being looked at.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return float("nan")
    obs = abs(np.median(x) - np.median(y))
    pool = np.concatenate([x, y])
    nx = x.size
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        p = rng.permutation(pool)
        if abs(np.median(p[:nx]) - np.median(p[nx:])) >= obs - 1e-12:
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


def cliffs_delta(x, y):
    """How often a value from one group exceeds one from the other.

    Reported instead of a standardised mean difference because these
    distributions are skewed and often contain zeros, where a mean-based
    effect size misleads. +1 means every cell in the second group is higher.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    gt = sum((y[:, None] > x[None, :]).sum(axis=1))
    lt = sum((y[:, None] < x[None, :]).sum(axis=1))
    return float(gt - lt) / (x.size * y.size)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="unpaired comparison of independently detected populations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--csv", type=Path, nargs="+", required=True)
    p.add_argument("--labels", nargs="*", default=None)
    p.add_argument("--out", type=Path, required=True, help="output stem")
    p.add_argument("--value", default="auc_per_min_dff")
    p.add_argument("--brightness", default="raw_F0",
                   help="column used to check for survivorship among the "
                        "detected cells")
    p.add_argument("--ylabel", default=None)
    p.add_argument("--color", nargs="*", default=None)
    p.add_argument("--point-size", type=float, default=26.0)
    p.add_argument("--jitter", type=float, default=0.10)
    p.add_argument("--log", action="store_true")
    p.add_argument("--n-perm", type=int, default=20000)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--title", default=None)
    args = p.parse_args(argv)

    labels = args.labels or [c.parent.parent.name for c in args.csv]
    if len(labels) != len(args.csv):
        print("ERROR: --labels must match --csv", file=sys.stderr)
        return 2

    groups, brights = [], []
    for c in args.csv:
        cp = c.expanduser().resolve()
        if not cp.exists():
            print(f"ERROR: no such file {cp}", file=sys.stderr)
            return 2
        v, b = read_group(cp, args.value, args.brightness)
        groups.append(v)
        brights.append(b)

    ylab = args.ylabel or {
        "auc_per_min_dff": r"AUC (%$\Delta$F/F$\cdot$s / min)",
        "auc_per_min_df": r"$\Delta$F AUC (a.u.$\cdot$s / min)",
        "spike_rate_per_min": "inferred rate (a.u. / min)",
    }.get(args.value, args.value)

    print(f"{'group':18s} {'n ROI':>6} {'median':>10} {'IQR':>18} "
          f"{'zero':>6} {'median F0':>10}")
    rows = []
    for lab, v, b in zip(labels, groups, brights):
        q1, q3 = (np.percentile(v, [25, 75]) if v.size else (np.nan, np.nan))
        med = float(np.median(v)) if v.size else np.nan
        mb = float(np.nanmedian(b)) if np.isfinite(b).any() else np.nan
        n0 = int((v == 0).sum())
        rows.append({"group": lab, "n_roi": int(v.size),
                     "median": round(med, 4),
                     "q1": round(float(q1), 4), "q3": round(float(q3), 4),
                     "n_zero": n0,
                     "median_brightness": (None if not np.isfinite(mb)
                                           else round(mb, 2))})
        print(f"{lab:18s} {v.size:6d} {med:10.2f} "
              f"[{q1:7.2f},{q3:7.2f}] {n0:6d} {mb:10.1f}")

    # --- the comparison, and the caveat that goes with it -------------------
    comps = []
    if len(groups) >= 2:
        print()
        for j in range(1, len(groups)):
            a, b_ = groups[0], groups[j]
            pv = permutation_p(a, b_, args.n_perm)
            dl = cliffs_delta(a, b_)
            comps.append({"vs": labels[j], "p": round(float(pv), 5),
                          "cliffs_delta": round(float(dl), 4)})
            print(f"{labels[0]} vs {labels[j]}: median "
                  f"{np.median(a):.2f} -> {np.median(b_):.2f}, "
                  f"p = {pv:.4f} (permutation), Cliff's delta = {dl:+.2f}")

        n0, nj = groups[0].size, groups[-1].size
        b0 = np.nanmedian(brights[0])
        bj = np.nanmedian(brights[-1])
        if np.isfinite(b0) and np.isfinite(bj) and n0 > 0:
            d_n = (nj - n0) / n0 * 100
            d_b = (bj - b0) / max(abs(b0), 1e-9) * 100
            print(f"\ndetection: {n0} -> {nj} cells ({d_n:+.0f}%), "
                  f"median brightness of those cells {d_b:+.0f}%")
            if d_n < -15 and d_b > 10:
                print("  Fewer cells were detected and the ones that were are "
                      "brighter, so part of\n  any rise in activity is the dim "
                      "cells dropping out rather than the\n  remaining cells "
                      "doing more.")
            elif d_n > 15 and d_b < -10:
                print("  More cells were detected and they are dimmer, so part "
                      "of any fall in\n  activity is dim cells entering the "
                      "sample rather than the population\n  becoming quieter.")

    # --- figure --------------------------------------------------------------
    try:
        import figstyle_tshino as FS
        FS.set_style()
    except Exception:  # noqa: BLE001
        plt.rcParams.update({"font.family": "sans-serif",
                             "font.sans-serif": ["Arial", "Liberation Sans",
                                                 "DejaVu Sans"]})
    plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none",
                         "font.size": 9})

    cols = (list(args.color) if args.color
            else [plt.get_cmap("tab10")(i % 10) for i in range(len(groups))])
    fig, ax = plt.subplots(1, 2, figsize=(4.2 + 1.5 * len(groups), 4.6),
                           gridspec_kw={"width_ratios": [1.5, 1.0]})

    rng = np.random.default_rng(0)
    for i, (lab, v) in enumerate(zip(labels, groups)):
        x = i + 1 + rng.uniform(-args.jitter, args.jitter, v.size)
        ax[0].scatter(x, v, s=args.point_size, color=cols[i], alpha=0.75,
                      edgecolor="0.3", linewidth=0.4, zorder=3, clip_on=False)
        if v.size:
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            ax[0].plot([i + 0.72, i + 1.28], [med, med], color="0.15", lw=2.2,
                       zorder=4)
            ax[0].plot([i + 1, i + 1], [q1, q3], color="0.15", lw=1.0, zorder=2)
        ax[0].text(i + 1, ax[0].get_ylim()[1], f"n={v.size}", ha="center",
                   va="bottom", fontsize=8)
    ax[0].set_xticks(range(1, len(groups) + 1))
    ax[0].set_xticklabels(labels, fontsize=9)
    ax[0].set_ylabel(ylab)
    ax[0].set_xlim(0.5, len(groups) + 0.5)
    if args.log:
        ax[0].set_yscale("log")
    else:
        allv = np.concatenate(groups) if groups else np.array([0.0])
        lo = min(0.0, float(allv.min()))
        ax[0].set_ylim(lo - abs(lo) * 0.05 - 1e-9,
                       float(allv.max()) * 1.12 if allv.max() > 0 else 1.0)
    for side in ("top", "right"):
        ax[0].spines[side].set_visible(False)
    ax[0].set_title("each point is one cell, detected in that acquisition only",
                    fontsize=9, loc="left")

    # detection count and brightness, the two things that make this unpaired
    # comparison misreadable if they move
    xs = np.arange(1, len(groups) + 1)
    ax[1].bar(xs - 0.18, [g.size for g in groups], width=0.34, color="0.55",
              label="cells detected")
    ax[1].set_ylabel("cells detected")
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels(labels, fontsize=8)
    tw = ax[1].twinx()
    tw.plot(xs, [np.nanmedian(b) if np.isfinite(b).any() else np.nan
                 for b in brights], marker="o", color="C3", lw=1.8,
            label="median F0 of those cells")
    tw.set_ylabel("median F0 (a.u.)", color="C3")
    tw.tick_params(axis="y", labelcolor="C3")
    for side in ("top",):
        ax[1].spines[side].set_visible(False)
    ax[1].set_title("what was counted, and how bright it was",
                    fontsize=9, loc="left")

    ttl = args.title or "independent detection per acquisition"
    if comps:
        ttl += f"   {labels[0]} vs {labels[-1]}: p = {comps[-1]['p']:.3f}, " \
               f"delta = {comps[-1]['cliffs_delta']:+.2f}"
    fig.suptitle(ttl + "   (unpaired: the cells are not the same)",
                 fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    with open(out.with_suffix(".json"), "w") as fh:
        json.dump({"value": args.value, "groups": rows, "comparisons": comps,
                   "note": "populations detected independently in each "
                           "acquisition; unpaired, and subject to a "
                           "detectability bias when brightness changes."},
                  fh, indent=2)
    with open(out.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "roi", args.value, args.brightness])
        for lab, v, b in zip(labels, groups, brights):
            for i, (vv, bb) in enumerate(zip(v, b), start=1):
                w.writerow([lab, i, round(float(vv), 4),
                            (None if not np.isfinite(bb) else round(float(bb), 2))])
    print(f"\nwrote {out.with_suffix('.png')}, .pdf, .csv and .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
