#!/usr/bin/env python
"""Show what each detection run found, on the images the detector actually saw.

One row per run: the mean image, the ROI outlines on top of it, and the count.
The top row shows the three background images Suite2p computes, so the cells
that are visibly present can be compared against the cells that were detected.

`meanImg` is what anatomical detection segments and is activity-independent.
`max_proj` shows active somata far more clearly but weights detection by
activity. `Vcorr` is the correlation map, where a cell appears wherever pixels
covary in time, so it reveals cells that are dim in the mean image but active.
A cell obvious in Vcorr yet missing from the ROI set is a detection failure, not
an expression failure -- the distinction that decides whether to adjust
parameters or to change the acquisition.

Example
-------
    python fig_roi_compare.py --dataset <dataset> \
        --runs s2p_series s2p_cp-1 s2p_cp-2 s2p_cp-4 \
        --out <dataset>/work/roi_compare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_run(plane: Path) -> dict:
    out: dict = {"plane": plane}
    for name in ("stat", "iscell", "F"):
        f = plane / f"{name}.npy"
        if f.exists():
            out[name] = np.load(f, allow_pickle=True)
    for cand in ("reg_outputs.npy", "detect_outputs.npy", "ops.npy", "db.npy"):
        f = plane / cand
        if f.exists():
            d = np.load(f, allow_pickle=True).item()
            for k in ("meanImg", "meanImgE", "max_proj", "Vcorr", "refImg",
                      "Ly", "Lx", "xrange", "yrange"):
                if k in d and k not in out:
                    out[k] = d[k]
    return out


def roi_edge(entry, shape):
    m = np.zeros(shape, bool)
    m[entry["ypix"], entry["xpix"]] = True
    inner = (np.roll(m, 1, 0) & np.roll(m, -1, 0)
             & np.roll(m, 1, 1) & np.roll(m, -1, 1))
    return m & ~inner


def show(ax, img, clip, title, cmap="gray"):
    if img is None:
        ax.text(0.5, 0.5, "not available", ha="center", va="center",
                transform=ax.transAxes, color="0.5", fontsize=9)
    else:
        img = np.asarray(img, float)
        vmin, vmax = np.percentile(img, list(clip))
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="compare detection runs on the detector's own images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--runs", nargs="+", required=True,
                   help="folder names under <dataset>/work/ (or full paths)")
    p.add_argument("--out", type=Path, required=True, help="output stem, no extension")
    p.add_argument("--clip", type=float, nargs=2, default=[0.5, 99.8],
                   metavar=("LO", "HI"))
    p.add_argument("--all-roi", action="store_true",
                   help="outline every detected ROI, not only accepted ones")
    p.add_argument("--zoom", type=int, nargs=4, default=None,
                   metavar=("X0", "X1", "Y0", "Y1"),
                   help="crop to this pixel box, to inspect a region closely")
    args = p.parse_args(argv)

    ds = args.dataset.expanduser().resolve()
    runs = []
    for r in args.runs:
        base = Path(r) if Path(r).is_absolute() else ds / "work" / r
        plane = base if base.name == "plane0" else base / "suite2p" / "plane0"
        if not (plane / "stat.npy").exists():
            print(f"skip {r}: no stat.npy under {plane}", file=sys.stderr)
            continue
        runs.append((r, load_run(plane)))
    if not runs:
        print("ERROR: no usable runs", file=sys.stderr)
        return 2

    ref = runs[0][1]
    shape = None
    for k in ("meanImg", "max_proj", "Vcorr"):
        if ref.get(k) is not None:
            shape = np.asarray(ref[k]).shape
            break
    if shape is None:
        shape = (int(ref.get("Ly", 128)), int(ref.get("Lx", 128)))

    plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none",
                         "font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"]})

    ncol = max(3, len(runs))
    fig, axes = plt.subplots(2, ncol, figsize=(3.4 * ncol, 7.4), squeeze=False)

    # top row: the images the detector had to work with
    show(axes[0][0], ref.get("meanImg"), args.clip,
         "meanImg  (activity-independent)")
    show(axes[0][1], ref.get("max_proj"), args.clip,
         "max_proj  (active somata)", cmap="magma")
    show(axes[0][2], ref.get("Vcorr"), (1, 99.5),
         "Vcorr  (correlated pixels)", cmap="viridis")
    for j in range(3, ncol):
        axes[0][j].axis("off")

    # bottom row: one panel per run
    summary = []
    for j, (name, r) in enumerate(runs):
        ax = axes[1][j]
        show(ax, r.get("meanImg"), args.clip, "")
        stat = r.get("stat")
        iscell = r.get("iscell")
        if stat is None:
            continue
        keep = (np.ones(len(stat), bool) if args.all_roi or iscell is None
                else iscell[:, 0].astype(bool))
        sel = np.flatnonzero(keep)
        cmap = plt.get_cmap("turbo")
        cols = [cmap(v) for v in np.linspace(0.08, 0.92, max(sel.size, 1))]
        diam = []
        for k, i in enumerate(sel):
            ys, xs = np.nonzero(roi_edge(stat[i], shape))
            ax.plot(xs, ys, ".", ms=1.4, color=cols[k], alpha=0.95)
            cy, cx = float(np.mean(stat[i]["ypix"])), float(np.mean(stat[i]["xpix"]))
            ax.text(cx + 3, cy - 3, str(k + 1), color=cols[k], fontsize=6,
                    fontweight="bold")
            diam.append(2 * np.sqrt(len(stat[i]["xpix"]) / np.pi))
        med = float(np.median(diam)) if diam else float("nan")
        ax.set_title(f"{name}\n{sel.size} shown / {len(stat)} detected"
                     f"   median {med:.1f} px", fontsize=9)
        ax.set_xlim(0, shape[1])
        ax.set_ylim(shape[0], 0)
        summary.append({"run": name, "n_detected": int(len(stat)),
                        "n_shown": int(sel.size),
                        "median_diameter_px": None if not diam else round(med, 2)})
    for j in range(len(runs), ncol):
        axes[1][j].axis("off")

    if args.zoom:
        x0, x1, y0, y1 = args.zoom
        for row in axes:
            for ax in row:
                if ax.images:
                    ax.set_xlim(x0, x1)
                    ax.set_ylim(y1, y0)

    fig.suptitle(f"{ds.name}   detection comparison"
                 + ("   (all detected ROIs)" if args.all_roi else "   (accepted ROIs)"),
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"{'run':24s} {'detected':>9} {'shown':>7} {'median px':>10}")
    for s in summary:
        print(f"{s['run']:24s} {s['n_detected']:>9} {s['n_shown']:>7} "
              f"{s['median_diameter_px']!s:>10}")
    with open(out.with_suffix(".json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {out.with_suffix('.png')} and .pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
