#!/usr/bin/env python
"""Mean image of each acquisition, side by side on one intensity scale.

Built from Suite2p's registered binary, so the comparison is between images the
registration has already aligned; whatever difference remains is not lateral
motion.

The scale is shared across every panel and stated in the title. Scaling each
panel to its own range would make a preparation that dimmed by a tenth look
identical to one that did not, which is the whole question here.

Three quantities separate the two explanations for a recording that gets
fainter. Photobleaching removes signal without blurring it: mean intensity
falls, sharpness holds, and the image still matches the first acquisition.
Axial drift moves the plane away from the cells: intensity falls AND sharpness
falls, because what is left is the out-of-focus skirt of the cells, and the
correlation with the first acquisition decays. Two-dimensional registration
cannot correct axial movement, so this is the distinction that decides whether
a decline in fluorescence is something the analysis must account for.

Example
-------
    python fig_mean_per_run.py \
        --s2p-dir <dataset>/work/s2p_series/suite2p/plane0 \
        --ledger <dataset>/work/s2p_series/frame_ledger.csv \
        --dataset <dataset> --out <dataset>/work/mean_per_run
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


def load_binary(plane: Path, chan2=False):
    d: dict = {}
    for cand in ("reg_outputs.npy", "ops.npy", "db.npy", "detect_outputs.npy"):
        f = plane / cand
        if f.exists():
            d.update(np.load(f, allow_pickle=True).item())
    name = "data_chan2.bin" if chan2 else "data.bin"
    path = plane / name
    if not path.exists():
        key = "reg_file_chan2" if chan2 else "reg_file"
        if d.get(key) and Path(d[key]).exists():
            path = Path(d[key])
    if not path.exists():
        raise FileNotFoundError(
            f"no registered binary at {plane / name}; re-run Suite2p with "
            "delete_bin False")
    ly, lx = int(d["Ly"]), int(d["Lx"])
    n = path.stat().st_size // (ly * lx * 2)
    return np.memmap(path, dtype=np.int16, mode="r", shape=(n, ly, lx)), d


def sharpness(im):
    """Mean gradient magnitude, normalised by mean intensity.

    Dividing by the mean makes it independent of overall brightness, so a
    uniformly dimmer image scores the same and only genuine blurring moves it.
    """
    gy, gx = np.gradient(im.astype(np.float64))
    return float(np.mean(np.hypot(gx, gy)) / max(im.mean(), 1e-9))


def roi_edge(entry, shape, crop):
    (y0, y1), (x0, x1) = crop
    m = np.zeros(shape, bool)
    m[entry["ypix"], entry["xpix"]] = True
    inner = (np.roll(m, 1, 0) & np.roll(m, -1, 0)
             & np.roll(m, 1, 1) & np.roll(m, -1, 1))
    return (m & ~inner)[y0:y1, x0:x1]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="mean image per acquisition, one shared intensity scale",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--s2p-dir", type=Path, required=True)
    p.add_argument("--ledger", type=Path, default=None)
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True, help="output stem")
    p.add_argument("--clip", type=float, nargs=2, default=[0.5, 99.8],
                   metavar=("LO_PCT", "HI_PCT"))
    p.add_argument("--cmap", default="gray")
    p.add_argument("--diff-cmap", default="RdBu_r")
    p.add_argument("--rois", action="store_true", help="outline ROIs on each panel")
    p.add_argument("--all-roi", action="store_true")
    p.add_argument("--roi-color", default="#ffd200")
    p.add_argument("--zoom", type=int, nargs=4, default=None,
                   metavar=("X0", "X1", "Y0", "Y1"))
    p.add_argument("--no-diff", action="store_true")
    p.add_argument("--chan2", action="store_true")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args(argv)

    plane = args.s2p_dir.expanduser().resolve()
    mov, ops = load_binary(plane, args.chan2)
    n_t, ly, lx = mov.shape

    yr, xr = ops.get("yrange"), ops.get("xrange")
    crop = ((int(yr[0]), int(yr[1])) if yr is not None else (0, ly),
            (int(xr[0]), int(xr[1])) if xr is not None else (0, lx))
    (y0, y1), (x0, x1) = crop

    ledger_path = args.ledger or (plane.parent.parent / "frame_ledger.csv")
    if Path(ledger_path).exists():
        with open(ledger_path) as fh:
            rows = list(csv.DictReader(fh))
        segs = [(r["source_file"], int(r["frame_start"]), int(r["frame_end"]) + 1)
                for r in rows]
    else:
        print(f"no ledger at {ledger_path}", file=sys.stderr)
        return 2
    n_run = len(segs)
    print(f"{n_t} frames, {n_run} acquisition(s), valid region "
          f"y {y0}:{y1}  x {x0}:{x1}")

    # --- mean image per acquisition ----------------------------------------
    means, stats = [], []
    for k, (name, a, b) in enumerate(segs, start=1):
        m = np.asarray(mov[a:b, y0:y1, x0:x1], np.float64).mean(axis=0)
        means.append(m)
        stats.append({"run": k, "file": name, "frames": b - a,
                      "mean": float(m.mean()), "sharpness": sharpness(m)})
    ref = means[0]
    for s, m in zip(stats, means):
        s["corr_with_run1"] = float(np.corrcoef(ref.ravel(), m.ravel())[0, 1])
        s["mean_pct_of_run1"] = s["mean"] / max(stats[0]["mean"], 1e-9) * 100
        s["sharpness_pct_of_run1"] = (s["sharpness"]
                                      / max(stats[0]["sharpness"], 1e-9) * 100)

    print(f"\n{'run':>3} {'mean':>9} {'% of r1':>9} {'sharpness':>10} "
          f"{'% of r1':>9} {'corr w/ r1':>11}")
    for s in stats:
        print(f"{s['run']:>3} {s['mean']:9.1f} {s['mean_pct_of_run1']:8.1f}% "
              f"{s['sharpness']:10.4f} {s['sharpness_pct_of_run1']:8.1f}% "
              f"{s['corr_with_run1']:11.4f}")

    d_int = stats[-1]["mean_pct_of_run1"] - 100
    d_shp = stats[-1]["sharpness_pct_of_run1"] - 100
    d_cor = stats[-1]["corr_with_run1"]
    if d_int > -3:
        verdict = "the preparation held its brightness"
    elif d_shp > -5 and d_cor > 0.95:
        verdict = ("intensity fell while sharpness and the match to the first "
                   "acquisition held: consistent with photobleaching, not with "
                   "the plane moving away from the cells")
    elif d_shp < -10 or d_cor < 0.9:
        verdict = ("intensity and sharpness both fell and the image drifted "
                   "away from the first acquisition: consistent with axial "
                   "movement, which two-dimensional registration does not "
                   "correct")
    else:
        verdict = "intermediate; neither explanation is clean on its own"
    print(f"\nrun 1 -> run {n_run}: intensity {d_int:+.1f}%, sharpness "
          f"{d_shp:+.1f}%, correlation {d_cor:.4f}")
    print(f"-> {verdict}")

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

    stack = np.stack(means)
    lo, hi = np.percentile(stack, list(args.clip))
    print(f"\nshared intensity range [{lo:.1f}, {hi:.1f}] for all panels")

    edges = None
    if args.rois and (plane / "stat.npy").exists():
        stat = np.load(plane / "stat.npy", allow_pickle=True)
        ic = np.load(plane / "iscell.npy")
        if not args.all_roi:
            stat = stat[ic[:, 0].astype(bool)]
        edges = np.zeros(means[0].shape, bool)
        for s_ in stat:
            edges |= roi_edge(s_, (ly, lx), crop)

    n_row = 1 if args.no_diff else 2
    fig = plt.figure(figsize=(2.6 * n_run + 1.6, 2.9 * n_row + 2.4))
    gs = fig.add_gridspec(n_row + 1, n_run, height_ratios=[1] * n_row + [0.85],
                          hspace=0.32, wspace=0.06)

    for k, (m, s) in enumerate(zip(means, stats)):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(m, cmap=args.cmap, vmin=lo, vmax=hi,
                       interpolation="nearest")
        if edges is not None:
            ys, xs = np.nonzero(edges)
            ax.plot(xs, ys, ".", ms=0.7, color=args.roi_color, alpha=0.9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"run {s['run']}\n{s['mean_pct_of_run1']:.0f}% brightness, "
                     f"{s['sharpness_pct_of_run1']:.0f}% sharpness", fontsize=8)
        if args.zoom:
            zx0, zx1, zy0, zy1 = args.zoom
            ax.set_xlim(zx0 - x0, zx1 - x0)
            ax.set_ylim(zy1 - y0, zy0 - y0)
        if k == n_run - 1:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    if not args.no_diff:
        diffs = [m - ref for m in means]
        lim = float(np.percentile(np.abs(np.stack(diffs[1:])), 99.5)) or 1.0
        for k, (dm, s) in enumerate(zip(diffs, stats)):
            ax = fig.add_subplot(gs[1, k])
            im = ax.imshow(dm, cmap=args.diff_cmap, vmin=-lim, vmax=lim,
                           interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"run {s['run']} minus run 1", fontsize=8)
            if args.zoom:
                zx0, zx1, zy0, zy1 = args.zoom
                ax.set_xlim(zx0 - x0, zx1 - x0)
                ax.set_ylim(zy1 - y0, zy0 - y0)
            if k == n_run - 1:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    axm = fig.add_subplot(gs[n_row, :])
    runs = [s["run"] for s in stats]
    axm.plot(runs, [s["mean_pct_of_run1"] for s in stats], marker="o", lw=1.8,
             color="C3", label="mean intensity")
    axm.plot(runs, [s["sharpness_pct_of_run1"] for s in stats], marker="s",
             lw=1.8, color="C0", label="sharpness")
    axm.plot(runs, [s["corr_with_run1"] * 100 for s in stats], marker="^",
             lw=1.8, color="C2", label="correlation with run 1")
    axm.axhline(100, color="0.8", lw=0.8, zorder=0)
    axm.set_xticks(runs)
    axm.set_xlabel("acquisition")
    axm.set_ylabel("percent of run 1")
    axm.legend(fontsize=8, frameon=False, ncol=3)
    for side in ("top", "right"):
        axm.spines[side].set_visible(False)
    axm.set_title(verdict, fontsize=9, loc="left")

    ds_name = args.dataset.name if args.dataset else plane.parent.parent.name
    fig.suptitle(f"{ds_name}   mean image per acquisition, registered, "
                 f"one shared intensity scale [{lo:.0f}, {hi:.0f}]",
                 fontsize=10, x=0.01, ha="left")
    fig.subplots_adjust(left=0.03, right=0.94, top=0.90, bottom=0.07)

    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    with open(out.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(stats[0].keys()))
        w.writeheader()
        w.writerows(stats)
    with open(out.with_suffix(".json"), "w") as fh:
        json.dump({"stats": stats, "shared_range": [lo, hi],
                   "verdict": verdict}, fh, indent=2)
    print(f"wrote {out.with_suffix('.png')}, .pdf, .csv and .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
