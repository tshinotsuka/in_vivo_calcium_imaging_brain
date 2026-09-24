#!/usr/bin/env python
"""Representative traces for chosen ROIs, with the map that says which cells they are.

A representative trace is an argument, not an illustration: it claims that
these cells behaved the way the summary says the population did. So the figure
carries what is needed to check that claim. The ROI map marks which cells were
picked, in the same colours as their traces. The acquisition boundaries are
drawn on the time axis, because a change that coincides with a boundary is a
different kind of observation from one that does not. And each trace is
labelled with its own value of the summary statistic, so a reader can see
whether the cells shown sit near the middle of the distribution or at its edge.

Choosing the ROIs by hand is the point; --roi takes whichever ones the argument
is about. When no ROIs are given the script picks the ones closest to the
median change, which is the least misleading automatic choice, and says so in
the title rather than leaving it implied.

Example
-------
    python fig_representative.py --s2p-dir <plane> --dataset <dataset> \
        --ledger <...>/frame_ledger.csv --roi 4 11 17 --neucoeff 0 \
        --auc-csv <...>/auc_per_roi_per_run.csv \
        --out <dataset>/work/representative
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_Z = {1.0: -2.3263, 5.0: -1.6449, 10.0: -1.2816, 20.0: -0.8416,
      25.0: -0.6745, 50.0: 0.0}


def _z_of(pct):
    try:
        from scipy.stats import norm
        return float(norm.ppf(pct / 100.0))
    except ImportError:
        return _Z[min(_Z, key=lambda x: abs(x - pct))]


def robust_sd(F):
    d = np.abs(np.diff(F, axis=1))
    return np.median(d, axis=1) / (np.sqrt(2) * 0.6745)


def rolling_baseline(F, win, pct=10.0, correct_bias=True):
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
    if correct_bias:
        out = out + (abs(_z_of(pct)) * robust_sd(F))[:, None]
    return out


def roi_edge(entry, shape, off=(0, 0)):
    m = np.zeros(shape, bool)
    yy = np.asarray(entry["ypix"]) - off[0]
    xx = np.asarray(entry["xpix"]) - off[1]
    k = (yy >= 0) & (yy < shape[0]) & (xx >= 0) & (xx < shape[1])
    m[yy[k], xx[k]] = True
    inner = (np.roll(m, 1, 0) & np.roll(m, -1, 0)
             & np.roll(m, 1, 1) & np.roll(m, -1, 1))
    return m & ~inner


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="representative traces for chosen ROIs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--s2p-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="output stem")
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--ledger", type=Path, default=None)
    p.add_argument("--fs", type=float, default=None)
    p.add_argument("--pixel-size-um", type=float, default=None)
    p.add_argument("--roi", type=int, nargs="*", default=None,
                   help="ROI numbers to show, as numbered in the analysis CSVs")
    p.add_argument("--n-auto", type=int, default=4,
                   help="how many to pick when --roi is not given")
    p.add_argument("--auc-csv", type=Path, default=None,
                   help="auc_per_roi_per_run.csv, to label each trace with its "
                        "own summary value and to pick typical ROIs")
    p.add_argument("--value", default="auc_per_min_dff")
    p.add_argument("--neucoeff", type=float, default=0.0)
    p.add_argument("--baseline-window-s", type=float, default=45.0)
    p.add_argument("--baseline-percentile", type=float, default=10.0)
    p.add_argument("--all-roi", action="store_true",
                   help="ROI numbers refer to all detected ROIs rather than "
                        "the accepted ones; match whatever the analysis used")
    p.add_argument("--frames", type=int, nargs=2, default=None,
                   metavar=("A", "B"))
    p.add_argument("--events-csv", type=Path, default=None,
                   help="events.csv, to shade the intervals that were counted")
    p.add_argument("--scale-bar-pct", type=float, default=50.0)
    p.add_argument("--scale-bar-s", type=float, default=60.0)
    p.add_argument("--cmap", default="turbo")
    p.add_argument("--colors", nargs="*", default=None,
                   help="explicit colours, one per ROI")
    p.add_argument("--trace-lw", type=float, default=0.6)
    p.add_argument("--event-color", default="#ff4b00")
    p.add_argument("--event-alpha", type=float, default=0.45)
    p.add_argument("--map-width", type=float, default=0.28,
                   help="fraction of the figure width taken by the ROI map")
    p.add_argument("--no-map", action="store_true")
    p.add_argument("--height-per-roi", type=float, default=1.15)
    p.add_argument("--width", type=float, default=13.0)
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args(argv)

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

    ledger = args.ledger or (plane.parent.parent / "frame_ledger.csv")
    segs = []
    if Path(ledger).exists():
        with open(ledger) as fh:
            for r in csv.DictReader(fh):
                segs.append((r["source_file"], int(r["frame_start"]),
                             int(r["frame_end"]) + 1))
    if not segs:
        segs = [("all", 0, n_t)]

    # --- summary values, for labels and for the automatic pick --------------
    vals = {}
    if args.auc_csv and Path(args.auc_csv).exists():
        with open(args.auc_csv) as fh:
            for r in csv.DictReader(fh):
                try:
                    vals[(int(r["roi"]), int(r["run"]))] = float(r[args.value])
                except (KeyError, TypeError, ValueError):
                    pass

    if args.roi:
        sel = np.array([r - 1 for r in args.roi], int)
        bad = [r for r in args.roi if not 1 <= r <= n_roi]
        if bad:
            print(f"ERROR: ROI {bad} outside 1..{n_roi}", file=sys.stderr)
            return 2
        how = f"{len(sel)} ROI(s) chosen by hand"
    else:
        sel = np.array([], int)
        how = ""
        if vals and len(segs) > 1:
            first, last = 1, len(segs)
            ratio = np.array([
                (vals.get((i + 1, last), np.nan)
                 / vals.get((i + 1, first), np.nan))
                if vals.get((i + 1, first), 0) else np.nan
                for i in range(n_roi)])
            ok = np.isfinite(ratio)
            if ok.any():
                med = float(np.nanmedian(ratio[ok]))
                order = np.argsort(np.abs(ratio - med))
                sel = np.array([i for i in order if ok[i]][:args.n_auto], int)
                how = (f"{sel.size} ROI(s) nearest the median change "
                       f"(x{med:.2f} from acquisition {first} to {last})")
        if sel.size == 0 and vals:
            # No ROI has a ratio to be typical of, which happens when the first
            # acquisition detected nothing: every denominator is zero. Fall
            # back to the most active cells over the whole recording and say
            # that is what these are, rather than implying they are typical.
            total = np.array([sum(v for (i2, _), v in vals.items() if i2 == i + 1)
                              for i in range(n_roi)])
            sel = np.argsort(-total)[:args.n_auto]
            sel = np.array([i for i in sel if total[i] > 0], int)
            how = (f"{sel.size} most active ROI(s); no ROI had a ratio to be "
                   "typical of, because the first acquisition detected nothing")
        if sel.size == 0:
            sel = np.arange(min(args.n_auto, n_roi))
            how = f"first {sel.size} ROI(s); no summary given to choose from"
    print(how)

    a, b = args.frames if args.frames else (0, n_t)
    a, b = max(0, a), min(n_t, b)

    Fc = F - args.neucoeff * Fneu
    F0 = np.empty_like(Fc)
    win = int(round(args.baseline_window_s * fs))
    for _, s0, s1 in segs:
        F0[:, s0:s1] = rolling_baseline(Fc[:, s0:s1], win,
                                        args.baseline_percentile)
    d = (Fc - F0) / np.maximum(F0, 1.0) * 100.0

    ev_by_roi = {}
    if args.events_csv and Path(args.events_csv).exists():
        with open(args.events_csv) as fh:
            for r in csv.DictReader(fh):
                ev_by_roi.setdefault(int(r["roi"]), []).append(
                    (int(r["onset_frame"]), int(r["offset_frame"])))

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

    if args.colors:
        colors = list(args.colors)[:sel.size]
        colors += [plt.get_cmap(args.cmap)(v)
                   for v in np.linspace(0.1, 0.9, max(sel.size - len(colors), 1))]
        colors = colors[:sel.size]
    else:
        colors = [plt.get_cmap(args.cmap)(v)
                  for v in np.linspace(0.08, 0.92, max(sel.size, 1))]

    h = max(args.height_per_roi * sel.size + 1.4, 3.2)
    fig = plt.figure(figsize=(args.width, h))
    if args.no_map or "meanImg" not in ops:
        gs = fig.add_gridspec(1, 1)
        ax_tr = fig.add_subplot(gs[0, 0])
        ax_map = None
    else:
        gs = fig.add_gridspec(1, 2,
                              width_ratios=[args.map_width, 1 - args.map_width],
                              wspace=0.04)
        ax_map = fig.add_subplot(gs[0, 0])
        ax_tr = fig.add_subplot(gs[0, 1])

    if ax_map is not None:
        img = np.asarray(ops["meanImg"], float)
        lo, hi = np.percentile(img, [0.5, 99.8])
        ax_map.imshow(img, cmap="gray", vmin=lo, vmax=hi,
                      interpolation="nearest")
        shape = img.shape
        off = (0, 0)
        if ops.get("Ly") and shape != (int(ops["Ly"]), int(ops["Lx"])) \
                and ops.get("yrange") is not None:
            off = (int(ops["yrange"][0]), int(ops["xrange"][0]))
        for s_ in stat:
            ys, xs = np.nonzero(roi_edge(s_, shape, off))
            ax_map.plot(xs, ys, ".", ms=0.5, color="0.55", alpha=0.7)
        for k, i in enumerate(sel):
            ys, xs = np.nonzero(roi_edge(stat[i], shape, off))
            ax_map.plot(xs, ys, ".", ms=1.6, color=colors[k])
            ax_map.text(float(np.mean(stat[i]["xpix"])) - off[1] + 4,
                        float(np.mean(stat[i]["ypix"])) - off[0] - 4,
                        str(i + 1), color=colors[k], fontsize=8,
                        fontweight="bold")
        if px:
            bar = 50.0 / px
            y0 = shape[0] * 0.95
            x0 = shape[1] * 0.05
            ax_map.plot([x0, x0 + bar], [y0, y0], "-", color="w", lw=2.5,
                        solid_capstyle="butt")
            ax_map.text(x0 + bar / 2, y0 - 3, "50 um", color="w", ha="center",
                        va="bottom", fontsize=7)
        ax_map.set_xlim(0, shape[1])
        ax_map.set_ylim(shape[0], 0)
        ax_map.axis("off")
        ax_map.set_title("chosen ROIs in colour, the rest in grey",
                         fontsize=8, loc="left")

    t = (np.arange(a, b) - a) / fs
    span = float(np.percentile(d[sel, a:b], 99.5))
    step = max(span * 1.25, args.scale_bar_pct * 1.4)
    for k, i in enumerate(sel):
        off_y = -step * k
        ax_tr.plot(t, d[i, a:b] + off_y, lw=args.trace_lw, color=colors[k])
        for on, of in ev_by_roi.get(i + 1, []):
            if a <= on and of <= b:
                sl = slice(on - a, of - a)
                ax_tr.fill_between(t[sl], off_y, d[i, a:b][sl] + off_y,
                                   color=args.event_color,
                                   alpha=args.event_alpha, lw=0)
        lab = f"ROI {i + 1}"
        if vals:
            vv = [vals.get((i + 1, k2 + 1)) for k2 in range(len(segs))]
            vv = [x for x in vv if x is not None]
            if len(vv) >= 2 and not np.allclose(vv, 0):
                lab += f"   {vv[0]:.0f} to {vv[-1]:.0f}"
            elif vv and not np.allclose(vv, 0):
                lab += f"   {vv[0]:.0f}"
        ax_tr.text(0, off_y + step * 0.42, lab, fontsize=8, color=colors[k],
                   fontweight="bold", va="bottom")

    for _, s0, _ in segs[1:]:
        if a <= s0 < b:
            ax_tr.axvline((s0 - a) / fs, color="0.75", ls=":", lw=0.9, zorder=0)
    for k, (name, s0, s1) in enumerate(segs, start=1):
        if s1 <= a or s0 >= b:
            continue
        mid = ((max(s0, a) + min(s1, b)) / 2 - a) / fs
        ax_tr.text(mid, step * 0.6, str(k), fontsize=8, color="0.45",
                   ha="center")

    ax_tr.set_xlim(0, t[-1])
    ax_tr.set_ylim(-step * (sel.size - 1) - step * 0.7, step * 0.95)
    ax_tr.set_yticks([])
    for side in ("top", "right", "left"):
        ax_tr.spines[side].set_visible(False)
    ax_tr.set_xlabel("time (s)")

    # scale bars instead of axes, so the traces are read as shapes
    xb = t[-1] * 0.995
    yb = -step * (sel.size - 1) - step * 0.5
    ax_tr.plot([xb, xb], [yb, yb + args.scale_bar_pct], "-", color="k", lw=1.8,
               clip_on=False)
    ax_tr.text(xb - t[-1] * 0.004, yb + args.scale_bar_pct / 2,
               f"{args.scale_bar_pct:g}% ", ha="right", va="center", fontsize=8)
    ax_tr.plot([xb - args.scale_bar_s, xb], [yb, yb], "-", color="k", lw=1.8,
               clip_on=False)
    ax_tr.text(xb - args.scale_bar_s / 2, yb - step * 0.12,
               f"{args.scale_bar_s:g} s", ha="center", va="top", fontsize=8)

    ds_name = args.dataset.name if args.dataset else plane.parent.parent.name
    sub = f"{ds_name}   {how}   r = {args.neucoeff:g}"
    if ev_by_roi:
        sub += "   shaded = counted toward the AUC"
    fig.suptitle(sub, fontsize=9, x=0.01, ha="left")
    fig.subplots_adjust(left=0.02, right=0.97, top=0.90, bottom=0.12)

    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.with_suffix('.png')} and .pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
