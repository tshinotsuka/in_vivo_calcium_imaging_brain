#!/usr/bin/env python
"""The inferred spike trains, one figure per acquisition.

A sparse non-negative train is better shown as a raster than as a stacked
trace: almost every frame is zero, so a line plot spends its height on the
baseline and leaves the events as hairlines. Marker size carries the inferred
amplitude, because the rate is a sum of those amplitudes and not a count of
frames, and a raster that ignored them would not match the number underneath
it.

Three things sit around the raster. The ROI map, coloured by that
acquisition's rate on a scale shared with every other acquisition, so a dim
panel means a quiet acquisition rather than a differently scaled one. The
summed train, which shows whether the population moved together or whether a
few cells carry the total. And the rate per ROI as points rather than bars,
which keeps the cells that inferred nothing visible at zero instead of hiding
them against the axis.

The values are in arbitrary units. Deconvolution recovers relative changes far
better than absolute rates, so these are for comparing acquisitions with each
other, not for stating a firing rate.

Example
-------
    python fig_spikes_per_run.py --s2p-dir <dataset>/work/spks_series/suite2p/plane0 \
        --ledger <dataset>/work/spks_series/frame_ledger.csv \
        --dataset <dataset> --all-roi --out <dataset>/work/spikes_per_run
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
        description="spike trains per acquisition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--s2p-dir", type=Path, required=True,
                   help="plane directory holding spks.npy")
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--ledger", type=Path, default=None)
    p.add_argument("--fs", type=float, default=None)
    p.add_argument("--pixel-size-um", type=float, default=None)
    p.add_argument("--all-roi", action="store_true")
    p.add_argument("--style", choices=["raster", "stacked"], default="raster")
    p.add_argument("--sort", choices=["index", "rate", "position"],
                   default="index",
                   help="row order in the raster; 'rate' uses the whole "
                        "recording so the rows mean the same thing in every "
                        "panel")
    p.add_argument("--bin-s", type=float, default=1.0,
                   help="bin width for the summed train")
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--cmap-vmax", type=float, default=None)
    p.add_argument("--color-tick", default="#1a1a1a")
    p.add_argument("--color-pop", default="#005aff")
    p.add_argument("--color-mean", default="#ff4b00")
    p.add_argument("--color-bg", default="gray")
    p.add_argument("--tick-scale", type=float, default=1.0,
                   help="multiplier on the raster marker size")
    p.add_argument("--point-size", type=float, default=26.0)
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args(argv)

    plane = args.s2p_dir.expanduser().resolve()
    spk_path = plane / "spks.npy"
    if not spk_path.exists():
        print(f"ERROR: no spks.npy in {plane}", file=sys.stderr)
        return 2
    S = np.load(spk_path).astype(np.float32)
    iscell = np.load(plane / "iscell.npy")
    stat = np.load(plane / "stat.npy", allow_pickle=True)
    ops = {}
    for cand in ("reg_outputs.npy", "detect_outputs.npy", "ops.npy", "db.npy"):
        f = plane / cand
        if f.exists():
            ops.update(np.load(f, allow_pickle=True).item())

    keep = (np.ones(S.shape[0], bool) if args.all_roi
            else iscell[:, 0].astype(bool))
    if keep.size != S.shape[0]:
        keep = np.ones(S.shape[0], bool)
    S = S[keep]
    stat = stat[keep] if len(stat) == keep.size else stat
    n_roi, n_t = S.shape

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
    else:
        segs = [("all", 0, n_t)]

    print(f"{n_roi} ROIs x {n_t} frames at {fs:.4g} Hz, "
          f"{len(segs)} acquisition(s)")
    print(f"non-zero frames: {float((S > 0).mean()) * 100:.2f}%")

    rate = np.zeros((n_roi, len(segs)))
    for k, (_, a, b) in enumerate(segs):
        rate[:, k] = S[:, a:b].sum(axis=1) / ((b - a) / fs / 60.0)

    if args.sort == "rate":
        order = np.argsort(-rate.mean(axis=1))
    elif args.sort == "position":
        order = np.argsort([float(np.mean(s["ypix"])) for s in stat])
    else:
        order = np.arange(n_roi)

    vmax = args.cmap_vmax or float(np.percentile(rate, 99)) or 1.0
    cmap = plt.get_cmap(args.cmap)

    try:
        import figstyle_tshino as FS
        FS.set_style()
    except Exception:  # noqa: BLE001
        plt.rcParams.update({"font.family": "sans-serif",
                             "font.sans-serif": ["Arial", "Liberation Sans",
                                                 "DejaVu Sans"]})
    plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none",
                         "font.size": 8})

    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    shape = (np.asarray(ops["meanImg"]).shape if "meanImg" in ops
             else (int(ops.get("Ly", 128)), int(ops.get("Lx", 128))))
    map_off = (0, 0)
    if ops.get("Ly") and shape != (int(ops["Ly"]), int(ops["Lx"])) \
            and ops.get("yrange") is not None:
        map_off = (int(ops["yrange"][0]), int(ops["xrange"][0]))

    amp_hi = float(np.percentile(S[S > 0], 99)) if (S > 0).any() else 1.0
    bin_f = max(int(round(args.bin_s * fs)), 1)

    print(f"\n{'run':>3} {'file':40s} {'rate':>10} {'s.e.m.':>8} {'silent':>7}")
    for k, (name, a, b) in enumerate(segs, start=1):
        vals = rate[:, k - 1]
        t = (np.arange(a, b) - a) / fs
        dur = (b - a) / fs

        fig = plt.figure(figsize=(15, max(0.30 * n_roi + 4.2, 7.0)))
        gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 2.5],
                              height_ratios=[max(n_roi * 0.30, 4.0), 1.1, 1.6],
                              wspace=0.14, hspace=0.30)
        axm = fig.add_subplot(gs[0, 0])
        axr = fig.add_subplot(gs[0, 1])
        axp = fig.add_subplot(gs[1, 1], sharex=axr)
        axb = fig.add_subplot(gs[2, :])

        # ROI map, coloured by this acquisition's rate
        if "meanImg" in ops:
            im = np.asarray(ops["meanImg"], float)
            lo, hi = np.percentile(im, [0.5, 99.8])
            axm.imshow(im, cmap=args.color_bg, vmin=lo, vmax=hi,
                       interpolation="nearest")
        for i in range(n_roi):
            c = cmap(min(max(vals[i], 0.0) / vmax, 1.0))
            ys, xs = np.nonzero(roi_edge(stat[i], shape, map_off))
            axm.plot(xs, ys, ".", ms=1.4, color=c)
            axm.text(float(np.mean(stat[i]["xpix"])) - map_off[1] + 3,
                     float(np.mean(stat[i]["ypix"])) - map_off[0] - 3,
                     str(i + 1), fontsize=6, fontweight="bold", color=c)
        if px:
            bar = 50.0 / px
            axm.plot([shape[1] * 0.06, shape[1] * 0.06 + bar],
                     [shape[0] * 0.94] * 2, "-", color="w", lw=2.5)
            axm.text(shape[1] * 0.06 + bar / 2, shape[0] * 0.94 - 3, "50 um",
                     color="w", ha="center", va="bottom", fontsize=7)
        axm.set_xlim(0, shape[1])
        axm.set_ylim(shape[0], 0)
        axm.axis("off")
        axm.set_title("ROIs coloured by inferred rate", fontsize=9)
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(vmin=0, vmax=vmax))
        cb = fig.colorbar(sm, ax=axm, fraction=0.045, pad=0.02)
        cb.set_label("rate (a.u. / min)", fontsize=7)
        cb.ax.tick_params(labelsize=6)

        # the trains
        if args.style == "raster":
            for row, i in enumerate(order):
                idx = np.flatnonzero(S[i, a:b])
                if idx.size == 0:
                    continue
                amp = S[i, a:b][idx]
                axr.scatter(t[idx], np.full(idx.size, row),
                            s=np.clip(amp / amp_hi, 0.08, 1.4) * 9
                            * args.tick_scale,
                            marker="|", linewidths=0.8, color=args.color_tick)
            axr.set_ylim(n_roi - 0.5, -0.5)
            axr.set_yticks(np.arange(n_roi))
            axr.set_yticklabels([str(order[r] + 1) for r in range(n_roi)],
                                fontsize=5.5)
            axr.tick_params(axis="y", length=0)
            axr.set_ylabel("ROI")
            axr.set_title("each tick is one inferred event; its size is the "
                          "inferred amplitude", fontsize=9, loc="left")
        else:
            step = max(float(np.percentile(S[:, a:b], 99.9)), 1e-6) * 1.4
            for row, i in enumerate(order):
                axr.plot(t, S[i, a:b] - step * row, lw=0.4,
                         color=args.color_tick)
            axr.set_yticks([-step * r for r in range(n_roi)])
            axr.set_yticklabels([str(order[r] + 1) for r in range(n_roi)],
                                fontsize=5.5)
            axr.tick_params(axis="y", length=0)
            axr.set_ylabel("ROI")
        for side in ("top", "right"):
            axr.spines[side].set_visible(False)
        axr.set_xlim(0, dur)
        axr.tick_params(labelbottom=False)

        # the summed train
        edges = np.arange(0, (b - a) + bin_f, bin_f)
        pop = np.array([S[:, a + e0:a + min(e0 + bin_f, b - a)].sum()
                        for e0 in edges[:-1]]) / (bin_f / fs) / n_roi
        axp.step(edges[:-1] / fs, pop, where="post", lw=0.8,
                 color=args.color_pop)
        axp.set_ylabel(f"summed\n(a.u./s, {args.bin_s:g} s bins)", fontsize=7)
        axp.set_xlabel("time within acquisition (s)")
        axp.set_xlim(0, dur)
        for side in ("top", "right"):
            axp.spines[side].set_visible(False)

        # rate per ROI
        xs_ = np.arange(1, n_roi + 1)
        axb.scatter(xs_, vals, s=args.point_size, zorder=3,
                    color=[cmap(min(max(v, 0.0) / vmax, 1.0)) for v in vals],
                    edgecolor="0.25", linewidth=0.4, clip_on=False)
        mu = float(vals.mean())
        sem = float(vals.std(ddof=1) / np.sqrt(n_roi)) if n_roi > 1 else 0.0
        axb.axhline(mu, color=args.color_mean, lw=1.4, zorder=2,
                    label=f"mean {mu:.1f} $\\pm$ {sem:.1f} (s.e.m.)")
        axb.axhspan(mu - sem, mu + sem, color=args.color_mean, alpha=0.18,
                    lw=0, zorder=1)
        axb.set_xlabel("ROI")
        axb.set_ylabel("rate (a.u. / min)")
        axb.set_xticks(xs_)
        axb.tick_params(axis="x", labelsize=6)
        axb.set_xlim(0.3, n_roi + 0.7)
        axb.set_ylim(bottom=min(0.0, float(vals.min()) * 1.1))
        for side in ("top", "right"):
            axb.spines[side].set_visible(False)
        axb.legend(fontsize=7, frameon=False)

        n_silent = int((vals == 0).sum())
        n_ev = int((S[:, a:b] > 0).sum())
        fig.suptitle(f"{name}   frames [{a} .. {b - 1}]   {dur / 60:.1f} min   "
                     f"{n_roi} ROIs, {n_ev} inferred events, {n_silent} silent  "
                     f" mean rate {mu:.1f} a.u./min   (arbitrary units)",
                     fontsize=9.5, x=0.01, ha="left")
        fig.subplots_adjust(left=0.04, right=0.97, top=0.94, bottom=0.06)
        stem = out / f"spikes_run{k:02d}"
        fig.savefig(stem.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"{k:>3} {name[:40]:40s} {mu:10.2f} {sem:8.2f} {n_silent:7d}"
              f"  -> {stem.name}.png/.pdf")

    # --- across acquisitions -------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6),
                           gridspec_kw={"width_ratios": [1.2, 1.0]})
    runs = np.arange(1, len(segs) + 1)
    rng_j = np.random.default_rng(0)
    jit = rng_j.uniform(-0.12, 0.12, n_roi)
    for i in range(n_roi):
        ax[0].plot(runs + jit[i], rate[i], lw=0.5, color="0.80", zorder=1)
        ax[0].scatter(runs + jit[i], rate[i], s=12, color="0.45", zorder=2,
                      linewidth=0)
    m = rate.mean(axis=0)
    se = (rate.std(axis=0, ddof=1) / np.sqrt(n_roi) if n_roi > 1
          else np.zeros(len(segs)))
    ax[0].errorbar(runs, m, yerr=se, color=args.color_mean, lw=2.2, marker="o",
                   ms=6, capsize=4, zorder=3, elinewidth=1.6,
                   label=f"mean $\\pm$ s.e.m. (n = {n_roi} ROIs)")
    ax[0].set_xlabel("acquisition")
    ax[0].set_ylabel("inferred rate (a.u. / min)")
    ax[0].set_xticks(runs)
    ax[0].set_xlim(0.5, len(segs) + 0.5)
    ax[0].set_ylim(bottom=min(0.0, float(rate.min()) * 1.1))
    ax[0].legend(fontsize=8, frameon=False)
    for side in ("top", "right"):
        ax[0].spines[side].set_visible(False)
    ax[0].set_title("rate per ROI", fontsize=10, loc="left")

    im = ax[1].imshow(rate[order], aspect="auto", cmap=args.cmap, vmin=0,
                      vmax=vmax, interpolation="nearest")
    ax[1].set_xticks(np.arange(len(segs)))
    ax[1].set_xticklabels(runs)
    ax[1].set_yticks(np.arange(n_roi))
    ax[1].set_yticklabels([str(i + 1) for i in order], fontsize=5.5)
    ax[1].set_xlabel("acquisition")
    ax[1].set_ylabel(f"ROI (by {args.sort})")
    cb = fig.colorbar(im, ax=ax[1], fraction=0.046)
    cb.set_label("rate (a.u. / min)", fontsize=8)
    ax[1].set_title("every ROI, every acquisition", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out / "spikes_summary.png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out / "spikes_summary.pdf", bbox_inches="tight")
    plt.close(fig)

    with open(out / "spikes_per_run.json", "w") as fh:
        json.dump({"s2p_dir": str(plane), "n_roi": n_roi, "n_frames": n_t,
                   "fs_hz": fs, "style": args.style, "sort": args.sort,
                   "cmap_vmax_used": round(float(vmax), 4),
                   "rate_mean_by_run": [round(float(x), 4) for x in m],
                   "rate_sem_by_run": [round(float(x), 4) for x in se],
                   "note": "inferred rates in arbitrary units; deconvolution "
                           "recovers relative change far better than absolute "
                           "rate."}, fh, indent=2)
    print(f"\nwrote {len(segs)} per-acquisition figures and a summary to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
