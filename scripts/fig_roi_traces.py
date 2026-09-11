#!/usr/bin/env python
"""ROI map + per-ROI dF/F traces + treadmill, on one time axis.

Left: the mean image with the plotted ROIs outlined and numbered. Right: one
dF/F trace per ROI, stacked, with the treadmill signal on top. Colours and
numbers match across the two, so "this cell on the tissue" and "this time
course" are the same object.

Time alignment uses the rising edges of ``frame_clock`` in the Data Recorder
HDF5, not the recorder start: the acquisition head pad is non-zero, so anchoring
on the recorder start would shift every trace by an unknown offset. Edge count
is checked against the number of imaging frames and the script stops on a
mismatch rather than silently stretching one signal onto the other.

Outputs PNG (300 dpi) and PDF next to it. The PDF is vector with embedded fonts,
so panels can be composed in Illustrator without rasterising text.

Example
-------
    python fig_roi_traces.py \
        --s2p-dir <dataset>/work/s2p_func/suite2p/plane0 \
        --dataset <dataset> \
        --h5 <dataset>/raw/sub-sk28_ses-01_cond-qc_run-01_00002.h5 \
        --n-roi 6 \
        --out <dataset>/work/fig_roi_traces
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# traces
# ---------------------------------------------------------------------------


def rolling_baseline(F: np.ndarray, win: int, pct: float = 10.0) -> np.ndarray:
    """Percentile baseline in a sliding window, evaluated on a coarse grid."""
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


def robust_sd(x: np.ndarray, axis=-1) -> np.ndarray:
    """Noise scale from the median absolute frame-to-frame difference.

    Transients inflate a plain standard deviation, so a trace with strong
    activity would appear to have a large noise floor and its amplitude in SD
    units would be understated.
    """
    d = np.abs(np.diff(x, axis=axis))
    return np.median(d, axis=axis) / (np.sqrt(2) * 0.6745)


# ---------------------------------------------------------------------------
# recorder alignment
# ---------------------------------------------------------------------------


def frame_times_from_clock(sig: np.ndarray, fs: float) -> np.ndarray:
    """Recorder time (s) of each frame, from the rising edges of frame_clock."""
    lo, hi = float(np.min(sig)), float(np.max(sig))
    th = lo + 0.5 * (hi - lo)
    above = sig >= th
    idx = np.flatnonzero(~above[:-1] & above[1:]) + 1
    return idx / fs


def load_recorder(h5_path: Path, n_frames: int, clock: str, speed: str, direction: str | None):
    """Return (frame_times_s, speed_per_frame, info) or (None, None, info)."""
    info: dict = {"h5": str(h5_path)}
    try:
        import datarecorder_loader as DR
    except ImportError:
        info["error"] = ("datarecorder_loader not importable; install the acquisition "
                         "repo with `pip install -e ../microscope_control_dev`")
        return None, None, info

    d = DR.load_datarecorder(str(h5_path))
    fs = float(d["samplerate"])
    info["names"] = list(d["names"])
    info["samplerate"] = fs

    if clock not in d["signals"]:
        info["error"] = f"{clock!r} not in {info['names']}"
        return None, None, info

    ft = frame_times_from_clock(np.asarray(d["signals"][clock], float), fs)
    info["n_edges"] = int(ft.size)
    if ft.size != n_frames:
        info["error"] = (f"{ft.size} clock edges but {n_frames} imaging frames; "
                         "the recorder and the movie are not the same acquisition")
        return None, None, info

    if speed not in d["signals"]:
        info["error"] = f"{speed!r} not in {info['names']}"
        return ft, None, info

    sp = np.asarray(d["signals"][speed], float)
    # average within each inter-frame interval so the behaviour trace lives on
    # exactly the imaging time base
    edges = np.r_[ft, ft[-1] + (ft[-1] - ft[-2] if ft.size > 1 else 1.0 / fs)]
    idx = np.clip((edges * fs).astype(int), 0, sp.size)
    per_frame = np.array([sp[a:b].mean() if b > a else np.nan
                          for a, b in zip(idx[:-1], idx[1:])])
    info["speed_raw"] = {"mean": float(sp.mean()), "sd": float(sp.std()),
                         "min": float(sp.min()), "max": float(sp.max())}
    if direction and direction in d["signals"]:
        dr = np.asarray(d["signals"][direction], float)
        info["dir_raw"] = {"mean": float(dr.mean()), "sd": float(dr.std())}
    return ft, per_frame, info


def looks_inactive(stats: dict | None, span_v: float = 0.05) -> bool:
    """Whether the behaviour channel spans less than a plausible signal range."""
    if not stats:
        return True
    return (stats["max"] - stats["min"]) < span_v


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------


def roi_outline(stat_entry, shape):
    """Boolean mask edge pixels for one ROI, for a light outline."""
    m = np.zeros(shape, bool)
    m[stat_entry["ypix"], stat_entry["xpix"]] = True
    edge = m.copy()
    inner = (
        np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)
    )
    edge &= ~inner
    return m, edge


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="ROI map + traces + treadmill on one time axis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--s2p-dir", type=Path, required=True, help="suite2p/plane0 directory")
    p.add_argument("--dataset", type=Path, default=None, help="dataset dir, for raw/metadata.yaml")
    p.add_argument("--h5", type=Path, default=None, help="Data Recorder HDF5 for the same acquisition")
    p.add_argument("--fs", type=float, default=None, help="imaging frame rate; resolved from metadata when omitted")
    p.add_argument("--pixel-size-um", type=float, default=None)
    p.add_argument("--n-roi", type=int, default=6, help="number of ROIs to plot, chosen by peak/noise")
    p.add_argument("--all", dest="all_traces", action="store_true",
                   help="plot every ROI, split across pages of --per-page")
    p.add_argument("--per-page", type=int, default=10,
                   help="ROIs per page when plotting more than fit on one figure")
    p.add_argument("--order", choices=["snr", "index", "position"], default="snr",
                   help="page ordering: by peak/noise, by ROI index, or top-to-bottom in the field")
    p.add_argument("--roi-ids", type=int, nargs="+", default=None, help="explicit ROI indices (into the accepted set)")
    p.add_argument("--neucoeff", type=float, default=0.7)
    p.add_argument("--baseline-window-s", type=float, default=45.0)
    p.add_argument("--units", choices=["sd", "pct"], default="sd", help="trace scale: noise SD or percent dF/F")
    p.add_argument("--tmax-s", type=float, default=None, help="plot only the first N seconds")
    p.add_argument("--clock-name", default="frame_clock")
    p.add_argument("--speed-name", default="treadmill_speed")
    p.add_argument("--dir-name", default="treadmill_dir")
    p.add_argument("--all-roi", action="store_true", help="draw from every ROI, not only iscell-accepted")
    p.add_argument("--cmap", default="turbo", help="colormap for ROI identity")
    p.add_argument("--out", type=Path, required=True, help="output path stem (no extension)")
    args = p.parse_args(argv)

    s2p = args.s2p_dir.expanduser().resolve()
    F = np.load(s2p / "F.npy")
    Fneu = np.load(s2p / "Fneu.npy")
    stat = np.load(s2p / "stat.npy", allow_pickle=True)
    iscell = np.load(s2p / "iscell.npy")

    ops = {}
    for cand in ("reg_outputs.npy", "detect_outputs.npy", "ops.npy", "db.npy"):
        if (s2p / cand).exists():
            ops.update(np.load(s2p / cand, allow_pickle=True).item())

    keep = np.ones(F.shape[0], bool) if args.all_roi else iscell[:, 0].astype(bool)
    F, Fneu, stat = F[keep], Fneu[keep], stat[keep]
    n_roi, n_frames = F.shape
    if n_roi == 0:
        print("ERROR: no ROIs to plot", file=sys.stderr)
        return 2

    # --- acquisition parameters ---------------------------------------------
    fs, px = args.fs, args.pixel_size_um
    if (fs is None or px is None) and args.dataset:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        try:
            from run_roi_suite2p import resolve_from_metadata
            info = resolve_from_metadata(args.dataset.expanduser().resolve() / "raw" / "metadata.yaml")
            fs = fs if fs is not None else info.get("fs_hz")
            px = px if px is not None else info.get("pixel_size_um")
        except Exception as e:  # noqa: BLE001
            print(f"could not read metadata.yaml: {e}", file=sys.stderr)
    if fs is None:
        print("ERROR: frame rate unresolved; pass --fs or --dataset", file=sys.stderr)
        return 2
    print(f"ROIs {n_roi}   frames {n_frames}   {n_frames / fs / 60:.1f} min at {fs:.4g} Hz")

    # --- traces --------------------------------------------------------------
    Fc = (F - args.neucoeff * Fneu).astype(np.float32)
    F0 = rolling_baseline(Fc, int(round(args.baseline_window_s * fs)))
    dff = (Fc - F0) / np.maximum(F0, 1.0)
    sd = robust_sd(dff)
    peak = np.percentile(dff, 99.5, axis=1)
    snr = peak / np.maximum(sd, 1e-9)

    if args.roi_ids:
        sel = np.array([i for i in args.roi_ids if 0 <= i < n_roi], int)
        if sel.size != len(args.roi_ids):
            print("WARNING: some --roi-ids are out of range and were dropped", file=sys.stderr)
    elif args.all_traces:
        sel = np.arange(n_roi)
    else:
        sel = np.argsort(-snr)[: max(1, args.n_roi)]

    if args.order == "snr":
        sel = sel[np.argsort(-snr[sel])]
    elif args.order == "index":
        sel = np.sort(sel)
    else:  # position: top of the field first, so page order follows the map
        cy = np.array([float(np.mean(stat[i]["ypix"])) for i in sel])
        sel = sel[np.argsort(cy)]
    print(f"plotting {sel.size} ROI(s), ordered by {args.order}")

    t = np.arange(n_frames) / fs
    nshow = n_frames if args.tmax_s is None else min(n_frames, int(args.tmax_s * fs))

    # --- recorder ------------------------------------------------------------
    speed, rinfo = None, {}
    if args.h5:
        _, speed, rinfo = load_recorder(
            args.h5.expanduser().resolve(), n_frames,
            args.clock_name, args.speed_name, args.dir_name)
        if "error" in rinfo:
            print(f"recorder: {rinfo['error']}", file=sys.stderr)
        if rinfo.get("n_edges"):
            print(f"recorder: {rinfo['n_edges']} clock edges matched {n_frames} frames")

    sp_stats = rinfo.get("speed_raw")
    speed_dead = looks_inactive(sp_stats)
    if speed is not None and sp_stats:
        print(f"treadmill_speed raw: mean={sp_stats['mean']:.4g} sd={sp_stats['sd']:.4g} "
              f"range={sp_stats['min']:.4g}..{sp_stats['max']:.4g} V")
        if speed_dead:
            print("  the channel spans well under 0.05 V, which is an offset and noise\n"
                  "  rather than an encoder signal -- plotted, but labelled as such")

    # --- figure --------------------------------------------------------------
    try:
        import figstyle_tshino as FS
        FS.set_style()
    except Exception:  # noqa: BLE001
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        })
    plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none",
                         "font.size": 9, "axes.labelsize": 9})

    # A page holds a limited number of stacked traces before they become
    # unreadable, so long ROI lists are split. Every page draws the full field
    # with the other pages' ROIs greyed out, so no page loses spatial context.
    per_page = max(1, args.per_page)
    pages = [sel[i:i + per_page] for i in range(0, sel.size, per_page)]
    n_pages = len(pages)
    label_of = {int(i): k + 1 for k, i in enumerate(sel)}

    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    written = []

    for page_no, page_sel in enumerate(pages, start=1):
        n_tr = page_sel.size + (1 if speed is not None else 0)
        fig = plt.figure(figsize=(11.0, max(4.2, 0.62 * n_tr + 1.6)))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1.0, 2.0], wspace=0.10, figure=fig)
        axL = fig.add_subplot(gs[0, 0])
        gsR = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0, 1])
        axR = fig.add_subplot(gsR[0, 0])

        cmap = plt.get_cmap(args.cmap)
        colors = [cmap(v) for v in np.linspace(0.08, 0.92, page_sel.size)]

        # left: mean image + outlined ROIs
        img = ops.get("meanImg")
        if img is not None:
            vmin, vmax = np.percentile(img, [1, 99.5])
            axL.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
            shape = img.shape
        else:
            shape = (int(ops.get("Ly", 128)), int(ops.get("Lx", 128)))
            axL.set_facecolor("k")
        # the ROIs on other pages, faint, so each page still shows the whole field
        for i in sel:
            if i in set(page_sel.tolist()):
                continue
            _, edge = roi_outline(stat[i], shape)
            ys, xs = np.nonzero(edge)
            axL.plot(xs, ys, ".", ms=0.5, color="0.45", alpha=0.55)

        for k, i in enumerate(page_sel):
            _, edge = roi_outline(stat[i], shape)
            ys, xs = np.nonzero(edge)
            axL.plot(xs, ys, ".", ms=0.8, color=colors[k])
            cy, cx = float(np.mean(stat[i]["ypix"])), float(np.mean(stat[i]["xpix"]))
            axL.text(cx + 4, cy - 4, str(label_of[i]), color=colors[k], fontsize=9,
                     fontweight="bold", ha="left", va="bottom")
        if px:
            bar_um = 50.0
            bar_px = bar_um / px
            y0 = shape[0] * 0.94
            x0 = shape[1] * 0.06
            axL.plot([x0, x0 + bar_px], [y0, y0], "-", color="w", lw=2.5,
                     solid_capstyle="butt")
            axL.text(x0 + bar_px / 2, y0 - 3, f"{bar_um:g} um", color="w",
                     ha="center", va="bottom", fontsize=8)
        axL.set_xlim(0, shape[1])
        axL.set_ylim(shape[0], 0)
        axL.set_axis_off()

        # right: stacked traces
        if args.units == "sd":
            traces = dff[page_sel] / np.maximum(sd[page_sel][:, None], 1e-9)
            scale_lab, bar_val = "SD", 5.0
        else:
            traces = dff[page_sel] * 100.0
            scale_lab, bar_val = "% dF/F", 50.0

        span = float(np.percentile(traces[:, :nshow], 99.5))
        step = max(span * 1.15, bar_val * 1.2)
        ytick_pos, ytick_lab = [], []

        for k in range(page_sel.size):
            off = -step * k
            axR.plot(t[:nshow], traces[k, :nshow] + off, lw=0.6, color=colors[k])
            ytick_pos.append(off)
            ytick_lab.append(str(label_of[page_sel[k]]))

        if speed is not None:
            off = -step * page_sel.size
            s = np.asarray(speed[:nshow], float)
            rng = np.nanmax(s) - np.nanmin(s)
            s_n = (s - np.nanmin(s)) / (rng if rng > 0 else 1.0) * step * 0.8
            axR.plot(t[:nshow], s_n + off, lw=0.6, color="0.25")
            lab = "treadmill" + ("\n(no signal)" if speed_dead else "")
            ytick_pos.append(off)
            ytick_lab.append(lab)

        axR.set_yticks(ytick_pos)
        axR.set_yticklabels(ytick_lab)
        axR.tick_params(axis="y", length=0)
        axR.set_xlabel("time (s)")
        for side in ("top", "right", "left"):
            axR.spines[side].set_visible(False)

        # vertical scale bar for the traces
        xb = t[:nshow][-1] * 1.01
        axR.plot([xb, xb], [0, bar_val], "-", color="k", lw=1.6, clip_on=False)
        axR.text(xb * 1.004, bar_val / 2, f" {bar_val:g} {scale_lab}", va="center",
                 ha="left", fontsize=8, clip_on=False)
        axR.set_xlim(0, t[:nshow][-1])

        sub = args.dataset.name if args.dataset else s2p.parent.parent.name
        page_tag = f"   |   page {page_no}/{n_pages}" if n_pages > 1 else ""
        title = (f"{sub}   ROI {', '.join(str(label_of[i]) for i in page_sel)}"
                 f"  of {n_roi}   {n_frames / fs / 60:.1f} min at {fs:.4g} Hz"
                 f"   |   r = {args.neucoeff:g}, rolling F0 {args.baseline_window_s:g} s")
        if px:
            title += f"   |   {px:.3g} um/px"
        title += page_tag
        fig.suptitle(title, fontsize=9.5, x=0.01, ha="left")
        fig.subplots_adjust(left=0.02, right=0.93, top=0.90, bottom=0.10)

        stem = out if n_pages == 1 else out.with_name(f"{out.name}_p{page_no}")
        fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        written.append(stem)
        print(f"  page {page_no}/{n_pages}: ROI "
              f"{', '.join(str(label_of[int(i)]) for i in page_sel)} -> {stem.name}.png/.pdf")

    print(f"wrote {len(written)} page(s) under {out.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
