#!/usr/bin/env python
"""Write the registered movie out as an AVI, for viewing and for talks.

Reads Suite2p's registered binary (data.bin) rather than the raw TIFFs, so what
the video shows is what the analysis actually saw: registered, cropped to the
valid region, and de-interleaved to the functional channel.

Two display choices matter and are therefore explicit rather than automatic.
The intensity range is fixed from a sample of frames spanning the whole
recording and held constant, so that a dimming preparation looks like a dimming
preparation instead of being silently re-normalised frame by frame. Playback
speed is stated in the overlay, because a movie sped up 8-fold makes slow drift
look like activity and slow activity look like noise.

Example
-------
    python make_movie.py --s2p-dir <dataset>/work/s2p_series/suite2p/plane0 \
        --dataset <dataset> --out <dataset>/results/sk52_registered.avi \
        --speed 8 --rois --ledger <dataset>/work/s2p_series/frame_ledger.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np


def load_binary(plane: Path, chan2: bool = False):
    """Memory-map Suite2p's registered binary and read its shape from the ops."""
    d: dict = {}
    for cand in ("reg_outputs.npy", "ops.npy", "db.npy", "detect_outputs.npy"):
        f = plane / cand
        if f.exists():
            d.update(np.load(f, allow_pickle=True).item())
    name = "data_chan2.bin" if chan2 else "data.bin"
    path = plane / name
    if not path.exists():
        for k in ("reg_file_chan2" if chan2 else "reg_file",):
            if d.get(k) and Path(d[k]).exists():
                path = Path(d[k])
                break
    if not path.exists():
        raise FileNotFoundError(
            f"no registered binary at {plane / name}. Suite2p deletes it when "
            "delete_bin is set; re-run with delete_bin False, or pass --tif.")
    ly, lx = int(d["Ly"]), int(d["Lx"])
    n = path.stat().st_size // (ly * lx * 2)
    mm = np.memmap(path, dtype=np.int16, mode="r", shape=(n, ly, lx))
    return mm, d


def scale_bounds(mov, lo_pct, hi_pct, n_sample=300):
    """One intensity range for the whole movie, from frames spread across it.

    Rescaling each frame to its own range would hide exactly the thing a long
    recording is being inspected for: a gradual loss of brightness.
    """
    idx = np.linspace(0, mov.shape[0] - 1, min(n_sample, mov.shape[0]), dtype=int)
    sample = np.asarray(mov[idx], np.float32)
    return (float(np.percentile(sample, lo_pct)),
            float(np.percentile(sample, hi_pct)))


def roi_outline_mask(stat, shape, crop):
    """Boolean image of ROI boundary pixels, in cropped coordinates."""
    (y0, y1), (x0, x1) = crop
    out = np.zeros((y1 - y0, x1 - x0), bool)
    for s in stat:
        m = np.zeros(shape, bool)
        m[s["ypix"], s["xpix"]] = True
        inner = (np.roll(m, 1, 0) & np.roll(m, -1, 0)
                 & np.roll(m, 1, 1) & np.roll(m, -1, 1))
        e = m & ~inner
        out |= e[y0:y1, x0:x1]
    return out


def compare_movies(args, cv2):
    """Play several versions of one recording side by side, in step.

    The point of the comparison is what a denoiser did to the images, and that
    is only visible if the versions are shown at the same moment on the same
    intensity scale. Scaling each tile separately would hide a change in
    absolute brightness, which is one of the things a denoiser can do without
    announcing it.
    """
    import tifffile

    labels = args.compare_labels or [m.stem for m in args.compare]
    if len(labels) != len(args.compare):
        print("ERROR: --compare-labels must match --compare", file=sys.stderr)
        return 2

    ops, stat = {}, None
    if args.s2p_dir:
        plane = args.s2p_dir.expanduser().resolve()
        for cand in ("reg_outputs.npy", "detect_outputs.npy", "ops.npy", "db.npy"):
            f = plane / cand
            if f.exists():
                ops.update(np.load(f, allow_pickle=True).item())
        if args.rois and (plane / "stat.npy").exists():
            stat = np.load(plane / "stat.npy", allow_pickle=True)
            ic = np.load(plane / "iscell.npy")
            if not args.all_roi:
                stat = stat[ic[:, 0].astype(bool)]

    movs, shapes = [], set()
    for m in args.compare:
        mm = tifffile.memmap(str(m)) if m.suffix.lower() in (".tif", ".tiff") \
            else None
        if mm is None:
            print(f"ERROR: {m} is not a TIFF", file=sys.stderr)
            return 2
        movs.append(mm)
        shapes.add(mm.shape)
        print(f"  {m.name:44s} {mm.shape} {mm.dtype}")
    if len(shapes) > 1:
        print(f"ERROR: the movies differ in shape: {shapes}", file=sys.stderr)
        return 2
    n_t, ny, nx = movs[0].shape

    a, b = args.frames if args.frames else (0, n_t)
    a, b = max(0, a), min(n_t, b)
    ds = max(int(args.downsample), 1)
    n_out = (b - a) // ds

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
    if fs is None:
        print("ERROR: pass --fs or --dataset", file=sys.stderr)
        return 2

    # intensity range
    idx = np.linspace(a, b - 1, min(200, b - a), dtype=int)
    ranges = []
    for mm in movs:
        smp = np.asarray(mm[idx], np.float32)
        ranges.append((float(np.percentile(smp, args.clip[0])),
                       float(np.percentile(smp, args.clip[1]))))
    if args.per_movie_scale:
        print("\nper-tile intensity ranges: "
              + ", ".join(f"{l} [{lo:.0f}, {hi:.0f}]"
                          for l, (lo, hi) in zip(labels, ranges)))
    else:
        lo = min(r[0] for r in ranges)
        hi = max(r[1] for r in ranges)
        ranges = [(lo, hi)] * len(movs)
        print(f"\nshared intensity range [{lo:.0f}, {hi:.0f}] for every tile")

    # ROI outlines, shifted if the movies are the cropped valid region
    edges = None
    if stat is not None:
        off = (0, 0)
        if ops.get("Ly") and (ny, nx) != (int(ops["Ly"]), int(ops["Lx"])) \
                and ops.get("yrange") is not None:
            off = (int(ops["yrange"][0]), int(ops["xrange"][0]))
            print(f"ROI coordinates shifted by {off} to match the cropped movies")
        edges = np.zeros((ny, nx), bool)
        for s_ in stat:
            m = np.zeros((ny, nx), bool)
            yy = np.asarray(s_["ypix"]) - off[0]
            xx = np.asarray(s_["xpix"]) - off[1]
            k = (yy >= 0) & (yy < ny) & (xx >= 0) & (xx < nx)
            m[yy[k], xx[k]] = True
            inner = (np.roll(m, 1, 0) & np.roll(m, -1, 0)
                     & np.roll(m, 1, 1) & np.roll(m, -1, 1))
            edges |= m & ~inner
        print(f"outlining {len(stat)} ROI(s) on every tile")

    lut = None
    if args.cmap:
        import matplotlib.cm as cm
        lut = (np.asarray(cm.get_cmap(args.cmap)(np.linspace(0, 1, 256)))[:, :3]
               * 255).astype(np.uint8)[:, ::-1]

    cols = max(1, min(args.grid_cols, len(movs)))
    rows = int(np.ceil(len(movs) / cols))
    tw, th = nx * args.scale, ny * args.scale
    lab_h = 0 if args.no_overlay else max(int(0.10 * th), 20)
    foot = 0 if args.no_overlay else max(int(0.06 * th), 16)
    gw, gh = cols * tw, rows * (th + lab_h) + foot
    fsc = max(lab_h * 0.030, 0.35) if lab_h else 0.4

    fps_out = args.fps_out or min(fs * args.speed / ds, 60.0)
    eff = fps_out * ds / fs
    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*args.codec),
                         fps_out, (gw, gh), isColor=True)
    if not vw.isOpened():
        print(f"ERROR: could not open {out} with codec {args.codec}",
              file=sys.stderr)
        return 2
    print(f"\nwriting {n_out} frames at {fps_out:.1f} fps ({eff:.0f}x), "
          f"{gw} x {gh} px -> {out.name}")

    rc, gc, bc = args.roi_color
    for k in range(n_out):
        s0 = a + k * ds
        canvas = np.zeros((gh, gw, 3), np.uint8)
        for j, (mm, (lo, hi), lab) in enumerate(zip(movs, ranges, labels)):
            chunk = np.asarray(mm[s0:s0 + ds], np.float32)
            img = chunk.mean(axis=0) if ds > 1 else chunk[0]
            v = np.clip((img - lo) / max(hi - lo, 1e-9), 0, 1)
            g8 = (v * 255).astype(np.uint8)
            fr = (lut[g8] if lut is not None
                  else np.repeat(g8[:, :, None], 3, axis=2))
            if edges is not None:
                fr[edges] = (bc, gc, rc)
            fr = cv2.resize(fr, (tw, th), interpolation=cv2.INTER_NEAREST)
            r, c = divmod(j, cols)
            yo_ = r * (th + lab_h)
            canvas[yo_:yo_ + th, c * tw:(c + 1) * tw] = fr
            if lab_h:
                cv2.putText(canvas, lab[:22], (c * tw + 6,
                                               yo_ + th + int(lab_h * 0.74)),
                            cv2.FONT_HERSHEY_SIMPLEX, fsc, (255, 255, 255), 1,
                            cv2.LINE_AA)
        if foot:
            t = s0 / fs
            scale_txt = ("per-tile scale" if args.per_movie_scale
                         else "shared intensity scale")
            cv2.putText(canvas, f"{int(t // 60):02d}:{t % 60:04.1f}    "
                                f"{eff:.0f}x    {scale_txt}",
                        (8, gh - int(foot * 0.28)), cv2.FONT_HERSHEY_SIMPLEX,
                        fsc * 0.9, (190, 190, 190), 1, cv2.LINE_AA)
        vw.write(canvas)
        if n_out >= 20 and k % max(n_out // 10, 1) == 0:
            print(f"  {k / n_out * 100:3.0f}%", end="\r", flush=True)
    vw.release()
    print(f"\nwrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    if not args.per_movie_scale:
        print("  Every tile is on the same scale, so a tile that looks "
              "brighter is brighter\n  and a cell that leaves its outline has "
              "moved in that version.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="export the registered movie as AVI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--s2p-dir", type=Path, default=None,
                   help="suite2p/plane0 directory holding data.bin")
    p.add_argument("--tif", type=Path, nargs="+", default=None,
                   help="read these TIFFs instead of the registered binary")
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--ledger", type=Path, default=None,
                   help="frame_ledger.csv, to label each acquisition in the overlay")
    p.add_argument("--out", type=Path, required=True, help="output .avi")
    p.add_argument("--fs", type=float, default=None, help="acquisition frame rate")
    p.add_argument("--speed", type=float, default=8.0,
                   help="playback speed relative to real time")
    p.add_argument("--fps-out", type=float, default=None,
                   help="output frame rate; default is fs * speed, capped at 60")
    p.add_argument("--frames", type=int, nargs=2, default=None, metavar=("A", "B"),
                   help="export only frames [A, B)")
    p.add_argument("--downsample", type=int, default=1,
                   help="average this many frames together before writing")
    p.add_argument("--clip", type=float, nargs=2, default=[1.0, 99.5],
                   metavar=("LO_PCT", "HI_PCT"))
    p.add_argument("--cmap", default=None,
                   help="matplotlib colormap name; default is grayscale")
    p.add_argument("--scale", type=int, default=4,
                   help="integer upscaling, so a 128 px field is visible")
    p.add_argument("--rois", action="store_true", help="draw ROI outlines")
    p.add_argument("--all-roi", action="store_true")
    p.add_argument("--roi-color", type=int, nargs=3, default=[255, 210, 0],
                   metavar=("R", "G", "B"))
    p.add_argument("--no-overlay", action="store_true",
                   help="omit the time and speed text")
    p.add_argument("--codec", default="MJPG",
                   help="FourCC; MJPG is widely playable, XVID is smaller")
    p.add_argument("--chan2", action="store_true")
    p.add_argument("--compare", type=Path, nargs="+", default=None,
                   metavar="MOVIE",
                   help="tile several versions of the same recording side by "
                        "side, played in step. Use with --s2p-dir so the ROI "
                        "outlines and the valid region come from the detection "
                        "run that all of them share")
    p.add_argument("--compare-labels", nargs="*", default=None)
    p.add_argument("--per-movie-scale", action="store_true",
                   help="scale each tile to its own range. Off by default: a "
                        "denoiser that changes absolute brightness should look "
                        "like it changed absolute brightness")
    p.add_argument("--per-run", action="store_true",
                   help="also write one AVI per acquisition, alongside the "
                        "combined view")
    p.add_argument("--grid", action="store_true",
                   help="tile the acquisitions side by side and play them "
                        "together, so the same cells can be compared directly "
                        "between acquisitions rather than from memory")
    p.add_argument("--grid-cols", type=int, default=3)
    p.add_argument("--grid-out", type=Path, default=None,
                   help="path for the tiled AVI (default: <out stem>_grid.avi)")
    args = p.parse_args(argv)

    try:
        import cv2
    except ImportError:
        print("ERROR: opencv is required. pip install opencv-python-headless",
              file=sys.stderr)
        return 2

    if args.compare:
        return compare_movies(args, cv2)

    # --- source ------------------------------------------------------------
    ops: dict = {}
    stat = None
    if args.tif:
        import tifffile
        stacks = []
        for f in args.tif:
            with tifffile.TiffFile(f) as tf:
                stacks.append(tf.asarray())
        mov = np.concatenate(stacks, axis=0)
        print(f"source: {len(args.tif)} TIFF(s), {mov.shape[0]} frames")
    elif args.s2p_dir:
        plane = args.s2p_dir.expanduser().resolve()
        mov, ops = load_binary(plane, args.chan2)
        print(f"source: registered binary, {mov.shape[0]} frames "
              f"{mov.shape[1]} x {mov.shape[2]} px")
        sf = plane / "stat.npy"
        if args.rois and sf.exists():
            stat = np.load(sf, allow_pickle=True)
            ic = np.load(plane / "iscell.npy")
            if not args.all_roi:
                stat = stat[ic[:, 0].astype(bool)]
    else:
        print("ERROR: pass --s2p-dir or --tif", file=sys.stderr)
        return 2

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
    if fs is None:
        print("ERROR: pass --fs or --dataset", file=sys.stderr)
        return 2

    a, b = (args.frames if args.frames else (0, mov.shape[0]))
    a, b = max(0, a), min(mov.shape[0], b)
    n_src = b - a

    # acquisition labels
    marks = []
    if args.ledger and Path(args.ledger).exists():
        with open(args.ledger) as fh:
            for r in csv.DictReader(fh):
                marks.append((int(r["frame_start"]), int(r["frame_end"]),
                              r["source_file"]))

    # --- valid region -------------------------------------------------------
    ly, lx = mov.shape[1], mov.shape[2]
    yr = ops.get("yrange")
    xr = ops.get("xrange")
    crop = ((int(yr[0]), int(yr[1])) if yr is not None else (0, ly),
            (int(xr[0]), int(xr[1])) if xr is not None else (0, lx))
    (y0, y1), (x0, x1) = crop
    if (y1 - y0, x1 - x0) != (ly, lx):
        print(f"cropping to the registration's valid region: "
              f"y {y0}:{y1}  x {x0}:{x1}  ({y1 - y0} x {x1 - x0} px)")

    lo, hi = scale_bounds(mov, args.clip[0], args.clip[1])
    print(f"intensity range fixed at [{lo:.0f}, {hi:.0f}] for the whole movie")

    lut = None
    if args.cmap:
        import matplotlib.cm as cm
        lut = (np.asarray(cm.get_cmap(args.cmap)(np.linspace(0, 1, 256)))[:, :3]
               * 255).astype(np.uint8)[:, ::-1]   # to BGR

    edges = roi_outline_mask(stat, (ly, lx), crop) if stat is not None else None
    if edges is not None:
        print(f"drawing outlines for {len(stat)} ROI(s)")

    def render(src_a, src_b, label_fn=None):
        """Frames for [src_a, src_b), scaled, ROI-outlined and labelled."""
        step = max(int(args.downsample), 1)
        for k in range((src_b - src_a) // step):
            s0 = src_a + k * step
            chunk = np.asarray(mov[s0:s0 + step, y0:y1, x0:x1], np.float32)
            img = chunk.mean(axis=0) if step > 1 else chunk[0]
            v = np.clip((img - lo) / max(hi - lo, 1e-9), 0, 1)
            g8 = (v * 255).astype(np.uint8)
            fr = (lut[g8] if lut is not None
                  else np.repeat(g8[:, :, None], 3, axis=2))
            if edges is not None:
                fr[edges] = (bc, gc, rc)
            yield s0, fr

    ds = max(int(args.downsample), 1)
    n_out = n_src // ds
    fps_out = args.fps_out or min(fs * args.speed / ds, 60.0)
    eff_speed = fps_out * ds / fs
    h_img, w = (y1 - y0) * args.scale, (x1 - x0) * args.scale
    # the overlay gets its own bar rather than being drawn over the data, so it
    # never hides a cell and stays legible whatever the image scaling is
    bar = 0 if args.no_overlay else max(int(0.09 * h_img), 22)
    h = h_img + bar
    font_scale = max(bar * 0.026, 0.35)

    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    vw = cv2.VideoWriter(str(out_path), fourcc, fps_out, (w, h), isColor=True)
    if not vw.isOpened():
        print(f"ERROR: could not open a writer for {out_path} with codec "
              f"{args.codec}. Try --codec XVID or MJPG.", file=sys.stderr)
        return 2

    print(f"writing {n_out} frames at {fps_out:.1f} fps "
          f"({eff_speed:.1f}x real time), {w} x {h} px -> {out_path.name}")

    rc, gc, bc = args.roi_color
    for k, (s0, frame) in enumerate(render(a, a + n_out * ds)):
        frame = cv2.resize(frame, (w, h_img), interpolation=cv2.INTER_NEAREST)
        if bar:
            frame = np.vstack([frame, np.zeros((bar, w, 3), np.uint8)])

        if not args.no_overlay:
            t = s0 / fs
            label = f"{int(t // 60):02d}:{t % 60:04.1f}   {eff_speed:.0f}x"
            for m0, m1, name in marks:
                if m0 <= s0 <= m1:
                    label += f"   {name}"
                    break
            cv2.putText(frame, label, (6, h - int(bar * 0.3)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(frame)
        if n_out >= 20 and k % max(n_out // 10, 1) == 0:
            print(f"  {k / n_out * 100:3.0f}%", end="\r", flush=True)
    vw.release()

    size_mb = out_path.stat().st_size / 1e6
    print(f"\nwrote {out_path}  ({size_mb:.1f} MB, "
          f"{n_out / fps_out:.1f} s of playback for "
          f"{n_src / fs / 60:.1f} min of recording)")

    if not marks:
        if args.per_run or args.grid:
            print("no ledger given, so the acquisitions cannot be separated; "
                  "skipping --per-run / --grid", file=sys.stderr)
        return 0

    # --- one file per acquisition -----------------------------------------
    if args.per_run:
        for k, (m0, m1, name) in enumerate(marks, start=1):
            sub = out_path.with_name(f"{out_path.stem}_run{k:02d}.avi")
            w2 = cv2.VideoWriter(str(sub), fourcc, fps_out, (w, h), isColor=True)
            if not w2.isOpened():
                print(f"could not open {sub}", file=sys.stderr)
                continue
            for s0, fr in render(m0, m1 + 1):
                fr = cv2.resize(fr, (w, h_img), interpolation=cv2.INTER_NEAREST)
                if bar:
                    fr = np.vstack([fr, np.zeros((bar, w, 3), np.uint8)])
                if not args.no_overlay:
                    t = (s0 - m0) / fs
                    cv2.putText(fr, f"{int(t // 60):02d}:{t % 60:04.1f}   "
                                    f"{eff_speed:.0f}x   {name}",
                                (6, h - int(bar * 0.3)),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                (255, 255, 255), 1, cv2.LINE_AA)
                w2.write(fr)
            w2.release()
            print(f"  run {k}: {sub.name}  ({sub.stat().st_size / 1e6:.1f} MB)")

    # --- all acquisitions tiled and played together ------------------------
    if args.grid:
        # Each tile starts at its own acquisition's first frame, so frame t of
        # the tiled movie shows the same elapsed time in every acquisition.
        # Comparing brightness or motion between acquisitions from separate
        # files means comparing from memory; side by side it is direct.
        lens = [m1 - m0 + 1 for m0, m1, _ in marks]
        n_tile = min(lens) // ds
        cols = max(1, min(args.grid_cols, len(marks)))
        rows = int(np.ceil(len(marks) / cols))
        tw, th = (x1 - x0) * args.scale, (y1 - y0) * args.scale
        lab_h = 0 if args.no_overlay else max(int(0.10 * th), 20)
        foot = 0 if args.no_overlay else max(int(0.06 * th), 16)
        gw, gh = cols * tw, rows * (th + lab_h) + foot
        gpath = args.grid_out or out_path.with_name(f"{out_path.stem}_grid.avi")
        gw_ = cv2.VideoWriter(str(gpath), fourcc, fps_out, (gw, gh), isColor=True)
        if not gw_.isOpened():
            print(f"could not open {gpath}", file=sys.stderr)
            return 0
        print(f"\ntiling {len(marks)} acquisitions {rows}x{cols}, "
              f"{n_tile} frames each -> {gpath.name}")
        gens = [render(m0, m0 + n_tile * ds) for m0, _, _ in marks]
        fsc = max(lab_h * 0.030, 0.35)
        for t in range(n_tile):
            canvas = np.zeros((gh, gw, 3), np.uint8)
            for j, g in enumerate(gens):
                try:
                    s0, fr = next(g)
                except StopIteration:
                    continue
                fr = cv2.resize(fr, (tw, th), interpolation=cv2.INTER_NEAREST)
                r, c = divmod(j, cols)
                yo_ = r * (th + lab_h)
                canvas[yo_:yo_ + th, c * tw:(c + 1) * tw] = fr
                if lab_h:
                    # the label has to fit inside its own tile, otherwise it
                    # runs into the neighbour and neither is readable
                    txt = f"run {j + 1}"
                    (tws, _), _ = cv2.getTextSize(
                        txt, cv2.FONT_HERSHEY_SIMPLEX, fsc, 1)
                    idx = marks[j][2]
                    m_ = re.search(r"_(\d+)\.tif{1,2}$", idx)
                    if m_:
                        extra = f"  ({m_.group(1)})"
                        if tws + len(extra) * fsc * 18 < tw - 10:
                            txt += extra
                    cv2.putText(canvas, txt,
                                (c * tw + 6, yo_ + th + int(lab_h * 0.74)),
                                cv2.FONT_HERSHEY_SIMPLEX, fsc,
                                (255, 255, 255), 1, cv2.LINE_AA)
            if foot:
                tt = t * ds / fs
                cv2.putText(canvas,
                            f"{int(tt // 60):02d}:{tt % 60:04.1f} into each "
                            f"acquisition    {eff_speed:.0f}x    shared "
                            f"intensity scale",
                            (8, gh - int(foot * 0.28)),
                            cv2.FONT_HERSHEY_SIMPLEX, fsc * 0.9,
                            (190, 190, 190), 1, cv2.LINE_AA)
            gw_.write(canvas)
            if n_tile >= 20 and t % max(n_tile // 10, 1) == 0:
                print(f"  {t / n_tile * 100:3.0f}%", end="\r", flush=True)
        gw_.release()
        print(f"\nwrote {gpath}  ({gpath.stat().st_size / 1e6:.1f} MB, "
              f"{gw} x {gh} px)")
        print("  intensity range and ROI outlines are shared across tiles, so a "
              "tile that\n  looks dimmer is dimmer and a cell that leaves its "
              "outline has moved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
