#!/usr/bin/env python
"""Per-acquisition traces with the AUC integration windows drawn on top.

One figure per acquisition. Every ROI is plotted, stacked, over the shared
baseline and window settings used by run_auc.py, and the window boundaries are
drawn as vertical lines with each window's AUC printed above the panel. Nothing
is summarised across ROIs or across acquisitions: the point is to see which part
of which trace each number came from.

Both baselines are available. The rolling baseline follows slow multiplicative
changes, so bleaching and focus drift cancel and only activity remains; the
fixed baseline keeps them, and a decline that appears there but not under the
rolling baseline is not yet distinguishable from the recording getting dimmer.
The raw F panel at the bottom is what separates those two readings, so it is
drawn on the same time axis rather than in a separate figure.

Example
-------
    python fig_traces_per_run.py \
        --s2p-dir <dataset>/work/s2p_series/suite2p/plane0 \
        --dataset <dataset> \
        --ledger <dataset>/work/s2p_series/frame_ledger.csv \
        --all-roi --out <dataset>/work/traces
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_auc import (  # noqa: E402  - shared definitions, not reimplemented here
    auc_of, dff, fixed_baseline, robust_sd, rolling_baseline,
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="per-acquisition traces with AUC windows marked",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--s2p-dir", type=Path, required=True)
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--ledger", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--fs", type=float, default=None)
    p.add_argument("--baseline", choices=["rolling", "fixed"], default="rolling")
    p.add_argument("--window-s", type=float, default=60.0)
    p.add_argument("--baseline-window-s", type=float, default=45.0)
    p.add_argument("--baseline-percentile", type=float, default=10.0)
    p.add_argument("--neucoeff", type=float, default=0.7)
    p.add_argument("--units", choices=["sd", "pct"], default="pct")
    p.add_argument("--all-roi", action="store_true")
    p.add_argument("--max-per-page", type=int, default=25)
    args = p.parse_args(argv)

    s2p = args.s2p_dir.expanduser().resolve()
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    F = np.load(s2p / "F.npy")
    Fneu = np.load(s2p / "Fneu.npy")
    iscell = np.load(s2p / "iscell.npy")
    keep = np.ones(F.shape[0], bool) if args.all_roi else iscell[:, 0].astype(bool)
    F, Fneu = F[keep].astype(np.float32), Fneu[keep].astype(np.float32)
    n_roi, n_frames = F.shape

    fs = args.fs
    if fs is None and args.dataset:
        try:
            from run_roi_suite2p import resolve_from_metadata
            fs = resolve_from_metadata(
                args.dataset.expanduser().resolve() / "raw" / "metadata.yaml").get("fs_hz")
        except Exception as e:  # noqa: BLE001
            print(f"metadata unreadable: {e}", file=sys.stderr)
    if fs is None:
        print("ERROR: pass --fs or --dataset", file=sys.stderr)
        return 2

    ledger_path = args.ledger or (s2p.parent.parent / "frame_ledger.csv")
    if not ledger_path.exists():
        print(f"ERROR: no frame ledger at {ledger_path}", file=sys.stderr)
        return 2
    with open(ledger_path) as fh:
        rows = list(csv.DictReader(fh))
    segs = [(r["source_file"], int(r["frame_start"]), int(r["frame_end"]) + 1)
            for r in rows]

    win_b = int(round(args.baseline_window_s * fs))
    win_w = int(round(args.window_s * fs))
    print(f"ROIs {n_roi}   frames {n_frames}   {len(segs)} acquisition(s)")
    print(f"baseline: {args.baseline}   AUC window {args.window_s:g} s "
          f"({win_w} frames)")

    # baselines computed exactly as run_auc.py does, per acquisition
    Fc = F - args.neucoeff * Fneu
    roll = np.empty_like(Fc)
    for _, a, b in segs:
        roll[:, a:b] = rolling_baseline(Fc[:, a:b], win_b, args.baseline_percentile)
    d_roll = dff(Fc, roll) * 100.0
    a0, b0 = segs[0][1], segs[0][2]
    d_fix = dff(Fc, fixed_baseline(Fc, a0, b0, args.baseline_percentile)) * 100.0
    d = d_roll if args.baseline == "rolling" else d_fix

    if args.units == "sd":
        sd = robust_sd(d)
        traces = d / np.maximum(sd[:, None], 1e-9)
        ulab, bar = "SD", 5.0
    else:
        traces = d
        ulab, bar = r"% $\Delta$F/F", 50.0

    try:
        import figstyle_tshino as FS
        FS.set_style()
    except Exception:  # noqa: BLE001
        plt.rcParams.update({"font.family": "sans-serif",
                             "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"]})
    plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 8})

    cmap = plt.get_cmap("turbo")
    colors = [cmap(v) for v in np.linspace(0.08, 0.92, n_roi)]
    span = float(np.percentile(traces, 99.5))
    step = max(span * 1.1, bar * 1.2)

    written = []
    for k, (name, a, b) in enumerate(segs, start=1):
        t = (np.arange(a, b) - a) / fs
        edges = np.arange(a, b + 1, win_w)
        if edges[-1] != b:
            edges = np.append(edges, b)
        wins = [(w0, w1) for w0, w1 in zip(edges[:-1], edges[1:])
                if w1 - w0 >= win_w // 2]

        fig, ax = plt.subplots(
            2, 1, figsize=(13, max(5.0, 0.34 * n_roi + 3.0)),
            sharex=True, gridspec_kw={"height_ratios": [max(n_roi * 0.34, 4), 1.5]})

        # traces
        yt, yl = [], []
        for i in range(n_roi):
            off = -step * i
            ax[0].plot(t, traces[i, a:b] + off, lw=0.45, color=colors[i])
            yt.append(off)
            yl.append(str(i + 1))
        ax[0].set_yticks(yt)
        ax[0].set_yticklabels(yl, fontsize=6)
        ax[0].tick_params(axis="y", length=0)
        ax[0].set_ylabel("ROI")
        for side in ("top", "right", "left"):
            ax[0].spines[side].set_visible(False)

        # AUC windows: shaded alternately, boundary lines, value printed above
        ymin, ymax = ax[0].get_ylim()
        for wi, (w0, w1) in enumerate(wins):
            x0, x1 = (w0 - a) / fs, (w1 - a) / fs
            if wi % 2:
                ax[0].axvspan(x0, x1, color="0.85", alpha=0.35, zorder=0)
            ax[0].axvline(x0, color="0.4", lw=0.7, ls="--", zorder=1)
            val = float(auc_of(d[:, w0:w1], fs).mean())
            ax[0].text((x0 + x1) / 2, ymax, f"{val:.2f}", ha="center", va="bottom",
                       fontsize=7, color="C3")
        ax[0].axvline((wins[-1][1] - a) / fs, color="0.4", lw=0.7, ls="--")

        run_auc_val = float(auc_of(d[:, a:b], fs).mean())
        ax[0].set_title(
            f"{name}   frames [{a} .. {b - 1}]   {n_roi} ROIs   "
            f"{args.baseline} F0, r = {args.neucoeff:g}\n"
            f"red = mean {ulab} integrated over each {args.window_s:g} s window "
            f"(dashed = window edges);  whole acquisition = {run_auc_val:.2f}",
            fontsize=9, loc="left")

        # scale bar
        xb = t[-1] * 1.01
        ax[0].plot([xb, xb], [0, bar], "-", color="k", lw=1.6, clip_on=False)
        ax[0].text(xb * 1.004, bar / 2, f" {bar:g} {ulab}", va="center", ha="left",
                   fontsize=7, clip_on=False)

        # raw F, undivided by any baseline
        ax[1].plot(t, F[:, a:b].mean(0), lw=0.6, color="C3", label="F")
        ax[1].plot(t, Fneu[:, a:b].mean(0), lw=0.6, color="C1", label="Fneu")
        for w0, _ in wins:
            ax[1].axvline((w0 - a) / fs, color="0.4", lw=0.7, ls="--")
        ax[1].set_ylabel("raw (mean)")
        ax[1].set_xlabel("time within acquisition (s)")
        ax[1].legend(fontsize=7, loc="upper right")
        ax[1].set_xlim(0, t[-1])

        fig.tight_layout()
        stem = out / f"traces_run{k:02d}"
        fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        written.append(stem)
        print(f"  run {k}: {name}  AUC {run_auc_val:8.3f}  "
              f"{len(wins)} window(s) -> {stem.name}.png/.pdf")

    print(f"\nwrote {len(written)} figure(s) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
