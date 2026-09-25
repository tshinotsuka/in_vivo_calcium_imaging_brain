#!/usr/bin/env python
"""What a threshold-free AUC actually integrates, drawn on the traces.

With no threshold nothing is excluded: the AUC is the integral of dF/F over the
whole acquisition, so every excursion contributes, upward and downward alike.
That is the property that makes it independent of a detection threshold, and it
is also the thing that has to be checked before the number is trusted.

The check is the balance between the two. Noise contributes upward and downward
excursions in roughly equal measure, so it largely cancels and leaves the real
transients behind. But "largely" is doing the work: if the positive and
negative areas are both large and nearly equal, the AUC is a small difference
between two big numbers and the noise has not cancelled so much as been
subtracted from itself, which is a far weaker position than it looks from the
final value alone.

So each trace is drawn with its positive area shaded one colour and its
negative area the other, and the two totals are reported next to their
difference. A ratio near one means the AUC rests on cancellation; a positive
area clearly larger than the negative one means it rests on signal.

Example
-------
    python fig_auc_area.py --labels raw CNMF DeepCAD \
        --s2p-dir <...>/s2p_series/suite2p/plane0 \
                  <...>/s2p_cnmf/suite2p/plane0 \
                  <...>/s2p_deepcad/suite2p/plane0 \
        --dataset <dataset> --ledger <...>/frame_ledger.csv \
        --all-roi --neucoeff 0 --out <dataset>/work/auc_area
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

_Z = {1.0: -2.3263, 5.0: -1.6449, 8.0: -1.4051, 10.0: -1.2816,
      15.0: -1.0364, 20.0: -0.8416, 25.0: -0.6745, 50.0: 0.0}


def _z_of(pct):
    try:
        from scipy.stats import norm
        return float(norm.ppf(pct / 100.0))
    except ImportError:
        return _Z[min(_Z, key=lambda x: abs(x - pct))]


def robust_sd(F):
    """Noise scale from frame-to-frame differences. NOT valid after smoothing."""
    d = np.abs(np.diff(F, axis=1))
    return np.median(d, axis=1) / (np.sqrt(2) * 0.6745)


def sd_from_values(F):
    """Noise scale from the spread of the values, valid on a smoothed trace."""
    F = np.asarray(F, float)
    return np.array([1.4826 * np.median(np.abs(x - np.median(x))) for x in F])


def rolling_baseline(F, win, pct=10.0, correct_bias=True):
    """Percentile baseline, corrected for the bias a low percentile carries.

    A low percentile of a noisy trace sits about |z| * sigma below the resting
    level, which adds a constant positive offset to dF/F that is not activity
    and that grows as the baseline dims. Adding it back removes the offset.
    """
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
    if correct_bias and pct < 50:
        out = out + (abs(_z_of(pct)) * sd_from_values(F))[:, None]
    return out


def fixed_baseline(F, a, b, pct=50.0, correct_bias=False):
    v = np.percentile(F[:, a:b], pct, axis=1).astype(np.float32)
    if correct_bias and pct < 50:
        v = v + abs(_z_of(pct)) * sd_from_values(F[:, a:b])
    return v[:, None]


def load_traces(plane: Path, neucoeff, all_roi):
    F = np.load(plane / "F.npy").astype(np.float32)
    Fneu = np.load(plane / "Fneu.npy").astype(np.float32)
    iscell = np.load(plane / "iscell.npy")
    keep = np.ones(F.shape[0], bool) if all_roi else iscell[:, 0].astype(bool)
    return (F[keep] - neucoeff * Fneu[keep]).astype(np.float32)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="what the threshold-free AUC integrates",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--s2p-dir", type=Path, nargs="+", required=True)
    p.add_argument("--labels", nargs="*", default=None)
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--ledger", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--fs", type=float, default=None)
    p.add_argument("--neucoeff", type=float, default=0.0)
    p.add_argument("--baseline", choices=["rolling", "fixed"], default="rolling")
    p.add_argument("--baseline-window-s", type=float, default=45.0)
    p.add_argument("--baseline-percentile", type=float, default=50.0,
                   help="50 (the median) keeps the positive and negative areas "
                        "comparable: pure noise then gives a ratio of one")
    p.add_argument("--bias-correction", dest="bias", action="store_true",
                   help="only meaningful below the 50th percentile; the sigma "
                        "it needs is estimated from the values, not from "
                        "frame-to-frame differences, so it stays valid on a "
                        "denoised trace")
    p.add_argument("--all-roi", action="store_true")
    p.add_argument("--trace-rois", type=int, nargs="*", default=None,
                   help="ROIs to draw; default is the first --n-trace")
    p.add_argument("--n-trace", type=int, default=6)
    p.add_argument("--run", type=int, default=1,
                   help="acquisition to draw the traces from")
    p.add_argument("--color-pos", default="#2c7fb8")
    p.add_argument("--color-neg", default="#d95f02")
    p.add_argument("--alpha", type=float, default=0.55)
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args(argv)

    labels = args.labels or [d.parent.parent.name for d in args.s2p_dir]
    if len(labels) != len(args.s2p_dir):
        print("ERROR: --labels must match --s2p-dir", file=sys.stderr)
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

    plane0 = args.s2p_dir[0].expanduser().resolve()
    ledger = args.ledger or (plane0.parent.parent / "frame_ledger.csv")
    segs = []
    if Path(ledger).exists():
        with open(ledger) as fh:
            for r in csv.DictReader(fh):
                segs.append((r["source_file"], int(r["frame_start"]),
                             int(r["frame_end"]) + 1))
    win_b = int(round(args.baseline_window_s * fs))

    variants = []
    for plane, lab in zip(args.s2p_dir, labels):
        Fc = load_traces(plane.expanduser().resolve(), args.neucoeff, args.all_roi)
        if not segs:
            segs = [("all", 0, Fc.shape[1])]
        if args.baseline == "rolling":
            F0 = np.empty_like(Fc)
            for _, a, b in segs:
                F0[:, a:b] = rolling_baseline(Fc[:, a:b], win_b,
                                              args.baseline_percentile, args.bias)
        else:
            a0, b0 = segs[0][1], segs[0][2]
            F0 = np.repeat(fixed_baseline(Fc, a0, b0, args.baseline_percentile,
                                          args.bias), Fc.shape[1], axis=1)
        d = (Fc - F0) / np.maximum(F0, 1.0) * 100.0
        variants.append((lab, d))
        print(f"{lab:12s} {d.shape[0]} ROIs x {d.shape[1]} frames")

    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    # --- the split, per variant and per acquisition -------------------------
    rows = []
    print(f"\n{'variant':12s} {'run':>3} {'positive':>10} {'negative':>10} "
          f"{'net AUC':>10} {'neg/pos':>8}")
    for lab, d in variants:
        for k, (name, a, b) in enumerate(segs, start=1):
            seg = d[:, a:b]
            dur = (b - a) / fs
            pos = float(np.clip(seg, 0, None).sum()) / fs / dur
            neg = float(np.clip(seg, None, 0).sum()) / fs / dur
            net = pos + neg
            ratio = abs(neg) / max(pos, 1e-9)
            rows.append({"variant": lab, "run": k, "file": name,
                         "positive_area": round(pos, 3),
                         "negative_area": round(neg, 3),
                         "net_auc": round(net, 3),
                         "neg_over_pos": round(ratio, 4)})
            print(f"{lab:12s} {k:>3} {pos:10.2f} {neg:10.2f} {net:10.2f} "
                  f"{ratio:8.3f}")

    print("\nWith a median baseline, pure noise gives neg/pos = 1 by "
          "construction. A ratio\nnear one therefore means the net AUC is "
          "what survives the cancellation of two\nnearly equal areas, rather "
          "than resting on signal.")

    with open(out / "auc_area_split.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # --- per-ROI net AUC, in the same format the line figure reads ----------
    for lab, d in variants:
        prows = []
        for k, (name, a, b) in enumerate(segs, start=1):
            dur = (b - a) / fs
            seg = d[:, a:b]
            for i in range(d.shape[0]):
                prows.append({
                    "roi": i + 1, "run": k, "file": name,
                    "auc_net": round(float(seg[i].sum()) / fs / dur, 4),
                    "auc_positive": round(float(np.clip(seg[i], 0, None).sum())
                                          / fs / dur, 4),
                    "auc_negative": round(float(np.clip(seg[i], None, 0).sum())
                                          / fs / dur, 4)})
        with open(out / f"auc_per_roi_{lab.replace(' ', '_')}.csv", "w",
                  newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(prows[0].keys()))
            w.writeheader()
            w.writerows(prows)

    # --- figure --------------------------------------------------------------
    try:
        import figstyle_tshino as FS
        FS.set_style()
    except Exception:  # noqa: BLE001
        plt.rcParams.update({"font.family": "sans-serif",
                             "font.sans-serif": ["Arial", "Liberation Sans",
                                                 "DejaVu Sans"]})
    plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none",
                         "font.size": 8})

    k = min(max(args.run, 1), len(segs)) - 1
    name, a, b = segs[k]
    t = (np.arange(a, b) - a) / fs
    n_roi = variants[0][1].shape[0]
    sel = (np.array(args.trace_rois, int) - 1 if args.trace_rois
           else np.arange(min(args.n_trace, n_roi)))
    sel = np.array([i for i in sel if 0 <= i < n_roi], int)

    nv = len(variants)
    fig, axes = plt.subplots(sel.size, nv, figsize=(6.2 * nv, 1.5 * sel.size),
                             squeeze=False, sharex=True)
    span = max(float(np.percentile(np.abs(variants[0][1][sel, a:b]), 99.5)), 1.0)
    for j, (lab, d) in enumerate(variants):
        for r, i in enumerate(sel):
            ax = axes[r, j]
            y = d[i, a:b]
            ax.plot(t, y, lw=0.4, color="0.25")
            ax.fill_between(t, 0, np.clip(y, 0, None), color=args.color_pos,
                            alpha=args.alpha, lw=0)
            ax.fill_between(t, 0, np.clip(y, None, 0), color=args.color_neg,
                            alpha=args.alpha, lw=0)
            ax.axhline(0, color="0.5", lw=0.6)
            pos = float(np.clip(y, 0, None).sum()) / fs / ((b - a) / fs)
            neg = float(np.clip(y, None, 0).sum()) / fs / ((b - a) / fs)
            ax.set_ylim(-span * 1.1, span * 1.6)
            ax.set_title(f"ROI {i + 1}   +{pos:.1f}  {neg:.1f}  "
                         f"net {pos + neg:+.1f}", fontsize=7, loc="left")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if j == 0:
                ax.set_ylabel(r"%$\Delta$F/F", fontsize=7)
            if r == 0:
                ax.text(0.5, 1.55, lab, transform=ax.transAxes, ha="center",
                        va="bottom", fontsize=10, fontweight="bold")
    for j in range(nv):
        axes[-1, j].set_xlabel("time within acquisition (s)")
    handles = [plt.Line2D([0], [0], color=c, lw=6, alpha=args.alpha, label=t_)
               for c, t_ in ((args.color_pos, "counted as positive"),
                             (args.color_neg, "counted as negative"))]
    axes[0, 0].legend(handles=handles, fontsize=7, ncol=2, frameon=False,
                      loc="upper left")
    fig.suptitle(f"{name}   everything is integrated: the AUC is the blue area "
                 f"minus the orange one   ({args.baseline} F0"
                 + ("" if args.bias else ", bias correction off") + ")",
                 fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "auc_area_traces.png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out / "auc_area_traces.pdf", bbox_inches="tight")
    plt.close(fig)

    # --- the split across acquisitions --------------------------------------
    fig, axes = plt.subplots(1, nv, figsize=(4.6 * nv, 4.2), squeeze=False,
                             sharey=True)
    runs = np.arange(1, len(segs) + 1)
    for j, (lab, _) in enumerate(variants):
        ax = axes[0, j]
        sub = [r for r in rows if r["variant"] == lab]
        ax.plot(runs, [r["positive_area"] for r in sub], marker="o",
                color=args.color_pos, lw=1.8, label="positive")
        ax.plot(runs, [-r["negative_area"] for r in sub], marker="s",
                color=args.color_neg, lw=1.8, label="negative (sign flipped)")
        ax.plot(runs, [r["net_auc"] for r in sub], marker="^", color="C3",
                lw=2.2, label="net AUC")
        ax.axhline(0, color="0.7", lw=0.7)
        ax.set_xticks(runs)
        ax.set_xlabel("acquisition")
        ax.set_title(lab, fontsize=10, loc="left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if j == 0:
            ax.set_ylabel(r"area per second (%$\Delta$F/F)")
            ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "auc_area_split.png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out / "auc_area_split.pdf", bbox_inches="tight")
    plt.close(fig)

    with open(out / "auc_area.json", "w") as fh:
        json.dump({"baseline": args.baseline, "neucoeff": args.neucoeff,
                   "bias_correction": args.bias, "fs_hz": fs, "rows": rows},
                  fh, indent=2)
    print(f"\nwrote auc_area_traces.png, auc_area_split.png and CSVs to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
