#!/usr/bin/env python
"""Per-ROI activity across acquisitions, from auc_per_roi_per_run.csv.

Every ROI is drawn as its own line, so a change in the mean can be read as
what it is: the same cells moving together, a few cells moving a long way, or
cells moving in opposite directions that happen to average to a decline. A bar
of the mean shows none of those apart, and with 25 ROIs there is no reason to
hide them.

The mean and its standard error are drawn as a separate line on top rather than
as error bars on the individual points, because the two are different
quantities: the spread of the ROI lines is the variability between cells, while
the error bars are the uncertainty in the mean.

Several CSVs can be given at once to compare processing variants. They are
plotted in separate panels sharing one y axis, since the comparison is only
meaningful at the same scale.

Example
-------
    python fig_auc_lines.py \
        --csv <dataset>/work/event_auc_r0/auc_per_roi_per_run.csv \
        --out <dataset>/work/auc_lines

    python fig_auc_lines.py --labels raw CNMF \
        --csv <...>/event_auc_r0/auc_per_roi_per_run.csv \
              <...>/event_auc_cnmf/auc_per_roi_per_run.csv \
        --out <dataset>/work/auc_compare
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


def load(path: Path, value: str):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path} is empty")
    if value not in rows[0]:
        raise KeyError(f"{path} has no column {value!r}; available: "
                       + ", ".join(rows[0]))
    rois = sorted({int(r["roi"]) for r in rows})
    if rois and rois[0] == 0:
        # some writers index ROIs from zero and others from one; shift so the
        # labels mean the same thing when panels are compared
        rois = [r + 1 for r in rois]
        for r in rows:
            r["roi"] = str(int(r["roi"]) + 1)
    runs = sorted({int(r["run"]) for r in rows})
    M = np.full((len(rois), len(runs)), np.nan)
    ri = {r: i for i, r in enumerate(rois)}
    ci = {c: i for i, c in enumerate(runs)}
    for r in rows:
        try:
            M[ri[int(r["roi"])], ci[int(r["run"])]] = float(r[value])
        except (TypeError, ValueError):
            pass
    return M, np.array(rois), np.array(runs)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="per-ROI lines with the mean and s.e.m. drawn separately",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--csv", type=Path, nargs="+", required=True)
    p.add_argument("--labels", nargs="*", default=None)
    p.add_argument("--out", type=Path, required=True, help="output stem")
    p.add_argument("--value", nargs="+", default=["auc_per_min_dff"],
                   help="column to plot. Give one name, or one per CSV when "
                        "the panels should show different quantities")
    p.add_argument("--ylabel", default=None,
                   help="default is chosen from --value")
    p.add_argument("--xlabel", default="acquisition")
    p.add_argument("--color-roi", default="0.60")
    p.add_argument("--color-mean", default="C3")
    p.add_argument("--roi-lw", type=float, default=0.8)
    p.add_argument("--roi-alpha", type=float, default=0.75)
    p.add_argument("--roi-ms", type=float, default=2.8)
    p.add_argument("--mean-lw", type=float, default=2.4)
    p.add_argument("--mean-ms", type=float, default=7.0)
    p.add_argument("--errorbar", choices=["sem", "sd", "ci95", "none"],
                   default="sem")
    p.add_argument("--log", action="store_true", help="log y axis")
    p.add_argument("--ymin", type=float, default=None)
    p.add_argument("--ymax", type=float, default=None)
    p.add_argument("--label-rois", action="store_true",
                   help="write the ROI number at the end of each line")
    p.add_argument("--normalise", action="store_true",
                   help="divide each ROI by its own first acquisition")
    p.add_argument("--figsize", type=float, nargs=2, default=None,
                   metavar=("W", "H"))
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--style", default=None)
    p.add_argument("--rcparams", default=None)
    args = p.parse_args(argv)

    labels = args.labels or [c.parent.name for c in args.csv]
    if len(labels) != len(args.csv):
        print("ERROR: --labels must match the number of --csv", file=sys.stderr)
        return 2

    values = (args.value if len(args.value) == len(args.csv)
              else args.value[:1] * len(args.csv))
    if len(args.value) not in (1, len(args.csv)):
        print("ERROR: --value must be one name, or one per --csv",
              file=sys.stderr)
        return 2
    mats = []
    for c, v in zip(args.csv, values):
        try:
            mats.append(load(c.expanduser().resolve(), v))
        except (KeyError, ValueError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2

    ylab = args.ylabel
    if ylab is None:
        args_value = values[0]
        ylab = {
            "auc_per_min_dff": r"AUC (%$\Delta$F/F$\cdot$s / min)",
            "auc_rolling": r"AUC, rolling F0 (mean %$\Delta$F/F)",
            "auc_fixed":   r"AUC, fixed F0 (mean %$\Delta$F/F)",
            "spike_rate_per_min": "inferred rate (a.u. / min)",
            "resid_sd": r"residual s.d. (%$\Delta$F/F)",
            "auc_per_min_df":  r"$\Delta$F AUC (a.u.$\cdot$s / min)",
            "n_events": "events per acquisition",
            "nu": r"$\nu$ (%$\cdot$Hz$^{-1/2}$)",
            "raw_F": "raw F (a.u.)",
            "raw_F0": "F0 (a.u.)",
        }.get(args_value, args_value)
    if args.normalise:
        ylab += "\n(fraction of acquisition 1)"

    try:
        import figstyle_tshino as FS
        FS.set_style()
    except Exception:  # noqa: BLE001
        plt.rcParams.update({"font.family": "sans-serif",
                             "font.sans-serif": ["Arial", "Liberation Sans",
                                                 "DejaVu Sans"]})
    plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none"})
    if args.style:
        plt.style.use(args.style)
    if args.rcparams:
        with open(args.rcparams) as fh:
            plt.rcParams.update(json.load(fh))

    n = len(mats)
    fs_ = args.figsize or (4.6 * n + 0.6, 4.6)
    fig, axes = plt.subplots(1, n, figsize=fs_, squeeze=False, sharey=True)

    summary = []
    for ax, (M, rois, runs), lab, val in zip(axes[0], mats, labels, values):
        if args.normalise:
            M = M / np.where(np.abs(M[:, 0:1]) > 1e-12, M[:, 0:1], np.nan)
        for i in range(M.shape[0]):
            ax.plot(runs, M[i], lw=args.roi_lw, color=args.color_roi,
                    alpha=args.roi_alpha, marker="o", ms=args.roi_ms,
                    markeredgewidth=0, zorder=1)
            if args.label_rois and np.isfinite(M[i, -1]):
                ax.annotate(str(rois[i]), (runs[-1], M[i, -1]),
                            xytext=(4, 0), textcoords="offset points",
                            fontsize=5.5, color=args.color_roi,
                            va="center", zorder=1)
        m = np.nanmean(M, axis=0)
        k = np.sum(np.isfinite(M), axis=0)
        sd = np.nanstd(M, axis=0, ddof=1)
        if args.errorbar == "sem":
            err, etxt = sd / np.sqrt(np.maximum(k, 1)), "s.e.m."
        elif args.errorbar == "sd":
            err, etxt = sd, "s.d."
        elif args.errorbar == "ci95":
            err, etxt = 1.96 * sd / np.sqrt(np.maximum(k, 1)), "95% CI"
        else:
            err, etxt = None, None
        ax.errorbar(runs, m, yerr=err, color=args.color_mean, lw=args.mean_lw,
                    marker="o", ms=args.mean_ms, capsize=4, elinewidth=1.6,
                    zorder=3,
                    label=(f"mean $\\pm$ {etxt} (n = {int(k.max())} ROIs)"
                           if etxt else f"mean (n = {int(k.max())} ROIs)"))
        ax.set_xlabel(args.xlabel)
        ax.set_xticks(runs)
        ax.set_xlim(runs[0] - 0.4, runs[-1] + 0.4)
        if args.log:
            ax.set_yscale("log")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        # A relative change needs a starting value to be relative to. When the
        # first acquisition is zero -- no events detected in the baseline --
        # there is no percentage to report, and saying so is the answer rather
        # than an error.
        chg = ((m[-1] - m[0]) / abs(m[0]) * 100) if abs(m[0]) > 1e-12 else np.nan
        chg_txt = f"({chg:+.0f}%)" if np.isfinite(chg) else "(no baseline to compare against)"
        n_down = int(np.sum(M[:, -1] < M[:, 0]))
        ax.set_title(f"{lab}"
                     + (f"  [{val}]" if len(set(values)) > 1 else "")
                     + f"\nmean {m[0]:.1f} -> {m[-1]:.1f} {chg_txt};   "
                     f"{n_down} of {M.shape[0]} ROIs lower at the end",
                     fontsize=9, loc="left")
        ax.legend(fontsize=8, frameon=False)
        summary.append({"label": lab, "value": val, "n_roi": int(M.shape[0]),
                        "mean_by_run": [round(float(x), 4) for x in m],
                        "sem_by_run": [round(float(x), 4)
                                       for x in (sd / np.sqrt(np.maximum(k, 1)))],
                        "pct_change_first_to_last": (None if not np.isfinite(chg)
                                                     else round(float(chg), 2)),
                        "n_roi_lower_at_end": n_down})

    axes[0][0].set_ylabel(ylab)
    if args.ymin is not None or args.ymax is not None:
        axes[0][0].set_ylim(args.ymin, args.ymax)
    elif not args.log:
        lo = min(0.0, float(np.nanmin([m[0].min() for m in mats])))
        hi = float(np.nanmax([m[0].max() for m in mats]))
        axes[0][0].set_ylim(bottom=lo, top=(hi * 1.05 if hi > lo else lo + 1.0))
    fig.tight_layout()

    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    for s in summary:
        pc = s["pct_change_first_to_last"]
        pct_txt = f"({pc:+.0f}%)" if pc is not None else "(no baseline)"
        print(f"{s['label']:20s} n={s['n_roi']:3d}  "
              f"{s['mean_by_run'][0]:8.2f} -> {s['mean_by_run'][-1]:8.2f}  "
              f"{pct_txt:>14s}  "
              f"{s['n_roi_lower_at_end']}/{s['n_roi']} lower")
    with open(out.with_suffix(".json"), "w") as fh:
        json.dump({"values": values, "panels": summary}, fh, indent=2)
    print(f"\nwrote {out.with_suffix('.png')}, .pdf and .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
