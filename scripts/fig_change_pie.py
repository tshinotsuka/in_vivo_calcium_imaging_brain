#!/usr/bin/env python
"""How many ROIs went up, down, or neither, against a common baseline.

The hard part of this figure is not the pie, it is the word "unchanged". A
fixed cutoff — ten per cent, say — states an answer rather than measuring one,
and its arbitrariness disappears from view once the slices are drawn. So the
comparison is made against what the baseline acquisition itself could have
produced: that ROI's events are resampled to give the range a repeat of the
baseline would fall in, and an ROI counts as changed only when the later
acquisition lands outside it.

A cell that is merely noisy therefore lands in "unchanged", where it belongs,
and a cell with a small but consistent shift can still count as changed. Every
panel is compared against the same baseline, so the panels form a time course
rather than a chain of successive differences.

Each pie is drawn above a scatter of the two acquisitions against each other,
because proportions alone say nothing about size: three ROIs that halved and
three that fell by a thousandth make the same slice.

Example
-------
    python fig_change_pie.py --auc-dir <dataset>/work/eauc_fix_series \
        --ledger <dataset>/work/s2p_series/frame_ledger.csv \
        --dataset <dataset> --from-run 1 --to-run all \
        --color-up '#ff4b00' --color-down '#005aff' \
        --out <dataset>/work/change_pie
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


def read_events(path: Path):
    ev = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            ev.append((int(r["roi"]), int(r["onset_frame"]),
                       float(r["area_pct_s"])))
    return ev


def read_per_run(path: Path, value):
    M = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                M[(int(r["roi"]), int(r["run"]))] = float(r[value])
            except (KeyError, TypeError, ValueError):
                pass
    return M


def bootstrap_interval(areas, duration_min, n_boot=10000, alpha=0.05, seed=0):
    """Range a repeat of this acquisition would fall in, for one ROI.

    The AUC is a count of events multiplied by their mean size, and both vary
    from one acquisition to the next. Resampling a fixed number of events
    captures only the second, and collapses to nothing when the events happen
    to be the same size; drawing the count from a Poisson distribution puts the
    counting variability back, which for a sparse process is usually the larger
    part.

    The question is whether a later acquisition could have come from the same
    cell behaving the same way, which needs a prediction interval rather than a
    confidence one: the later acquisition has its own sampling variability, and
    the baseline only estimates the rate rather than fixing it. Each draw
    therefore resamples twice, once for the rate the baseline was estimating
    and once for what a repeat of it would produce.

    Coverage falls short of nominal when events are few, because resampling
    sizes from a short list underestimates how far that list's own tail
    extends. Simulation puts the true coverage of a nominal 95% interval at
    roughly 82% at five events, 85% at twenty and 90% at eighty, so the test is
    somewhat liberal. That figure is reported alongside the result.
    """
    a = np.asarray(areas, float)
    n = a.size
    if n == 0:
        # a repeat of a rate of zero is zero, so any activity later is a change
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    obs = a.sum() / duration_min
    draws = np.empty(n_boot)
    for j in range(n_boot):
        c1 = rng.poisson(n)
        pop = a if c1 == 0 else rng.choice(a, size=c1, replace=True)
        c2 = rng.poisson(n)
        draws[j] = 0.0 if c2 == 0 else rng.choice(pop, size=c2, replace=True).sum()
    draws /= duration_min
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(obs), float(lo), float(hi)


def coverage_at(n_events):
    """Simulated coverage of a nominal 95% interval, by event count."""
    return float(np.interp(n_events, [5, 10, 20, 40, 80], [82, 84, 85, 86, 90]))


def event_areas(events, a, b):
    out = {}
    for roi, onset, area in events:
        if a <= onset < b:
            out.setdefault(roi, []).append(area)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="ROIs up, down or unchanged against a common baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--auc-dir", type=Path, required=True,
                   help="directory holding events.csv and auc_per_roi_per_run.csv")
    p.add_argument("--ledger", type=Path, default=None)
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True, help="output stem")
    p.add_argument("--fs", type=float, default=None)
    p.add_argument("--from-run", type=int, default=1, help="the baseline")
    p.add_argument("--to-run", nargs="*", default=None,
                   help="acquisitions to compare against the baseline, or "
                        "'all' for every other one. Default is the last")
    p.add_argument("--value", default="auc_per_min_dff")
    p.add_argument("--method", choices=["baseline-bootstrap", "threshold"],
                   default="baseline-bootstrap")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--threshold", type=float, default=0.2,
                   help="for --method threshold: fractional change counted as changed")
    p.add_argument("--color-up", default="#ff4b00")
    p.add_argument("--color-same", default="#bdbdbd")
    p.add_argument("--color-down", default="#005aff")
    p.add_argument("--point-size", type=float, default=42.0)
    p.add_argument("--label", default=None)
    p.add_argument("--log-scatter", action="store_true")
    p.add_argument("--no-scatter", action="store_true",
                   help="pies only, in a single row")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args(argv)

    auc_dir = args.auc_dir.expanduser().resolve()
    per_run_path = auc_dir / "auc_per_roi_per_run.csv"
    if not per_run_path.exists():
        print(f"ERROR: no auc_per_roi_per_run.csv in {auc_dir}", file=sys.stderr)
        return 2
    M = read_per_run(per_run_path, args.value)
    if not M:
        print(f"ERROR: no column {args.value!r} in {per_run_path}",
              file=sys.stderr)
        return 2
    rois = sorted({r for r, _ in M})
    runs = sorted({k for _, k in M})
    n_roi = len(rois)

    r_from = args.from_run
    if args.to_run is None:
        targets = [runs[-1]]
    elif len(args.to_run) == 1 and str(args.to_run[0]).lower() == "all":
        targets = [r for r in runs if r != r_from]
    else:
        targets = [int(x) for x in args.to_run]
    if r_from not in runs or any(t not in runs for t in targets):
        print(f"ERROR: acquisitions {runs} do not include {r_from} and "
              f"{targets}", file=sys.stderr)
        return 2

    v = {k: np.array([M.get((i, k), np.nan) for i in rois]) for k in runs}
    v_from = v[r_from]

    # --- the baseline's own interval, computed once -------------------------
    ci_lo = ci_hi = None
    if args.method == "baseline-bootstrap":
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
        ev_path = auc_dir / "events.csv"
        ledger = args.ledger
        if ledger is None:
            for cand in (auc_dir / "frame_ledger.csv",
                         auc_dir.parent / "frame_ledger.csv"):
                if cand.exists():
                    ledger = cand
                    break
        if fs is None or not ev_path.exists() or ledger is None \
                or not Path(ledger).exists():
            print("ERROR: this method needs events.csv, a frame ledger and a "
                  "frame rate.\n  Pass --ledger and --fs, or use "
                  "--method threshold.", file=sys.stderr)
            return 2
        events = read_events(ev_path)
        segs = []
        with open(ledger) as fh:
            for r in csv.DictReader(fh):
                segs.append((int(r["frame_start"]), int(r["frame_end"]) + 1))
        a1, b1 = segs[r_from - 1]
        dur1 = (b1 - a1) / fs / 60.0
        areas1 = event_areas(events, a1, b1)

        ci_lo = np.zeros(n_roi)
        ci_hi = np.zeros(n_roi)
        n_ev1 = np.zeros(n_roi, int)
        for k, roi in enumerate(rois):
            aa = areas1.get(roi, [])
            n_ev1[k] = len(aa)
            _, lo, hi = bootstrap_interval(aa, dur1, args.n_boot, args.alpha,
                                           seed=k)
            ci_lo[k], ci_hi[k] = lo, hi
        med_n = int(np.median(n_ev1))
        cov = coverage_at(med_n)
        print(f"baseline is acquisition {r_from}: median {med_n} events per ROI")
        print(f"  a nominal {int((1 - args.alpha) * 100)}% interval covers a "
              f"repeat about {cov:.0f}% of the time at this count, so the test "
              "is\n  somewhat liberal: it calls change a little more often "
              "than stated")
        thin = int((n_ev1 < 5).sum())
        if thin:
            print(f"  {thin} ROI(s) had fewer than 5 events in the baseline; "
                  "their intervals rest on\n  very few values and those "
                  "classifications are weak")
        detail = (f"{int((1 - args.alpha) * 100)}% interval for a repeat of "
                  f"acquisition {r_from} (covers ~{cov:.0f}%)")
    else:
        detail = (f"a change of more than {args.threshold * 100:.0f}% counts, "
                  "which is a choice, not a measurement")

    # --- classify each target against that same baseline --------------------
    results = []
    print(f"\n{'vs':>4} {'increased':>10} {'unchanged':>10} {'decreased':>10}")
    for t in targets:
        v_to = v[t]
        if args.method == "baseline-bootstrap":
            sig = (v_to < ci_lo) | (v_to > ci_hi)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                sig = np.abs((v_to - v_from) / np.abs(v_from)) > args.threshold
        up = sig & (v_to > v_from)
        down = sig & (v_to < v_from)
        counts = [int(up.sum()), int((~sig).sum()), int(down.sum())]
        results.append({"to": t, "v_to": v_to, "up": up, "down": down,
                        "counts": counts})
        print(f"{t:>4} {counts[0]:10d} {counts[1]:10d} {counts[2]:10d}")

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

    names = ["increased", "unchanged", "decreased"]
    cols = [args.color_up, args.color_same, args.color_down]
    n_panel = len(results)
    n_row = 1 if args.no_scatter else 2
    panel_w = 3.5
    fig, axes = plt.subplots(n_row, n_panel,
                             figsize=(panel_w * n_panel, 3.4 * n_row + 0.6),
                             squeeze=False)

    # one scale across every scatter, so the panels compare with each other as
    # well as with the baseline
    allv = np.concatenate([v_from] + [r["v_to"] for r in results])
    if args.log_scatter:
        pos = allv[allv > 0]
        base = [pos.min(), pos.max()] if pos.size else [1e-3, 1.0]
        span = np.log10(base[1] / max(base[0], 1e-12))
        pad = max(span * 0.08, 0.05)
        lim = [base[0] * 10 ** (-pad), base[1] * 10 ** pad]
    else:
        vmax = float(np.nanmax(allv))
        vmin = float(min(0.0, np.nanmin(allv)))
        # half a marker width converted from points into data units, so a point
        # sitting on an axis is drawn as a circle and not as a half circle
        frac = (np.sqrt(args.point_size) / 2 + 2.0) / (panel_w * 0.72 * 72)
        pad = max(vmax - vmin, 1e-9) * max(frac, 0.035)
        lim = [vmin - pad, vmax + pad]

    for j, res in enumerate(results):
        counts = res["counts"]
        keep = [c > 0 for c in counts]
        ax = axes[0, j]
        ax.pie([c for c, k in zip(counts, keep) if k],
               labels=[f"{n}\n{c}/{n_roi}"
                       for n, c, k in zip(names, counts, keep) if k],
               colors=[c for c, k in zip(cols, keep) if k],
               autopct="%1.0f%%", startangle=90, counterclock=False,
               wedgeprops={"edgecolor": "white", "linewidth": 1.2},
               textprops={"fontsize": 8})
        ax.set_title(f"acquisition {r_from} to {res['to']}", fontsize=10)

        if args.no_scatter:
            continue
        ax = axes[1, j]
        if ci_lo is not None:
            o = np.argsort(v_from)
            ax.fill_between(v_from[o], ci_lo[o], ci_hi[o], color="0.88",
                            alpha=0.8, lw=0, zorder=0)
        ax.plot(lim, lim, ls="--", color="0.6", lw=1.0, zorder=1)
        colour = np.where(res["up"], args.color_up,
                          np.where(res["down"], args.color_down,
                                   args.color_same))
        ax.scatter(v_from, res["v_to"], s=args.point_size, c=colour,
                   edgecolor="0.3", linewidth=0.5, zorder=3, clip_on=False)
        if args.log_scatter:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f"acquisition {r_from}")
        if j == 0:
            ax.set_ylabel("later acquisition")
        else:
            ax.tick_params(labelleft=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    if not args.no_scatter:
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], ls="--", color="0.6", label="no change")]
        if ci_lo is not None:
            handles.append(plt.Rectangle((0, 0), 1, 1, color="0.88",
                                         label="repeat of the baseline"))
        axes[1, 0].legend(handles=handles, fontsize=7, frameon=False,
                          loc="upper left")

    title = args.label or auc_dir.name
    fig.suptitle(f"{title}   baseline = acquisition {r_from}   {detail}",
                 fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    with open(out.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["roi", "from_run", "to_run", "v_from", "v_to", "ratio",
                    "ci_lo", "ci_hi", "classification"])
        for res in results:
            for i, roi in enumerate(rois):
                cls = ("increased" if res["up"][i] else
                       "decreased" if res["down"][i] else "unchanged")
                ratio = (res["v_to"][i] / v_from[i]
                         if abs(v_from[i]) > 1e-12 else np.nan)
                w.writerow([roi, r_from, res["to"], round(v_from[i], 4),
                            round(res["v_to"][i], 4),
                            (None if not np.isfinite(ratio) else round(ratio, 4)),
                            (None if ci_lo is None else round(ci_lo[i], 4)),
                            (None if ci_hi is None else round(ci_hi[i], 4)),
                            cls])
    with open(out.with_suffix(".json"), "w") as fh:
        json.dump({"auc_dir": str(auc_dir), "value": args.value,
                   "from_run": r_from, "to_runs": targets, "n_roi": n_roi,
                   "method": args.method, "detail": detail,
                   "counts": {str(r["to"]): dict(zip(names, r["counts"]))
                              for r in results}}, fh, indent=2)
    print(f"\nwrote {out.with_suffix('.png')}, .pdf, .csv and .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
