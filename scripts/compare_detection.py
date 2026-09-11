#!/usr/bin/env python
"""Compare every detection run under a dataset's work/ directory.

Reads each ``work/s2p_*/roi_run_record.json``, optionally runs the feasibility
QC for runs that lack it, and prints one table plus a CSV.

The column that decides the question is ``diam_px`` against ``soma_px``. A run
whose ROIs are far smaller than a cell body has found neuropil fragments, and
its ROI count is meaningless however large it is. A second check catches the
opposite failure: if the accepted ROIs would cover more of the field than a
plausible cell density allows, the detector is over-segmenting.

Usage:
    python compare_detection.py <dataset_dir> [--soma-um 15] [--qc]
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def load_qc(qc_json: Path) -> dict:
    try:
        return json.load(open(qc_json))
    except Exception:  # noqa: BLE001
        return {}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="compare suite2p detection runs")
    p.add_argument("dataset", type=Path)
    p.add_argument("--soma-um", type=float, default=15.0)
    p.add_argument("--qc", action="store_true",
                   help="run qc_feasibility.py for any run that has no QC yet")
    p.add_argument("--qc-script", default="qc_feasibility.py")
    args = p.parse_args(argv)

    D = args.dataset.expanduser().resolve()
    work = D / "work"
    runs = sorted(d for d in work.glob("s2p*") if (d / "roi_run_record.json").exists())
    if not runs:
        print(f"no runs with roi_run_record.json under {work}", file=sys.stderr)
        return 2

    rows = []
    for r in runs:
        rec = json.load(open(r / "roi_run_record.json"))
        tag = r.name.replace("s2p_", "").replace("s2p", "default")
        qc_dir = work / f"qc_{r.name}"
        qc_json = qc_dir / "feasibility_summary.json"

        if args.qc and not qc_json.exists():
            plane = r / "suite2p" / "plane0"
            print(f"running QC for {tag} ...")
            subprocess.run(
                [sys.executable, args.qc_script, "--s2p-dir", str(plane),
                 "--dataset", str(D), "--out", str(qc_dir)],
                capture_output=True, text=True,
            )
        q = load_qc(qc_json)

        det = (rec.get("settings") or {}).get("detection", {})
        sp = det.get("sparsery_settings", {})
        cp = det.get("cellpose_settings", {})
        px = rec.get("pixel_size_um")
        ny, nx = rec.get("frame_shape_px", [0, 0])

        n_acc = rec.get("n_roi_iscell") or 0
        diam = rec.get("median_roi_diameter_px") or (q.get("rois") or {}).get("median_roi_diameter_px")
        npix = rec.get("median_roi_npix")
        soma_px = (args.soma_um / px) if px else None
        coverage = (n_acc * npix / (ny * nx)) if (npix and ny and nx) else None

        rows.append({
            "run": tag,
            "algo": det.get("algorithm"),
            "scale": sp.get("spatial_scale") if det.get("algorithm") == "sparsery" else cp.get("img"),
            "diameter_set": rec.get("diameter_px"),
            "thresh": det.get("threshold_scaling"),
            "n_detected": rec.get("n_roi_detected"),
            "n_accepted": n_acc,
            "diam_px": round(diam, 1) if diam else None,
            "npix": npix,
            "coverage": round(coverage, 2) if coverage else None,
            "nu": round((q.get("nu") or {}).get("median", float("nan")), 2) if q else None,
            "pairwise_r": round((q.get("brain_state") or {}).get("median_pairwise_corr", float("nan")), 3) if q else None,
            "peak_over_nu": round((q.get("transients") or {}).get("median_peak_over_nu", float("nan")), 1) if q else None,
        })

    px = rows and json.load(open(runs[0] / "roi_run_record.json")).get("pixel_size_um")
    soma_px = (args.soma_um / px) if px else None
    soma_npix = 3.14159 * (soma_px / 2) ** 2 if soma_px else None

    hdr = ["run", "algo", "scale", "diameter_set", "thresh", "n_detected",
           "n_accepted", "diam_px", "npix", "coverage", "nu", "pairwise_r", "peak_over_nu"]
    widths = {h: max(len(h), *(len(str(r.get(h))) for r in rows)) for h in hdr}
    print()
    if soma_px:
        print(f"pixel size {px:.3g} um/px -> a {args.soma_um:g} um soma is "
              f"{soma_px:.1f} px across, {soma_npix:.0f} px in area")
        print("diam_px near that value means somata; far below means fragments.")
        print("coverage = accepted ROIs x npix / frame area; above ~0.5 is implausible.")
    print()
    print("  ".join(h.ljust(widths[h]) for h in hdr))
    print("  ".join("-" * widths[h] for h in hdr))
    for r in rows:
        print("  ".join(str(r.get(h)).ljust(widths[h]) for h in hdr))

    if soma_px:
        print()
        ok = [r for r in rows
              if r["diam_px"] and 0.6 * soma_px <= r["diam_px"] <= 1.6 * soma_px
              and (r["coverage"] or 0) < 0.5]
        if ok:
            best = max(ok, key=lambda r: r["n_accepted"])
            print(f"cell-body-sized and plausibly covered: "
                  f"{', '.join(r['run'] for r in ok)}")
            print(f"most ROIs among those: {best['run']} "
                  f"({best['n_accepted']} accepted, {best['diam_px']} px)")
        else:
            print("No run produced cell-body-sized ROIs at plausible coverage.")
            print("At this pixel size the somata may simply be too small to")
            print("segment; a higher zoom on the next acquisition is the lever.")

    out = work / "detection_comparison.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=hdr)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
