"""Significant calcium transients, after Magnus et al. 2019.

The published method in one place, so the offset-sign correction and the
false-positive controls apply wherever events are counted.

Detection: an event begins at `onset_sd` standard deviations above the mean and
ends only once the trace has fallen `offset_sd` standard deviations BELOW it.
The offset threshold is on the far side of baseline deliberately, so an event is
not terminated by a decay that is still well above rest. Reading it as being
above the mean cuts every event short: in synthetic data the fraction passing a
0.5 s minimum fell from 74% to 20%, and marginal events were kept or dropped
according to how the noise happened to fall.

Significance: noise and axial motion push the trace up and down about equally,
so downward excursions are a null distribution measured from the same trace
rather than assumed. Three controls are available, because the published one
needs more events than a short recording provides.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np

from .traces import robust_sd, sd_from_values


@dataclass
class Event:
    roi: int
    onset: int
    offset: int
    amplitude: float      # peak dF/F, percent
    duration_s: float
    area: float           # integral of dF/F, percent-seconds
    area_df: float = 0.0  # the same window in raw units, a.u.-seconds


def _scan(trace, sd, on_k, off_k, sign):
    """Excursions crossing on_k*sd that end only past -off_k*sd."""
    x = np.asarray(trace, float) * sign
    hi, lo = on_k * sd, -off_k * sd
    above = x > hi
    spans, i, n = [], 0, x.size
    while i < n:
        if not above[i]:
            i += 1
            continue
        a = i
        while a > 0 and x[a - 1] > lo:
            a -= 1
        b = i
        while b < n - 1 and x[b + 1] > lo:
            b += 1
        spans.append((a, b + 1))
        i = b + 1
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def detect(d_pct, fs, *, df_raw=None, onset_sd=3.0, offset_sd=0.5,
           min_duration_s=0.5, fp_max=0.05, fp_method="cumulative",
           amp_bin_sd=0.5, dur_bin_s=0.25, min_bin_count=5, roi_alpha=0.05,
           n_passes=2):
    """Significant transients per ROI, plus the statistics behind the decision.

    fp_method:
      bin         the published amplitude-by-duration test. Needs enough events
                  per bin to estimate a rate; with a short recording a bin
                  holding one positive and one negative reads as 100% and is
                  discarded, while one positive and none reads as 0% and is kept.
      cumulative  one amplitude threshold fitted to all events at once, far
                  better determined when events are scarce.
      none        no population control; the per-ROI test below is the only one.

    The per-ROI test runs regardless. The population controls hold the rate over
    all ROIs together, which leaves individual ROIs unprotected: silent cells,
    and ROIs that are not cells, contribute only false positives while the
    active cells keep the pooled rate low. Within one ROI the null is simple --
    up and down excursions are equally likely -- so a binomial test on its own
    counts asks whether it has more upward excursions than chance. An ROI that
    fails contributes no events, and so a rate of zero rather than a spurious one.
    """
    d_pct = np.asarray(d_pct, np.float32)
    n_roi, n_t = d_pct.shape
    sd = robust_sd(d_pct)
    mask = np.zeros_like(d_pct, bool)
    pos_all, neg_all = [], []

    for p in range(max(n_passes, 1)):
        if p > 0:
            # Recompute from event-free frames so large transients do not raise
            # the bar against themselves. Values, not differences: a smoothed
            # trace has correlated neighbours and a difference-based estimate
            # collapses.
            sd = sd_from_values(d_pct, mask)
        pos_all, neg_all = [], []
        mask = np.zeros_like(d_pct, bool)
        for i in range(n_roi):
            s = float(max(sd[i], 1e-9))
            tr = d_pct[i] - float(np.median(d_pct[i]))
            for a, b in _scan(tr, s, onset_sd, offset_sd, +1):
                pos_all.append((i, a, b, float(tr[a:b].max()) / s,
                                (b - a) / fs, float(tr[a:b].sum()) / fs))
                mask[i, a:b] = True
            for a, b in _scan(tr, s, onset_sd, offset_sd, -1):
                neg_all.append((i, a, b, float(-tr[a:b].min()) / s,
                                (b - a) / fs, float(-tr[a:b].sum()) / fs))
                mask[i, a:b] = True

    def binned(evs):
        out = {}
        for e in evs:
            if e[4] < min_duration_s:
                continue
            out.setdefault((int(e[3] / amp_bin_sd), int(e[4] / dur_bin_s)),
                           []).append(e)
        return out

    pb, nb = binned(pos_all), binned(neg_all)
    kept_bins, rates, cum_thresh = set(), {}, None

    if fp_method == "none":
        kept_bins = set(pb)
    elif fp_method == "cumulative":
        pos = np.array([e[3] for e in pos_all if e[4] >= min_duration_s])
        neg = np.array([e[3] for e in neg_all if e[4] >= min_duration_s])
        cum_thresh = float("inf")
        for a in (np.unique(np.round(np.sort(pos), 2)) if pos.size
                  else np.array([])):
            npos = int((pos >= a).sum())
            if npos >= 5 and int((neg >= a).sum()) / max(npos, 1) < fp_max:
                cum_thresh = float(a)
                break
        for key, evs in pb.items():
            if any(e[3] >= cum_thresh for e in evs):
                kept_bins.add(key)
                rates[key] = 0.0
            else:
                rates[key] = 1.0
    else:
        for key, evs in pb.items():
            if len(evs) < min_bin_count:
                sp = sum(len(v) for k, v in pb.items() if k[0] == key[0])
                sn = sum(len(v) for k, v in nb.items() if k[0] == key[0])
                fp = sn / max(sp, 1)
            else:
                fp = len(nb.get(key, [])) / max(len(evs), 1)
            rates[key] = fp
            if fp < fp_max:
                kept_bins.add(key)

    kept = [e for key in kept_bins for e in pb[key]
            if cum_thresh is None or e[3] >= cum_thresh]

    rejected_rois = set()
    if roi_alpha and roi_alpha > 0:
        thr = cum_thresh if cum_thresh is not None else 0.0
        for i in range(n_roi):
            npos = sum(1 for e in kept if e[0] == i)
            nneg = sum(1 for e in neg_all
                       if e[0] == i and e[4] >= min_duration_s and e[3] >= thr
                       and (fp_method != "bin"
                            or (int(e[3] / amp_bin_sd),
                                int(e[4] / dur_bin_s)) in kept_bins))
            n = npos + nneg
            if n == 0:
                continue
            pval = sum(comb(n, k) for k in range(npos, n + 1)) / (2.0 ** n)
            if pval > roi_alpha:
                rejected_rois.add(i)

    def df_area(i, a, b):
        return 0.0 if df_raw is None else float(df_raw[i, a:b].sum()) / fs

    events = [Event(roi=e[0], onset=e[1], offset=e[2],
                    amplitude=e[3] * float(sd[e[0]]),
                    duration_s=e[4], area=e[5], area_df=df_area(e[0], e[1], e[2]))
              for e in kept if e[0] not in rejected_rois]
    events.sort(key=lambda e: (e.roi, e.onset))

    stats = {
        "n_positive_raw": len(pos_all), "n_negative_raw": len(neg_all),
        "n_bins": len(pb), "n_bins_kept": len(kept_bins),
        "n_events_kept": len(events),
        "median_sd_pct": float(np.median(sd)),
        "fp_method": fp_method, "roi_alpha": roi_alpha,
        "n_roi_rejected": len(rejected_rois),
        "min_duration_s": min_duration_s,
        "cumulative_amplitude_threshold_sd": (
            None if cum_thresh is None else round(cum_thresh, 3)),
        "false_positive_rates": {f"{k[0]}_{k[1]}": round(v, 4)
                                 for k, v in sorted(rates.items())},
    }
    return events, stats


def rate_per_min(events, n_roi, a, b, fs, field="area"):
    """Cumulative event area within [a, b), per minute, per ROI."""
    dur_min = (b - a) / fs / 60.0
    out = np.zeros(n_roi, float)
    for e in events:
        if e.onset >= a and e.offset <= b:
            out[e.roi] += getattr(e, field)
    return out / max(dur_min, 1e-9)
