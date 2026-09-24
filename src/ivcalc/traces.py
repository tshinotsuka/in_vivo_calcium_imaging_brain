"""Trace processing: baselines, noise, dF/F, filtering.

Single source for every quantity that turns raw fluorescence into something
comparable. These functions were previously copied into each script that needed
them, and the copies drifted: seven copies of the rolling baseline had become
six different implementations, so a correction applied to the analysis did not
reach the figures drawn from it. Anything that computes a baseline, a noise
level or a dF/F imports it from here.
"""

from __future__ import annotations

import numpy as np

# Standard-normal quantiles for the percentiles a baseline is usually taken at,
# so the bias correction works without scipy.
_Z = {1.0: -2.3263, 5.0: -1.6449, 8.0: -1.4051, 10.0: -1.2816,
      15.0: -1.0364, 20.0: -0.8416, 25.0: -0.6745, 50.0: 0.0}


def z_of(pct: float) -> float:
    """Standard-normal quantile at a percentile."""
    try:
        from scipy.stats import norm
        return float(norm.ppf(pct / 100.0))
    except ImportError:
        return _Z[min(_Z, key=lambda x: abs(x - pct))]


def robust_sd(F: np.ndarray) -> np.ndarray:
    """Noise scale per row, from the median absolute frame-to-frame difference.

    Transients inflate a plain standard deviation, so an active trace would
    appear noisier than a quiet one and be held to a higher threshold. The
    difference-based estimate is insensitive to them.

    Not valid on a trace that has already been smoothed: smoothing correlates
    neighbouring samples, the differences shrink, and this reads far below the
    true spread. Use `sd_from_values` there.
    """
    d = np.abs(np.diff(np.asarray(F, float), axis=1))
    return np.median(d, axis=1) / (np.sqrt(2) * 0.6745)


def sd_from_values(F: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Noise scale per row from the spread of the values themselves.

    Valid whether or not the trace has been filtered, but it needs the events
    excluded (via `mask`) or it measures the activity as well as the noise.
    """
    F = np.asarray(F, float)
    out = np.empty(F.shape[0], float)
    for i in range(F.shape[0]):
        v = F[i] if mask is None else F[i][~mask[i]]
        if v.size < 10:
            v = F[i]
        out[i] = 1.4826 * np.median(np.abs(v - np.median(v)))
        if not np.isfinite(out[i]) or out[i] <= 0:
            out[i] = max(robust_sd(F[i][None, :])[0], 1e-9)
    return out


def percentile_baseline(F: np.ndarray, win: int, pct: float = 10.0,
                        correct_bias: bool = True) -> np.ndarray:
    """Percentile baseline in a sliding window, corrected for percentile bias.

    A low percentile of a noisy trace lands about |z| * sigma BELOW the resting
    level, so dF/F acquires a constant positive offset of |z| * sigma / F0. That
    offset is not activity, and because F0 shrinks with photobleaching while
    sigma does not shrink as fast, it GROWS over a long recording. Uncorrected,
    a synthetic recording whose activity fell 40% reported a 14% RISE. Adding
    |z| * sigma back removes it.

    A median (pct=50) carries no such bias and is left alone.
    """
    F = np.asarray(F, np.float32)
    n = F.shape[1]
    win = max(int(win), 3)
    step = max(win // 4, 1)
    centres = np.arange(0, n, step)
    vals = np.empty((F.shape[0], centres.size), np.float32)
    for j, c in enumerate(centres):
        a, b = max(0, c - win // 2), min(n, c + win // 2 + 1)
        vals[:, j] = np.percentile(F[:, a:b], pct, axis=1)
    out = np.empty_like(F)
    for i in range(F.shape[0]):
        out[i] = np.interp(np.arange(n), centres, vals[i])
    if correct_bias and pct < 50:
        out = out + (abs(z_of(pct)) * robust_sd(F))[:, None]
    return out


def fixed_baseline(F: np.ndarray, a: int, b: int, pct: float = 10.0,
                   correct_bias: bool = True) -> np.ndarray:
    """One baseline per row from frames [a, b), with the same bias correction."""
    F = np.asarray(F, np.float32)
    v = np.percentile(F[:, a:b], pct, axis=1).astype(np.float32)
    if correct_bias and pct < 50:
        v = v + abs(z_of(pct)) * robust_sd(F[:, a:b])
    return v[:, None]


def baseline_per_segment(F: np.ndarray, segs, win: int, pct: float = 10.0,
                         correct_bias: bool = True) -> np.ndarray:
    """Rolling baseline computed within each acquisition, never across a join.

    A window spanning a boundary would smear a step in brightness -- from
    refocusing, or a changed laser setting -- across both acquisitions, turning
    an instrument change into an apparent slow drift in each.
    """
    F = np.asarray(F, np.float32)
    out = np.empty_like(F)
    for _, a, b in segs:
        out[:, a:b] = percentile_baseline(F[:, a:b], win, pct, correct_bias)
    return out


def dff(F: np.ndarray, F0: np.ndarray, floor: float = 1.0,
        percent: bool = True) -> np.ndarray:
    """Relative fluorescence change. The floor guards a baseline near zero."""
    d = (np.asarray(F, float) - np.asarray(F0, float)) / np.maximum(F0, floor)
    return d * 100.0 if percent else d


def nu(d_pct: np.ndarray, fs: float) -> np.ndarray:
    """Noise standardised for frame rate, in percent per root hertz.

    Dividing by the square root of the frame rate makes the number comparable
    between recordings taken at different rates: the photon count per unit time
    is what the optics set, not the count per frame.
    """
    return np.median(np.abs(np.diff(np.asarray(d_pct, float), axis=1)),
                     axis=1) / np.sqrt(fs)


def mean_dff(d_pct: np.ndarray, fs: float) -> np.ndarray:
    """Integrated dF/F per second of recording: percent-seconds per second."""
    d_pct = np.asarray(d_pct, float)
    return d_pct.sum(axis=1) / fs / (d_pct.shape[1] / fs)


def matched_filter(x: np.ndarray, tau_frames: float) -> np.ndarray:
    """Correlate each trace with the indicator's own transient shape.

    A calcium transient rises within a frame and decays with the indicator's
    time constant, so noise that is white across frames is suppressed by about
    the square root of the number of frames the decay spans while the transient
    survives. At 13 Hz with a 0.27 s decay that is roughly a factor of two,
    which is the largest gain available without changing the acquisition.

    Applied symmetrically: a causal filter would delay every onset by about one
    time constant.
    """
    x = np.asarray(x, np.float32)
    if tau_frames <= 0:
        return x
    n = max(int(np.ceil(6 * tau_frames)), 3)
    h = np.exp(-np.arange(n) / tau_frames)
    h = h / h.sum()          # unit area, so a slow transient keeps its amplitude
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        v = np.pad(x[i], n, mode="reflect")
        out[i] = np.convolve(v, h[::-1], mode="same")[n:-n]
    return out


def exp_smooth(x: np.ndarray, tau_frames: float) -> np.ndarray:
    """Causal exponential smoothing, as in the published event method."""
    x = np.asarray(x, np.float32)
    if tau_frames <= 0:
        return x
    a = float(np.exp(-1.0 / tau_frames))
    out = np.empty_like(x)
    out[:, 0] = x[:, 0]
    for t in range(1, x.shape[1]):
        out[:, t] = a * out[:, t - 1] + (1 - a) * x[:, t]
    return out


def smooth(x: np.ndarray, kind: str, *, tau_s: float, fs: float) -> np.ndarray:
    """Dispatch to a filter by name, so callers share one vocabulary."""
    if kind == "matched":
        return matched_filter(x, tau_s * fs)
    if kind == "exp":
        return exp_smooth(x, tau_s * fs)
    if kind in ("none", None):
        return np.asarray(x, np.float32)
    raise ValueError(f"unknown smoothing {kind!r}")
