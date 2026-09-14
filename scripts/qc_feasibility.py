#!/usr/bin/env python
"""Feasibility QC for a calcium imaging preparation.

Answers one question: is this recording usable for calcium imaging at all?
No drug, no conditions, no group comparison -- just whether real neural signal
is present and how good it is.

The decisive test is panel 5. ROIs and traces can be produced from pure noise;
what cannot be produced from noise is correlation structure. Under anaesthesia
cortical populations are strongly synchronised, so neighbouring cells should
correlate far more than time-shuffled surrogates. If they do not, the traces are
noise regardless of how many ROIs were detected.

Panels
------
1. Mean image, ROI map, ROI size distribution in pixels. Somata spanning fewer
   than roughly 8 px make anatomical detection unreliable and indicate the
   pixel size should be checked against the ScanImage header.
2. Detector range: fraction of saturated and of near-zero pixels. Clipping at
   the top censors bright transients; a floor at zero indicates the PMT offset
   or gain needs adjusting.
3. nu, the standardised noise level: median absolute frame-to-frame difference
   of dF/F divided by sqrt(fs), in %/sqrt(Hz). This is the number that sets the
   achievable sensitivity and the input to any later power calculation.
4. Photostability: raw F over the recording, as a percentage change from the
   first minute. Separates bleaching from a genuine baseline.
5. Signal reality: distribution of pairwise correlations between ROIs, against
   time-shuffled surrogates, plus the somatic-to-neuropil correlation.
6. Transients: example traces, peak dF/F distribution, and the fraction of ROIs
   showing at least one excursion above k * nu.

Example
-------
    python qc_feasibility.py \
        --s2p-dir <dataset>/work/s2p/suite2p/plane0 \
        --fs 13.0107 \
        --out <dataset>/results/qc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# helpers
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


def dff(F: np.ndarray, F0: np.ndarray, floor: float = 1.0) -> np.ndarray:
    return (F - F0) / np.maximum(F0, floor)


def nu_of(dff_pct: np.ndarray, fs: float) -> np.ndarray:
    """Standardised noise in %/sqrt(Hz). Input must be dF/F in percent."""
    return np.median(np.abs(np.diff(dff_pct, axis=1)), axis=1) / np.sqrt(fs)


def phase_randomised(X: np.ndarray, rng) -> np.ndarray:
    """Surrogate preserving each row's power spectrum but destroying phase.

    This is the surrogate the asymmetry test needs. Preserving the spectrum
    preserves the autocorrelation, so a surrogate has the same apparent time
    constants as the data; only the waveform shape is destroyed. Noise that
    merely happens to be smooth therefore cannot pass, while a genuine
    fast-rise / slow-decay calcium transient can.
    """
    n = X.shape[1]
    Z = np.fft.rfft(X, axis=1)
    amp = np.abs(Z)
    ph = rng.uniform(0, 2 * np.pi, size=Z.shape)
    ph[:, 0] = 0.0
    if n % 2 == 0:
        ph[:, -1] = 0.0
    return np.fft.irfft(amp * np.exp(1j * ph), n=n, axis=1).astype(np.float32)


def circular_shuffle(X: np.ndarray, rng) -> np.ndarray:
    """Roll each row by an independent random lag.

    Destroys between-ROI synchrony while preserving each row entirely. Used
    only to characterise population state, never as a feasibility gate: a
    lightly anaesthetised or near-awake cortex is desynchronised, so low
    pairwise correlation is a statement about brain state, not about whether
    calcium signal is present.
    """
    out = np.empty_like(X)
    n = X.shape[1]
    for i in range(X.shape[0]):
        out[i] = np.roll(X[i], int(rng.integers(n // 10, n - n // 10)))
    return out


def skewness(X: np.ndarray) -> np.ndarray:
    """Per-row skewness. Calcium transients rise above baseline, so real
    traces are positively skewed; symmetric noise is not."""
    m = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd = np.maximum(sd, 1e-12)
    return (((X - m) / sd) ** 3).mean(axis=1)


def autocorr_halfwidth(X: np.ndarray, fs: float, max_lag_s: float = 3.0) -> np.ndarray:
    """Timescale of the correlated component, in seconds.

    Shot noise is white, so it contributes only at lag zero. Normalising the
    autocorrelation by its value at lag one therefore removes the noise floor
    and leaves the timescale of whatever structure is actually shared across
    frames. Returned value is the lag at which that normalised curve first
    falls below 0.5. White noise gives one frame; a genuine indicator
    transient gives something on the order of its decay constant.
    """
    max_lag = max(int(max_lag_s * fs), 3)
    Xc = X - X.mean(axis=1, keepdims=True)
    denom = np.maximum((Xc * Xc).sum(axis=1), 1e-12)

    lags = np.arange(1, max_lag)
    r = np.empty((X.shape[0], lags.size), dtype=np.float64)
    for j, lag in enumerate(lags):
        r[:, j] = (Xc[:, :-lag] * Xc[:, lag:]).sum(axis=1) / denom

    r1 = r[:, 0].copy()
    out = np.full(X.shape[0], np.nan, dtype=np.float32)
    # rows with no correlated component at all: report one frame
    dead = r1 <= 1e-6
    out[dead] = 1.0 / fs

    rn = r / np.maximum(r1[:, None], 1e-12)
    prev = np.ones(X.shape[0])
    for j, lag in enumerate(lags):
        cur = rn[:, j]
        hit = np.isnan(out) & (cur < 0.5)
        if hit.any():
            frac = (prev[hit] - 0.5) / np.maximum(prev[hit] - cur[hit], 1e-12)
            out[hit] = (lag - 1 + frac) / fs + 1.0 / fs
        prev = cur
    out[np.isnan(out)] = max_lag / fs
    return out


def upper_offdiag(C: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(C, k=1)
    v = C[iu]
    return v[np.isfinite(v)]


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------


def _roi_edge(entry, shape):
    """Boundary pixels of one ROI mask, for an outline that does not hide the cell."""
    m = np.zeros(shape, bool)
    m[entry["ypix"], entry["xpix"]] = True
    inner = (np.roll(m, 1, 0) & np.roll(m, -1, 0)
             & np.roll(m, 1, 1) & np.roll(m, -1, 1))
    return m & ~inner


def panel_rois(ops, stat, iscell, out: Path, clip=(1.0, 99.5)):
    """Mean image, ROI outlines over it, and the ROI size distribution.

    The middle panel outlines each ROI and numbers it rather than filling it in:
    the question this panel answers is whether the detected ROIs sit on the
    bright cells actually visible in the mean image, and a filled overlay hides
    exactly the pixels needed to judge that. The left panel is left clean for
    the same reason -- counting cells by eye there is the comparison that says
    whether detection is missing cells or the expression is sparse.
    """
    img = ops.get("meanImg")
    keep = iscell[:, 0].astype(bool)
    npix = np.array([len(s["xpix"]) for s in stat])
    diam = 2 * np.sqrt(npix / np.pi)  # equivalent-circle diameter, px

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    shape = img.shape if img is not None else (
        int(ops.get("Ly", 512)), int(ops.get("Lx", 512)))
    if img is not None:
        vmin, vmax = np.percentile(img, list(clip))
        for ax in axes[:2]:
            ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")

    n_keep = int(keep.sum())
    cmap = plt.get_cmap("turbo")
    colors = [cmap(v) for v in np.linspace(0.08, 0.92, max(n_keep, 1))]
    k = 0
    for s, ok in zip(stat, keep):
        if not ok:
            continue
        ys, xs = np.nonzero(_roi_edge(s, shape))
        axes[1].plot(xs, ys, ".", ms=1.6, color=colors[k], alpha=0.95)
        axes[1].text(float(np.mean(s["xpix"])) + 3, float(np.mean(s["ypix"])) - 3,
                     str(k + 1), color=colors[k], fontsize=7, fontweight="bold",
                     ha="left", va="bottom")
        k += 1

    axes[0].set_title(f"mean image (display {clip[0]:g}-{clip[1]:g} pct) "
                      "-- count the bright cells here")
    axes[1].set_title(f"accepted ROIs outlined  n={n_keep} / {len(stat)} detected")
    for ax in axes[:2]:
        ax.set_xlim(0, shape[1])
        ax.set_ylim(shape[0], 0)
        ax.axis("off")

    axes[2].hist(diam[keep], bins=25, color="0.35")
    axes[2].axvline(8, color="r", ls="--", lw=1.2)
    axes[2].set_xlabel("equivalent ROI diameter (px)")
    axes[2].set_ylabel("ROIs")
    axes[2].set_title(f"median {np.median(diam[keep]):.1f} px  (red = 8 px)")
    fig.tight_layout()
    fig.savefig(out / "fq1_rois.png", dpi=150)
    plt.close(fig)
    return {
        "n_detected": int(len(stat)),
        "n_accepted": int(keep.sum()),
        "median_roi_diameter_px": float(np.median(diam[keep])) if keep.any() else None,
        "frac_roi_under_8px": float((diam[keep] < 8).mean()) if keep.any() else None,
    }


def panel_range(ops, out: Path, bit_max: int):
    img = ops.get("meanImg")
    reg = ops.get("reg_file")
    stats = {"source": None}
    if reg and Path(reg).exists():
        mm = np.memmap(
            reg, dtype=np.int16, mode="r",
            shape=(int(ops["nframes"]), int(ops["Ly"]), int(ops["Lx"])),
        )
        idx = np.linspace(0, mm.shape[0] - 1, min(300, mm.shape[0]), dtype=int)
        sample = np.asarray(mm[idx]).astype(np.float32)
        del mm
        stats["source"] = "binary"
    elif img is not None:
        sample = img.astype(np.float32)[None]
        stats["source"] = "meanImg"
    else:
        return stats

    hi = float(np.percentile(sample, 99.99))
    sat = float((sample >= bit_max * 0.995).mean())
    zero = float((sample <= 0).mean())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(sample.ravel(), bins=200, color="0.35", log=True)
    axes[0].axvline(bit_max, color="r", ls="--", lw=1.2, label=f"full scale {bit_max}")
    axes[0].set_xlabel("pixel value")
    axes[0].set_ylabel("count (log)")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"saturated {sat*100:.3f}%   at/below zero {zero*100:.2f}%")

    axes[1].imshow(sample.max(axis=0), cmap="magma")
    axes[1].axis("off")
    axes[1].set_title("max projection (sampled frames)")
    fig.tight_layout()
    fig.savefig(out / "fq2_detector_range.png", dpi=150)
    plt.close(fig)

    stats.update({"p99_99": hi, "frac_saturated": sat, "frac_at_or_below_zero": zero})
    return stats


def panel_nu(nu: np.ndarray, out: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(nu, bins=40, color="0.35")
    med = float(np.median(nu))
    ax.axvline(med, color="r", lw=1.5)
    ax.set_xlabel(r"$\nu$  (%$\cdot$Hz$^{-1/2}$)")
    ax.set_ylabel("ROIs")
    ax.set_title(rf"standardised noise: median $\nu$ = {med:.2f}")
    fig.tight_layout()
    fig.savefig(out / "fq3_noise_level.png", dpi=150)
    plt.close(fig)
    return {
        "median": med,
        "q25": float(np.percentile(nu, 25)),
        "q75": float(np.percentile(nu, 75)),
    }


def panel_stability(F, Fneu, fs, out: Path):
    t = np.arange(F.shape[1]) / fs / 60.0
    mF = F.mean(0)
    ref = mF[: max(int(60 * fs), 1)].mean()
    pct = (mF - ref) / max(ref, 1.0) * 100.0

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    axes[0].plot(t, mF, lw=0.7, color="C0", label="F")
    axes[0].plot(t, Fneu.mean(0), lw=0.7, color="C1", label="Fneu")
    axes[0].set_ylabel("raw fluorescence")
    axes[0].legend(fontsize=8)
    axes[1].plot(t, pct, lw=0.7, color="C3")
    axes[1].axhline(0, color="0.5", ls="--", lw=0.8)
    axes[1].set_ylabel("change from first minute (%)")
    axes[1].set_xlabel("time (min)")
    axes[0].set_title("photostability")
    fig.tight_layout()
    fig.savefig(out / "fq4_photostability.png", dpi=150)
    plt.close(fig)

    tail = pct[-max(int(60 * fs), 1):].mean()
    return {"pct_change_end_vs_start": float(tail)}


def panel_signal_reality(d_pct, Fneu_d, nu, fs, out: Path, tau_s: float, seed: int = 0):
    """Is calcium signal present? Tests that do not assume a brain state.

    Population synchrony is deliberately not used here. A lightly anaesthetised
    or near-awake cortex is desynchronised, so low pairwise correlation would
    make a synchrony-based gate reject perfectly good data. Each test below is
    per-ROI and holds whether or not the population is coherent.
    """
    rng = np.random.default_rng(seed)
    surr = phase_randomised(d_pct, rng)

    sk_real = skewness(d_pct)
    sk_surr = skewness(surr)
    ac_real = autocorr_halfwidth(d_pct, fs)
    ac_surr = autocorr_halfwidth(surr, fs)

    sn = np.array([np.corrcoef(d_pct[i], Fneu_d[i])[0, 1] for i in range(d_pct.shape[0])])
    sn = sn[np.isfinite(sn)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    bins = np.linspace(min(sk_surr.min(), sk_real.min()), np.percentile(sk_real, 99.5) + 0.5, 50)
    axes[0].hist(sk_surr, bins=bins, alpha=0.6, density=True, color="0.6", label="phase-randomised")
    axes[0].hist(sk_real, bins=bins, alpha=0.6, density=True, color="C0", label="observed")
    axes[0].axvline(0, color="k", lw=0.8)
    axes[0].set_xlabel(r"skewness of $\Delta$F/F")
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"median {np.median(sk_real):.2f} vs surrogate {np.median(sk_surr):.2f}")

    axes[1].hist(ac_surr, bins=40, alpha=0.6, density=True, color="0.6", label="phase-randomised")
    axes[1].hist(ac_real, bins=40, alpha=0.6, density=True, color="C0", label="observed")
    axes[1].axvline(1.0 / fs, color="r", ls=":", lw=1.2, label="one frame")
    axes[1].axvline(tau_s, color="g", ls="--", lw=1.2, label=f"indicator {tau_s:g}s")
    axes[1].set_xlabel("autocorrelation half-width (s)")
    axes[1].legend(fontsize=7)
    axes[1].set_title(f"median {np.median(ac_real):.2f}s")

    axes[2].hist(sn, bins=40, color="0.35")
    axes[2].set_xlabel("soma - neuropil correlation")
    axes[2].set_ylabel("ROIs")
    axes[2].set_title(f"median {np.median(sn):.3f}")

    fig.tight_layout()
    fig.savefig(out / "fq5_signal_reality.png", dpi=150)
    plt.close(fig)

    return {
        "median_skew": float(np.median(sk_real)),
        "median_skew_surrogate": float(np.median(sk_surr)),
        "frac_roi_skew_above_surrogate_p99": float(
            (sk_real > np.percentile(sk_surr, 99)).mean()
        ),
        "median_autocorr_halfwidth_s": float(np.median(ac_real)),
        "median_autocorr_halfwidth_surrogate_s": float(np.median(ac_surr)),
        "one_frame_s": float(1.0 / fs),
        "indicator_tau_s": tau_s,
        "median_soma_neuropil_corr": float(np.median(sn)),
    }


def panel_brain_state(d_pct, fs, out: Path, seed: int = 0):
    """Characterise the population state. Descriptive, never a pass/fail gate.

    High synchrony indicates deep anaesthesia with up/down structure; low
    synchrony indicates a light plane or a near-awake animal. Both are
    recordable; the number is reported so the state is on record rather than
    assumed.
    """
    rng = np.random.default_rng(seed)
    C = np.corrcoef(d_pct)
    real = upper_offdiag(C)
    surr = upper_offdiag(np.corrcoef(circular_shuffle(d_pct, rng)))
    pop = d_pct.mean(axis=0)

    t = np.arange(pop.size) / fs
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    show = slice(0, min(pop.size, int(120 * fs)))
    axes[0].plot(t[show], pop[show], lw=0.7, color="C0")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel(r"population mean $\Delta$F/F (%)")
    axes[0].set_title("population trace (first 2 min)")

    bins = np.linspace(-0.4, 1.0, 60)
    axes[1].hist(surr, bins=bins, alpha=0.6, density=True, color="0.6", label="shuffled")
    axes[1].hist(real, bins=bins, alpha=0.6, density=True, color="C0", label="observed")
    axes[1].set_xlabel("pairwise correlation")
    axes[1].legend(fontsize=8)
    axes[1].set_title(f"synchrony: median r = {np.median(real):.3f}")

    freqs = np.fft.rfftfreq(pop.size, 1.0 / fs)
    psd = np.abs(np.fft.rfft(pop - pop.mean())) ** 2
    m = (freqs > 0.01) & (freqs < fs / 2)
    axes[2].loglog(freqs[m], psd[m], lw=0.7, color="C3")
    axes[2].set_xlabel("frequency (Hz)")
    axes[2].set_ylabel("power")
    axes[2].set_title("population spectrum")

    fig.tight_layout()
    fig.savefig(out / "fq7_brain_state.png", dpi=150)
    plt.close(fig)

    lo = (freqs > 0.1) & (freqs < 1.0)
    hi = (freqs >= 1.0) & (freqs < min(5.0, fs / 2))
    ratio = float(psd[lo].sum() / max(psd[hi].sum(), 1e-12))
    return {
        "median_pairwise_corr": float(np.median(real)),
        "median_shuffled_corr": float(np.median(surr)),
        "synchrony_excess": float(np.median(real) - np.median(surr)),
        "slow_to_fast_power_ratio": ratio,
        "interpretation": (
            "high synchrony and slow-dominated power suggest a deep plane with "
            "up/down structure; low synchrony suggests a light plane or a "
            "near-awake animal. Not a feasibility criterion."
        ),
    }


def panel_transients(d_pct, nu, fs, out: Path, k: float, n_show: int):
    peak = np.percentile(d_pct, 99.5, axis=1)
    snr = peak / np.maximum(nu, 1e-9)
    thresh = k * nu[:, None]
    active = (d_pct > thresh).any(axis=1)

    idx = np.argsort(-snr)[:n_show]
    t = np.arange(d_pct.shape[1]) / fs

    fig = plt.figure(figsize=(16, 7))
    ax0 = fig.add_subplot(1, 2, 1)
    for j, i in enumerate(idx):
        ax0.plot(t, d_pct[i] + j * 60, lw=0.5)
    ax0.set_xlabel("time (s)")
    ax0.set_ylabel(r"$\Delta$F/F (%), offset")
    ax0.set_title(f"{len(idx)} ROIs with the highest peak / $\\nu$")

    ax1 = fig.add_subplot(2, 2, 2)
    ax1.hist(peak, bins=40, color="0.35")
    ax1.set_xlabel(r"99.5th percentile $\Delta$F/F (%)")
    ax1.set_ylabel("ROIs")
    ax1.set_title(f"median peak {np.median(peak):.1f}%")

    ax2 = fig.add_subplot(2, 2, 4)
    ax2.hist(snr, bins=40, color="0.35")
    ax2.axvline(k, color="r", ls="--", lw=1.2)
    ax2.set_xlabel(r"peak $\Delta$F/F / $\nu$")
    ax2.set_ylabel("ROIs")
    ax2.set_title(f"active {active.mean()*100:.0f}% of ROIs at {k:g}x $\\nu$")
    fig.tight_layout()
    fig.savefig(out / "fq6_transients.png", dpi=150)
    plt.close(fig)

    return {
        "median_peak_dff_pct": float(np.median(peak)),
        "median_peak_over_nu": float(np.median(snr)),
        "active_fraction": float(active.mean()),
        "threshold_k": k,
    }


# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Feasibility QC for calcium imaging",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--s2p-dir", type=Path, required=True, help="suite2p/plane0 directory")
    p.add_argument("--fs", type=float, default=None, help="frame rate in Hz; resolved from --dataset when omitted")
    p.add_argument("--dataset", type=Path, default=None, help="workspace dataset dir, used to read raw/metadata.yaml")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--neucoeff", type=float, default=0.7)
    p.add_argument("--baseline-window-s", type=float, default=45.0)
    p.add_argument("--baseline-percentile", type=float, default=10.0)
    p.add_argument("--tau", type=float, default=0.27, help="indicator decay time constant in s (GCaMP8s ~0.27)")
    p.add_argument("--transient-k", type=float, default=5.0, help="threshold in units of nu for calling an ROI active")
    p.add_argument("--n-example-traces", type=int, default=12)
    p.add_argument("--bit-max", type=int, default=32767, help="detector full scale in raw units")
    p.add_argument("--all-roi", action="store_true", help="use every ROI, not only iscell-accepted ones")
    p.add_argument("--tau-tolerance", type=float, default=4.0,
                   help="how many indicator decay times the autocorrelation may "
                        "span before the dominant fluctuation is too slow to be "
                        "a calcium transient")
    p.add_argument("--min-skew", type=float, default=1.0,
                   help="minimum median skewness; calcium traces rise above "
                        "baseline and fall back, so they are positively skewed")
    p.add_argument("--img-clip", type=float, nargs=2, default=[1.0, 99.5],
                   metavar=("LO_PCT", "HI_PCT"),
                   help="display percentiles for the mean image; widen to see dim cells")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if args.fs is None:
        if args.dataset is None:
            print("ERROR: pass --fs or --dataset (to read raw/metadata.yaml)", file=sys.stderr)
            return 2
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from run_roi_suite2p import resolve_from_metadata
        info = resolve_from_metadata(args.dataset.expanduser().resolve() / "raw" / "metadata.yaml")
        args.fs = info.get("fs_hz")
        if args.fs is None:
            print("ERROR: frame rate not found in metadata.yaml", file=sys.stderr)
            return 2
        print(f"fs resolved from metadata: {args.fs}")

    s2p = args.s2p_dir.expanduser().resolve()
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    F = np.load(s2p / "F.npy")
    Fneu = np.load(s2p / "Fneu.npy")
    stat = np.load(s2p / "stat.npy", allow_pickle=True)
    iscell = np.load(s2p / "iscell.npy")
    ops = {}
    for cand in ("reg_outputs.npy", "detect_outputs.npy", "ops.npy", "db.npy"):
        if (s2p / cand).exists():
            ops.update(np.load(s2p / cand, allow_pickle=True).item())
    if not ops:
        print(f"WARNING: neither ops.npy nor db.npy in {s2p}; "
              "mean image and binary-based panels will be skipped", file=sys.stderr)

    keep = np.ones(F.shape[0], bool) if args.all_roi else iscell[:, 0].astype(bool)
    if keep.sum() < 5:
        print("FAIL: fewer than 5 accepted ROIs. Nothing to assess.")
        return 1

    Fk = F[keep].astype(np.float32)
    Fnk = Fneu[keep].astype(np.float32)
    n_roi, n_frames = Fk.shape
    dur_min = n_frames / args.fs / 60.0
    print(f"ROIs {n_roi} ({'all' if args.all_roi else 'accepted'})   frames {n_frames}   {dur_min:.1f} min at {args.fs} Hz")

    win = int(round(args.baseline_window_s * args.fs))

    # traces
    Fc = Fk - args.neucoeff * Fnk
    d = dff(Fc, rolling_baseline(Fc, win, args.baseline_percentile)) * 100.0
    dn = dff(Fnk, rolling_baseline(Fnk, win, args.baseline_percentile)) * 100.0
    nu = nu_of(d, args.fs)

    res = {
        "s2p_dir": str(s2p),
        "fs_hz": args.fs,
        "n_frames": int(n_frames),
        "duration_min": float(dur_min),
        "neucoeff": args.neucoeff,
    }
    res["rois"] = panel_rois(ops, stat[keep], iscell[keep], out, clip=tuple(args.img_clip))
    res["detector_range"] = panel_range(ops, out, args.bit_max)
    res["nu"] = panel_nu(nu, out)
    res["photostability"] = panel_stability(Fk, Fnk, args.fs, out)
    res["signal_reality"] = panel_signal_reality(d, dn, nu, args.fs, out, args.tau, args.seed)
    res["brain_state"] = panel_brain_state(d, args.fs, out, args.seed)
    res["transients"] = panel_transients(d, nu, args.fs, out, args.transient_k, args.n_example_traces)

    # --- verdict -------------------------------------------------------------
    checks = []
    sr = res["signal_reality"]
    checks.append((
        "waveform asymmetry",
        sr["frac_roi_skew_above_surrogate_p99"] > 0.30,
        f"{sr['frac_roi_skew_above_surrogate_p99']*100:.0f}% of ROIs exceed the "
        f"phase-randomised 99th pct (skew {sr['median_skew']:.2f} vs {sr['median_skew_surrogate']:.2f})",
    ))
    # The timescale has to sit between two bounds, not merely above one.
    # Anything as fast as a single frame is noise; anything far slower than the
    # indicator's own decay is not a calcium transient either, whatever else it
    # may be. A lower bound alone passes slow baseline wander, which is exactly
    # what a detector built for fast transients will then fail to find.
    _ac = sr["median_autocorr_halfwidth_s"]
    _hi = args.tau * args.tau_tolerance
    checks.append((
        "temporal structure",
        (_ac > 3.0 / args.fs) and (_ac <= _hi),
        f"autocorr half-width {_ac:.2f}s; expected between "
        f"{3.0 / args.fs:.2f}s (one frame) and {_hi:.2f}s "
        f"({args.tau_tolerance:g}x the {args.tau:g}s indicator decay)"
        + ("" if _ac <= _hi else
           f" -- {_ac / args.tau:.0f}x the indicator, so the dominant "
           "fluctuation is not a calcium transient"),
    ))
    checks.append((
        "transient shape",
        sr["median_skew"] >= args.min_skew,
        f"median skewness {sr['median_skew']:.2f}, expected at least "
        f"{args.min_skew:g} for traces dominated by rise-and-decay transients",
    ))
    tr = res["transients"]
    checks.append((
        "transients present",
        tr["median_peak_over_nu"] > args.transient_k,
        f"median peak / nu = {tr['median_peak_over_nu']:.1f}, active {tr['active_fraction']*100:.0f}%",
    ))
    checks.append((
        "ROI size adequate",
        (res["rois"]["median_roi_diameter_px"] or 0) >= 8,
        f"median ROI diameter {res['rois']['median_roi_diameter_px']:.1f} px",
    ))
    dr = res["detector_range"]
    if "frac_saturated" in dr:
        checks.append((
            "no heavy saturation",
            dr["frac_saturated"] < 0.01,
            f"saturated {dr['frac_saturated']*100:.3f}%",
        ))
    ps = res["photostability"]["pct_change_end_vs_start"]
    checks.append((
        "photostable",
        ps > -25.0,
        f"raw F change over recording {ps:+.1f}%",
    ))

    print("\n--- feasibility ---")
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'CHECK'}] {name:28s} {detail}")
    res["checks"] = [{"name": n, "pass": bool(o), "detail": dd} for n, o, dd in checks]

    bs = res["brain_state"]
    print("\n--- population state (descriptive, not a criterion) ---")
    print(f"  median pairwise r      {bs['median_pairwise_corr']:+.3f}  (shuffled {bs['median_shuffled_corr']:+.3f})")
    print(f"  slow/fast power ratio  {bs['slow_to_fast_power_ratio']:.1f}")
    print("  high synchrony + slow-dominated power -> deep plane with up/down structure")
    print("  low synchrony -> light plane or near-awake; both are recordable")

    with open(out / "feasibility_summary.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {out / 'feasibility_summary.json'} and 7 figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
