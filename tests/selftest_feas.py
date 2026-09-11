"""Synthetic self-test: the gate must accept DESYNCHRONISED real signal
and reject smooth noise, i.e. it must not depend on brain state."""
import numpy as np
import qc_feasibility as Q

fs = 13.0
rng = np.random.default_rng(3)
n_roi, n_t = 80, 8000
tau_s = 0.27


def ca_traces(sync: float, seed: int):
    """Calcium-like traces: fast rise, slow decay. `sync` = shared event fraction."""
    r = np.random.default_rng(seed)
    kern = np.exp(-np.arange(0, 6 * tau_s * fs) / (tau_s * fs))
    kern[0] = 0.0
    kern[:2] = np.linspace(0, 1, 2)          # ~1 frame rise
    shared = np.zeros(n_t)
    shared[r.choice(n_t, 60, replace=False)] = 1.0
    out = np.empty((n_roi, n_t))
    for i in range(n_roi):
        own = np.zeros(n_t)
        own[r.choice(n_t, 60, replace=False)] = 1.0
        ev = sync * shared + (1 - sync) * own
        out[i] = 40 * np.convolve(ev, kern, mode="same") + r.normal(0, 3, n_t)
    return out


def smooth_noise(seed: int):
    """Gaussian noise low-passed to the SAME timescale as the calcium traces.
    Symmetric waveform: this is what the asymmetry test must reject."""
    from scipy.ndimage import gaussian_filter1d
    r = np.random.default_rng(seed)
    return gaussian_filter1d(r.normal(0, 20, (n_roi, n_t)), tau_s * fs, axis=1)


def gate(D, seed=0):
    r = np.random.default_rng(seed)
    sur = Q.phase_randomised(D, r)
    sk, sks = Q.skewness(D), Q.skewness(sur)
    frac = float((sk > np.percentile(sks, 99)).mean())
    ac = float(np.median(Q.autocorr_halfwidth(D, fs)))
    sync = float(np.median(Q.upper_offdiag(np.corrcoef(D))))
    return frac, ac, sync


print(f"{'case':28s} {'skew-frac':>10s} {'ac_hw(s)':>9s} {'sync r':>8s}   verdict")
results = {}
for label, D, expect in [
    ("synchronised Ca (deep)", ca_traces(0.9, 11), True),
    ("desynchronised Ca (awake)", ca_traces(0.0, 12), True),
    ("smooth noise, matched tau", smooth_noise(13), False),
]:
    frac, ac, sync = gate(D)
    ok = frac > 0.30 and ac > 3.0 / fs
    print(f"{label:28s} {frac:10.2f} {ac:9.2f} {sync:8.3f}   {'PASS' if ok else 'reject'}")
    assert ok == expect, (label, frac, ac)
    results[label] = sync

# the decisive property: the gate is state-independent
assert results["desynchronised Ca (awake)"] < 0.05, results
print("\nPASS  desynchronised real signal accepted despite near-zero synchrony "
      f"(r = {results['desynchronised Ca (awake)']:+.3f})")
print("PASS  smooth noise with matched autocorrelation rejected")

# --- phase randomisation preserves the spectrum ---------------------------
X = ca_traces(0.5, 21)
S = Q.phase_randomised(X, np.random.default_rng(0))
pr = np.abs(np.fft.rfft(X, axis=1)); ps = np.abs(np.fft.rfft(S, axis=1))
assert np.allclose(pr, ps, rtol=1e-5, atol=1e-5)
print("PASS  phase_randomised preserves the power spectrum exactly")

# --- autocorr half-width sanity ------------------------------------------
white = np.random.default_rng(0).normal(0, 1, (20, 5000))
assert np.median(Q.autocorr_halfwidth(white, fs)) < 2.0 / fs
print("PASS  white noise decorrelates within one frame")

# --- nu ---------------------------------------------------------------------
s = 4.0
exp = 0.6745 * s * np.sqrt(2) / np.sqrt(fs)
got = float(np.median(Q.nu_of(np.random.default_rng(0).normal(0, s, (100, 20000)), fs)))
assert abs(got - exp) / exp < 0.03
print(f"PASS  nu_of ({got:.3f} vs {exp:.3f})")

print("\nALL PASS")
