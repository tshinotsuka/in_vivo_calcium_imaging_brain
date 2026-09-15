"""Synthetic self-test: event detection and AUC/min against known ground truth."""
import numpy as np
import run_event_auc as E

fs = 13.2908
rng = np.random.default_rng(0)

# --- 1. AUC/min recovers a known injected area -----------------------------
# rectangular transients of known area, well above noise
n_roi, n_t = 6, int(300*fs)
amp, dur_s, n_ev = 60.0, 1.5, 10           # 60 %dF/F for 1.5 s, 10 times
d = rng.normal(0, 2.0, (n_roi, n_t))
onsets = np.linspace(100, n_t-200, n_ev).astype(int)
w = int(dur_s*fs)
for i in range(n_roi):
    for o in onsets:
        d[i, o:o+w] += amp
ev, kept, st = E.detect_events(d, fs, min_dur_s=0.5)
auc = E.auc_per_min(ev, n_roi, 0, n_t, fs)
expect = n_ev * amp * (w/fs) / (n_t/fs/60)
print(f"injected {n_ev} x {amp:.0f}%dF/F x {w/fs:.2f}s over {n_t/fs/60:.1f} min")
print(f"  expected AUC/min {expect:.1f}   measured {auc.mean():.1f} "
      f"({auc.mean()/expect*100:.0f}%)")
assert abs(auc.mean()-expect)/expect < 0.20, (auc.mean(), expect)
assert st["n_events_kept"] >= n_roi*n_ev*0.9, st["n_events_kept"]
print("PASS  AUC/min recovers the injected area")

# --- 2. pure noise yields (almost) nothing ---------------------------------
noise = rng.normal(0, 2.0, (n_roi, n_t))
ev_n, _, st_n = E.detect_events(noise, fs, min_dur_s=0.5)
auc_n = E.auc_per_min(ev_n, n_roi, 0, n_t, fs)
print(f"  pure noise: {st_n['n_events_kept']} events kept of "
      f"{st_n['n_positive_raw']} raw, AUC/min {auc_n.mean():.2f}")
assert auc_n.mean() < 0.05*expect, auc_n.mean()
print("PASS  false-positive control rejects noise")

# --- 3. the null really is symmetric ---------------------------------------
assert abs(st_n['n_positive_raw']-st_n['n_negative_raw'])/max(st_n['n_positive_raw'],1) < 0.35, st_n
print(f"PASS  positive and negative excursions balance in noise "
      f"({st_n['n_positive_raw']} vs {st_n['n_negative_raw']})")

# --- 4. AUC is insensitive to a rising noise floor -------------------------
# same events, but the second half has 2x the noise: threshold-free integration
# would inflate, event-based should not
d2 = d.copy()
d2[:, n_t//2:] += rng.normal(0, 2.0, (n_roi, n_t-n_t//2))
ev2, _, _ = E.detect_events(d2, fs, min_dur_s=0.5)
a1 = E.auc_per_min(ev2, n_roi, 0, n_t//2, fs).mean()
a2 = E.auc_per_min(ev2, n_roi, n_t//2, n_t, fs).mean()
naive1 = d2[:, :n_t//2].mean()*60
naive2 = d2[:, n_t//2:].mean()*60
print(f"  noise doubles in the 2nd half: event AUC {a1:.0f} -> {a2:.0f} "
      f"({(a2-a1)/a1*100:+.0f}%)")
assert abs(a2-a1)/a1 < 0.25, (a1, a2)
print("PASS  event-based AUC is robust to a changing noise floor")

# --- 5. baseline / smoothing helpers ---------------------------------------
flat = np.full((4, 3000), 1000.0, np.float32) + rng.normal(0, 30, (4, 3000))
b = E.percentile_filter(flat, int(15*fs), 50.0)
assert abs(np.median(b)-1000) < 5, np.median(b)
print(f"PASS  median baseline unbiased ({np.median(b):.1f} vs 1000)")
sm = E.exp_smooth(rng.normal(0,1,(3,5000)).astype(np.float32), 0.2*fs)
assert sm.std() < 0.6
print("PASS  exponential smoothing reduces variance")
print("\nALL PASS")

# --- 6. matched filtering in a regime where detection actually fails -------
print("\n--- low-SNR regime (peak transient = 2.5x the per-frame noise) ---")

def low_snr(seed, peak_over_noise=2.5, n_ev=15, nr=8):
    r = np.random.default_rng(seed)
    kern = np.exp(-np.arange(70)/(0.27*fs)); kern[0]=0; kern[:2]=np.linspace(0,1,2)
    sigma = 10.0
    out = np.empty((nr, n_t))
    for i in range(nr):
        ev = np.zeros(n_t); ev[r.choice(n_t, n_ev, replace=False)] = 1
        sig = np.convolve(ev, kern, 'same')
        out[i] = peak_over_noise*sigma*sig/sig.max() + r.normal(0, sigma, n_t)
    return out.astype(np.float32), nr

D, nr = low_snr(7)
res = {}
for lab in ('none','exp','matched'):
    Ds = (E.exp_smooth(D, 0.2*fs) if lab=='exp'
          else E.matched_filter(D, 0.27*fs) if lab=='matched' else D)
    for meth in ('bin','cumulative'):
        ev,_,st = E.detect_events(Ds, fs, min_dur_s=0.5, fp_method=meth)
        a = E.auc_per_min(ev, nr, 0, n_t, fs)
        res[(lab,meth)] = st['n_events_kept']
        print(f"  {lab:8s} {meth:11s} events {st['n_events_kept']:4d}/{nr*15}  "
              f"ROIs>0 {int((a>0).sum()):2d}/{nr}  AUC {a.mean():7.1f}")
assert res[('matched','cumulative')] > res[('none','bin')], res
print(f"PASS  matched + cumulative finds more of the injected events "
      f"({res[('none','bin')]} -> {res[('matched','cumulative')]} of {nr*15})")

# and must not manufacture events from noise
pure = np.random.default_rng(11).normal(0, 10.0, (nr, n_t)).astype(np.float32)
for lab in ('none','matched'):
    Ps = E.matched_filter(pure, 0.27*fs) if lab=='matched' else pure
    for meth in ('bin','cumulative'):
        ev,_,st = E.detect_events(Ps, fs, min_dur_s=0.5, fp_method=meth)
        print(f"  pure noise {lab:8s} {meth:11s}: {st['n_events_kept']:3d} kept "
              f"of {st['n_positive_raw']:4d} raw")
        assert st['n_events_kept'] <= 3, (lab, meth, st['n_events_kept'])
print("PASS  neither smoothing nor the cumulative test manufactures events")
print("\nALL PASS")
