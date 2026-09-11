"""Synthetic self-test: frame_clock edge detection and per-frame averaging."""
import numpy as np, fig_roi_traces as G

fs_rec, n_frames, fs_img = 5000.0, 400, 13.29
# square-wave frame clock with known edge times
t = np.arange(int(n_frames/fs_img*fs_rec)+50)/fs_rec
edges_true = np.arange(n_frames)/fs_img + 0.0137          # arbitrary head pad
clk = np.zeros_like(t)
for e in edges_true:
    i = int(e*fs_rec); clk[i:i+8] = 5.0
ft = G.frame_times_from_clock(clk, fs_rec)
assert ft.size == n_frames, (ft.size, n_frames)
assert np.abs(ft - edges_true).max() < 2/fs_rec, np.abs(ft-edges_true).max()
print(f"PASS  frame_clock edges: {ft.size} found, max error "
      f"{np.abs(ft-edges_true).max()*1e3:.2f} ms (head pad {ft[0]*1e3:.1f} ms recovered)")

# robust_sd must ignore transients
rng = np.random.default_rng(0)
x = rng.normal(0, 2.0, (5, 20000))
assert abs(np.median(G.robust_sd(x)) - 2.0)/2.0 < 0.05
x2 = x.copy(); x2[:, ::500] += 60            # sparse large transients
plain = x2.std(axis=1)
rob = G.robust_sd(x2)
assert np.median(rob) < 2.3 and np.median(plain) > 3.0, (np.median(rob), np.median(plain))
print(f"PASS  robust_sd ignores transients ({np.median(rob):.2f} vs plain {np.median(plain):.2f})")

# rolling baseline on flat data invents nothing
flat = np.full((4, 4000), 1000.0, np.float32) + rng.normal(0,1,(4,4000))
d = (flat - G.rolling_baseline(flat, int(45*fs_img)))/1000*100
assert np.percentile(d, 99.5) < 5
print("PASS  rolling baseline flat -> no transients")

# dead-channel detector
assert G.looks_inactive({"min":0.0037,"max":0.0519,"mean":0.0297,"sd":0.0031})
assert not G.looks_inactive({"min":0.0,"max":3.3,"mean":1.1,"sd":1.2})
assert G.looks_inactive(None)
print("PASS  looks_inactive flags the 0.05 V-span channel, passes a 0-3.3 V one")
print("\nALL PASS")
