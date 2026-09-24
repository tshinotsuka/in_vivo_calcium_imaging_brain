"""The core module against known ground truth, and against duplication."""
import ast
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ivcalc import events, traces  # noqa: E402

fs = 13.2908
CORE = {"rolling_baseline", "percentile_baseline", "fixed_baseline",
        "robust_sd", "roi_edge", "_z_of", "z_of", "dff", "auc_of",
        "matched_filter", "exp_smooth", "detect_events", "auc_per_min"}


def test_no_duplicate_definitions():
    """No script may redefine a core function.

    This is the test that keeps the package honest. Seven copies of the rolling
    baseline had drifted into six implementations before it existed, so the
    correction applied to the analysis never reached the figures.
    """
    repo = Path(__file__).resolve().parents[1]
    offenders = []
    for f in sorted((repo / "scripts").rglob("*.py")):
        try:
            tree = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for n in tree.body:
            if isinstance(n, ast.FunctionDef) and n.name in CORE:
                offenders.append(f"{f.relative_to(repo)}::{n.name}")
    assert not offenders, (
        "these redefine a core function instead of importing it from ivcalc:\n  "
        + "\n  ".join(offenders))


def test_percentile_bias_removed():
    rng = np.random.default_rng(0)
    sigma, F0 = 40.0, 800.0
    flat = np.full((30, 6000), F0, np.float32) + rng.normal(0, sigma, (30, 6000))
    raw = traces.percentile_baseline(flat, int(45 * fs), 10.0, correct_bias=False)
    cor = traces.percentile_baseline(flat, int(45 * fs), 10.0, correct_bias=True)
    off_raw = float(np.mean(traces.dff(flat, raw)))
    off_cor = float(np.mean(traces.dff(flat, cor)))
    assert off_raw > 4.0, off_raw
    assert abs(off_cor) < 0.6, off_cor
    # and the offset must not grow as the preparation dims
    dim = np.full((30, 6000), F0 * 0.6, np.float32) + rng.normal(0, sigma, (30, 6000))
    off_dim = float(np.mean(traces.dff(
        dim, traces.percentile_baseline(dim, int(45 * fs), 10.0, False))))
    assert off_dim > off_raw * 1.4, (off_dim, off_raw)
    off_dim_c = float(np.mean(traces.dff(
        dim, traces.percentile_baseline(dim, int(45 * fs), 10.0, True))))
    assert abs(off_dim_c - off_cor) < 0.6


def test_robust_sd_recovers_sigma():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 7.5, (20, 8000))
    assert abs(np.median(traces.robust_sd(x)) - 7.5) / 7.5 < 0.05


def test_sd_from_values_valid_after_smoothing():
    """The difference-based estimate collapses on a smoothed trace."""
    rng = np.random.default_rng(2)
    x = rng.normal(0, 10.0, (10, 8000)).astype(np.float32)
    sm = traces.matched_filter(x, 0.27 * fs)
    assert np.median(traces.robust_sd(sm)) < 0.5 * np.median(traces.sd_from_values(sm))


def test_offset_threshold_is_below_the_mean():
    """Events must not end where an exponential decay is still above baseline."""
    rng = np.random.default_rng(3)
    kern = np.exp(-np.arange(70) / (0.27 * fs)); kern[0] = 0
    d = np.empty((10, 8000), np.float32)
    for i in range(10):
        ev = np.zeros(8000); ev[rng.choice(8000, 40, replace=False)] = 1
        d[i] = 35 * np.convolve(ev, kern, "same") + rng.normal(0, 8, 8000)
    sd = traces.robust_sd(d)
    dur = [(b - a) / fs for i in range(10)
           for a, b in events._scan(d[i] - np.median(d[i]), float(sd[i]),
                                    3.0, 0.5, +1)]
    assert np.median(dur) > 0.5, np.median(dur)


def test_events_recover_injected_area():
    rng = np.random.default_rng(4)
    n_roi, n_t = 6, int(300 * fs)
    amp, w = 60.0, int(1.5 * fs)
    d = rng.normal(0, 2.0, (n_roi, n_t))
    for o in np.linspace(100, n_t - 200, 10).astype(int):
        d[:, o:o + w] += amp
    ev, st = events.detect(d, fs)
    got = events.rate_per_min(ev, n_roi, 0, n_t, fs).mean()
    want = 10 * amp * (w / fs) / (n_t / fs / 60)
    assert abs(got - want) / want < 0.2, (got, want)


def test_noise_yields_nothing():
    rng = np.random.default_rng(5)
    noise = rng.normal(0, 2.0, (6, int(300 * fs)))
    _, st = events.detect(noise, fs)
    assert st["n_events_kept"] == 0, st["n_events_kept"]
    # and the null really is symmetric
    r = st["n_negative_raw"] / max(st["n_positive_raw"], 1)
    assert 0.6 < r < 1.6, r


def test_matched_filter_helps_at_low_snr():
    rng = np.random.default_rng(6)
    n_t = int(300 * fs)
    kern = np.exp(-np.arange(70) / (0.27 * fs)); kern[0] = 0
    d = np.empty((8, n_t), np.float32)
    for i in range(8):
        ev = np.zeros(n_t); ev[rng.choice(n_t, 25, replace=False)] = 1
        sig = np.convolve(ev, kern, "same")
        d[i] = 2.5 * 10.0 * sig / sig.max() + rng.normal(0, 10.0, n_t)
    plain, _ = events.detect(d, fs)
    filt, _ = events.detect(traces.matched_filter(d, 0.27 * fs), fs)
    # at this signal-to-noise the unfiltered traces yield nothing at all
    assert len(plain) == 0 and len(filt) > 10, (len(plain), len(filt))
    # and the filter must not manufacture events from noise alone
    pure = np.random.default_rng(7).normal(0, 10.0, (8, n_t)).astype(np.float32)
    ev_n, _ = events.detect(traces.matched_filter(pure, 0.27 * fs), fs)
    assert len(ev_n) == 0, len(ev_n)


if __name__ == "__main__":
    import traceback
    ok = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
                ok += 1
            except AssertionError as e:
                print(f"FAIL  {name}: {e}")
            except Exception:  # noqa: BLE001
                print(f"ERROR {name}")
                traceback.print_exc()
    print(f"\n{ok} passed")
