# Moving a script onto ivcalc

Each script keeps its command-line interface and loses its private copies of
the core functions. The interfaces do not change, so nothing downstream has to
be touched at the same time.

## What to delete from a script, and what to import instead

| delete from the script | import instead |
|---|---|
| `rolling_baseline`, `percentile_filter` | `traces.percentile_baseline`, `traces.baseline_per_segment` |
| `fixed_baseline` | `traces.fixed_baseline` |
| `robust_sd` | `traces.robust_sd`, `traces.sd_from_values` |
| `_z_of`, `_Z` | `traces.z_of` |
| `dff` | `traces.dff` |
| `auc_of` | `traces.mean_dff` |
| `nu_of` | `traces.nu` |
| `matched_filter`, `exp_smooth` | `traces.smooth` |
| `detect_events`, `_scan`, `Event` | `events.detect`, `events.Event` |
| `auc_per_min` | `events.rate_per_min` |
| `roi_edge` | `viz.roi_edge` |
| `apply_style` and the figstyle try/except | `viz.apply_style` |
| `load_plane`, the ops.npy fallback loop | `io.load_plane` |
| `from run_roi_suite2p import resolve_from_metadata` | `io.resolve_fs`, `io.read_metadata` |
| the ledger reader | `io.load_ledger`, `io.find_ledger` |
| `open_binary` | `io.open_binary` |

## The header every script gets

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ivcalc import events, io, traces, viz
```

Once `pip install -e .` has been run in the repo the path line is unnecessary,
but leaving it in means a script still runs from a clone that has not been
installed.

## Behaviour that changes on migration

Three scripts were computing something different from the analysis they
illustrate, and adopting the package corrects them:

- `fig_roi_traces.py` and `qc_feasibility.py` use an **uncorrected** percentile
  baseline. Their dF/F carries a positive offset that grows as the recording
  dims. Figures drawn before the migration overstate activity late in a series.
- `qc_traces.py` and `run_auc.py` predate `run_event_auc.py` and duplicate it.
  Delete both; `run_event_auc.py --fp-method none` covers the threshold-free case.
- `run_roi_suite2p.py` and `mc_scanimage.py` belong to the CaImAn-first route,
  which `run_suite2p_series.py` replaced. Keep them only for the registration
  comparison, under `scripts/legacy/`.

## Order

1. `traces` first: it has no dependencies and the most copies.
2. `io` next: it removes the import of a script as a library.
3. `events`, then `viz`.
4. Delete the superseded scripts.
5. `pytest` — `test_no_duplicate_definitions` fails until a migration is complete,
   so it tells you what is left.
