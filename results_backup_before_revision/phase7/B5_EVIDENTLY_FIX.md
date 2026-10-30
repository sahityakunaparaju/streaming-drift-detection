# B5 (Evidently) Integration — Fix Documentation

## Outcome

**B5 now runs on the real Evidently API and produces real, non-zero
DRIFT/NO_DRIFT decisions for all 110 windows of the committed Phase 7
seed-42 test set** (0 UNAVAILABLE). The row in
`results/phase7/tables/ladder.csv` and the `ladder` block in
`results/phase7/results.json` are updated.

Important: B5's default-config numbers are degenerate (flags every
window). This is a REAL Evidently behavior, not an integration
failure — see "Default-config caveat" below. The paper must report it
that way.

## Root cause (two stacked bugs + one environment issue)

### 1. Package not installed — and the install was broken by a PyPI namespace conflict

`pip show evidently` → not installed. Installing `evidently==0.7.21`
pulled `litestar`, which imports the classic `multipart` API
(`MultipartSegment`, `PushMultipartParser`, ...). The `multipart`
package was republished on PyPI as a breaking rewrite (1.x/2.0.0)
that no longer exports those names. The environment ALSO has
`python-multipart==0.0.6` (a dependency of fastapi/starlette) which
installs a `multipart/` package directory that shadows the `multipart.py`
module from the `multipart` wheel. Net effect:

```
ImportError: cannot import name 'MultipartSegment' from 'multipart'
  (C:\Python312\Lib\site-packages\multipart\__init__.py)
```

This is known upstream: litestar-org/litestar issue #4000,
pypa/packaging-problems issue #818.

**Fix:** downgraded `litestar` to `2.13.0` (self-contained
`_multipart.py`, no `multipart` dependency; still satisfies
evidently's `litestar>=2.8.3`) and pinned `evidently==0.6.7` (the
last version with the `evidently.metric_preset` + `evidently.report`
API the B5 code targets). `python-multipart` left untouched.

### 2. API moved — `evidently.metric_preset` does not exist in 0.7.x

`evidently 0.7.21` restructured the public API
(`evidently.presets` / `evidently.sdk`); `evidently.metric_preset`
and `evidently.report` were removed. Resolved by pinning 0.6.7.

### 3. Two real bugs in `evaluation/baseline.py::run_b5_evidently`

a) **Wrong result index.** It read `summary["metrics"][0]["result"]`
   and looked for `drift_by_columns` there, but the DataDriftPreset
   emits the per-column breakdown in the metric whose result carries
   that key (metrics[1] in 0.6.x). Fixed to scan all metrics for the
   one containing `drift_by_columns`.

b) **Column-shape crash on schema windows.** `numeric_columns` was
   computed from `reference_df` only; a schema-`removed` window lacks
   those columns and crashed with
   `KeyError: "['temp'] not in index"`. Fixed to intersect with
   `current_df.columns` (Evidently requires matching schemas).

## Exact numbers on the committed seed-42 test set (110 windows)

| metric | raw | adjusted |
|---|---|---|
| Precision | 0.5455 | 0.5319 |
| Recall | 1.0000 | 1.0000 |
| F1 | 0.7059 | 0.6944 |
| FPR | 1.0000 | 1.0000 |
| TP / FP / TN / FN | 60 / 50 / 0 / 0 | 50 / 44 / 0 / 0 |
| flagged detected / missed | 16 / 0 | — |
| lag (mean/median/p95) | 0.0 / 0.0 / 0.0 (16/16 streams detected) | — |

## Default-config caveat (must be in the paper)

With Evidently's default `DataDriftPreset` (KS-based per-column
drift at the default threshold), **every window is flagged as DRIFT,
including the clean and normal_noise controls** — FPR = 1.0. Verified
this is inherent Evidently behavior, not an integration artifact: 3
random 1000-row slices drawn from the reference dataset ITSELF all
report `dataset_drift=True, 5/5 columns drifted`. Cause: the
reference distribution is built from the full 405,184-row dataset
while windows are 1,000 rows; the default two-sample tests have
enormous power at that size ratio and reject the null for any
sampling variation.

Consequences for the paper:
- B5 with default config is an always-DRIFT baseline on this dataset;
  it cannot discriminate drift and its ladder row is trivially
  R=1.0 / FPR=1.0.
- To make B5 a meaningful comparator, Evidently must be calibrated
  for the reference/window size ratio (e.g. subsample the reference
  to window scale, or raise `num_stattest_threshold` /
  `drift_share`). That is a baseline-configuration decision, not an
  integration fix, and was left untouched per scope.

## Files changed

- `evaluation/baseline.py` — `run_b5_evidently`: scan all metrics for
  `drift_by_columns`; intersect numeric columns with current frame.
- `results/phase7/tables/ladder.csv` — B5 row now real numbers.
- `results/phase7/results.json` — B5 ladder block now real numbers.
- Environment: `evidently==0.6.7`, `litestar==2.13.0` (pinned).
