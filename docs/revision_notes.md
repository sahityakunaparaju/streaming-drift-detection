# Revision notes — streaming drift detection

Date: 2026-10-29

This document records the methodological revision described in the review.
Everything here is measured, not targeted. Where the revision made a number
worse, it is reported as worse.

## Backups (made before any change)

- `results_backup_before_revision/` — full copy of the old `results/` tree.
- `configs/thresholds_v1_frozen.yaml` — the old frozen thresholds
  (`ks_floor=0.50`, `psi_scale=72.0`, `chi2_scale=32.0`,
  `decision_threshold=0.25`).

No old result file was deleted.

## Step 1 — Reference builder

New module `detectors/reference.py`, configured by `configs/reference.yaml`.
Three modes:

| mode | how a window is compared |
|---|---|
| `global` | one global reference sample (legacy) |
| `per_device` | window split by device; each slice vs its own device reference; slices < 30 rows skipped and logged |
| `rolling` | window vs the immediately preceding clean 1000-row block (falls back to `global` before any history) |

Categorical columns (`device`) are always compared globally: a device slice
contains only one device value, so a per-device chi-square test would be
meaningless. That is also why category shift is still detectable under
`per_device`.

### Row splits (all disjoint)

| split | rows | purpose |
|---|---|---|
| reference | `[250000, 325000)` | reference sample |
| calibration | `[325000, 385000)` | threshold derivation only |
| clean holdout | `[385000, 405184)` | out-of-sample clean FPR report |
| evaluation matrix | `[0, 100000)` | Phase 7 matrix windows |
| evaluation lag | `[120000, 250000)` | Phase 7 lag streams |

Calibration seed is `1234567`, deliberately not one of the ten evaluation
seeds `[42, 123, 7, 11, 2024, 314, 2718, 999, 555, 777]`.

### Step 1b measurement (clean windows only)

`experiments/measure_clean_noise.py`, 3 calibration seeds × 60 clean
windows × 7 numeric columns. Outputs:
`results/revision/clean_baseline_noise.csv`,
`clean_baseline_noise_summary.csv`, `clean_baseline_noise.png`.

Mean clean-window KS per column (lower is better):

| column | global | per_device | rolling |
|---|---:|---:|---:|
| co | 0.4234 | 0.9221 | **0.2201** |
| humidity | 0.2691 | 0.8506 | **0.2142** |
| light | 0.0169 | 0.0176 | **0.0149** |
| lpg | 0.4234 | 0.9279 | **0.2201** |
| motion | 0.0014 | 0.0047 | **0.0016** |
| smoke | 0.4234 | 0.9279 | **0.2201** |
| temp | 0.2575 | 0.8538 | **0.2357** |
| **mean** | 0.2593 | 0.6435 | **0.1610** |

Mean clean PSI (numeric columns): global ≈ 8.5, per_device ≈ 15.0,
rolling ≈ 2.2.

**Chosen mode: `rolling`** (lowest clean KS and PSI). It is the only mode
that is also *valid* for a local, time-ordered stream. `per_device` is much
worse, not because of an implementation bug, but because the window-level
data is quantized: a 1000-row window of one device is nearly a point mass
at a quantized sensor level, while that device's 75k-row reference spans
many levels.

Honest note on the target: the review asked for clean KS "well below 0.2".
Rolling reaches 0.161 on average but the gas-sensor columns remain ~0.22,
and no mode reaches below 0.2 for those columns. This is a property of the
dataset (slow regime shifts plus aggressive quantization: `co`/`lpg`/`smoke`
are the same underlying gas sensor and `smoke≈4·co`, `lpg≈1.5·co`). It was
not forced down by tuning.

## Step 2 — Per-column scoring

`detectors/statistical_detector.py` now computes, per column:

- KS statistic and p-value, PSI (diagnostics);
- standardized mean difference `smd = (mean_c - mean_r)/std_r`;
- Wasserstein distance normalized by `std_r`;
- `effect_size = max(|smd|, wasserstein_norm)`.

KS is deliberately **not** part of the effect size: because the telemetry is
quantized, a clean local window can have KS ≈ 1 against a neighbouring
quantization level while the actual distribution shift is tiny. KS and PSI
are reported for diagnosis.

Decision per column:

1. p-value passes Benjamini–Hochberg correction across all columns
   (`alpha=0.01`, `statistical.correction=benjamini-hochberg`);
2. effect size ≥ the column's clean-calibrated threshold.

A window is statistically drifted if **any** column triggers. The triggering
column names are stored in `details["trigger_columns"]` and written to every
per-window `rows.csv` (`trigger_columns` column).

Categorical columns use a chi-square p-value plus Cramer's V
`sqrt(chi2/(n·(k−1)))` as the scale-free effect. A 3-category `device` and a
16-category `wd` now use the same rule; there is no frozen `chi2_scale`.

Schema and semantic channels keep their event logic, but their scores are now
continuous (severity-weighted: high 1.0, medium 0.6, low 0.3) so the unified
score is meaningful rather than `schema_bool + stat_max + semantic_bool`.

### Final decision rule (single source of truth, mirrored in `detectors/unified.py`)

```
DRIFT if any of:
  * any schema event
  * any semantic contract event or value-contract violation
  * any column's p-value passes BH AND its effect >= its calibrated threshold
```

The old `unified.threshold = 0.5` branch was unreachable and has been
removed. `unified_score` remains a reporting aggregate (mean of the three
continuous channel scores).

## Step 3 — Threshold derivation

`experiments/tuning_pass.py` (1873 lines, tuned on a mix of clean and
drifted windows) is replaced by `experiments/derive_thresholds.py`
(`tuning_pass.py` is now a thin delegating shim).

Derivation: 79 consecutive clean 1000-row calibration windows, rolling mode,
no injected drift. Per-column threshold = 99th percentile of the clean
effect-size distribution (target clean per-column FPR ≈ 1%), with a small
floor of 0.02 to guard against zero thresholds.

| column | clean mean | clean p95 | clean max | threshold |
|---|---:|---:|---:|---:|
| co | 0.0815 | 0.2127 | 0.4114 | 0.3405 |
| device | 0.0035 | 0.0072 | 0.0084 | 0.0200 (floor) |
| humidity | 0.0749 | 0.1449 | 0.2328 | 0.2032 |
| light | 0.0299 | 0.0580 | 0.3428 | 0.3009 |
| lpg | 0.0842 | 0.2139 | 0.4254 | 0.3637 |
| motion | 0.0136 | 0.0840 | 0.1102 | 0.1016 |
| smoke | 0.0837 | 0.2139 | 0.4228 | 0.3594 |
| temp | 0.0821 | 0.1415 | 0.1782 | 0.1685 |

Clean holdout window FPR (rows 385000–405184, never used for derivation):
**0.0000 over 19 windows** (0 column triggers). Full audit:
`results/revision/threshold_derivation.json`,
`results/revision/thresholds_derived.csv`.

## Step 4 — Honest metrics and re-run

Raw metrics only: every window counts, including windows tagged
`expected_false_negative`. The flag is retained only as an analysis column.
The new revision runner (`experiments/run_revision_ladder.py`) never reads it
into a denominator.

### Ladder, raw, 10 seeds (mean ± std)

| config | precision | recall | F1 | FPR |
|---|---:|---:|---:|---:|
| B0_null | 1.000 ± 0.000 | 0.050 ± 0.000 | 0.095 ± 0.000 | 0.000 ± 0.000 |
| B1_schema | 1.000 ± 0.000 | 0.250 ± 0.000 | 0.400 ± 0.000 | 0.000 ± 0.000 |
| B2_statistical | 0.785 ± 0.056 | 0.492 ± 0.032 | 0.604 ± 0.035 | 0.164 ± 0.051 |
| B3_schema+statistical | 0.840 ± 0.043 | 0.705 ± 0.021 | 0.766 ± 0.025 | 0.164 ± 0.051 |
| B4_unified | 0.869 ± 0.033 | 0.937 ± 0.019 | 0.901 ± 0.015 | 0.172 ± 0.051 |

### Before → after (raw means)

| metric | B2 v1 | B2 v2 | B4 v1 | B4 v2 |
|---|---:|---:|---:|---:|
| precision | 0.9729 | 0.7848 | 0.9922 | 0.8689 |
| recall | 0.2333 | **0.4917** | 0.8333 | **0.9367** |
| F1 | 0.3761 | **0.6039** | 0.9055 | 0.9009 |
| FPR | 0.0080 | 0.1640 | 0.0080 | 0.1720 |

The statistical channel is no longer deaf: recall roughly doubles. B4 recall
also rises, but B4 F1 is essentially unchanged (0.9055 → 0.9009) because the
extra detections are partly paid for by clean false positives. Precision and
FPR got worse. This is the honest trade-off and is not hidden.

### Statistical subtype recall by severity (B2, raw, mean ± std over 10 seeds)

| subtype | low | medium | high |
|---|---:|---:|---:|
| mean_shift | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| variance_shift | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| distribution_shift | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.50 ± 0.50 |
| category_shift | 0.00 ± 0.00 | 0.90 ± 0.30 | 1.00 ± 0.00 |

Old 10-seed means (statistical detector, all severities, raw): mean_shift
0.10, variance_shift 0.02, distribution_shift 0.08, category_shift 0.96.

(With the full unified detector, distribution_shift is 0.30/0.20/0.60 at
low/medium/high and category_shift 0.00/0.90/1.00, because the semantic
range check sometimes catches the shape-transform outliers. The statistical
channel alone is the honest number for "can the statistical detector see
this?".)

### Where the revision is still weak (plainly)

1. **Distribution shift** remains mostly undetected by the statistical
   channel. `inject_distribution_shift` applies a monotone power transform
   around the mean and then *rescales to preserve the standard deviation*;
   the mean and normalized Wasserstein effect therefore stay around
   0.05–0.15, below the calibrated clean-noise band (temp threshold 0.1685).
   KS does move (0.24–0.35), but clean rolling KS is 0.22 on average with a
   95th percentile of 0.32, so KS is not separable either. Detecting this
   subtype needs a shape-specific statistic (e.g. tail/quantile-ratio or
   higher-moment test) with its own clean calibration; it was not invented
   here.
2. **Low-severity category shift** (10% replacement) is missed. With 1000
   rows the chi-square test has too little power and the effect (Cramer's V
   ≈ 0.016) is below the calibrated categorical threshold (0.02 floor).
3. **Clean-window FPR is high**: 0.16–0.17 on average (range 0.08–0.26
   across seeds, 50 clean windows per seed). Almost all false positives come
   from the three collinear gas-sensor columns (`co`, `lpg`, `smoke`), whose
   local quantization-level jumps in the evaluation regions are larger than
   in the calibration region. The calibration-region holdout FPR is 0.0, so
   this is a real cross-region generalization gap, not an in-sample
   artefact. It is reported rather than tuned away with region-specific
   thresholds.
4. Schema recall at the ladder level is unchanged (schema events are exact);
   B1 recall 0.25 is simply its share of the mixed window set.
5. Detection lag is still ~0; drift is injected at full magnitude at onset.
   No claim of early detection is made.

## Tests

`PYTHONPATH=. python -m pytest tests -q` — all tests pass after the
revision (see final run in the session). New/updated tests:

- `tests/test_reference.py` — reference modes, min-row skipping, rolling
  previous-window comparison, per-column trigger, BH correction, Cramer's V.
- `tests/test_statistical_detector.py` — effect-size fields, clean effect
  size, quantization case where KS is high but effect is small.
- `tests/test_thresholds.py` — new keys, per-column calibrated thresholds,
  disjoint splits, frozen v1 preservation, no legacy constants, no dead
  unified threshold.
- `tests/test_unified.py` — continuous channel scores, OR-of-channels rule,
  removed unified-threshold path.
- `tests/test_phase8_generator.py` — updated to the new reference model and
  per-column statistical gate (the blind-spot windows are still
  statistically silent).

## Not re-run (out of scope)

- Phase 8 semantic blind-spot experiments, Phase 9 ablation, Phase 10/13
  and the Kafka/Spark/Postgres streaming overhead results were not re-run
  and their historical files were left untouched (per the ground rules).
- The 10-seed ladder was re-run only for B0–B4 (B5 Evidently remains the
  strawman documented in the review).
- The historical Phase 8/13 diagnostic scripts
  (`experiments/run_phase8.py`, `experiments/verify_phase8_variation.py`,
  `experiments/diagnose_phase13_b4.py`) still read the removed v1 keys
  (`ks_floor`, `psi_scale`, `chi2_scale`, `unified.threshold`). They are
  preserved as historical diagnostics and are not part of the revision;
  the frozen v1 values remain available in
  `configs/thresholds_v1_frozen.yaml` if they are ever re-run.

# Revision v3 — reducing clean-window false positives (options a-d)

Date: 2026-10-30. Task: reduce B4 clean-window FPR (0.172 raw on
evaluation clean windows in v2, 0.0 on the calibration holdout) with
options (a) gas-sensor grouping, (b) k-of-n persistence, (c)
multi-block threshold calibration, (d) a quantile sweep — WITHOUT
using any evaluation window (matrix rows [0, 100000), lag rows
[120000, 250000)) for any choice. The evaluation ladder was run
exactly once, after the config was frozen. `configs/
thresholds_v1_frozen.yaml` remains untouched; the v2 derivatives and
results were kept in `results/revision/`.

## Protocol and regions (all disjoint, none inside an evaluation region)

| role | rows |
|---|---|
| one-region calibration (= v2) | `[325000, 385000)` |
| option-(c) calibration blocks | `[101000, 110000)`, `[250000, 265000)`, `[270000, 285000)`, `[290000, 303000)`, `[325000, 340000)`, `[355000, 370000)` |
| validation clean windows | `[111000, 115000)` (4 matrix-style) + the 12 clean stream windows below |
| validation drift matrix | `[115000, 120000)` — 5 disjoint base windows x 12 statistical matrix cells injected with the same injectors |
| validation lag streams | `[304000, 324000)` — 4 five-window streams, same format as the evaluation ones (onset 3) |
| clean holdout (report only) | `[385000, 405184)` |

Experiment: `experiments/revision3_fpr_options.py` (full grid:
4 threshold variants x 3 quantiles x group on/off x persistence
on/off = 48 candidate configs). Artifacts:
`results/revision/validation_fpr_options/`
(`comparison_table.csv`, `selection.json`, `validation_outcomes.csv`).

## Calibration-block evidence (the generalization gap, from non-evaluation rows only)

Per-block clean gas-group effect sizes (max over co/lpg/smoke per
window; rolling references, first window of each block skipped):

| block | usable windows | gas p95 | gas max |
|---|---:|---:|---:|
| one region `[325000, 385000)` (= v2 basis) | 59 | 0.213 | 0.425 |
| `[101000, 110000)` (adjacent to the eval matrix region) | 8 | 0.509 | **0.511** |
| `[250000, 265000)` | 14 | 0.148 | 0.159 |
| `[270000, 285000)` | 14 | 0.047 | 0.050 |
| `[290000, 303000)` | 13 | 0.344 | 0.345 |
| `[325000, 340000)` | 14 | 0.213 | 0.215 |
| `[355000, 370000)` | 14 | 0.286 | 0.319 |

The mid-dataset block shows clean gas effects up to 0.511 — 20% above
the one-region max and far above v2's thresholds (0.340-0.364) —
confirming the cross-region gas-jump problem against rows that are
NOT evaluation rows. v2's thresholds were lucky for their region.

## Options implemented (config switches in configs/thresholds.yaml)

- **(a) `statistical.gas_group`** (`detectors/reference.py`): co/lpg/
  smoke become ONE composite gate. Group effect = max member effect;
  group p-value = min member p-value (one BH entry); ONE group
  threshold calibrated on the clean group-statistic distribution.
  Members keep their metrics for diagnosis but cannot trigger.
- **(b) `statistical.persistence`** (`detectors/persistence.py` +
  `experiments/run_phase7.py::run_detector_over_manifest`): alarm only
  if the statistical trigger fired in k of the last n windows.
  Lag-stream history = the stream's own evaluated windows; isolated
  matrix windows reconstruct history from the preceding n-1 RAW blocks
  (the same never-write-back convention as rolling references). k=2,
  n=3, so a sustained drift adds exactly +1 window of detection lag.
- **(c) `statistical.calibration.{mode,blocks,pooling}`**
  (`experiments/derive_thresholds.py`, CLI): single region, or >= 4
  disjoint blocks with `pooled`, `max_blocks` (max of per-block
  quantiles), or `gas_envelope` (gas columns get the max-over-blocks
  envelope, other columns keep the one-region quantile).
- **(d) quantile sweep** 0.99 / 0.995 / 0.999 reported per option
  (full table in `comparison_table.csv`).

## Validation comparison (condensed; group ON, persistence OFF rows)

Validation clean FPR is 0.0 for ALL 48 candidates (16 clean validation
windows; the mid-dataset gas regime is calm — the failing regime is
documented to live in the evaluation regions only). The validation
drift positives (60 windows) therefore discriminate on recall/cost:

| variant | q | group thr | val recall | val F1 | val clean margin |
|---|---:|---:|---:|---:|---:|
| single (v2 basis) | 0.99 | 0.3637 | 47/60 | 0.8785 | 0.4981 |
| single | 0.995 | 0.3945 | 46/60 | 0.8679 | 0.4819 |
| single | 0.999 | 0.4192 | 46/60 | 0.8679 | 0.4713 |
| multi_pooled | 0.99 | 0.3863 | 38/60 | 0.7755 | 0.5019 |
| multi_maxblocks | 0.99 | 0.4976 | 34/60 | 0.7234 | 0.5019 |
| gas_envelope | 0.99 | 0.4976 | 45/60 | 0.8571 | 0.4981 |
| gas_envelope | 0.995 | 0.5044 | 44/60 | 0.8462 | 0.4819 |
| gas_envelope | 0.999 | 0.5099 | 43/60 | 0.8350 | 0.4713 |

(group OFF rows are equal on validation except 2 gas-only-detected
positives; multi_pooled raises temp to ~0.23 and costs 9 low-severity
positives; multi_maxblocks for ALL columns is worse.)

**(b) persistence (k=2, n=3), reported honestly:** it never improved
validation (alone it scores F1 = 0.0: isolated drifted windows are
suppressed by construction, and on the 16 clean windows it changes
nothing). Its measured cost is the +1 window lag on the streams (the
evaluation lag numbers below come from the chosen config, where
persistence is OFF, so the reported lag remains ~0). The switch is
implemented and tested; it is documented as evaluated-and-rejected.

## Selection (validation only; no evaluation window used)

1. **Rule v1 (declared first; outcome reported and REJECTED):**
   rank by validation F1 desc, clean FPR asc, margin, lag, simplicity.
   Because every candidate has validation FPR 0, this degenerates into
   "maximize recall" and would return the v2 baseline unchanged
   (`single|q=0.99|group=off|persist=off`), leaving the documented
   evaluation FPR in place by construction. Unfit as a fix; recorded
   in `selection.json` as `winner_v1`.
2. **Rule v2 (declared BEFORE the evaluation run; the binding rule):**
   restrict to validation FPR = 0 and validation F1 >= 0.95 x the v2
   baseline's F1 (floor 0.8346); rank by gas-group threshold
   DESCENDING (robustness of the documented failure channel against
   every calibration block), then margin, lag, group ON, persistence
   OFF. Winner:
   `gas_envelope|q=0.999|group=on|persist=off` (threshold 0.5099;
   validation F1 0.8350 — it passes the floor by 4e-4, borderline by
   construction and disclosed as such).

## Chosen configuration (configs/thresholds.yaml v3)

- Gas columns co/lpg/smoke and the gas-group gate: max-over-blocks
  envelope over the one region + 6 blocks at q = 0.999 computed on the
  GROUP statistic; the binding value is the C_mid block's 0.5099.
- All other columns: one-region quantile at q = 0.999.

| column | v2 threshold | v3 threshold |
|---|---:|---:|
| co | 0.3405 | 0.5017 |
| lpg | 0.3637 | 0.5099 |
| smoke | 0.3594 | 0.5084 |
| gas_group (enabled) | — | **0.5099** |
| humidity | 0.2032 | 0.2298 |
| light | 0.3009 | 0.3386 |
| motion | 0.1016 | 0.1093 |
| temp | 0.1685 | 0.1772 |
| device (floor) | 0.0200 | 0.0200 |

Reproduction:
`PYTHONPATH=. python3 experiments/derive_thresholds.py
--calibration-mode multi_block --quantile 0.999 --pooling
gas_envelope --blocks "101000-110000;250000-265000;270000-285000;
290000-303000;325000-340000;355000-370000" --group-enabled
--persistence-k 2 --persistence-n 3 --output-root results/revision3`
(default invocation without flags still reproduces the v2 values;
verified: zero difference, holdout FPR report 0.0 over 19 windows).

## 10-seed evaluation ladder (run ONCE, after freezing; raw metrics, mean +/- std)

| config | precision | recall | F1 | FPR | lag mean | lag median (p95) |
|---|---:|---:|---:|---:|---:|---|
| B0_null | 1.000 ± 0.000 | 0.050 ± 0.000 | 0.095 ± 0.000 | 0.000 ± 0.000 | 0.000 | 0.0 (0.00) |
| B1_schema | 1.000 ± 0.000 | 0.250 ± 0.000 | 0.400 ± 0.000 | 0.000 ± 0.000 | 0.000 | 0.0 (0.00) |
| B2_statistical | 0.842 ± 0.055 | 0.470 ± 0.030 | 0.603 ± 0.034 | 0.108 ± 0.044 | 0.112 | 0.0 (0.51) |
| B3_schema+statistical | 0.887 ± 0.041 | 0.693 ± 0.021 | 0.778 ± 0.024 | 0.10
