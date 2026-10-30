# Phase 7 Multi-Seed Robustness Sweep (Section Y) — Report

**Generator fix applied:** `build_phase7_set()` now samples window
START POSITIONS per seed (Section Y real fix), so different seeds draw
genuinely different, non-overlapping windows from the dataset. This
replaces the old behavior where only internal injector randomness
varied inside fixed windows (std = 0 by construction).

**Second fix applied (mean_shift flagging):** `drift_injection/flagging.py`
now flags `mean_*` windows as expected false negatives too, using the
SAME existing mechanism as variance/distribution shift — because the
mean_shift blind spot was diagnosed as the same aggregation-masking
limitation (see `results/phase7/diagnose_mean_shift.json` and section
6 below). No new mechanism was invented; the shared rule was extended
by one prefix (`mean_`).

- Old (fixed placement) outputs preserved in `results/phase7/multiseed_old/`.
- New (sampled placement) outputs: this directory.
- Thresholds / injection magnitudes / severity definitions:
  **unchanged** (`configs/thresholds.yaml` untouched).

---

## 1. What changed in the generator

`drift_injection/phase7_generator.py`:

- **Matrix windows (30):** starts sampled via the new
  `sample_non_overlapping_starts(rng, 30, WINDOW_SIZE, 0, MATRIX_REGION)`
  helper — 30 distinct offsets in `[0, 100000 - 30*1000]`, sorted, then
  spaced by `window_size`. Guarantees non-overlap and containment.
- **Lag streams (16 × 5 = 80):** each stream's base sampled within
  `[LAG_REGION_BASE=120000, LAG_REGION_END=250000)` with
  `LAG_STREAM_STRIDE = 5*1000` rows per stream; 16 non-overlapping
  stream blocks per seed.
- Window counts, subtype/severity distribution, and
  `expected_false_negative` flagging are identical across seeds
  (verified: 110 windows = 30 matrix + 80 lag, **24 flagged** — 8
  mean_shift + 8 variance_shift + 8 distribution_shift — every seed).
- Dataset is 405,184 rows; all windows fit with large margin.

Verified properties:

| seed | n_windows | matrix | lag | flagged | matrix start range | lag base range | any overlap |
|---|---|---|---|---|---|---|---|
| 42 | 110 | 30 | 80 | 24 (8 mean + 8 var + 8 dist) | 6014–97276 | 124296–243778 | no |
| 123 | 110 | 30 | 80 | 24 (8 mean + 8 var + 8 dist) | 1080–93620 | 120771–241166 | no |
| 7 / 11 / 2024 / 314 / 2718 / 999 / 555 / 777 | 110 | 30 | 80 | 24 (8 mean + 8 var + 8 dist) | (varies per seed) | (varies per seed) | no |

Seeds used (same list as before): `[42, 123, 7, 11, 2024, 314, 2718,
999, 555, 777]`.

## 2. Baseline ladder — mean +/- std across 10 seeds (sampled placement)

| Config | Variant | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|
| B0_null | raw | 1.0000 ± 0.0000 | 0.0500 ± 0.0000 | 0.0952 ± 0.0000 | 0.0000 ± 0.0000 |
| B0_null | adjusted | 1.0000 ± 0.0000 | 0.0600 ± 0.0000 | 0.1132 ± 0.0000 | 0.0000 ± 0.0000 |
| B1_schema | raw | 1.0000 ± 0.0000 | 0.2500 ± 0.0000 | 0.4000 ± 0.0000 | 0.0000 ± 0.0000 |
| B1_schema | adjusted | 1.0000 ± 0.0000 | 0.3000 ± 0.0000 | 0.4615 ± 0.0000 | 0.0000 ± 0.0000 |
| B2_statistical | raw | 0.9729 ± 0.0333 | 0.2333 ± 0.0149 | 0.3761 ± 0.0201 | 0.0080 ± 0.0098 |
| B2_statistical | adjusted | 0.9786 ± 0.0327 | 0.2889 ± 0.0172 | 0.4457 ± 0.0207 | 0.0073 ± 0.0112 |
| B3_schema_statistical | raw | 0.9866 ± 0.0165 | 0.4800 ± 0.0100 | 0.6457 ± 0.0100 | 0.0080 ± 0.0098 |
| B3_schema_statistical | adjusted | 0.9897 ± 0.0158 | 0.6178 ± 0.0089 | 0.7606 ± 0.0069 | 0.0073 ± 0.0112 |
| **B4_unified** | raw | 0.9922 ± 0.0096 | 0.8333 ± 0.0298 | 0.9055 ± 0.0177 | 0.0080 ± 0.0098 |
| **B4_unified** | adjusted | 0.9935 ± 0.0100 | 0.9956 ± 0.0089 | **0.9945 ± 0.0055** | 0.0073 ± 0.0112 |
| B5_evidently | raw/adj | 0.5455 / 1.0000 / 0.7059 raw, FPR 1.0000 — real Evidently decisions (fixed; flags every window under default config, see `B5_EVIDENTLY_FIX.md`). This sweep's B5 row was generated BEFORE the fix (n=0); the fixed numbers are on the committed seed-42 set in `results/phase7/results.json`. | | | |
| semantic-only | raw | 1.0000 ± 0.0000 | 0.4900 ± 0.0309 | 0.6571 ± 0.0276 | 0.0000 ± 0.0000 |
| semantic-only | adjusted | 1.0000 ± 0.0000 | 0.5556 ± 0.0000 | 0.7143 ± 0.0000 | 0.0000 ± 0.0000 |

Detection lag (over detected lag streams): B4 mean 0.0333 ± 0.0447,
median 0.0000 ± 0.0000, p95 0.1900 ± 0.3015 windows; B4 detects 15 of
16 streams at every seed (the never-detected stream is mean_shift).
Semantic lag mean 0.0680 ± 0.0746, p95 0.3450 ± 0.3622.

**Note on adjusted metrics after the flagging change:** the mean_shift
windows are now excluded from adjusted denominators the same way as
variance/distribution shift, so B4's adjusted recall rises to
0.9956 ± 0.0089 and adjusted F1 to 0.9945 ± 0.0055. The residual
adjusted FN (~0.4% of positives) is the category_shift window that the
statistical channel misses in some seeds — the only non-flagged
subtype with imperfect recall.

## 3. Subtype recall — mean +/- std across 10 seeds

Complete per-seed values in `aggregate_subtype_recall.csv`. Summary of
the cells that now carry genuine variance (all others are ± 0.0000):

| Subtype | Detector | Variant | Mean | Std | min–max |
|---|---|---|---|---|---|
| category_shift | statistical/combined | raw & adj | 0.960 | 0.080 | 0.800–1.000 |
| distribution_shift | combined | raw | 0.420 | 0.275 | 0.000–1.000 |
| distribution_shift | semantic | raw | 0.340 | 0.237 | 0.000–0.600 |
| distribution_shift | statistical | raw | 0.080 | 0.160 | 0.000–0.400 |
| variance_shift | combined | raw | 0.520 | 0.160 | 0.400–0.800 |
| variance_shift | semantic | raw | 0.500 | 0.134 | 0.400–0.800 |
| variance_shift | statistical | raw | 0.020 | 0.060 | 0.000–0.200 |
| mean_shift | statistical/combined | raw | 0.100 | 0.100 | 0.000–0.200 |
| mean_shift | statistical/combined | adj | 0.000 | 0.000 | 0.000–0.000 (excluded — now flagged) |
| mean_shift | semantic | raw | 0.040 | 0.080 | 0.000–0.200 |
| mean_shift | semantic | adj | 0.000 | 0.000 | 0.000–0.000 (excluded — now flagged) |
| added | statistical | raw & adj | 0.067 | 0.200 | 0.000–0.667 |

Rock-solid (std = 0.0000 across all 10 seeds): all schema events via
schema/combined (1.000), all semantic rules via semantic/combined
(unit_change, unit_conversion, business_meaning_change, range_change,
allowed_domain_change, range_violation, domain_violation — 1.000),
domain_violation via statistical (1.000), and the adjusted-recall of
all flagged variance/distribution/mean windows (0.000 by design —
excluded from adjusted metrics).

## 4. High-variance cells (std > 20% of mean) — MUST be caveated

`high_variance.csv` lists 21 flagged cells (down from 24 — the three
mean_shift **adjusted**-recall cells dropped out because those windows
are now excluded from adjusted metrics, making the adjusted mean 0.000
with no relative std):

**Ladder:**
- FPR for B2 / B3 / B4 (raw 0.0080 ± 0.0098, adjusted 0.0068 ± 0.0104):
  near-zero mean makes the relative std large; interpret as
  "0–2 FP windows out of 50 clean windows, seed-dependent".
- B4 detection lag mean 0.0333 ± 0.0447 and p95 0.1900 ± 0.3015, and
  semantic lag mean/p95 — lag is near 0 in most seeds with occasional
  slower streams.

**Subtype recall (the important ones for the paper):**
- `distribution_shift` / combined raw 0.420 ± 0.275 (rel 0.65) and
  semantic raw 0.340 ± 0.237 — the flagged "expected FN" windows get
  caught through the semantic channel in some seeds and not others.
- `variance_shift` / combined raw 0.520 ± 0.160 (rel 0.31), semantic
  raw 0.500 ± 0.134 — same story, more pronounced.
- `mean_shift` / statistical & combined raw 0.100 ± 0.100 (rel 1.00)
  and semantic raw 0.040 ± 0.080 — near-zero mean with real variance;
  a single-seed number would be anywhere from 0 to 0.2. (Adjusted
  variants are 0.000 ± 0.000 — now flagged, excluded.)
- `added` / statistical 0.067 ± 0.200 (rel 3.00) — one seed detected
  2/3 added windows statistically, most seeds 0/3.
- `distribution_shift` / statistical raw 0.080 ± 0.160 and
  `variance_shift` / statistical raw 0.020 ± 0.060.

These must be reported as mean ± std with the caveat that the
single-window-set estimate is not representative on its own.

## 5. Before / after: B4 headline

| Metric | Old (fixed placement, seed 42) | Sampled placement, 10 seeds (pre-flagging) | Sampled placement, 10 seeds (post-flagging) |
|---|---|---|---|
| B4 adjusted F1 | 0.9375 ± 0.0000 | 0.9476 ± 0.0084 | **0.9945 ± 0.0055** (seed42 1.0000, min 0.9888, max 1.0000) |
| B4 adjusted precision | 0.9783 ± 0.0000 | 0.9936 ± 0.0098 | 0.9935 ± 0.0100 |
| B4 adjusted recall | 0.9000 ± 0.0000 | 0.9060 ± 0.0156 | 0.9956 ± 0.0089 |
| B4 raw F1 | 0.9107 ± 0.0000 | 0.9055 ± 0.0177 | 0.9055 ± 0.0177 (unchanged) |
| B4 raw precision | 0.9808 ± 0.0000 | 0.9922 ± 0.0096 | 0.9922 ± 0.0096 (unchanged) |
| B4 raw recall | 0.8500 ± 0.0000 | 0.8333 ± 0.0298 | 0.8333 ± 0.0298 (unchanged) |
| B4 FPR (raw) | 0.0200 ± 0.0000 | 0.0080 ± 0.0098 | 0.0080 ± 0.0098 (unchanged) |

**Verdict: the headline holds up under sampling variation, and the
mean_shift flagging fix moves the ADJUSTED number decisively up.** The
sampled-placement mean (0.9476 ± 0.0084) confirmed the fixed-placement
point estimate was mildly conservative. After flagging mean_shift as
an expected FN (same aggregation limitation as variance/distribution
shift), the adjusted F1 becomes **0.9945 ± 0.0055** with adjusted
recall 0.9956 ± 0.0089 — the only remaining adjusted FN is the
category_shift window the statistical channel misses in some seeds.
Raw metrics are completely unchanged by the flagging fix (flagging
only affects the adjusted denominators).

**For the paper:** quote **0.9945 ± 0.0055 (adjusted) /
0.9055 ± 0.0177 (raw)** — but only alongside the honest caveat that
the adjusted number excludes mean/variance/distribution-shift windows
documented as aggregation limitations (24 of 110 windows), and that
mean_shift's true (raw) recall is 0.100 ± 0.100.

## 6. The mean_shift diagnosis and flagging fix

`results/phase7/diagnose_mean_shift.json` contains the per-window
per-column KS/PSI audit across all 10 seeds. The root cause is the
**same aggregation-masking limitation** as variance/distribution shift
— NOT a distinct bug:

- **Injected magnitude is too small at low/medium severity for the
  shifted column to win the max-over-columns aggregation.** Temp's
  KS after a mean shift is 0.27–0.48 at low/medium severity, below the
  frozen `ks_floor = 0.50` in `configs/thresholds.yaml`.
- **The shifted column's KS sits at or below the sampling-noise floor
  of the OTHER columns.** co/lpg/smoke carry baseline KS of
  ~0.36–0.50 on clean rows (sampling noise at n=1000 vs a 405K
  reference), and co owns the column-max KS in 9/10 medium windows.
  Example (seed 42, medium): shifted temp KS = 0.3649, PSI = 10.90;
  co max KS = 0.4455, PSI = 11.64 — the argmax column is co, not temp,
  so the statistical detector scores the wrong column's signal
  (statistical_score 0.162, NO_DRIFT).
- **PSI does not move** for mean shifts (scale 72.0), and high severity
  is marginal — temp KS 0.49–0.80 crosses the 0.25 decision score
  only above ~0.625.

Per the task: since this is the same root cause as variance/distribution
shift, the EXISTING `expected_false_negative_condition()` /
`expected_false_negative_reason()` mechanism in
`drift_injection/flagging.py` was extended to `mean_*` (one prefix),
and `MEAN_FN_REASON` documents the diagnosis. No new mechanism, no
threshold change. The generator test and the Phase 6 flagging test were
updated to match; all 103 tests pass.

Effect on ground truth and adjusted metrics (verified across all 10
seeds): 16 → **24 flagged windows** per seed (8 mean_shift added),
B4 adjusted F1 0.9476 → **0.9945**, adjusted recall 0.9060 → 0.9956.
Raw metrics unchanged.

## 7. B5 (Evidently) status

`results/phase7/B5_EVIDENTLY_FIX.md` documents the full fix. Summary:
root cause was environmental + API-mismatch, now resolved:

1. `evidently` was not installed. Installed `evidently==0.6.7` (the
   version whose `metric_preset` + `report` API the B5 runner expects;
   0.7.x restructured the API and dropped `metric_preset`).
2. A dependency conflict (litestar needing `MultipartSegment` from the
   `multipart` package that `python-multipart` shadows) was resolved by
   pinning `litestar==2.13.0`, whose `_multipart.py` is self-contained.
3. Two real bugs in `evaluation/baseline.py` were fixed: the drift
   decision read `metrics[0]` instead of scanning all metrics for
   `drift_by_columns`, and schema `removed` windows crashed because
   the B5 comparison used reference-only columns (now intersected).
4. B5 now runs on all 110 windows of the committed seed-42 set with
   real decisions; `results/phase7/results.json` has real numbers
   (raw precision 0.5455, recall 1.0000, FPR 1.0000).

**Caveat:** under Evidently's default `DataDriftPreset`, B5 flags
EVERY window including clean controls — KS with n=1000 vs a 405K-row
reference has extreme power. This is genuine Evidently default
behavior, not an integration bug, but it makes B5's decisions
uninformative (FPR = 1.0) at this reference/window size ratio. The
B5 row in THIS sweep's aggregate tables is still the pre-fix n=0
placeholder; folding real B5 into the next multi-seed run is
straightforward via the same runner.

## 8. What the sweep changes vs the previous (std = 0) sweep

- Genuine sampling variance now exists: subtype cells that were frozen
  at exactly 0 or 1 across seeds (mean_shift, distribution_shift,
  variance_shift, category_shift, added/statistical) now move per seed.
- The overall conclusions do NOT flip: B4 remains clearly best,
  schema/semantic rule detection remains perfect, and the three blind
  spots (mean_shift ~0.1 raw, variance/distribution/mean shifted
  adjusted-recall 0 by design) persist with honest uncertainty bands.
- The sweep now satisfies Section Y's actual requirement: mean ± std
  reflects real window-sampling variation, not design artifacts.
