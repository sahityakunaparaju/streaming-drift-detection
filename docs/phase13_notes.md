# Phase 13 — Secondary Dataset Sanity Check (Generalization)

## Objective

Confirm whether the Phase 6 frozen thresholds (`configs/thresholds.yaml`)
generalize to a genuinely different dataset, using a reduced-scope baseline
ladder (B0–B4) only.

## Dataset

**UCI Beijing Multi-Site Air-Quality Data** (Zhang et al., 2017, Proc. Royal
Society A), CC BY 4.0. 12 station CSVs merged into 420,768 rows × 14 columns.
Hourly measurements, March 2013 – February 2017.

Key differences from primary IoT dataset:
- **Country/sensor network**: Chinese urban air-quality monitoring vs. US
  home IoT sensors
- **Pollutants**: PM2.5, PM10, SO2, NO2, CO, O3 (chemical) vs. co, lpg,
  smoke (combustion)
- **Categorical field**: `wd` (wind direction, 16 compass points) — the
  primary dataset's categorical field was `device` (3 MAC addresses)
- **Null values**: 0.08–4.92% across pollutant columns; primary dataset had
  **zero** nulls everywhere
  
## Phase 13 Fixes Applied

Two issues in the original Phase 13 run were identified and fixed:

### Fix 1: Rate-Aware Nullability Baseline

**Root cause**: The schema detector compared binary `nullable` flags
(True/False) between reference and current schemas. The reference schema
marked columns as `nullable: True` if any null existed in the reference data,
while clean windows often had zero nulls in those same columns due to natural
variation in missing-data patterns. Every window triggered a
`nullability_change` event, producing FPR=1.0 for B1/B3/B4.

**Design decision**: Replaced the binary nullable comparison with a
**rate-aware comparison** using `NULLABILITY_RATE_TOLERANCE = 0.05` (5
percentage points). The reference schema now stores `missing_rate` per column
(computed by `build_current_schema()` in `evaluation/baseline.py`). The
comparison logic in `detectors/schema_detector.py` is:

- If reference `missing_rate` is **exactly 0.0**: flag if current rate > 0.0
  (zero tolerance for zero-null baselines — preserves primary dataset behavior)
- If reference `missing_rate` > 0.0: flag only if
  `abs(ref_rate - cur_rate) > 0.05` (5pp tolerance absorbs natural variation)
- If no `missing_rate` present (backward compat with hand-crafted schemas):
  fall back to existing boolean `nullable` comparison

**Tolerance justification**: Across all 1K windows in the first station
(Aotizhongxin, 35K rows), natural null-rate variation from the full-dataset
baseline reaches up to 25pp for individual columns (CO: 0–25.6%, O3: 0.1–
25.7%). A 5pp tolerance is a practical compromise: it absorbs small-sample
variation while still catching injected nullability changes (typically 15–20pp
injections). The trade-off is that some clean windows with anomalously high
null rates (e.g., CO at 10% vs. baseline 4.9%) may still false-positive.

### Fix 2: Generalized Category-Shift Column Selection

**Root cause**: `evaluation/baseline.py`'s `statistical_details_for()`
hardcoded `"device"` as the only categorical column for χ² testing:

```python
if "device" in reference_df.columns and "device" in current_df.columns:
    categorical["device"] = categorical_drift(...)
```

The secondary dataset has no `device` column; its categorical field is `wd`
(wind direction). The category-shift test silently no-opped.

**Fix**: Made `statistical_details_for()` accept an optional
`categorical_columns` parameter. When provided, it tests only those columns.
When `None`, it falls back to the legacy `device`-only behavior (backward
compatible with Phase 7). The parameter is threaded through
`run_b2_statistical()`, `run_b3_schema_statistical()`, and `run_b4_unified()`
as an optional keyword argument.

Categorical columns are discovered from the semantic contract using
`discover_categorical_columns()` in `run_phase13.py`, which looks for fields
with `data_role` containing `"categorical"` or with an `allowed_values` list.

## Side-by-Side Results: Primary vs. Secondary

### B0 (Null Monitor)

| Metric | Primary (Phase 7) | Secondary (Phase 13) |
|--------|-------------------|---------------------|
| Precision | 1.000 | 0.714 |
| Recall | 0.050 | 1.000 |
| F1 | 0.095 | 0.833 |
| FPR | 0.000 | 1.000 |

**Analysis**: B0 correctly reports **zero false positives** on the primary
dataset (which has zero nulls) but flags **every window** on the secondary
dataset because Beijing data has natural missing values (2–5% in pollutant
columns). B0 is a null presence monitor (`max_null_rate > 0.0`), so this is
**expected behavior** — it correctly detects that nulls exist.

### B1 (Schema Detector — Rate-Aware)

| Metric | Primary (Phase 7) | Secondary (Phase 13) |
|--------|-------------------|---------------------|
| Precision | 1.000 | 0.800 |
| Recall | 0.250 | 0.800 |
| F1 | 0.400 | 0.800 |
| FPR | 0.000 | **0.500** |

**Before fix**: FPR=1.0 (all clean windows flagged via binary nullable mismatch).
**After fix**: FPR=0.5 (one clean window correctly cleared, one still
false-positives due to CO/O3 null rates exceeding the 5pp tolerance).

B1 correctly detects:
- ✅ schema-added (column addition)
- ✅ stat-mean-shift (schema-level side effect)
- ✅ sem-unit-change (schema-level side effect)
- ✅ schema-nullability (injected 20% nulls in PM2.5)

B1 misses:
- ❌ cat-wd-shift (category shift doesn't alter schema structure — expected)

The remaining false positive (clean-01) is caused by CO (10% null rate vs.
4.9% baseline, 5.1pp difference) and O3 (10% vs. 3.2%, 6.8pp difference)
exceeding the 5pp tolerance. This is natural variation in the data, not a
threshold bug.

### B2 (Statistical — KS + PSI + χ²)

| Metric | Primary (Phase 7) | Secondary (Phase 13) |
|--------|-------------------|---------------------|
| Precision | 0.933 | 0.714 |
| Recall | 0.233 | 1.000 |
| F1 | 0.373 | 0.833 |
| FPR | 0.020 | **1.000** |

**Before fix (no wd testing)**: P=1.000, FPR=0.000 — B2 was clean because
the χ² test silently no-opped (no `device` column in secondary data).
**After fix (wd tested)**: FPR=1.000 — the χ² test fires on **every** clean
window because the `wd` distribution (16 categories) naturally varies across
1K windows within the same station.

**Root cause of chi2 sensitivity**: With 16 wind-direction categories (df=15)
and 1000-row windows, the χ² test has enormous statistical power. Even natural
sample variation produces χ² values of 78–763 (all far exceeding the frozen
`chi2_scale=32.0` threshold, which was calibrated on the primary dataset's
3-category `device` field). The injected category-shift produces χ²=1103
(3.0× larger than the clean-window maximum of 372), confirming the detector
**does** detect real drift — but the χ² threshold is not calibrated for
16-category fields.

**B2 per-window detail**:
- ✅ clean-00 → NO_DRIFT for numeric, DRIFT for wd (χ² fires) → overall DRIFT
- ✅ clean-01 → same as above
- ✅ schema-added → DRIFT (cross-column numeric shift from added column)
- ✅ stat-mean-shift → DRIFT (PM2.5 mean shift detected by KS/PSI)
- ✅ cat-wd-shift → DRIFT (χ² detects injected category shift, χ²=1103)
- ✅ sem-unit-change → DRIFT (semantic; B2 shouldn't detect this — but the
  `inject_unit_change(..., metadata_only=True)` call doesn't change numeric
  values, so this must be coming from the categorical path or noise)

**Note on sem-unit-change**: The `inject_unit_change` with `metadata_only=True`
does not alter data values, so B2 should return NO_DRIFT. Investigation shows
the χ² test on `wd` fires because the semantic window (rows 5000–5999) has a
different wind-direction distribution from the reference — same natural-variation
issue as the clean windows. This is a false positive from the χ² path, not from
KS/PSI.

### B3 (Schema + Statistical)

| Metric | Primary (Phase 7) | Secondary (Phase 13) |
|--------|-------------------|---------------------|
| Precision | 0.966 | 0.714 |
| Recall | 0.467 | 1.000 |
| F1 | 0.629 | 0.833 |
| FPR | 0.020 | 1.000 |

Inherits from B1 (one false positive) and B2 (χ² false positives). The OR
combination means any component firing triggers DRIFT.

### B4 (Unified)

| Metric | Primary (Phase 7) | Secondary (Phase 13) |
|--------|-------------------|---------------------|
| Precision | 0.981 | 0.714 |
| Recall | 0.850 | 1.000 |
| F1 | 0.911 | 0.833 |
| FPR | 0.020 | 1.000 |

Inherits false positives from B1 (nullability) and B2 (χ² on wd).
Per-window component attribution: see "B4 False-Positive Attribution"
below — 50% statistical (χ²/wd), 50% schema (nullability), 0% semantic.

## Category-Shift Generalization

The Beijing dataset has a **real categorical field** (`wd`, wind direction, 16
categories). After the fix, the χ² test actually runs on `wd` for the first
time.

**Result**: The χ² test **detects the injected category shift** (χ²=1103, 3.0×
larger than the clean-window maximum of 372), confirming the detector works.
However, it **also fires on every clean window** because the frozen
`chi2_scale=32.0` was calibrated on 3-category `device` fields and is
overwhelmingly sensitive for 16-category fields with 1000-row samples.

**Diagnosis**: Under the null hypothesis (same distribution), expected χ² ≈
df = 15. With 1000 samples and 16 categories, even 1–2pp proportional
differences produce χ² > 50. The frozen scale of 32.0 maps this to a score of
1.0 (saturated). The chi2_scale would need to be ~800+ to discriminate
natural variation from injected drift on a 16-category field — a 25× increase
from the frozen value.

This is a **genuine generalization limitation**: the χ² threshold is
category-count-dependent, and the frozen value was tuned for the primary
dataset's specific cardinality.

## Nullability-Change Detection Confirmation

The injected nullability-change window (20% nulls injected into PM2.5, baseline
~2%) is **correctly detected** by B1 with rate-aware comparison:
- Reference PM2.5 missing_rate: 2.08%
- Injected window PM2.5 missing_rate: 20.0%
- Difference: 17.92pp > 5pp tolerance → **flagged** ✅

The fix did **not** break injected-drift detection for nullability.

## Verdict

### Generalization: Partially Confirmed — With Documented Limitations

| Aspect | Result | Notes |
|--------|--------|-------|
| Statistical thresholds (KS/PSI) | ✅ **Generalize** | Numeric drift detection works on different sensors/pollutants |
| Schema nullability check | ⚠️ **Partial** | Rate-aware fix reduces FPR from 1.0 to 0.5; remaining FP due to natural null-rate variation |
| Category-shift detection (χ²) | ⚠️ **Partial** | Detects injected drift but χ² threshold too sensitive for 16-category fields |
| Frozen chi2_scale | ❌ **Does not generalize** | Calibrated for 3-category `device`; overwhelms on 16-category `wd` |
| B0 null monitor | ✅ **Correct** | Works as designed; fires on natural nulls (expected) |

### What Was Wrong and How It Was Fixed

1. **Schema nullability false positives** (B1 FPR=1.0 → 0.5): The binary
   `nullable` flag comparison was replaced with rate-aware comparison using a
   5pp tolerance. The reference schema now stores `missing_rate` per column.
   Root cause: the primary dataset's zero-null characteristic made boolean
   comparison work perfectly; the secondary dataset's natural missingness
   exposed the limitation.

2. **Category-shift silent no-op** (B2 not testing wd): The hardcoded
   `"device"` column reference in `statistical_details_for()` was replaced
   with an optional `categorical_columns` parameter, discovered from the
   semantic contract. Root cause: the function was written for the primary
   dataset and never generalized.

### B4 False-Positive Attribution (Per-Window Breakdown)

**Deliverable statement**: *B4's secondary-dataset FPR=1.0 is **partially
explained** by the same χ²/wd-cardinality issue identified for B2.* The
statistical component's saturated χ² override fires in **100%** of B4's
false positives and **fully explains 50%** of them (clean-00); the other
50% (clean-01) is **not** explained by χ² — it is caused by an
**independent confound in the schema component** (rate-aware nullability
on CO/O3, the same source as B1's FPR=0.5). The semantic component
contributes **0%**.

**Method**: read-only diagnostic `experiments/diagnose_phase13_b4.py`
(no threshold or detector changes) reassembles B4 exactly as
`run_b4_unified()` does for each frozen window, records every condition
fired inside `detectors/unified.py::decide()`, and re-runs the decision
with the categorical (χ²) component removed from the statistical score
(numeric KS/PSI, schema, and semantic unchanged). Full output:
`results/phase13/diagnose_b4_fp.json`. The script reproduces the frozen
`results/phase13/results.json` decisions exactly (B4: 2/2 clean windows
FP, FPR=1.0).

Decision paths available in `decide()` (each is independently sufficient
for DRIFT; ALL fired conditions are recorded per window below):

1. **Max-confidence override** — any component score ≥ its max (1.0)
2. **Statistical path** — statistical_score ≥ 0.25
3. **Unified path** — unified_score ≥ 0.5

#### Q1 — Which component triggered each FP window?

| Window | Schema score / events | Statistical score (χ² on wd) | Semantic | Conditions fired in `decide()` | B4 |
|--------|----------------------|------------------------------|----------|--------------------------------|----|
| clean-00 | 0.0 — no events | **1.000** (χ²=248, p≈4e-44; ks_comp=0.000, psi_comp=0.122) | 0.0 | statistical **override** + statistical path (unified=0.333 < 0.5 — unified path alone would NOT fire) | DRIFT (FP) |
| clean-01 | **1.0** — CO 4.92%→10.00%, O3 3.16%→10.00% (both >5pp tolerance) | **1.000** (χ²=372, p≈5e-70; ks_comp=0.000, psi_comp=0.098) | 0.0 | schema **override** + statistical **override** + statistical path + unified path (0.667 ≥ 0.5) — either override alone is sufficient | DRIFT (FP) |

#### Q2 — Does every FP coincide with the statistical χ² firing?

**Yes — 2/2.** In both FP windows the statistical score is saturated at
1.000 *purely* by the χ²/wd component (χ²=248/372 vs.
`chi2_scale=32.0`, i.e. 7–12× the scale; ks_comp=0.000 and
psi_comp≈0.10 are far below the 0.25 decision threshold on their own).
This is the same wd-cardinality issue root-caused for B2 — and B2
indeed fires on exactly these same two windows (B2's FPR=1.0 covers the
same 2 negative windows).

**But statistical firing alone does not fully explain B4.** There is
**no FP where statistical did NOT fire**, yet on clean-01 B4 stays DRIFT
even with χ² removed, because the **schema** override (CO/O3
nullability) independently forces DRIFT. So schema contributes an
*independent* confound on top of χ² — this is the "additional confound"
branch of the question. The **semantic** component fires on 0/2 FP
windows and contributes nothing.

#### Q3 — Percentage breakdown (χ²-only counterfactual)

Counterfactual removes only the wd/χ² confound from the statistical
score (numeric KS/PSI kept; schema/semantic unchanged):

| Window | stat score w/o χ² | B4 decision w/o χ² | Outcome |
|--------|-------------------|--------------------|---------|
| clean-00 | 0.122 (unified=0.041) | NO_DRIFT | FP disappears → TN ✅ |
| clean-01 | 0.098 | DRIFT (schema override still fires) | FP persists ❌ |

- **50% of FPs (1/2 — clean-00) fully explained by the χ²/wd-
  cardinality issue**: would disappear if only the wd category-shift
  confound were fixed.
- **50% of FPs (1/2 — clean-01) NOT explained by χ²** — residual source:
  the **schema** detector's rate-aware nullability comparison flagging
  natural null-rate variation (CO 10% vs. 4.9% baseline, O3 10% vs.
  3.2%, both >5pp tolerance). Needs separate investigation; it is the
  same confound behind B1's FPR=0.5 (see the *Natural null-rate
  variation* limitation below).
- **0% explained by semantic.**

Corollaries from the same run:

- χ²-only counterfactual for B4: FPR 1.0 → 0.5 (clean-01 persists),
  recall 1.0 → 0.8 (cat-wd-shift becomes FN — stat 1.000 → 0.128).
- **B2's FPR=1.0 is 100% χ²-explained** (both clean windows drop to
  NO_DRIFT without χ²: 0.122/0.098 < 0.25 → FPR 0.0), whereas **B4's is
  only 50% χ²-explained** — B4's extra schema component is a second,
  independent FP source that B2 does not have.
- Removing **both** the χ² and schema nullability confounds drives B4's
  secondary FPR to **0.0** (only cat-wd-shift recall is lost).

**Reference-slice provenance (reproducibility note)**: χ² magnitudes
depend slightly on which 10K reference slice is used. Rows
**7000–17000** reproduce every χ² value recorded in this document
(247.6 / 372.1 / 1103.4 ≈ 248 / 372 / 1103), while the current
`run_phase13.main()` uses rows **6000–16000** (`ref_start = 6 *
WINDOW_SIZE`, which overlaps the schema-nullability window at rows
6000–7000 despite the non-overlap comment), giving
223.8 / 355.4 / 1081.8. The diagnostic runs **both** offsets: decisions
and FP attribution are **identical** under either (every χ² component
still saturates at 1.0 ≫ 32.0), so the 50/50 attribution does not
depend on this choice.

### Honest Limitations to Report

1. **The secondary-dataset FPR is attributable to specific components —
   not to "the unified detector" broadly** (verified by
   `experiments/diagnose_phase13_b4.py`, output in
   `results/phase13/diagnose_b4_fp.json`):
   - **Statistical component** (χ²/wd): fires in **100%** of B4's false
     positives (both clean windows, χ² component saturated at 1.0) and is
     the **sole** trigger for **50%** of them (clean-00). It is also the
     entire cause of B2's FPR=1.0 (B2 would drop to FPR=0.0 without it).
   - **Schema component** (rate-aware nullability): the residual **50%**
     of B4's FPs (clean-01 — CO/O3 natural null-rate variation >5pp),
     which fires **independently of χ²** and is the same source as B1's
     FPR=0.5. Would persist even with a perfectly calibrated
     categorical threshold.
   - **Semantic component**: **0%** of B4's FPs (never fires on clean
     windows).

2. **The χ² threshold (chi2_scale=32.0) is category-count-dependent**. It
   was tuned for 3 categories and does not generalize to 16+ categories
   without rescaling. A production system would need cardinality-aware
   thresholding (e.g., chi2_scale ∝ df). Drives B2's FPR=1.0 and 50% of
   B4's FPs (see Limitation 1).

3. **Natural null-rate variation in the Beijing dataset is large** (up to
   25pp across 1K windows), making any fixed-rate nullability tolerance a
   compromise between false positives and missed detections. Drives B1's
   FPR=0.5 and the other 50% of B4's FPs (see Limitation 1).

4. **B0's behavior is correct but uninformative** on datasets with natural
   nulls — it always fires. A more useful null monitor would compare against
   a baseline null rate rather than zero.

## Deliverables

| Artifact | Path |
|----------|------|
| Merged dataset | `data/secondary/reference_data.csv` |
| Dataset README | `data/secondary/README.md` |
| Semantic contract | `configs/semantic_contract_secondary.yaml` |
| Reference profile | `results/phase13/reference_profile.json` |
| Schema reference | `results/phase13/schema_reference.yaml` |
| Injected windows | `results/phase13/windows/` |
| Ground truth | `results/phase13/ground_truth.csv` |
| B0–B4 results | `results/phase13/results.json` |
| Experiment script | `experiments/run_phase13.py` |
| B4 FP attribution diagnostic (read-only) | `experiments/diagnose_phase13_b4.py` |
| B4 FP attribution output | `results/phase13/diagnose_b4_fp.json` |

## Test Suite

**136/136 passed** (130 original + 6 new rate-aware nullability tests).
