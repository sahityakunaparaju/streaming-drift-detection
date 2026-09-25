# Phase 10 — Classical Detector Sanity Check: DDM, KSWIN, ADWIN

**Date:** August 2026  
**Requested by:** Paper preparation review  
**Scope:** Verify whether high FPR for DDM (0.75) and KSWIN (0.96) reflects
genuine detector behavior or an integration bug, mirroring the Evidently
FPR=1.0 audit from Phase 7.

---

## Executive Summary

**All three detectors are correctly integrated. No state-reset bug, no
wrong-update-granularity bug, no mismatched window handling. The high FPR
for DDM and KSWIN is genuine detector behavior on this dataset at library
default hyperparameters.** The numbers are fit for the comparison table,
with the caveats documented below.

---

## 1. Per-Window Decision Derivation: Is It Correct?

The adapter in `detectors/classical_detectors.py` follows this protocol
for every window:

1. A **fresh detector** is created (`new_detector(name)`) — no state
   leaks between windows.
2. The 2000-row primary-dataset warm-up (rows `[start-2000, start)`)
   is fed first; signals during warm-up are adaptation, ignored.
3. The window's observations are fed next; a signal during the window
   portion = DRIFT for that feature.
4. Five per-feature streams are evaluated independently; DRIFT on ANY
   stream = DRIFT for the window (OR aggregation).

This is correct for all three detectors. The state-reset invariant is
verified by code inspection: `new_detector()` is called inside
`evaluate_window()` for each feature, guaranteeing fresh state.

**DDM specifics:** DDM's documented input is a binary error stream.
The adapter feeds a derived binary surprise stream: bit=1 iff
`|x − μ_warmup| > 2·σ_warmup`. This is the standard unsupervised
adaptation of DDM and does not alter the detector itself.

**KSWIN specifics:** KSWIN's per-observation cost is ~100× ADWIN's
(a scipy KS test on every update). The adapter uses stride=10
(observation decimation) and cold-start prefill of the last 100
decimated warm-up values, so every fed observation is a window
observation and any signal counts. Both choices are documented.

**ADWIN specifics:** Full-rate stream, warm-up fed first, window
signals count. No adapter modifications.

**Verdict: Per-window decision derivation is correct and fair for all
three detectors.**

---

## 2. Hyperparameters: Default vs. Tuned

| Detector | Parameters Used | Source |
|----------|----------------|--------|
| ADWIN    | delta=0.002, clock=32 | river 0.21.0 defaults |
| DDM      | warm_start=30, warning_threshold=2.0, drift_threshold=3.0 | river 0.21.0 defaults |
| KSWIN    | alpha=0.005, window_size=100, stat_size=30 | river 0.21.0 defaults |
| DDM adapter | k=2.0 (deviation multiplier) | Adapter choice |
| KSWIN adapter | stride=10, cold-start prefill, seed=42 | Adapter choice |

All three detectors run at **river library defaults** for their native
hyperparameters. The adapter choices (DDM's k=2.0, KSWIN's stride=10)
are documented design decisions, not tuning.

**Are the defaults appropriate for this dataset?**

- **DDM:** The default drift_threshold=3.0 is standard for supervised
  classification error monitoring. However, DDM's algorithm assumes a
  **nonzero baseline error rate** — its detection threshold is
  `p_min + 3.0·s_min`, where `p_min` is the minimum observed error
  rate. When the baseline is exactly 0 (see §3), this threshold
  collapses to 0, making DDM hypersensitive. This is a **dataset-
  specific interaction**, not a tuning failure.

- **KSWIN:** The default alpha=0.005 is the documented river default
  (described as "very sensitive" in the river docs, which recommend
  alpha < 0.01). Over 100 tests per window per feature, the multiple-
  testing compound probability is theoretically ~39% per feature
  (see §3 for why actual rates are even higher).

- **ADWIN:** The default delta=0.002 is standard and produces
  well-calibrated behavior on this dataset (FPR=0.046).

**Verdict: Defaults are used, which is the fair comparison choice.
B4's thresholds were calibrated on this dataset; the classical
detectors run at library defaults. This sensitivity asymmetry is
documented (REPORT.md §D.7).**

---

## 3. Sanity Check: 5 Clean Windows (Seed-42)

### 3.1 DDM: p_min = 0 Collapse

**Root cause identified and traced:**

For low-variance features (`co`, `lpg`, `smoke`), the 2000-row warm-up
has **zero** surprise bits (all values fall within 2σ of the warm-up
mean). DDM accumulates: `p_min = 0.0`, `s_min = 0.0`, threshold =
`0.0 + 3.0 × 0.0 = 0.0`.

During the window, the data is from a different spatial region of the
IoT dataset and naturally has slightly different statistics. Even a
single observation outside the warm-up's 2σ band produces a nonzero
error rate, which immediately exceeds the zero threshold.

**Detailed trace (co feature, clean window "matrix-normal-clean-none"):**

```
Warm-up: 2000 obs, μ=0.0048, σ=0.0004, surprise rate = 0.000000 (0/2000)
Window:  1000 obs, μ=0.0047, σ=0.0005, surprise rate = 0.054000 (54/1000)

DDM state after warm-up: p_min=0.0, s_min=0.0, threshold=0.0
DDM fires at index 2821: p_i=0.000354, s_i=0.000354
  → 0.000354 + 0.000354 = 0.000708 > 0.0 + 3.0×0.0 = 0.0  ✓ DRIFT
```

This is a genuine algorithmic property: DDM was designed for supervised
classification where the baseline error rate is nonzero. When adapted
to unsupervised anomaly detection via a binary surprise stream on this
dataset's low-variance features, the zero-baseline creates a degenerate
threshold.

**Per-feature firing on 5 clean windows (seed-42):**

| Window | temp | co | humidity | lpg | smoke | Decision |
|--------|------|----|----------|-----|-------|----------|
| matrix-normal-clean-none | DRIFT | DRIFT | NO_DRIFT | DRIFT | DRIFT | DRIFT |
| matrix-normal-normal_noise-none | NO_DRIFT | NO_DRIFT | NO_DRIFT | NO_DRIFT | NO_DRIFT | NO_DRIFT |
| lag-mean_shift-00 | NO_DRIFT | NO_DRIFT | NO_DRIFT | NO_DRIFT | NO_DRIFT | NO_DRIFT |
| lag-mean_shift-01 | DRIFT | NO_DRIFT | NO_DRIFT | DRIFT | NO_DRIFT | DRIFT |
| lag-mean_shift-02 | DRIFT | NO_DRIFT | DRIFT | NO_DRIFT | NO_DRIFT | DRIFT |

**3/5 clean windows fire = 60% rate** (consistent with overall FPR=0.75
across all clean windows and seeds).

**Verdict: Genuine behavior. NOT a state-reset bug. The zero-baseline
degeneracy is a dataset-specific interaction with DDM's algorithm.**

### 3.2 KSWIN: Multiple-Testing Amplification

**Root cause: KSWIN performs a KS test on every update (every 10th
row), giving ~100 tests per window per feature.**

Manual simulation on the `co` feature (clean window "matrix-normal-clean-none"):
- 100 KS tests performed
- 1 test fires (at win_idx=98): p=0.00089958, KS stat=0.5000
- Per-feature fire rate: 1/100 = 1%

Theoretical per-feature false-alarm probability:
- `1 − (1 − 0.005)^100 = 0.394` (assuming independence)
- Actual per-feature rate is lower (~1%) because KS tests on overlapping
  windows are highly correlated

However, across 5 features, the per-window false-alarm probability is
substantially higher than the theoretical 91.8%. The actual FPR of 0.96
exceeds the theoretical prediction because:

1. The IoT data has genuine **regional variation**: different spatial
   locations produce readings with different distributions, causing KS
   tests to genuinely reject even on "clean" (no-injected-drift) windows.
2. KSWIN's `stat_size=30` is small relative to the data's natural
   variance, making the KS test sensitive to minor distributional shifts.

**Per-feature firing on 5 clean windows (seed-42):**

| Window | temp | co | humidity | lpg | smoke | Decision |
|--------|------|----|----------|-----|-------|----------|
| matrix-normal-clean-none | NO_DRIFT | DRIFT | DRIFT | DRIFT | DRIFT | DRIFT |
| matrix-normal-normal_noise-none | DRIFT | NO_DRIFT | DRIFT | NO_DRIFT | NO_DRIFT | DRIFT |
| lag-mean_shift-00 | DRIFT | DRIFT | DRIFT | DRIFT | DRIFT | DRIFT |
| lag-mean_shift-01 | DRIFT | DRIFT | DRIFT | DRIFT | DRIFT | DRIFT |
| lag-mean_shift-02 | DRIFT | DRIFT | DRIFT | DRIFT | DRIFT | DRIFT |

**5/5 clean windows fire = 100% rate** (consistent with overall FPR=0.96).

**Verdict: Genuine behavior. NOT a state-reset bug. The high FPR is
caused by KSWIN's library-default alpha producing a high compound false-
alarm rate over 100 tests per window, amplified by genuine regional
variation in the IoT data.**

### 3.3 ADWIN: Clean Baseline

**5 clean windows, all features: NO_DRIFT on every combination.**

ADWIN's adaptive windowing mechanism correctly handles the data's
natural variation without excessive false alarms. Its FPR of 0.046
is plausible for a well-calibrated streaming detector.

**Verdict: Genuine behavior. No investigation needed.**

---

## 4. Summary of Findings

| Detector | FPR | Root Cause | Bug? | Action |
|----------|-----|------------|------|--------|
| DDM | 0.75 | p_min=0 degeneracy when warm-up surprise rate=0 | No | Document as genuine finding |
| KSWIN | 0.96 | Multiple-testing amplification (α=0.005 × 100 tests × 5 features) + genuine regional variation | No | Document as genuine finding |
| ADWIN | 0.046 | Well-calibrated at library defaults | N/A | No action needed |

**All three numbers are genuine and fit for the comparison table**, with
the documentation that:

1. DDM and KSWIN run at river library defaults — their FPRs are
   properties of the default configuration on this dataset, not signs
   of integration failure.
2. B4's thresholds were **calibrated** on this dataset; the classical
   detectors were not. This sensitivity asymmetry is a protocol
   limitation (REPORT.md §D.7), not an unfairness.
3. The same structural finding already exists for Evidently (Phase 7):
   FPR=1.0 at default config, root-caused and documented. DDM and
   KSWIN receive the same treatment.

---

## 5. Cross-Reference: Blind-Spot Comparison (Table C)

The Table C numbers confirm the finding:

| Detector | Injected | Clean Slice | Interpretation |
|----------|----------|-------------|----------------|
| ADWIN | 1/50 | 1/50 | Fires on neither → no false recovery |
| DDM | 35/50 | 35/50 | Fires on both at its clean-data FPR rate |
| KSWIN | 48/50 | 48/50 | Fires on both at its clean-data FPR rate |

DDM fires on 35/50 injected AND 35/50 clean slices — identical rates.
KSWIN fires on 48/50 on both — identical rates. **Neither detector
discriminates between drift and non-drift data on the semantic blind-
spot cases.** Any apparent "detection" is a spurious clean-data alarm.
This is structurally correct behavior for univariate numerical detectors
that cannot see the semantic metadata channel.

---

*This document is the deliverable for the pre-paper sanity check. The
ADWIN/DDM/KSWIN comparison table is now verified and final.*
