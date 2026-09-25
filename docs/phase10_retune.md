# Phase 10 Retune — Good-Faith Single-Adjustment Comparison

**Date:** August 2026  
**Scope:** One documented, non-iterative adjustment per detector to preempt the
reviewer question "would beating these change if you'd tuned them properly?"

---

## Protocol

- Same 10 seeds, same 110 windows/seed, same evaluation pipeline as Phase 10.
- ONE isolated parameter change per detector. No chaining, no grid search.
- If the change helps, we report it honestly. If it doesn't, we report that too.

---

## DDM: k=2.0 → k=3.0 (widen deviation threshold)

### Rationale

The Phase 10 root cause traced DDM's FPR=0.75 to a zero-baseline degeneracy:
for low-variance features (co, lpg, smoke), the 2000-row warm-up produces **zero**
surprise bits at k=2.0, setting p_min=0.0, s_min=0.0, and the drift threshold
to 0.0. Any nonzero surprise rate in the window immediately triggers drift.

Widening the deviation threshold to k=3.0 (still a standard choice in anomaly
detection, representing a 3σ band instead of 2σ) ensures the warm-up has a
nonzero surprise baseline, giving DDM a meaningful threshold to work with.

### What changed

| Parameter | Default | Retuned |
|-----------|---------|---------|
| DDM deviation k | 2.0 | 3.0 |

River DDM hyperparameters unchanged: warm_start=30, warning_threshold=2.0,
drift_threshold=3.0 (library defaults).

### Results (10 seeds, 110 windows/seed)

| Metric | Default (k=2.0) | Retuned (k=3.0) | B4 (reference) |
|--------|------------------|------------------|----------------|
| Raw F1 | 0.6537 ± 0.0358 | 0.6347 ± 0.0439 | 0.9055 ± 0.0177 |
| Raw Precision | 0.5577 ± 0.0248 | 0.5659 ± 0.0328 | 0.9922 ± 0.0096 |
| Raw Recall | 0.7900 ± 0.0559 | 0.7233 ± 0.0638 | 0.8333 ± 0.0298 |
| Raw FPR | 0.7500 ± 0.0300 | 0.6640 ± 0.0454 | 0.0080 ± 0.0098 |
| Adj F1 | 0.6259 ± 0.0385 | 0.6096 ± 0.0504 | 0.9945 ± 0.0055 |

### Outcome: (c) — Retuning barely changes anything

FPR dropped ~11% (0.75 → 0.66), but recall dropped ~8.5% (0.79 → 0.72).
Net F1 slightly *decreased* (0.654 → 0.635). The wider threshold suppresses
some false alarms but also misses more genuine drift — a net-zero trade.
B4's F1 (0.906) exceeds both DDM variants by >25 percentage points.
The zero-baseline degeneracy is partially mitigated but the fundamental
mismatch (DDM tracks cumulative error rate; IoT sensor data has genuine
regional variation that DDM interprets as "error") remains.

**One sentence:** DDM's FPR improves marginally at the cost of recall with no
net F1 gain; the algorithmic mismatch with this dataset is structural, not
a hyperparameter oversight.

---

## KSWIN: alpha=0.005 → alpha=0.0001 (Bonferroni correction)

### Rationale

The Phase 10 root cause traced KSWIN's FPR=0.96 to compound multiple testing:
with alpha=0.005, ~100 KS tests per window per feature, and OR-over-5-features,
the theoretical compound false-alarm probability is 91.8%.

Bonferroni correction: target ~5% compound rate across all features.  
Total tests per window = ~100 per feature × 5 features = 500.  
alpha_corrected = 0.05 / 500 = **0.0001**.

This is also the river docs example value (the docs note alpha should be
"below 0.01" and show an example with alpha=0.0001).

### What changed

| Parameter | Default | Retuned |
|-----------|---------|---------|
| KSWIN alpha | 0.005 | 0.0001 |

River KSWIN hyperparameters unchanged: window_size=100, stat_size=30
(library defaults). Adapter unchanged: stride=10, cold-start prefill, seed=42.

### Results (10 seeds, 110 windows/seed)

| Metric | Default (α=0.005) | Retuned (α=0.0001) | B4 (reference) |
|--------|--------------------|--------------------|----------------|
| Raw F1 | 0.6960 ± 0.0119 | 0.6428 ± 0.0259 | 0.9055 ± 0.0177 |
| Raw Precision | 0.5454 ± 0.0087 | 0.5770 ± 0.0295 | 0.9922 ± 0.0096 |
| Raw Recall | 0.9617 ± 0.0211 | 0.7267 ± 0.0327 | 0.8333 ± 0.0298 |
| Raw FPR | 0.9620 ± 0.0227 | 0.6420 ± 0.0729 | 0.0080 ± 0.0098 |
| Adj F1 | 0.6739 ± 0.0166 | 0.6088 ± 0.0268 | 0.9945 ± 0.0055 |

### Outcome: (c) — Retuning trades FPR for recall with no net F1 improvement

FPR dropped significantly (0.962 → 0.642, a 33% reduction), but recall also
dropped significantly (0.962 → 0.727, a 24% reduction). Net F1 decreased
(0.696 → 0.643). The tighter alpha suppresses false alarms but also makes
KSWIN less sensitive to genuine drift — a textbook precision-recall tradeoff.
Even at α=0.0001, KSWIN's FPR (0.642) is still 80× higher than B4's (0.008).
B4's F1 (0.906) exceeds both KSWIN variants by >26 percentage points.

The remaining high FPR at α=0.0001 is driven by genuine regional variation
in the IoT data: different spatial locations produce readings with different
distributions, and KSWIN's KS test detects these real differences as "drift"
even on value-unchanged windows. No single alpha value can fix this — it's
a property of the detector's sensitivity to distributional shifts in the data.

**One sentence:** KSWIN's FPR improves substantially at the cost of recall
with no net F1 gain; the remaining high FPR is genuine dataset behavior,
not a hyperparameter oversight.

---

## ADWIN: No change (already well-calibrated)

ADWIN's FPR=0.046 and recall=0.148 at library defaults are plausible and
require no retuning. Its adaptive windowing mechanism correctly handles the
data's natural variation.

---

## Side-by-Side Summary

| Detector | Variant | Raw F1 | Raw Recall | Raw FPR |
|----------|---------|--------|------------|---------|
| ADWIN | default | 0.250 | 0.148 | 0.046 |
| DDM | default (k=2) | 0.654 | 0.790 | 0.750 |
| DDM | retuned (k=3) | 0.635 | 0.723 | 0.664 |
| KSWIN | default (α=.005) | 0.696 | 0.962 | 0.962 |
| KSWIN | retuned (α=.0001) | 0.643 | 0.727 | 0.642 |
| **B4** | **existing validated** | **0.906** | **0.833** | **0.008** |

**Conclusion:** Both retuning attempts trade FPR for recall with no net F1
improvement. B4's advantage is robust to one-shot retuning of either
classical detector. The comparison table is final.

---

## Test Suite

```
PYTHONPATH=. python -m pytest tests/test_classical_detectors.py -v
```

All existing tests pass. The retune parameters do not alter any test because
tests exercise the default-configuration adapters (the retuned variants are
only used via the explicit ddm_k/kswin_alpha overrides in evaluate_window).

---

*This document is the deliverable for the Phase 10 retune follow-up.*
