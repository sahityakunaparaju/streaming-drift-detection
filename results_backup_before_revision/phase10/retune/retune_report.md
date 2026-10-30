# Phase 10 Retune — Good-Faith Single-Adjustment Comparison

One documented adjustment per detector, no iteration. DDM: k=2.0→3.0 (widen deviation threshold). KSWIN: alpha=0.005→0.0001 (Bonferroni correction for ~500 tests/window targeting ~5% compound FPR).

## Side-by-side comparison

| detector | variant | raw_f1 | raw_precision | raw_recall | raw_fpr | adjusted_f1 | adjusted_precision | adjusted_recall | adjusted_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ADWIN | default | 0.2496 +/- 0.0282 | 0.7995 +/- 0.0732 | 0.1483 +/- 0.0189 | 0.0460 +/- 0.0201 | 0.1770 +/- 0.0454 | 0.6935 +/- 0.1415 | 0.1022 +/- 0.0285 | 0.0512 +/- 0.0255 |
| DDM | default | 0.6537 +/- 0.0358 | 0.5577 +/- 0.0248 | 0.7900 +/- 0.0559 | 0.7500 +/- 0.0300 | 0.6259 +/- 0.0385 | 0.5259 +/- 0.0271 | 0.7733 +/- 0.0603 | 0.7634 +/- 0.0328 |
| DDM | retuned_k3 | 0.6347 +/- 0.0439 | 0.5659 +/- 0.0328 | 0.7233 +/- 0.0638 | 0.6640 +/- 0.0454 | 0.6096 +/- 0.0504 | 0.5353 +/- 0.0385 | 0.7089 +/- 0.0720 | 0.6732 +/- 0.0514 |
| KSWIN | default | 0.6960 +/- 0.0119 | 0.5454 +/- 0.0087 | 0.9617 +/- 0.0211 | 0.9620 +/- 0.0227 | 0.6739 +/- 0.0166 | 0.5213 +/- 0.0123 | 0.9533 +/- 0.0306 | 0.9610 +/- 0.0331 |
| KSWIN | retuned_alpha0001 | 0.6428 +/- 0.0259 | 0.5770 +/- 0.0295 | 0.7267 +/- 0.0327 | 0.6420 +/- 0.0729 | 0.6088 +/- 0.0268 | 0.5438 +/- 0.0291 | 0.6933 +/- 0.0407 | 0.6415 +/- 0.0764 |
| B4 | existing_validated | 0.9055 +/- 0.0177 | 0.9922 +/- 0.0096 | 0.8333 +/- 0.0298 | 0.0080 +/- 0.0098 | 0.9945 +/- 0.0055 | 0.9935 +/- 0.0100 | 0.9956 +/- 0.0089 | 0.0073 +/- 0.0112 |

## Interpretation

### DDM (k=2.0 → k=3.0)
- Default FPR: 0.7500 +/- 0.0300
- Retuned FPR: 0.6640 +/- 0.0454
- Outcome: see overall table for full metrics. One isolated k adjustment.

### KSWIN (alpha=0.005 → alpha=0.0001)
- Default FPR: 0.9620 +/- 0.0227
- Retuned FPR: 0.6420 +/- 0.0729
- Outcome: see overall table for full metrics. One isolated alpha adjustment.

### B4 (reference)
- FPR: 0.0080 +/- 0.0098
