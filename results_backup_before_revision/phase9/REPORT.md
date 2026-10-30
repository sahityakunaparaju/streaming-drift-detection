# Phase 9 — Ablation Study (Section U)

Adding semantic detection to the schema+statistical baseline improves adjusted F1 by 0.2339 +/- 0.0025 (B3 0.7606 +/- 0.0069 -> B4 0.9945 +/- 0.0055, 30.7% relative), with 100.0% of the adjusted recall gain concentrated in semantic-specific drift subtypes — corroborating the isolated Semantic Blind-Spot Recovery = 1.0 result from Phase 8 using independent evidence (raw F1: B3 0.6457 +/- 0.0100 -> B4 0.9055 +/- 0.0177, delta 0.2598 +/- 0.0156).

## Incremental deltas (per-seed mean +/- std, 10 seeds)

### Adjusted F1

- **B0_null** (null monitoring only): P=1.0000 +/- 0.0000, R=0.0667 +/- 0.0000, F1=0.1250 +/- 0.0000, FPR=0.0000 +/- 0.0000
- **B1_schema** (+ schema detection): P=1.0000 +/- 0.0000, R=0.3333 +/- 0.0000, F1=0.5000 +/- 0.0000, FPR=0.0000 +/- 0.0000
- **B2_statistical** (+ statistical detection (context row)): P=0.9786 +/- 0.0327, R=0.2889 +/- 0.0172, F1=0.4457 +/- 0.0207, FPR=0.0073 +/- 0.0112
- **B3_schema_statistical** (+ statistical on top of schema): P=0.9897 +/- 0.0158, R=0.6178 +/- 0.0089, F1=0.7606 +/- 0.0069, FPR=0.0073 +/- 0.0112
- **B4_unified** (+ semantic on top of schema+statistical): P=0.9935 +/- 0.0100, R=0.9956 +/- 0.0089, F1=0.9945 +/- 0.0055, FPR=0.0073 +/- 0.0112
- **B0_null->B1_schema** (adding schema detection): dF1=0.3750 +/- 0.0000 (300.0% relative), dR=0.2667 +/- 0.0000, dP=0.0000 +/- 0.0000, dFPR=0.0000 +/- 0.0000
- **B1_schema->B3_schema_statistical** (adding statistical detection (on top of schema)): dF1=0.2606 +/- 0.0069 (52.1% relative), dR=0.2844 +/- 0.0089, dP=-0.0103 +/- 0.0158, dFPR=0.0073 +/- 0.0112
- **B3_schema_statistical->B4_unified** (adding semantic detection (on top of schema+statistical)): dF1=0.2339 +/- 0.0025 (30.7% relative), dR=0.3778 +/- 0.0000, dP=0.0038 +/- 0.0058, dFPR=0.0000 +/- 0.0000

Full table: `ablation_table.csv` (raw and adjusted).

## B3->B4 subtype delta + Phase 8 cross-check

B3 (schema+statistical) per-subtype recall was NOT aggregated in Phase 7 (only schema/statistical/semantic/combined were); it is recomputed here from the cached per-window decisions in `results/phase7/multiseed/seed-<N>/rows.csv` — no new experiment was run. The B4 recomputation was validated against the cached 'combined' rows (0 mismatches).

`subtype_delta.csv`; concentration of the gain in semantic-drift subtypes:

- raw: total B3->B4 recall gain 5.8400, of which 5.0000 (85.6%) in semantic-drift subtypes
- adjusted: total B3->B4 recall gain 5.0000, of which 5.0000 (100.0%) in semantic-drift subtypes

Note: definition_change is a Phase 8-only subtype (absent from the Phase 7 matrix), so it cannot contribute to this cross-check; the check covers the other Phase 8 blind-spot subtypes (unit_change, business_meaning_change, range_change, allowed_domain_change) plus the value-level semantic subtypes (unit_conversion, range_violation, domain_violation).

## Detection lag

`lag.csv`; per-config lag mean/median/p95 across seeds:

| config | metric | mean | std | min | max | per_seed |
|---|---|---|---|---|---|---|
| B0_null | lag_mean | 0.0 | 0.0 | 0.0 | 0.0 | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] |
| B0_null | lag_median | 0.0 | 0.0 | 0.0 | 0.0 | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] |
| B0_null | lag_p95 | 0.0 | 0.0 | 0.0 | 0.0 | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] |
| B1_schema | lag_mean | 0.0 | 0.0 | 0.0 | 0.0 | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] |
| B1_schema | lag_median | 0.0 | 0.0 | 0.0 | 0.0 | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] |
| B1_schema | lag_p95 | 0.0 | 0.0 | 0.0 | 0.0 | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] |
| B2_statistical | lag_mean | 0.0 | 0.0 | 0.0 | 0.0 | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] |
| B2_statistical | lag_median | 0.0 | 0.0 | 0.0 | 0.0 | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] |
| B2_statistical | lag_p95 | 0.0 | 0.0 | 0.0 | 0.0 | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] |
| B3_schema_statistical | lag_mean | 0.0 | 0.0 | 0.0 | 0.0 | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] |
| B3_schema_statistical | lag_median | 0.0 | 0.0 | 0.0 | 0.0 | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] |
| B3_schema_statistical | lag_p95 | 0.0 | 0.0 | 0.0 | 0.0 | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] |
| B4_unified | lag_mean | 0.03334 | 0.044721 | 0.0 | 0.1333 | [0.0, 0.0, 0.0, 0.1333, 0.0667, 0.0, 0.0667, 0.0, 0.0, 0.0667] |
| B4_unified | lag_median | 0.0 | 0.0 | 0.0 | 0.0 | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] |
| B4_unified | lag_p95 | 0.19 | 0.301496 | 0.0 | 1.0 | [0.0, 0.0, 0.0, 1.0, 0.3, 0.0, 0.3, 0.0, 0.0, 0.3] |

Streams with lag > 0 (all in B4_unified — streams only B4 detects, first flagged at the second drifted window):

| seed | config | stream_id | lag |
|---|---|---|---|
| 11 | B4_unified | lag-variance_shift | 1 |
| 11 | B4_unified | lag-distribution_shift | 1 |
| 2024 | B4_unified | lag-variance_shift | 1 |
| 2718 | B4_unified | lag-variance_shift | 1 |
| 777 | B4_unified | lag-variance_shift | 1 |

Plain statement: lag is structurally ~0 in this design — every stream any config detects is flagged at or within one window of onset, and adding dimensions changes recall/precision, not lag. The only nonzero lags are B4-only detections (streams the lower rungs never detect at all), not delayed detection of streams the lower rungs already detect.