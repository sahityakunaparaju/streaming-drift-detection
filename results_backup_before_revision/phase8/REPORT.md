# Phase 8 — Semantic Blind-Spot Experiment (Section T)

Semantic Blind-Spot Recovery is a NOVEL, NON-STANDARD metric purpose-built for this thesis (Section T of the research plan). It is not a published literature metric and is not compared against any published baseline; it quantifies the coverage contribution of the semantic detector over the blind spot left by the schema and statistical detectors.

## Headline

**Semantic Blind-Spot Recovery = 1.0000 ± 0.0000 across 10 seeds**, computed over 50 isolated cases total (5 per seed), where schema and statistical detectors combined missed 50 of 50 cases (W) and the semantic detector recovered 50 of them (V).

This is a **NOVEL, NON-STANDARD metric** purpose-built for this thesis (Section T). It is not a published literature metric and is not compared against any published baseline.

## Per-seed recovery

| seed | z | n_isolated | n_excluded | w | v | recovery |
|---|---|---|---|---|---|---|
| 42 | 5 | 5 | 0 | 5 | 5 | 1.0 |
| 123 | 5 | 5 | 0 | 5 | 5 | 1.0 |
| 7 | 5 | 5 | 0 | 5 | 5 | 1.0 |
| 11 | 5 | 5 | 0 | 5 | 5 | 1.0 |
| 2024 | 5 | 5 | 0 | 5 | 5 | 1.0 |
| 314 | 5 | 5 | 0 | 5 | 5 | 1.0 |
| 2718 | 5 | 5 | 0 | 5 | 5 | 1.0 |
| 999 | 5 | 5 | 0 | 5 | 5 | 1.0 |
| 555 | 5 | 5 | 0 | 5 | 5 | 1.0 |
| 777 | 5 | 5 | 0 | 5 | 5 | 1.0 |

## Per-detector recall (isolated blind-spot cases)

| detector | config | recall_mean | recall_std | min | max | per_seed |
|---|---|---|---|---|---|---|
| schema | B1_schema | 0.0 | 0.0 | 0.0 | 0.0 | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] |
| statistical | B2_statistical | 0.0 | 0.0 | 0.0 | 0.0 | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] |
| semantic | semantic | 1.0 | 0.0 | 1.0 | 1.0 | [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] |
| combined | B4_unified | 1.0 | 0.0 | 1.0 | 1.0 | [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] |

## Isolation verification (per case, per seed)

Every blind-spot case was verified BEFORE metric computation: schema comparison identical (0 events), and actual measured KS/PSI/chi2 for the affected column fall below the frozen thresholds (ks_floor = 0.50, decision_threshold = 0.25, psi_scale = 72.0). Full per-case numbers: `isolation_verification.csv`.

### Excluded / flagged cases: 0

None — every blind-spot case was a clean isolation.

## Controls (clean / normal_noise)

| seed | event_id | set | subtype | feature | severity | ground_truth | start_row | schema_identical | schema_events | schema_event_types | statistical_score | max_ks | max_psi | max_chi2 | affected_ks | affected_psi | affected_chi2 | semantic_fired | semantic_event_types | isolated | B1_schema | B2_statistical | semantic | B4_unified |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | bs-clean | control | clean | None | none | NO_DRIFT | 385507 | True | 0 | [] | 0.189609 | 0.352139 | 13.651867 | 1.533078 | nan | nan | nan | False | [] | True | NO_DRIFT | NO_DRIFT | NO_DRIFT | NO_DRIFT |
| 42 | bs-normal_noise | control | normal_noise | temp | none | NO_DRIFT | 393128 | True | 0 | [] | 0.143452 | 0.24393 | 10.328574 | 0.937349 | nan | nan | nan | False | [] | True | NO_DRIFT | NO_DRIFT | NO_DRIFT | NO_DRIFT |
| 123 | bs-clean | control | clean | None | none | NO_DRIFT | 378346 | True | 0 | [] | 0.166926 | 0.421786 | 12.018661 | 3.394872 | nan | nan | nan | False | [] | True | NO_DRIFT | NO_DRIFT | NO_DRIFT | NO_DRIFT |
| 123 | bs-normal_noise | control | normal_noise | temp | none | NO_DRIFT | 397075 | True | 0 | [] | 0.152836 | 0.265818 | 11.004157 | 0.87652 | nan | nan | nan | False | [] | True | NO_DRIFT | NO_DRIFT | NO_DRIFT | NO_DRIFT |
| 7 | bs-clean | control | clean | None | none | NO_DRIFT | 395145 | True | 0 | [] | 0.151311 | 0.274965 | 10.894418 | 1.964619 | nan | nan | nan | False | [] | True | NO_DRIFT | NO_DRIFT | NO_DRIFT | NO_DRIFT |
| 7 | bs-normal_noise | control | normal_noise | temp | none | NO_DRIFT | 399871 | True | 0 | [] | 0.182922 | 0.312725 | 13.170369 | 0.137039 | nan | nan | nan | False | [] | True | NO_DRIFT | NO_DRIFT | NO_DRIFT | NO_DRIFT |
| 11 | bs-clean | control | clean | None | none | NO_DRIFT | 380681 | True | 0 | [] | 0.184049 | 0.271218 | 13.251514 | 4.024916 | nan | nan | nan | False | [] | True | NO_DRIFT | NO_DRIFT | NO_DRIFT | NO_DRIFT |
| 11 | bs-normal_noise | control | normal_noise | temp | none | NO_DRIFT | 388316 | True | 0 | [] | 0.187698 | 0.247045 | 13.514279 | 0.47065 | nan | nan | nan | False | [] | True | NO_DRIFT | NO_DRIFT | NO_DRIFT | NO_DRIFT |
| 2024 | bs-clean | control | clean | None | none | NO_DRIFT | 377836 | True | 0 | [] | 0.163852 | 0.330415 | 11.797342 | 2.810233 | nan | nan | nan | False | [] | True | NO_DRIFT | NO_DRIFT | NO_DRIFT | NO_DRIFT |
| 2024 | bs-normal_noise | control | normal_noise | temp | none | NO_DRIFT | 397045 | True | 0 | [] | 0.152822 | 0.263703 | 11.003194 | 0.678274 | nan | nan | nan | False | [] | True | NO_DRIFT | NO_DRIFT | NO_DRIFT | NO_DRIFT |
| 314 | bs-clean | control | clean | None | none | NO_DRIFT | 383797 | True | 0 | [] | 0.190247 | 0.396951 | 13.697799 | 0.642523 | nan | nan | nan | False | [] | True | NO_DRIFT | NO_DRIFT | NO_DRIFT | NO_DRIFT |
| 314 | bs-normal_noise | control | normal_noise | temp | none | NO_DRIFT | 398124 | True | 0 | [] | 0.189392 | 0.281041 | 13.636234 | 0.477659 | nan | nan | nan | False | [] | True | NO_DRIFT | NO_DRIFT | NO_DRIFT | NO_DRIFT |
| 2718 | bs-clean | control | clean | None | none | NO_DRIFT | 386224 | True | 0 | [] | 0.186236 | 0.330415 | 13.408959 | 1.682035 | nan | nan | nan | False | [] | True | NO_DRIFT | NO_DRIFT | NO_DRIFT | NO_DRIFT |
| 2718 | bs-normal_noise | control | normal_noise | temp | none | NO_DRIFT | 393631 | True | 0 | [] | 0.135158 | 0.255677 | 9.731382 | 0.728821 | nan | nan | nan | False | [] | True | NO_DRIFT | NO_DRIFT | NO_DRIFT | NO_DRIFT |
| 999 | bs-clean | control | clean | None | none | NO_DRIFT | 385888 | True | 0 | [] | 0.187007 | 0.329415 | 13.464515 | 1.34571 | nan | nan | nan | False | [] | True | NO_DRIFT | NO_DRIFT | NO_DRIFT | NO_DRIFT |
| 999 | bs-normal_noise | control | normal_noise | temp | none | NO_DRIFT | 389629 | True | 0 | [] | 0.158837 | 0.29042 | 11.436265 | 1.113699 | nan | nan | nan | False | [] | True | NO_DRIFT | NO_DRIFT | NO_DRIFT | NO_DRIFT |
| 555 | bs-clean | control | clean | None | none | NO_DRIFT | 390607 | True | 0 | [] | 0.190304 | 0.261819 | 13.701868 | 0.794205 | nan | nan | nan | False | [] | True | NO_DRIFT | NO_DRIFT | NO_DRIFT | NO_DRIFT |
| 555 | bs-normal_noise | control | normal_noise | temp | none | NO_DRIFT | 399875 | True | 0 | [] | 0.182911 | 0.309725 | 13.169588 | 0.137039 | nan | nan | nan | False | [] | True | NO_DRIFT | NO_DRIFT | NO_DRIFT | NO_DRIFT |
| 777 | bs-clean | control | clean | None | none | NO_DRIFT | 372775 | True | 0 | [] | 0.152837 | 0.238274 | 11.004242 | 3.25582 | nan | nan | nan | False | [] | True | NO_DRIFT | NO_DRIFT | NO_DRIFT | NO_DRIFT |
| 777 | bs-normal_noise | control | normal_noise | temp | none | NO_DRIFT | 396858 | True | 0 | [] | 0.153593 | 0.261988 | 11.058724 | 0.588587 | nan | nan | nan | False | [] | True | NO_DRIFT | NO_DRIFT | NO_DRIFT | NO_DRIFT |

## Method notes

- Definition-change gap: **option (a) taken** — a `definition` field was added to `temp` in `configs/semantic_contract.yaml` ("instantaneous_raw_sensor_value"), the reference profile was regenerated, and the semantic detector gained an exact-equality `definition_change` rule mirroring `business_meaning_change`. The frozen `configs/thresholds.yaml` was NOT modified; its Phase 6 note about the definition gap is superseded by this Phase 8 extension.
- Windows drawn from rows [320000, 405184), disjoint from Phase 7 regions; a full scan shows no 1000-row window there reaches the statistical decision threshold (max 0.215 < 0.25).
- Seed-aware window placement via `sample_non_overlapping_starts` (same construction as Phase 7, Section Y).
- Out of scope respected: no frozen threshold was modified, Phase 7 baseline results untouched, no ablation (Phase 9), mean/variance/distribution aggregation limitation not re-litigated.