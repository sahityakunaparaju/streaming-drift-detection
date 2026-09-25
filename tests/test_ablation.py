"""
Phase 9 (Section U) ablation tests.

Guards the ablation reframing: the delta table structure, the
sign/magnitude of the B3->B4 semantic contribution, the claim that
the adjusted B3->B4 recall gain is fully concentrated in semantic
subtypes (cross-checking Phase 8), and that the B4 recomputation from
cached per-window decisions matches the cached aggregates.
"""

import pandas as pd

from evaluation.ablation import (
    PHASE7_ROOT,
    aggregate_subtype_recall,
    build_ablation_table,
    build_subtype_delta,
    concentration,
    load_ladder,
    validate_against_cached,
)

SEEDS = [42, 123, 7, 11, 2024, 314, 2718, 999, 555, 777]


def _pipeline():
    ladder = load_ladder()
    ablation_df = build_ablation_table(ladder)
    subtype_recall = aggregate_subtype_recall(SEEDS)
    mismatches = validate_against_cached(subtype_recall, SEEDS)
    subtype_df = build_subtype_delta(subtype_recall)
    conc = concentration(subtype_df)
    return ablation_df, subtype_df, mismatches, conc


def test_ablation_table_has_all_steps():
    ablation_df, _subtype_df, _m, _c = _pipeline()
    for variant in ["raw", "adjusted"]:
        steps = set(
            ablation_df[
                (ablation_df["variant"] == variant)
                & (ablation_df["kind"] == "absolute")
            ]["step"]
        )
        assert {"B0_null", "B1_schema", "B2_statistical",
                "B3_schema_statistical", "B4_unified"} <= steps
        deltas = set(
            ablation_df[
                (ablation_df["variant"] == variant)
                & (ablation_df["kind"] == "delta")
            ]["step"]
        )
        assert deltas == {
            "B0_null->B1_schema",
            "B1_schema->B3_schema_statistical",
            "B3_schema_statistical->B4_unified",
        }


def test_semantic_step_improves_adjusted_f1():
    ablation_df, _subtype_df, _m, _c = _pipeline()
    row = ablation_df[
        (ablation_df["step"] == "B3_schema_statistical->B4_unified")
        & (ablation_df["variant"] == "adjusted")
    ].iloc[0]
    delta_mean = float(row["delta_f1"].split(" +/- ")[0])
    assert delta_mean > 0.2


def test_adjusted_gain_fully_concentrated_in_semantic_subtypes():
    """Phase 8 cross-check: 100% of the adjusted B3->B4 recall gain
    must sit in semantic-drift subtypes."""
    _ablation_df, _subtype_df, _m, conc = _pipeline()
    adjusted = conc["adjusted"]
    assert adjusted["semantic_share"] == 1.0
    assert adjusted["total_gain"] == 5.0


def test_b4_recompute_matches_cached_aggregates():
    _ablation_df, _subtype_df, mismatches, _c = _pipeline()
    assert mismatches == []


def test_subtype_delta_only_positive_in_semantic_subtypes():
    """Adjusted delta must be zero everywhere except semantic
    subtypes (B4 is a superset of B3, so deltas are >= 0)."""
    _ablation_df, subtype_df, _m, _c = _pipeline()
    adjusted = subtype_df[subtype_df["variant"] == "adjusted"]
    nonzero = adjusted[adjusted["delta_mean"] > 0]
    assert len(nonzero) > 0
    assert nonzero["semantic_subtype"].all()


def test_lag_data_exists_for_all_ladder_configs():
    ladder = load_ladder()
    lag_rows = ladder[ladder["variant"] == "lag"]
    configs = set(lag_rows["config"])
    assert {
        "B0_null", "B1_schema", "B2_statistical",
        "B3_schema_statistical", "B4_unified",
    } <= configs
