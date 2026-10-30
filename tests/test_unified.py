from detectors.unified import (
    channel_score,
    decide,
    statistical_score,
    unified_score,
)


def make_statistical_result(
    numeric=None,
    categorical=None,
    **extra,
):
    return {
        "numeric": numeric or {},
        "categorical": categorical or {},
        **extra,
    }


def test_schema_channel_is_continuous():
    assert channel_score(False) == 0.0
    assert channel_score(True) == 1.0
    assert channel_score([]) == 0.0
    assert channel_score([{"severity": "high"}]) == 1.0
    assert 0.0 < channel_score([{"severity": "low"}]) < 1.0
    # Severity weights are capped at 1.0.
    assert channel_score(
        [{"severity": "high"}, {"severity": "high"}]
    ) == 1.0


def test_unified_score_is_mean_of_continuous_components():
    result = unified_score(
        [{"severity": "high"}],
        make_statistical_result(statistical_score=0.5),
        [{"severity": "low"}],
    )
    expected = (
        result["schema_score"]
        + result["statistical_score"]
        + result["semantic_score"]
    ) / 3.0
    assert result["unified_score"] == expected
    assert 0.0 <= result["unified_score"] <= 1.0


def test_statistical_score_prefers_per_column_score():
    details = make_statistical_result(statistical_score=0.42)
    assert statistical_score(details) == 0.42


def test_statistical_score_legacy_effect_fallback():
    details = make_statistical_result(
        numeric={
            "temp": {
                "effect_size": 0.2,
                "threshold": 0.1,
            }
        }
    )
    assert statistical_score(details) == 1.0  # capped at 1


def test_statistical_score_legacy_raw_ks_fallback():
    details = make_statistical_result(
        numeric={
            "temp": {
                "ks_statistic": 0.01,
                "psi": 0.01,
            }
        }
    )
    score = statistical_score(details)
    assert 0.0 <= score < 1.0


def test_schema_channel_forces_drift():
    assert (
        decide(
            None,
            schema_score=0.3,
            semantic_score=0.0,
            statistical_score=0.0,
        )
        == "DRIFT"
    )


def test_semantic_channel_forces_drift_alone():
    assert (
        decide(
            None,
            schema_score=0.0,
            semantic_score=0.3,
            statistical_score=0.0,
        )
        == "DRIFT"
    )


def test_statistical_drift_flag_forces_drift_alone():
    assert (
        decide(
            None,
            schema_score=0.0,
            semantic_score=0.0,
            statistical_score=0.0,
            statistical_drift=True,
        )
        == "DRIFT"
    )


def test_statistical_score_gate_forces_drift():
    assert (
        decide(
            None,
            schema_score=0.0,
            semantic_score=0.0,
            statistical_score=1.0,
        )
        == "DRIFT"
    )


def test_no_channel_fires_is_no_drift():
    assert (
        decide(
            None,
            schema_score=0.0,
            semantic_score=0.0,
            statistical_score=0.5,
            statistical_drift=False,
        )
        == "NO_DRIFT"
    )
    assert (
        decide(
            None,
            schema_score=0.0,
            semantic_score=0.0,
            statistical_score=0.0,
        )
        == "NO_DRIFT"
    )


def test_unified_threshold_path_is_removed():
    """The old 0.5 unified-score branch must not decide anything."""
    assert (
        decide(
            1.0,
            threshold=0.5,
            schema_score=0.0,
            semantic_score=0.0,
            statistical_score=0.0,
            statistical_drift=False,
        )
        == "NO_DRIFT"
    )


def test_frozen_constants_match_config():
    from detectors.unified import STATISTICAL_DECISION_THRESHOLD
    from detectors.thresholds import THRESHOLDS

    assert (
        STATISTICAL_DECISION_THRESHOLD
        == THRESHOLDS["statistical"]["decision_threshold"]
    )
    assert STATISTICAL_DECISION_THRESHOLD > 0.0
