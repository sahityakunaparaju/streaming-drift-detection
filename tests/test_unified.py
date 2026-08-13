from detectors.unified import (
    unified_score,
    decide,
)


def make_statistical_result(
    numeric=None,
    categorical=None,
):
    return {
        "numeric": numeric or {},
        "categorical": categorical or {},
    }


def test_schema_score_one_when_schema_drift():

    result = unified_score(
        True,
        make_statistical_result(),
        False,
    )

    assert result["schema_score"] == 1.0


def test_schema_score_zero_when_no_schema_drift():

    result = unified_score(
        False,
        make_statistical_result(),
        False,
    )

    assert result["schema_score"] == 0.0


def test_semantic_score_one_when_semantic_drift():

    result = unified_score(
        False,
        make_statistical_result(),
        True,
    )

    assert result["semantic_score"] == 1.0


def test_semantic_score_zero_when_no_semantic_drift():

    result = unified_score(
        False,
        make_statistical_result(),
        False,
    )

    assert result["semantic_score"] == 0.0


def test_statistical_score_between_zero_and_one():

    results = [
        make_statistical_result(),
        make_statistical_result(
            numeric={
                "temp": {
                    "ks_statistic": 0.5,
                    "psi": 12.0,
                }
            }
        ),
        make_statistical_result(
            numeric={
                "temp": {
                    "ks_statistic": 1.0,
                    "psi": 100.0,
                }
            }
        ),
        make_statistical_result(
            categorical={
                "device": {
                    "chi2": 500.0,
                    "p_value": 0.0,
                }
            }
        ),
        make_statistical_result(
            numeric={
                "temp": {
                    "ks_statistic": 0.2,
                    "psi": 30.0,
                }
            },
            categorical={
                "device": {
                    "chi2": 15.0,
                    "p_value": 0.05,
                }
            }
        ),
    ]

    for statistical_result in results:

        result = unified_score(
            False,
            statistical_result,
            False,
        )

        assert (
            0.0
            <= result["statistical_score"]
            <= 1.0
        )


def test_unified_score_between_zero_and_one():

    results = [
        make_statistical_result(),
        make_statistical_result(
            numeric={
                "temp": {
                    "ks_statistic": 1.0,
                    "psi": 100.0,
                }
            }
        ),
        make_statistical_result(
            categorical={
                "device": {
                    "chi2": 200.0,
                    "p_value": 0.0,
                }
            }
        ),
    ]

    for statistical_result in results:

        result = unified_score(
            True,
            statistical_result,
            True,
        )

        assert (
            0.0
            <= result["unified_score"]
            <= 1.0
        )


def test_unified_score_is_average_of_components():

    statistical_result = make_statistical_result(
        numeric={
            "temp": {
                "ks_statistic": 0.5,
                "psi": 10.0,
            }
        }
    )

    result = unified_score(
        True,
        statistical_result,
        True,
    )

    expected = (
        result["schema_score"]
        + result["statistical_score"]
        + result["semantic_score"]
    ) / 3.0

    assert result["unified_score"] == expected


def test_schema_override_forces_drift_with_low_statistical():

    assert (
        decide(
            0.3,
            schema_score=1.0,
            semantic_score=0.0,
        )
        == "DRIFT"
    )


    assert (
        decide(
            0.0,
            schema_score=1.0,
            semantic_score=0.0,
        )
        == "DRIFT"
    )


def test_semantic_override_forces_drift_alone():

    assert (
        decide(
            0.1,
            schema_score=0.0,
            semantic_score=1.0,
        )
        == "DRIFT"
    )


def test_statistical_override_forces_drift_alone():

    assert (
        decide(
            0.1,
            schema_score=0.0,
            semantic_score=0.0,
            statistical_score=1.0,
        )
        == "DRIFT"
    )

    assert (
        decide(
            0.0,
            schema_score=0.0,
            semantic_score=0.0,
            statistical_score=1.0,
        )
        == "DRIFT"
    )


def test_no_override_keeps_threshold_behavior():

    assert (
        decide(
            0.6,
            schema_score=0.4,
            semantic_score=0.4,
        )
        == "DRIFT"
    )

    assert (
        decide(
            0.4,
            schema_score=0.4,
            semantic_score=0.4,
        )
        == "NO_DRIFT"
    )


def test_decide_at_threshold_is_drift():

    assert decide(0.5) == "DRIFT"


def test_decide_below_threshold_is_no_drift():

    assert decide(0.49) == "NO_DRIFT"


def test_decide_one_is_drift():

    assert decide(1.0) == "DRIFT"


def test_decide_zero_is_no_drift():

    assert decide(0.0) == "NO_DRIFT"


def test_statistical_decision_threshold_forces_drift():

    assert (
        decide(
            0.0,
            schema_score=0.0,
            semantic_score=0.0,
            statistical_score=0.3,
        )
        == "DRIFT"
    )


def test_statistical_below_decision_threshold_no_drift():

    assert (
        decide(
            0.0,
            schema_score=0.0,
            semantic_score=0.0,
            statistical_score=0.2,
        )
        == "NO_DRIFT"
    )


def test_statistical_decision_threshold_not_an_override():

    assert (
        decide(
            0.0,
            schema_score=0.0,
            semantic_score=0.0,
            statistical_score=0.99,
        )
        == "DRIFT"
    )


def test_frozen_constants_match_config():

    from detectors.unified import (
        KS_FLOOR,
        PSI_SCALE,
        CHI2_SCALE,
        STATISTICAL_DECISION_THRESHOLD,
        UNIFIED_THRESHOLD,
    )

    from detectors.thresholds import THRESHOLDS

    assert KS_FLOOR == THRESHOLDS["statistical"]["ks_floor"]
    assert PSI_SCALE == THRESHOLDS["statistical"]["psi_scale"]
    assert CHI2_SCALE == THRESHOLDS["statistical"]["chi2_scale"]
    assert (
        STATISTICAL_DECISION_THRESHOLD
        == THRESHOLDS["statistical"]["decision_threshold"]
    )
    assert UNIFIED_THRESHOLD == THRESHOLDS["unified"]["threshold"]
