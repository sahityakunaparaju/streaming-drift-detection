from detectors.thresholds import (
    THRESHOLDS,
    find_config_path,
)


def test_config_file_resolves():

    assert find_config_path().name == "thresholds.yaml"


def test_all_sections_present():

    assert set(THRESHOLDS.keys()) >= {
        "statistical",
        "semantic",
        "schema",
        "unified",
    }


def test_statistical_constants_in_range():

    statistical = THRESHOLDS["statistical"]

    assert 0.0 < statistical["ks_floor"] < 1.0
    assert statistical["psi_scale"] > 0.0
    assert statistical["chi2_scale"] > 0.0
    assert 0.0 <= statistical["decision_threshold"] <= 1.0
    assert 0.0 < statistical["categorical_p_value_threshold"] < 1.0
    assert statistical["psi_bins"] > 0


def test_semantic_tolerance_between_zero_and_one():

    fraction = THRESHOLDS["semantic"][
        "range_tolerance_fraction"
    ]

    assert 0.0 < fraction < 1.0


def test_schema_is_binary_no_threshold():

    schema = THRESHOLDS["schema"]

    assert schema["binary"] is True
    assert schema["threshold_required"] is False


def test_unified_threshold_between_zero_and_one():

    threshold = THRESHOLDS["unified"]["threshold"]

    assert 0.0 <= threshold <= 1.0


def test_unified_max_scores_are_one():

    unified = THRESHOLDS["unified"]

    assert unified["schema_max_score"] == 1.0
    assert unified["semantic_max_score"] == 1.0
    assert unified["statistical_max_score"] == 1.0
