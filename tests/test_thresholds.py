from pathlib import Path

import yaml

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

def test_schema_severity_labels_frozen():

    labels = THRESHOLDS["schema"]["severity_labels"]

    assert labels == {
        "added": "high",
        "removed": "high",
        "renamed": "high",
        "type_change": "medium",
        "nullability_change": "low",
    }

def test_schema_detector_emits_frozen_severities():

    import copy

    from detectors.schema_detector import (
        compare_schemas,
        detect_renames,
    )

    base = {
        "columns": {
            "temp": {
                "dtype": "float64",
                "nullable": False,
            },
            "motion": {
                "dtype": "bool",
                "nullable": False,
            },
        }
    }

    current = copy.deepcopy(base)
    current["columns"]["battery_level"] = {
        "dtype": "float64",
        "nullable": False,
    }
    assert (
        compare_schemas(base, current)[0]["severity"]
        == THRESHOLDS["schema"]["severity_labels"]["added"]
    )

    current = copy.deepcopy(base)
    del current["columns"]["motion"]
    assert (
        compare_schemas(base, current)[0]["severity"]
        == THRESHOLDS["schema"]["severity_labels"]["removed"]
    )

    current = copy.deepcopy(base)
    current["columns"]["temp"]["dtype"] = "string"
    assert (
        compare_schemas(base, current)[0]["severity"]
        == THRESHOLDS["schema"]["severity_labels"]["type_change"]
    )

    current = copy.deepcopy(base)
    current["columns"]["temp"]["nullable"] = True
    assert (
        compare_schemas(base, current)[0]["severity"]
        == THRESHOLDS["schema"]["severity_labels"]["nullability_change"]
    )

    events = detect_renames(
        {
            "columns": {
                "temp": {
                    "dtype": "float64",
                    "nullable": False,
                }
            }
        },
        {
            "columns": {
                "temperature": {
                    "dtype": "float64",
                    "nullable": False,
                }
            }
        },
        {"temp": "temperature"},
    )
    assert (
        events[0]["severity"]
        == THRESHOLDS["schema"]["severity_labels"]["renamed"]
    )

def test_severity_yaml_mirrors_thresholds():

    severity_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "severity.yaml"
    )

    with open(severity_path, "r") as file:
        severity = yaml.safe_load(file)

    assert (
        severity["schema"]
        == THRESHOLDS["schema"]["severity_labels"]
    )

def test_window_config_confirmed():

    window = THRESHOLDS["window"]

    assert window["size"] == 1000
    assert window["sweep_deferred"] is True

def test_unified_threshold_between_zero_and_one():

    threshold = THRESHOLDS["unified"]["threshold"]

    assert 0.0 <= threshold <= 1.0

def test_unified_max_scores_are_one():

    unified = THRESHOLDS["unified"]

    assert unified["schema_max_score"] == 1.0
    assert unified["semantic_max_score"] == 1.0
    assert unified["statistical_max_score"] == 1.0
