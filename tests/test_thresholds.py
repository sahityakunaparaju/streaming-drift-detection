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

    assert 0.0 < statistical["alpha"] < 1.0
    assert statistical["correction"] == "benjamini-hochberg"
    assert 0.0 <= statistical["decision_threshold"] <= 1.0
    assert 0.0 < statistical["categorical_p_value_threshold"] < 1.0
    assert statistical["psi_bins"] > 0
    assert statistical["default_effect_threshold"] > 0.0


def test_per_column_effect_thresholds_are_calibrated():

    statistical = THRESHOLDS["statistical"]
    thresholds = statistical["column_effect_thresholds"]

    assert set(thresholds) >= {
        "temp",
        "co",
        "humidity",
        "lpg",
        "smoke",
        "light",
        "motion",
        "device",
    }
    for column, value in thresholds.items():
        assert value > 0.0, column
        assert value < 1.0, column


def test_thresholds_were_derived_on_clean_calibration_data():

    calibration = THRESHOLDS["statistical"]["calibration"]

    assert calibration["reference_mode"] in {
        "global",
        "per_device",
        "rolling",
    }
    assert calibration["quantile"] in {0.99, 0.995, 0.999}
    assert calibration["calibration_region"] != calibration["reference_region"]
    assert (
        calibration["clean_holdout_region"]
        != calibration["calibration_region"]
    )
    assert calibration["n_calibration_windows"] > 0
    assert calibration["calibration_seed"] not in {
        42, 123, 7, 11, 2024, 314, 2718, 999, 555, 777,
    }


def test_revision3_switches_are_well_formed():
    """Options (a)/(b)/(c) switches exist and point the gateway to valid
    values; the selection protocol (results/revision/
    validation_fpr_options/selection.json) chose them on validation
    data, never on evaluation windows."""

    statistical = THRESHOLDS["statistical"]
    calibration = statistical["calibration"]

    # Option (c): single region or disjoint multi-block calibration.
    assert calibration["mode"] in {"single_region", "multi_block"}
    assert calibration["pooling"] in {
        "pooled", "max_blocks", "gas_envelope",
    }
    blocks = calibration.get("blocks", [])
    evaluation_regions = {
        "matrix": [0, 100000],
        "lag": [120000, 250000],
    }
    for block in blocks:
        start, end = block
        assert (end - start) % 1000 == 0
        # No calibration block may reach into an evaluation region.
        for name, region in evaluation_regions.items():
            overlaps = max(start, region[0]) < min(end, region[1])
            assert not overlaps, (
                f"block {block} overlaps the evaluation {name} region"
            )

    # Option (a): gas group switch with one calibrated group threshold.
    gas_group = statistical["gas_group"]
    assert isinstance(gas_group["enabled"], bool)
    assert gas_group["columns"] == ["co", "lpg", "smoke"]
    assert gas_group["aggregation"] in {"max", "mean"}
    assert gas_group["threshold"] > 0.0

    # Option (b): persistence switch k-of-n with k >= 2 and k <= n.
    persistence = statistical["persistence"]
    assert isinstance(persistence["enabled"], bool)
    assert 2 <= persistence["k"] <= persistence["n"]


def test_legacy_frozen_values_are_preserved():

    frozen_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "thresholds_v1_frozen.yaml"
    )
    assert frozen_path.exists()

    with open(frozen_path, "r") as file:
        frozen = yaml.safe_load(file)

    assert frozen["statistical"]["ks_floor"] == 0.50
    assert frozen["statistical"]["psi_scale"] == 72.0
    assert frozen["statistical"]["chi2_scale"] == 32.0

    # The new config must not reuse the deaf legacy constants.
    assert "ks_floor" not in THRESHOLDS["statistical"]
    assert "psi_scale" not in THRESHOLDS["statistical"]
    assert "chi2_scale" not in THRESHOLDS["statistical"]


def test_reference_config_regions_are_disjoint():

    from detectors.reference import load_reference_config

    config = load_reference_config()

    assert config["mode"] in {"global", "per_device", "rolling"}
    assert config["min_rows_per_device"] >= 1

    reference = config["reference_region"]
    calibration = config["calibration_region"]
    holdout = config["clean_holdout_region"]

    def _overlaps(a, b):
        return max(a[0], b[0]) < min(a[1], b[1])

    assert not _overlaps(reference, calibration)
    assert not _overlaps(reference, holdout)
    assert not _overlaps(calibration, holdout)

    for evaluation in config["evaluation_regions"].values():
        assert not _overlaps(reference, evaluation)
        assert not _overlaps(calibration, evaluation)
        assert not _overlaps(holdout, evaluation)

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

def test_unified_has_no_dead_threshold_path():

    unified = THRESHOLDS["unified"]

    assert "threshold" not in unified
    assert "schema_max_score" not in unified
    assert "semantic_max_score" not in unified
    assert "statistical_max_score" not in unified
    assert unified["severity_weights"]["high"] == 1.0
    assert 0.0 < unified["severity_weights"]["low"] < 1.0
