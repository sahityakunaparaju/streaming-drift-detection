import pandas as pd

from drift_injection.ground_truth import (
    GROUND_TRUTH_COLUMNS,
    log_injection,
    log_no_drift,
    load_ground_truth,
)

def test_log_drift_event(tmp_path):

    output = tmp_path / "ground_truth.csv"

    event = log_injection(
        drift_type="statistical",
        subtype="mean_shift",
        feature="temp",
        severity="medium",
        output_path=output,
        timestamp="2026-08-13T10:00:00+00:00",
    )

    assert event["ground_truth"] == "DRIFT"
    assert event["injected"] is True

    df = pd.read_csv(output)

    assert len(df) == 1
    assert df.iloc[0]["drift_type"] == "statistical"
    assert df.iloc[0]["subtype"] == "mean_shift"
    assert df.iloc[0]["feature"] == "temp"
    assert df.iloc[0]["severity"] == "medium"
    assert df.iloc[0]["ground_truth"] == "DRIFT"

def test_log_no_drift_event(tmp_path):

    output = tmp_path / "ground_truth.csv"

    event = log_no_drift(
        subtype="normal_noise",
        feature="temp",
        output_path=output,
        timestamp="2026-08-13T10:05:00+00:00",
    )

    assert event["ground_truth"] == "NO_DRIFT"
    assert event["injected"] is False

    df = pd.read_csv(output)

    assert len(df) == 1
    assert df.iloc[0]["drift_type"] == "normal"
    assert df.iloc[0]["subtype"] == "normal_noise"
    assert df.iloc[0]["severity"] == "none"
    assert df.iloc[0]["ground_truth"] == "NO_DRIFT"

def test_drift_and_no_drift_are_both_recorded(tmp_path):

    output = tmp_path / "ground_truth.csv"

    log_injection(
        drift_type="semantic",
        subtype="unit_change",
        feature="temp",
        severity="high",
        output_path=output,
    )

    log_no_drift(
        subtype="normal_noise",
        feature="temp",
        output_path=output,
    )

    df = load_ground_truth(output)

    assert len(df) == 2

    assert set(df["ground_truth"]) == {
        "DRIFT",
        "NO_DRIFT",
    }

def test_ground_truth_columns(tmp_path):

    output = tmp_path / "ground_truth.csv"

    log_injection(
        drift_type="schema",
        subtype="added",
        feature="battery_level",
        severity="high",
        output_path=output,
    )

    df = pd.read_csv(output)

    assert list(df.columns) == GROUND_TRUTH_COLUMNS

def test_expected_false_negative_flag_written(tmp_path):

    output = tmp_path / "ground_truth.csv"

    event = log_injection(
        drift_type="statistical",
        subtype="variance_shift",
        feature="temp",
        severity="medium",
        output_path=output,
        expected_false_negative=(
            "expected false negative — known aggregation limitation"
        ),
    )

    assert event["expected_false_negative"] != ""

    df = pd.read_csv(output)

    assert df.iloc[0]["expected_false_negative"] != ""

    assert (
        "expected false negative"
        in df.iloc[0]["expected_false_negative"]
    )

def test_distribution_shift_flag_written(tmp_path):

    output = tmp_path / "ground_truth.csv"

    event = log_injection(
        drift_type="statistical",
        subtype="distribution_shift",
        feature="temp",
        severity="medium",
        output_path=output,
        expected_false_negative=(
            "expected false negative — known aggregation limitation"
        ),
    )

    assert event["expected_false_negative"] != ""

    df = pd.read_csv(output)

    assert df.iloc[0]["subtype"] == "distribution_shift"
    assert (
        "expected false negative"
        in df.iloc[0]["expected_false_negative"]
    )

def test_default_flag_is_empty(tmp_path):

    output = tmp_path / "ground_truth.csv"

    log_injection(
        drift_type="schema",
        subtype="added",
        feature="battery_level",
        severity="high",
        output_path=output,
    )

    df = pd.read_csv(output)

    assert pd.isna(
        df.iloc[0]["expected_false_negative"]
    ) or df.iloc[0]["expected_false_negative"] == ""
