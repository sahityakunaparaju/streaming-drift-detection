import json

import pandas as pd

from drift_injection.phase7_generator import (
    build_phase7_set,
    build_matrix_spec,
)


def _build(tmp_path):
    return build_phase7_set(
        input_path="data/primary/iot_telemetry_data.csv",
        output_dir=str(tmp_path / "windows"),
        ground_truth_path=str(tmp_path / "ground_truth.csv"),
        manifest_path=str(tmp_path / "manifest.json"),
        seed=42,
    )


def test_expected_false_negative_propagates_to_every_window(tmp_path):

    manifest = _build(tmp_path)

    assert manifest["n_windows"] > 0

    for entry in manifest["windows"]:
        subtype = entry["subtype"]
        flagged = bool(entry["expected_false_negative"])
        if subtype.startswith("variance_") or subtype.startswith(
            "distribution_"
        ):
            # Mandatory Phase 7 precondition: variance_* and
            # distribution_* windows are ALWAYS flagged.
            assert flagged, (
                f"{entry['event_id']} ({subtype}) not flagged"
            )
        else:
            assert not flagged, (
                f"{entry['event_id']} ({subtype}) unexpectedly flagged"
            )


def test_flagging_matches_ground_truth_csv(tmp_path):

    manifest = _build(tmp_path)

    truth = pd.read_csv(tmp_path / "ground_truth.csv")

    assert len(truth) == manifest["n_windows"]

    for _, row in truth.iterrows():
        subtype = str(row["subtype"])
        raw_flag = row["expected_false_negative"]
        # Empty strings round-trip through CSV as NaN.
        flag = "" if pd.isna(raw_flag) else str(raw_flag)
        if subtype.startswith("variance_") or subtype.startswith(
            "distribution_"
        ):
            assert "expected false negative" in flag
        else:
            assert flag == ""


def test_distribution_shift_has_real_windows_at_all_severities(tmp_path):

    manifest = _build(tmp_path)

    # Phase 6 caveat: distribution shift previously had ZERO real
    # windows. Phase 7 must exercise it end-to-end.
    distribution_windows = [
        entry
        for entry in manifest["windows"]
        if entry["subtype"] == "distribution_shift"
        and entry["set"] == "matrix"
    ]

    severities = {entry["severity"] for entry in distribution_windows}

    assert {entry["subtype"] for entry in distribution_windows} == {
        "distribution_shift"
    }
    assert severities == {"low", "medium", "high"}

    # Lag streams also exercise it.
    assert any(
        entry["set"] == "lag"
        and entry["subtype"] == "distribution_shift"
        for entry in manifest["windows"]
    )


def test_matrix_spec_covers_expected_cells():

    spec = build_matrix_spec()

    cells = {(d, s) for d, s, _f, _sev in spec}

    for subtype in [
        "mean_shift",
        "variance_shift",
        "distribution_shift",
        "category_shift",
    ]:
        assert ("statistical", subtype) in cells

    assert ("normal", "clean") in cells
    assert ("normal", "normal_noise") in cells

    for severity in ["low", "medium", "high"]:
        assert any(
            d == "semantic"
            and s == "range_violation"
            and sev == severity
            for d, s, _f, sev in spec
        )
