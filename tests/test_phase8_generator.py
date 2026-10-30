"""
Phase 8 (Section T) blind-spot test-set tests.

Guards the centerpiece experiment's premise: every blind-spot case
must be a clean semantic-only signal (schema identical, measured
statistical signal below the frozen thresholds, semantic detector
fires), and the generator must produce the exact per-seed structure
(5 blind-spot + 2 controls).
"""

import json

import pandas as pd

from detectors.schema_detector import compare_schemas
from evaluation.baseline import (
    build_current_schema,
    statistical_details_for,
)
from experiments.run_phase7 import load_reference
from drift_injection.phase7_generator import apply_semantic_metadata
from drift_injection.phase8_generator import (
    build_phase8_set,
    BLIND_SPOT_CASES,
    CONTROLS,
)


def _build(tmp_path, seed=42):
    return build_phase8_set(
        input_path="data/primary/iot_telemetry_data.csv",
        output_dir=str(tmp_path / "windows"),
        ground_truth_path=str(tmp_path / "ground_truth.csv"),
        manifest_path=str(tmp_path / "manifest.json"),
        seed=seed,
    )


def test_set_structure(tmp_path):
    manifest = _build(tmp_path)

    assert manifest["n_windows"] == len(BLIND_SPOT_CASES) + len(CONTROLS)
    assert manifest["n_blind_spot"] == len(BLIND_SPOT_CASES)
    assert manifest["n_control"] == len(CONTROLS)

    subtypes = {
        entry["subtype"]
        for entry in manifest["windows"]
        if entry["set"] == "blind_spot"
    }
    assert subtypes == {case[0] for case in BLIND_SPOT_CASES}

    for entry in manifest["windows"]:
        assert (entry["metadata_file"] is not None) == (
            entry["set"] == "blind_spot"
        ), entry["event_id"]

    truth = pd.read_csv(tmp_path / "ground_truth.csv")
    assert len(truth) == manifest["n_windows"]
    assert (
        truth["ground_truth"].value_counts().to_dict()
        == {"DRIFT": len(BLIND_SPOT_CASES), "NO_DRIFT": len(CONTROLS)}
    )


def test_windows_drawn_from_phase8_region(tmp_path):
    manifest = _build(tmp_path)
    for entry in manifest["windows"]:
        assert 320000 <= entry["start_row"] < 405184


def test_seed_aware_placement_differs(tmp_path):
    m1 = _build(tmp_path, seed=42)
    m2 = _build(tmp_path, seed=777)
    starts_42 = {
        entry["event_id"]: entry["start_row"]
        for entry in m1["windows"]
    }
    starts_777 = {
        entry["event_id"]: entry["start_row"]
        for entry in m2["windows"]
    }
    assert starts_42 != starts_777


def test_blind_spot_cases_are_isolated(tmp_path):
    """
    The centerpiece guard: for every blind-spot window, schema is
    identical, the measured statistical signal is below the frozen
    thresholds, and the semantic detector fires on the metadata-only
    contract change.
    """
    manifest = _build(tmp_path)
    reference_df, profile, reference_model, raw_df = load_reference()

    for entry in manifest["windows"]:
        if entry["set"] != "blind_spot":
            continue
        event_id = entry["event_id"]
        window_df = manifest["_windows"][event_id]
        with open(entry["metadata_file"]) as file:
            metadata = json.load(file)

        schema_events = compare_schemas(
            profile["schema"],
            build_current_schema(window_df),
        )
        assert schema_events == [], f"{event_id}: schema changed"

        start = int(entry["start_row"])
        if reference_model.mode == "rolling" and start >= 1000:
            reference_model.set_rolling_reference(
                raw_df.iloc[start - 1000:start]
            )
        details = statistical_details_for(
            reference_df,
            window_df,
            reference_model=reference_model,
        )
        assert not details["statistical_drift"], (
            f"{event_id}: statistical channel fired on a blind-spot "
            f"window (triggers={details['trigger_columns']})"
        )

        current_contract = apply_semantic_metadata(
            profile["semantic_contract"],
            metadata,
        )
        from detectors.semantic_detector import (
            compare_semantic_contracts,
            validate_values_against_contract,
        )
        events = (
            compare_semantic_contracts(
                profile["semantic_contract"],
                current_contract,
            )
            + validate_values_against_contract(
                window_df,
                current_contract,
            )
        )
        assert events, f"{event_id}: semantic detector did not fire"
        assert metadata["change"] in {
            e["change"] for e in events
        }, f"{event_id}: wrong semantic event"


def test_controls_are_semantic_silent(tmp_path):
    manifest = _build(tmp_path)
    reference_df, profile, _reference_model, _raw_df = load_reference()

    from detectors.semantic_detector import (
        compare_semantic_contracts,
        validate_values_against_contract,
    )

    for entry in manifest["windows"]:
        if entry["set"] != "control":
            continue
        window_df = manifest["_windows"][entry["event_id"]]
        events = (
            compare_semantic_contracts(
                profile["semantic_contract"],
                profile["semantic_contract"],
            )
            + validate_values_against_contract(
                window_df,
                profile["semantic_contract"],
            )
        )
        assert events == [], f"{entry['event_id']}: control fired"
