from pathlib import Path

import pandas as pd
import numpy as np


from drift_injection.stream_builder import build_stream


def test_stream_builder(tmp_path):

    output_dir = tmp_path / "stream"

    ground_truth = (
        tmp_path / "ground_truth.csv"
    )

    windows = build_stream(
        input_path="data/primary/iot_telemetry_data.csv",
        output_dir=output_dir,
        ground_truth_path=ground_truth,
        window_size=1000,
        max_windows=10,
    )

    # --------------------------------------------------
    # Ten windows should be created
    # --------------------------------------------------

    assert len(windows) == 10

    files = list(
        output_dir.glob("window_*.csv")
    )

    assert len(files) == 10

    # --------------------------------------------------
    # Ground truth should contain one event per window
    # --------------------------------------------------

    truth = pd.read_csv(
        ground_truth
    )

    assert len(truth) == 10

    assert set(
        truth["ground_truth"]
    ) == {
        "DRIFT",
        "NO_DRIFT",
    }

    # --------------------------------------------------
    # Expected drift types
    # --------------------------------------------------

    drift_events = truth[
        truth["ground_truth"] == "DRIFT"
    ]

    assert len(drift_events) == 7

    # --------------------------------------------------
    # Normal windows
    # --------------------------------------------------

    normal_events = truth[
        truth["ground_truth"] == "NO_DRIFT"
    ]

    assert len(normal_events) == 3

    # --------------------------------------------------
    # Check semantic metadata-only window
    # --------------------------------------------------

    original = pd.read_csv(
        "data/primary/iot_telemetry_data.csv"
    ).iloc[
        7 * 1000:8 * 1000
    ].copy()

    semantic_window = pd.read_csv(
        output_dir / "window_007.csv"
    )

    assert np.allclose(
    original["temp"].to_numpy(),
    semantic_window["temp"].to_numpy(),
    rtol=1e-10,
    atol=1e-10,
    )

    # --------------------------------------------------
    # Check actual unit-conversion window
    # --------------------------------------------------

    conversion_window = pd.read_csv(
        output_dir / "window_008.csv"
    )

    assert not np.allclose(
    original["temp"].to_numpy(),
    conversion_window["temp"].to_numpy(),
    rtol=1e-10,
    atol=1e-10,
    )

    # --------------------------------------------------
    # Check schema addition
    # --------------------------------------------------

    added_window = pd.read_csv(
        output_dir / "window_002.csv"
    )

    assert "battery_level" in (
        added_window.columns
    )

    # --------------------------------------------------
    # Check schema removal
    # --------------------------------------------------

    removed_window = pd.read_csv(
        output_dir / "window_003.csv"
    )

    assert "motion" not in (
        removed_window.columns
    )