"""Tests for the revision reference builders and per-column decisions."""

import numpy as np
import pandas as pd

from detectors.reference import (
    ReferenceModel,
    numeric_columns_of,
    categorical_columns_of,
)
from detectors.statistical_detector import (
    benjamini_hochberg,
    categorical_column_metrics,
)


def make_telemetry(n=400, seed=0, device="A", shift=0.0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "device": [device] * n,
        "temp": rng.normal(22.0 + shift, 2.0, n),
        "co": rng.normal(0.004, 0.001, n),
    })


def make_reference():
    return pd.concat([
        make_telemetry(600, seed=1, device="A"),
        make_telemetry(600, seed=2, device="B"),
    ], ignore_index=True)


def test_numeric_and_categorical_columns():
    df = make_reference()
    assert "temp" in numeric_columns_of(df)
    assert "device" in categorical_columns_of(df)
    assert "device" not in numeric_columns_of(df)


def test_global_mode_clean_window_does_not_trigger():
    reference = make_reference()
    model = ReferenceModel(
        reference,
        mode="global",
        min_rows=30,
        config={
            "alpha": 0.01,
            "column_effect_thresholds": {"temp": 0.5, "co": 0.5, "device": 0.5},
        },
    )
    window = make_telemetry(300, seed=3, device="A")
    details = model.details(window)
    assert details["reference_mode"] == "global"
    assert details["statistical_drift"] is False


def test_per_device_mode_compares_device_slices():
    reference = make_reference()
    model = ReferenceModel(
        reference,
        mode="per_device",
        min_rows=30,
        config={
            "alpha": 0.01,
            "column_effect_thresholds": {"temp": 0.5, "co": 0.5, "device": 0.5},
        },
    )
    window = make_telemetry(300, seed=4, device="B")
    details = model.details(window)
    # Numeric metrics carry their device slice label.
    assert details["numeric"]["temp"]["slice"] == "B"
    assert details["statistical_drift"] is False


def test_per_device_skips_small_slices_and_logs():
    reference = make_reference()
    model = ReferenceModel(
        reference,
        mode="per_device",
        min_rows=30,
        config={"alpha": 0.01},
    )
    window = make_telemetry(100, seed=5, device="A")
    window = pd.concat(
        [window, make_telemetry(5, seed=6, device="B")],
        ignore_index=True,
    )
    details = model.details(window)
    reasons = [entry["reason"] for entry in details["skipped_slices"]]
    assert any("below min_rows" in reason for reason in reasons)


def test_rolling_uses_explicit_previous_window():
    reference = make_reference()
    model = ReferenceModel(
        reference,
        mode="rolling",
        min_rows=30,
        config={
            "alpha": 0.01,
            "column_effect_thresholds": {"temp": 0.5, "co": 0.5, "device": 0.5},
        },
    )
    previous = make_telemetry(400, seed=10, device="A")
    window = make_telemetry(400, seed=11, device="A")
    model.set_rolling_reference(previous)
    details = model.details(window)
    assert details["numeric"]["temp"]["slice"] == "rolling"
    assert details["statistical_drift"] is False


def test_rolling_flags_a_large_mean_shift_on_the_shifted_column():
    reference = make_telemetry(800, seed=1, device="A")
    model = ReferenceModel(
        reference,
        mode="rolling",
        min_rows=30,
        config={
            "alpha": 0.01,
            "column_effect_thresholds": {
                "temp": 0.3,
                "co": 0.3,
                "device": 0.3,
            },
        },
    )
    previous = make_telemetry(400, seed=12, device="A")
    window = make_telemetry(400, seed=13, device="A", shift=2.0)
    model.set_rolling_reference(previous)
    details = model.details(window)
    assert details["statistical_drift"] is True
    assert details["trigger_columns"] == ["temp"]
    assert details["column_flags"]["temp"] is True


def test_benjamini_hochberg():
    assert benjamini_hochberg([], 0.05) == []
    assert benjamini_hochberg([0.001, 0.5, 0.02], 0.05) == [
        True,
        False,
        True,
    ]
    assert benjamini_hochberg([0.5, 0.6], 0.01) == [False, False]


def test_categorical_cramers_v_is_scale_free():
    rng = np.random.default_rng(0)
    # Three vs sixteen categories: a similar proportional shift should
    # produce a comparable Cramer's V (not a chi2 that saturates).
    for k in (3, 16):
        reference = rng.choice(
            [f"c{i}" for i in range(k)], size=5000
        )
        current = rng.choice(
            [f"c{i}" for i in range(k)], size=5000
        )
        metrics = categorical_column_metrics(reference, current)
        assert 0.0 <= metrics["cramers_v"] < 0.2

    reference = np.array(["a"] * 800 + ["b"] * 200)
    current = np.array(["a"] * 500 + ["b"] * 500)
    shifted = categorical_column_metrics(reference, current)
    assert shifted["cramers_v"] > 0.3
    assert shifted["p_value"] < 1e-6
