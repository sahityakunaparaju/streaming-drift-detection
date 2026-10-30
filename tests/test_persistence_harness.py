"""Harness-level tests for the persistence switch (Revision v3, option b).

 Drives experiments.run_phase7.run_detector_over_manifest on a
 synthetic manifest: isolated matrix windows (history reconstructed
 from raw preceding blocks) and a 5-window lag stream (history =
 the stream's own evaluated windows).
"""

import numpy as np
import pandas as pd
import pytest

import detectors.thresholds as thresholds_module
from experiments.run_phase7 import run_detector_over_manifest
from detectors.reference import ReferenceModel


WINDOW = 1000


def _raw(n=8000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "device": rng.choice(["A", "B", "C"], size=n),
        "temp": rng.normal(22.0, 2.0, n),
    })


def _make_manifest(raw_df):
    """Two isolated matrix windows + one 5-window lag stream.

    The stream's last two windows carry a strong temp shift, so the
    base statistical trigger fires at positions 3 and 4 (onset 3).
    The matrix window at start 6000 carries the same shift.
    """

    def window(start, shift=0.0):
        df = raw_df.iloc[start:start + WINDOW].copy()
        if shift:
            df["temp"] = df["temp"] + shift
        return df

    stream_id = "lag-mean_shift"
    entries = []
    windows = {}
    for position in range(5):
        event_id = f"{stream_id}-{position:02d}"
        entries.append({
            "event_id": event_id,
            "set": "lag",
            "stream_id": stream_id,
            "stream_position": position,
            "drift_type": "statistical",
            "subtype": "mean_shift",
            "severity": "medium" if position >= 3 else "none",
            "injected": position >= 3,
            "ground_truth": "DRIFT" if position >= 3 else "NO_DRIFT",
            "expected_false_negative": "",
            "start_row": 1000 + position * WINDOW,
        })
        windows[event_id] = window(
            1000 + position * WINDOW,
            shift=5.0 if position >= 3 else 0.0,
        )
    entries.append({
        "event_id": "matrix-mean_shift-medium",
        "set": "matrix",
        "stream_id": None,
        "stream_position": None,
        "drift_type": "statistical",
        "subtype": "mean_shift",
        "severity": "medium",
        "injected": True,
        "ground_truth": "DRIFT",
        "expected_false_negative": "",
        "start_row": 6000,
    })
    windows["matrix-mean_shift-medium"] = window(6000, shift=5.0)
    return {
        "window_size": WINDOW,
        "windows": entries,
        "_windows": windows,
    }


def _base_state():
    raw_df = _raw()
    reference_df = raw_df.iloc[6000:7000].reset_index(drop=True)
    model = ReferenceModel(
        reference_df,
        mode="rolling",
        min_rows=30,
        history=1,
        config={"alpha": 0.01, "column_effect_thresholds":
                {"temp": 0.3}},
    )
    return raw_df, reference_df, model


def _patch_persistence(monkeypatch, enabled, k=2, n=3):
    monkeypatch.setattr(
        thresholds_module,
        "THRESHOLDS",
        {
            "statistical": {
                "persistence": {
                    "enabled": enabled, "k": k, "n": n,
                },
            },
        },
    )


def test_persistence_off_keeps_base_decisions(monkeypatch):
    _patch_persistence(monkeypatch, enabled=False)
    raw_df, reference_df, model = _base_state()
    manifest = _make_manifest(raw_df)
    rows = run_detector_over_manifest(
        "B2_statistical", reference_df, {}, manifest,
        reference_model=model, raw_df=raw_df,
    )
    by_id = {row["event_id"]: row for row in rows}
    assert by_id["lag-mean_shift-00"]["decision"] == "NO_DRIFT"
    assert by_id["lag-mean_shift-03"]["decision"] == "DRIFT"
    assert by_id["lag-mean_shift-04"]["decision"] == "DRIFT"
    assert by_id["matrix-mean_shift-medium"]["decision"] == "DRIFT"
    assert by_id["lag-mean_shift-03"]["persistence_alarm"] == ""


def test_persistence_on_requires_k_of_n(monkeypatch):
    _patch_persistence(monkeypatch, enabled=True, k=2, n=3)
    raw_df, reference_df, model = _base_state()
    manifest = _make_manifest(raw_df)
    rows = run_detector_over_manifest(
        "B2_statistical", reference_df, {}, manifest,
        reference_model=model, raw_df=raw_df,
    )
    by_id = {row["event_id"]: row for row in rows}
    # Stream onset: a single trigger window is one of the last n, so
    # the first drifted window does not alarm yet.
    assert by_id["lag-mean_shift-03"]["decision"] == "NO_DRIFT"
    assert by_id["lag-mean_shift-03"]["persistence_alarm"] == "False"
    # Second consecutive trigger window: k = 2 of n = 3 -> alarm,
    # i.e. the honest added detection lag is +1 window.
    assert by_id["lag-mean_shift-04"]["decision"] == "DRIFT"
    assert by_id["lag-mean_shift-04"]["persistence_alarm"] == "True"
    # Isolated matrix window: the raw preceding blocks are clean, so
    # the isolated trigger cannot alarm (documented trade-off).
    assert by_id["matrix-mean_shift-medium"]["decision"] == "NO_DRIFT"
