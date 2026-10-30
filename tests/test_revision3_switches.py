"""Tests for the Revision v3 FPR-reduction switches (options a-c).

Option (d) is a reporting sweep and carries no switch of its own.
All switch values are calibrated on calibration/validation regions
only; these tests exercise the MECHANICS, not region data.
"""

import numpy as np
import pandas as pd

from detectors.persistence import (
    added_lag_windows,
    persistence_alarm,
    persistence_last_n,
)
from detectors.reference import (
    GAS_GROUP_DEFAULT_COLUMNS,
    ReferenceModel,
    gas_group_config,
)
from experiments.derive_thresholds import (
    parse_blocks,
    derive_groups,
    group_effects_per_block,
)


def _telemetry(n=400, seed=0, device="A", temp_shift=0.0, gas_shift=0.0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "device": [device] * n,
        "temp": rng.normal(22.0 + temp_shift, 2.0, n),
        "co": rng.normal(0.004 + gas_shift, 0.001, n),
        "lpg": 1.5 * (rng.normal(0.004 + gas_shift, 0.001, n)),
        "smoke": 4.0 * (rng.normal(0.004 + gas_shift, 0.001, n)),
    })


def _threshold_config(tmp):
    return {
        "alpha": 0.01,
        "column_effect_thresholds": {
            "temp": tmp.get("temp", 0.5),
            "co": tmp.get("co", 0.5),
            "lpg": tmp.get("lpg", 0.5),
            "smoke": tmp.get("smoke", 0.5),
        },
    }


# ---------------------------------------------------------------------
# (a) gas group
# ---------------------------------------------------------------------

def test_gas_group_config_parser():
    assert gas_group_config({}) is None
    assert gas_group_config({"gas_group": {"enabled": False}}) is None
    group = gas_group_config({
        "gas_group": {
            "enabled": True,
            "threshold": 0.4,
        }
    })
    assert group["columns"] == GAS_GROUP_DEFAULT_COLUMNS
    assert group["aggregation"] == "max"
    assert group["threshold"] == 0.4


def test_firing_gas_window_triggers_one_group_not_three():
    reference = pd.concat([
        _telemetry(600, seed=1, device="A"),
        _telemetry(600, seed=2, device="B"),
    ], ignore_index=True)
    config = _threshold_config({"co": 0.1, "lpg": 0.1, "smoke": 0.1})
    config["gas_group"] = {
        "enabled": True,
        "threshold": 0.1,
        "aggregation": "max",
    }
    model = ReferenceModel(
        reference, mode="rolling", min_rows=30, config=config
    )
    previous = _telemetry(400, seed=3, device="A")
    window = _telemetry(400, seed=4, device="A", gas_shift=0.01)
    model.set_rolling_reference(previous)
    details = model.details(window)

    assert details["statistical_drift"] is True
    # ONE group trigger instead of three per-column triggers.
    assert details["trigger_columns"] == ["gas_group"]
    assert details["gas_group"]["members"] == ["co", "lpg", "smoke"]
    # The group effect is the max over the member effects.
    member_effects = details["gas_group"]["member_effects"]
    assert details["gas_group"]["effect_size"] == max(
        member_effects.values()
    )
    # Members cannot trigger on their own any more.
    assert details["column_flags"]["co"] is False
    assert details["column_flags"]["gas_group"] is True
    for member in ("co", "lpg", "smoke"):
        assert details["numeric"][member]["gated_by_group"] is True
        assert details["numeric"][member]["drift"] is False


def test_gas_group_off_keeps_legacy_behaviour():
    reference = pd.concat([
        _telemetry(600, seed=1, device="A"),
        _telemetry(600, seed=2, device="B"),
    ], ignore_index=True)
    config = _threshold_config({"co": 0.1, "lpg": 0.1, "smoke": 0.1})
    model = ReferenceModel(
        reference, mode="rolling", min_rows=30, config=config
    )
    previous = _telemetry(400, seed=3, device="A")
    window = _telemetry(400, seed=4, device="A", gas_shift=0.01)
    model.set_rolling_reference(previous)
    details = model.details(window)
    assert set(details["trigger_columns"]) == {"co", "lpg", "smoke"}
    assert details["gas_group"] is None


def test_temp_drift_still_triggers_under_gas_group():
    reference = pd.concat([
        _telemetry(600, seed=1, device="A"),
        _telemetry(600, seed=2, device="B"),
    ], ignore_index=True)
    config = _threshold_config({"temp": 0.3, "co": 0.1})
    config["gas_group"] = {"enabled": True, "threshold": 4.0}
    model = ReferenceModel(
        reference, mode="rolling", min_rows=30, config=config
    )
    previous = _telemetry(400, seed=3, device="A")
    window = _telemetry(400, seed=4, device="A", temp_shift=2.0)
    model.set_rolling_reference(previous)
    details = model.details(window)
    assert details["trigger_columns"] == ["temp"]
    assert details["statistical_drift"] is True


# ---------------------------------------------------------------------
# (b) persistence rule
# ---------------------------------------------------------------------

def test_persistence_alarm_rule():
    # 1 isolated trigger out of the last 3 windows: no alarm (k=2).
    assert persistence_alarm([False, True, False], 2, 3) is False
    # 2 of the last 3: alarm.
    assert persistence_alarm([True, False, True], 2, 3) is True
    assert persistence_alarm([True, True, False], 2, 3) is True
    # 3 of the last 3: alarm.
    assert persistence_alarm([True, True, True], 2, 3) is True
    # Start-up: fewer than k windows of history cannot alarm.
    assert persistence_alarm([True], 2, 3) is False
    assert persistence_alarm([False, True], 2, 3) is False
    # Long history is truncated to the last n.
    assert persistence_last_n([True] * 10, 3) == [True, True, True]
    assert persistence_alarm([False] * 10 + [True, True, False], 2, 3) \
        is True


def test_persistence_added_lag_is_honest():
    # A sustained drift alarms on the k-th trigger window.
    assert added_lag_windows(2) == 1
    assert added_lag_windows(3) == 2
    assert added_lag_windows(1) == 0


# ---------------------------------------------------------------------
# (c) multi-block calibration helpers
# ---------------------------------------------------------------------

def test_parse_blocks():
    blocks = parse_blocks(
        "101000-110000;250000-265000;290000-303000"
    )
    assert blocks == [[101000, 110000], [250000, 265000],
                      [290000, 303000]]


def test_group_effects_per_block_and_derive_groups():
    tables = [
        ("block1", pd.DataFrame({
            "window": [1, 1, 1, 2, 2, 2],
            "column": ["co", "lpg", "smoke"] * 2,
            "kind": "numeric",
            "effect_size": [0.10, 0.12, 0.30, 0.11, 0.13, 0.31],
        })),
        ("block2", pd.DataFrame({
            "window": [1, 1, 1],
            "column": ["co", "lpg", "smoke"],
            "kind": "numeric",
            "effect_size": [0.20, 0.21, 0.40],
        })),
    ]
    frames = group_effects_per_block(tables)
    assert list(frames[0]["group_effect"]) == [0.30, 0.31]
    assert list(frames[1]["group_effect"]) == [0.40]

    pooled, per_block = derive_groups(
        tables, quantile=0.99, pooling="pooled"
    )
    # 0.99 quantile of [0.30, 0.31, 0.40] with linear interpolation.
    assert pooled > 0.395
    assert per_block is None

    max_blocks, per_block = derive_groups(
        tables, quantile=0.99, pooling="max_blocks"
    )
    assert max_blocks >= pooled
    assert dict(per_block) is not None
