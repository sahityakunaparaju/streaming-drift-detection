"""
Phase 10 — classical drift-detector adapter tests.

Guards the river-based ADWIN / DDM / KSWIN adapters and the
comparison protocol invariants:

* each adapter exposes the update(observation) -> drift_detected
  interface;
* detectors fire on a synthetic mean shift and (where
  deterministic) not on stationary data;
* evaluation is deterministic and safe on absent/short streams;
* the Phase 8 blind-spot recovery definition holds structurally:
  a classical detector's decision on a value-unchanged injected
  window is IDENTICAL to its decision on the value-identical clean
  slice, so recovery (fired on injected, not on clean) is 0;
* per-window lag follows the same convention as B4's ladder lag.
"""

import numpy as np
import pandas as pd
import pytest

from detectors.classical_detectors import (
    DETECTOR_NAMES,
    evaluate_window,
    feature_values,
    new_detector,
    surprise_bits,
)
from evaluation.metrics import detection_lag_windows, evaluate

RNG = np.random.default_rng(7)


def make_df(values):
    return pd.DataFrame({"temp": values})


def stationary(mean=20.0, std=1.0, n=1000):
    return make_df(RNG.normal(mean, std, n))


def shifted(mean=20.0, std=1.0, shift=3.0, n=1000):
    return make_df(RNG.normal(mean + shift, std, n))


# ---------------------------------------------------------------- 1.
# Adapter interface


def test_adapters_expose_update_interface():
    for name in DETECTOR_NAMES:
        det = new_detector(name)
        assert callable(getattr(det, "update", None))
        assert hasattr(det, "drift_detected")


def test_kswin_cold_start_accepts_prefill():
    det = new_detector("kswin", window=[1.0] * 100)
    det.update(2.0)
    assert hasattr(det, "drift_detected")


# ---------------------------------------------------------------- 2.
# Drift detection behavior


def test_adwin_fires_on_mean_shift_not_stationary():
    warm = stationary(n=2000)
    assert evaluate_window(shifted(shift=3.0, n=1000), warm, "adwin")
    assert not evaluate_window(stationary(n=1000), warm, "adwin")


def test_ddm_fires_on_mean_shift_not_stationary():
    warm = stationary(n=2000)
    assert evaluate_window(shifted(shift=3.0, n=1000), warm, "ddm")
    assert not evaluate_window(stationary(n=1000), warm, "ddm")


def test_kswin_fires_on_mean_shift():
    warm = stationary(n=2000)
    assert evaluate_window(shifted(shift=5.0, n=1000), warm, "kswin")


# ---------------------------------------------------------------- 3.
# Determinism / no-data / short-stream safety


def test_deterministic_same_input_same_output():
    warm = stationary(n=2000)
    win = shifted(shift=3.0, n=1000)
    for name in DETECTOR_NAMES:
        first = evaluate_window(win, warm, name)
        second = evaluate_window(win, warm, name)
        assert first == second


def test_absent_column_is_unobservable():
    warm = make_df(np.zeros(100))
    win = make_df(np.zeros(100))  # missing nothing
    assert evaluate_window(win, warm, "adwin") is False
    # feature_values reports the feature unobservable when absent
    vals, ok = feature_values(pd.DataFrame({"other": [1.0]}), "temp")
    assert vals is None and ok is False


def test_nonnumeric_column_is_unobservable():
    df = pd.DataFrame({"temp": ["a", "b", "c"]})
    vals, ok = feature_values(df, "temp")
    assert vals is None and ok is False


def test_empty_window_does_not_crash():
    warm = stationary(n=200)
    empty = make_df(np.array([]))
    for name in DETECTOR_NAMES:
        assert evaluate_window(empty, warm, name) is False


def test_surprise_bits_math():
    ref = np.array([0.0] * 100)
    vals = np.array([0.0, 1.0, -1.0, 0.5])
    bits = surprise_bits(vals, ref, k=2.0)  # sigma=0 -> all zeros
    assert bits.tolist() == [0, 0, 0, 0]
    ref = np.array([0.0, 2.0] * 50)  # sigma=1
    bits = surprise_bits(np.array([5.0, 1.0]), ref, k=2.0)
    assert bits.tolist() == [1, 0]


# ---------------------------------------------------------------- 4.
# Metric compatibility (same evaluate() path as Phase 7/9)


def test_metrics_compatibility():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 1, 0]
    expected_fn = [False, False, False, False]
    summary = evaluate(y_true, y_pred, expected_fn)
    assert set(["raw", "adjusted"]) <= set(summary)
    assert summary["raw"]["recall"] == 0.5
    assert summary["raw"]["fpr"] == 0.5


def test_per_window_lag_convention():
    # 0/1 numeric convention (as used by evaluation/ablation.py and
    # the Phase 7 ladder): drift onset at position 3, first DRIFT at
    # position 4 -> lag 1; never detected -> None
    assert detection_lag_windows([0, 0, 0, 0, 1], onset_position=3) == 1
    assert detection_lag_windows([0, 0, 0, 0, 0], onset_position=3) is None
    # the runner converts "DRIFT"/"NO_DRIFT" strings to 0/1 first
    binary = [1 if d == "DRIFT" else 0
              for d in ["NO_DRIFT", "NO_DRIFT", "NO_DRIFT",
                        "NO_DRIFT", "DRIFT"]]
    assert detection_lag_windows(binary, onset_position=3) == 1


# ---------------------------------------------------------------- 5.
# Semantic blind-spot structural invariant (Phase 8 cross-check)


PHASE8_SEED42 = "results/phase8/seed-42"


@pytest.mark.skipif(
    not __import__("pathlib").Path(PHASE8_SEED42).is_dir(),
    reason="results/phase8 not generated",
)
def test_blind_spot_decisions_match_clean_slice():
    """The core honest-comparison invariant: on value-unchanged
    semantic windows, every classical detector makes the SAME
    decision on the injected window and the value-identical clean
    slice, so differential recovery is 0 for all of them."""
    import json
    from experiments.run_classical_comparison import (
        load_reference_df,
        local_warmup,
    )

    ref = load_reference_df()
    manifest = json.load(open(f"{PHASE8_SEED42}/manifest.json"))
    num_cols = ["temp", "co", "humidity", "lpg", "smoke", "ts"]
    bs = [e for e in manifest["windows"] if e["set"] == "blind_spot"]
    assert len(bs) == 5
    for entry in bs:
        win = pd.read_csv(entry["file"])
        s = entry["start_row"]
        clean = ref.iloc[s:s + len(win)].reset_index(drop=True)
        for col in num_cols:
            assert np.allclose(
                win[col].astype(float).to_numpy(),
                clean[col].astype(float).to_numpy(),
                equal_nan=True,
            ), f"{entry['event_id']}: values differ from clean slice"
        warm = local_warmup(ref, s)
        for name in DETECTOR_NAMES:
            dec = evaluate_window(win, warm, name)
            dec_clean = evaluate_window(clean, warm, name)
            assert dec == dec_clean, (
                f"{entry['event_id']} {name}: decision differs on "
                "value-identical data"
            )
            recovered = dec and not dec_clean
            assert recovered is False, (
                f"{entry['event_id']} {name}: spurious 'recovery'"
            )
