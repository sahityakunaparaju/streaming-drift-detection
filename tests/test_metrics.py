from evaluation.metrics import (
    confusion_counts,
    compute_metrics,
    detection_lag_windows,
    evaluate,
    lag_summary,
)


def test_confusion_counts():

    y_true = [1, 1, 1, 0, 0]
    y_pred = [1, 0, 0, 0, 1]

    counts = confusion_counts(y_true, y_pred)

    assert counts == {"tp": 1, "fp": 1, "tn": 1, "fn": 2}


def test_compute_metrics_perfect():

    metrics = compute_metrics(tp=7, fp=0, tn=3, fn=0)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["fpr"] == 0.0


def test_evaluate_raw_counts_flagged_miss_as_fn():

    y_true = [1, 1, 1, 0]
    y_pred = [1, 0, 0, 0]
    flagged = ["", "expected false negative — known aggregation limitation", "", ""]

    result = evaluate(y_true, y_pred, flagged)

    # Raw: the flagged miss counts as an ordinary FN.
    assert result["raw"]["tp"] == 1
    assert result["raw"]["fn"] == 2
    assert result["raw"]["recall"] == round(1 / 3, 6)

    # Adjusted: flagged window excluded from evaluation entirely.
    assert result["adjusted"]["tp"] == 1
    assert result["adjusted"]["fn"] == 1
    assert result["adjusted"]["recall"] == 0.5

    assert result["flagged"] == {
        "n_flagged": 1,
        "n_flagged_detected": 0,
        "n_flagged_missed": 1,
    }


def test_evaluate_flagged_detected_counts_in_raw_not_adjusted():

    y_true = [1, 1, 0]
    y_pred = [1, 1, 0]
    flagged = ["", "expected false negative — known aggregation limitation", ""]

    result = evaluate(y_true, y_pred, flagged)

    assert result["raw"]["tp"] == 2
    assert result["adjusted"]["tp"] == 1
    assert result["flagged"]["n_flagged_detected"] == 1


def test_evaluate_no_flag_is_raw_only():

    y_true = [1, 0, 1]
    y_pred = [1, 0, 0]

    result = evaluate(y_true, y_pred)

    assert result["raw"] == result["adjusted"]
    assert result["flagged"]["n_flagged"] == 0


def test_lag_summary_ignores_never_detected():

    summary = lag_summary([0, 1, None, 2])

    assert summary["n_detected"] == 3
    assert summary["n_never_detected"] == 1
    assert summary["mean"] == 1.0
    assert summary["median"] == 1.0
    assert summary["p95"] == 1.9


def test_lag_summary_empty():

    summary = lag_summary([None, None])

    assert summary["n_detected"] == 0
    assert summary["n_never_detected"] == 2
    assert summary["mean"] is None


def test_detection_lag_windows():

    assert detection_lag_windows([0, 0, 0, 1, 1], onset_position=3) == 0
    assert detection_lag_windows([0, 0, 0, 0, 1], onset_position=3) == 1
    assert detection_lag_windows([0, 0, 0, 0, 0], onset_position=3) is None
    assert detection_lag_windows(
        ["NO_DRIFT", "NO_DRIFT", "NO_DRIFT", "DRIFT", "DRIFT"],
        onset_position=3,
    ) == 0
