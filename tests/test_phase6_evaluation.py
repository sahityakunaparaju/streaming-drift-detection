import pandas as pd

from evaluation.phase6_evaluation import (
    average_precision,
    confusion_counts,
    compute_metrics,
    evaluate_policy_threshold,
    evaluate_threshold,
    ground_truth_binary,
    per_drift_type_table,
    roc_auc,
    score_distribution_table,
    threshold_sweep,
    threshold_values,
)


def test_ground_truth_mapping():

    assert ground_truth_binary("DRIFT") == 1
    assert ground_truth_binary("NO_DRIFT") == 0
    assert ground_truth_binary("drift") == 1
    assert ground_truth_binary("no_drift") == 0
    assert ground_truth_binary(None) == 0


def test_confusion_counts():

    y_true = [1, 1, 1, 0, 0, 0, 0]
    y_pred = [1, 1, 0, 0, 0, 1, 1]

    counts = confusion_counts(y_true, y_pred)

    assert counts["tp"] == 2
    assert counts["tn"] == 2
    assert counts["fp"] == 2
    assert counts["fn"] == 1


def test_metrics_perfect():

    metrics = compute_metrics(
        tp=7,
        tn=3,
        fp=0,
        fn=0,
    )

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["fpr"] == 0.0
    assert metrics["fnr"] == 0.0


def test_metrics_with_false_negative():

    metrics = compute_metrics(
        tp=6,
        tn=3,
        fp=0,
        fn=1,
    )

    assert metrics["precision"] == 1.0
    assert round(metrics["recall"], 6) == round(
        6 / 7,
        6,
    )
    assert metrics["fpr"] == 0.0
    assert round(metrics["fnr"], 6) == round(
        1 / 7,
        6,
    )


def test_metrics_zero_division_guarded():

    metrics = compute_metrics(
        tp=0,
        tn=0,
        fp=0,
        fn=10,
    )

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_evaluate_threshold_boundary_inclusive():

    y_true = [1, 0]
    scores = [0.5, 0.5]

    row = evaluate_threshold(y_true, scores, 0.5)

    assert row["tp"] == 1
    assert row["fp"] == 1


def test_threshold_sweep_shape_and_bounds():

    y_true = [1, 1, 0, 0]
    scores = [0.9, 0.6, 0.3, 0.1]

    rows = threshold_sweep(y_true, scores)

    assert len(rows) == len(threshold_values())
    assert rows[0]["threshold"] == 0.0
    assert rows[-1]["threshold"] == 1.0
    assert rows[0]["tp"] == 2
    assert rows[0]["fp"] == 2
    assert rows[-1]["tp"] == 0
    assert rows[-1]["fn"] == 2


def test_score_bounds_in_sweep():

    y_true = [1, 1, 0, 0]
    scores = [0.9, 0.6, 0.3, 0.1]

    for row in threshold_sweep(y_true, scores):

        for key in [
            "precision",
            "recall",
            "f1",
            "accuracy",
            "specificity",
            "fpr",
            "fnr",
        ]:
            assert 0.0 <= row[key] <= 1.0


def test_roc_auc_perfect_separation():

    y_true = [1, 1, 1, 0, 0, 0]
    scores = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]

    assert roc_auc(y_true, scores) == 1.0


def test_roc_auc_reversed_is_zero():

    y_true = [1, 1, 1, 0, 0, 0]
    scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]

    assert roc_auc(y_true, scores) == 0.0


def test_roc_auc_ties_give_half():

    y_true = [1, 1, 0, 0]
    scores = [0.5, 0.5, 0.5, 0.5]

    assert roc_auc(y_true, scores) == 0.5


def test_average_precision_perfect():

    y_true = [1, 1, 1, 0, 0, 0]
    scores = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]

    assert average_precision(y_true, scores) == 1.0


def test_policy_override_forces_drift_at_high_threshold():

    rows = [
        {
            "y_true": 1,
            "unified_score": 0.3333,
            "schema_score": 0.0,
            "semantic_score": 0.0,
            "statistical_score": 1.0,
        },
        {
            "y_true": 0,
            "unified_score": 0.05,
            "schema_score": 0.0,
            "semantic_score": 0.0,
            "statistical_score": 0.0,
        },
    ]

    row = evaluate_policy_threshold(rows, 1.0)

    assert row["tp"] == 1
    assert row["fp"] == 0


def test_score_distribution_table_groups():

    df = pd.DataFrame(
        {
            "event_id": [
                "window_000",
                "window_001",
                "window_002",
            ],
            "ground_truth": [
                "NO_DRIFT",
                "NO_DRIFT",
                "DRIFT",
            ],
            "schema_score": [0.0, 0.0, 1.0],
            "statistical_score": [0.1, 0.2, 0.3],
            "semantic_score": [0.0, 0.0, 0.0],
            "unified_score": [0.03, 0.07, 0.43],
        }
    )

    table = score_distribution_table(df)

    unified = table[
        table["score"] == "unified_score"
    ]
    no_drift = unified[
        unified["ground_truth"] == "NO_DRIFT"
    ].iloc[0]
    drift = unified[
        unified["ground_truth"] == "DRIFT"
    ].iloc[0]

    assert no_drift["count"] == 2
    assert drift["count"] == 1
    assert round(no_drift["mean"], 6) == round(0.05, 6)
    assert drift["max"] == 0.43


def test_per_drift_type_table():

    df = pd.DataFrame(
        {
            "event_id": [
                "window_002",
                "window_003",
                "window_004",
            ],
            "ground_truth": [
                "DRIFT",
                "DRIFT",
                "DRIFT",
            ],
            "drift_type": [
                "schema",
                "schema",
                "statistical",
            ],
            "subtype": [
                "added",
                "removed",
                "mean_shift",
            ],
            "feature": [
                "battery_level",
                "motion",
                "temp",
            ],
            "schema_score": [1.0, 1.0, 0.0],
            "statistical_score": [0.1, 0.2, 0.3],
            "semantic_score": [0.0, 0.0, 0.0],
            "unified_score": [0.4, 0.4, 0.1],
            "decision": [
                "DRIFT",
                "DRIFT",
                "NO_DRIFT",
            ],
            "channel": [
                "schema",
                "schema",
                "none",
            ],
        }
    )

    table = per_drift_type_table(df)

    added = table[
        (table["drift_type"] == "schema")
        & (table["subtype"] == "added")
    ].iloc[0]
    mean_shift = table[
        (table["drift_type"] == "statistical")
        & (table["subtype"] == "mean_shift")
    ].iloc[0]

    assert added["detected"] == 1
    assert added["missed"] == 0
    assert mean_shift["detected"] == 0
    assert mean_shift["missed"] == 1
