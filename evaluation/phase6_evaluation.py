import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from detectors.unified import (
    decide as policy_decide,
    KS_FLOOR,
    PSI_SCALE,
    CHI2_SCALE,
    STATISTICAL_DECISION_THRESHOLD,
)

OUTPUT_DIR = Path(
    os.environ.get(
        "PHASE6_OUTPUT_DIR",
        "results/phase6",
    )
)

POSITIVE = "DRIFT"

THRESHOLD_MIN = 0.00
THRESHOLD_MAX = 1.00
THRESHOLD_STEP = 0.01

METRIC_COLUMNS = [
    "threshold",
    "tp",
    "tn",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "specificity",
    "fpr",
    "fnr",
    "balanced_accuracy",
    "youden_j",
]

NUMERIC_COLUMNS = [
    "temp",
    "co",
    "humidity",
    "lpg",
    "smoke",
]


def ground_truth_binary(value):
    return (
        1
        if str(value).strip().upper() == POSITIVE
        else 0
    )


def confusion_counts(y_true, y_pred):
    pairs = list(zip(y_true, y_pred))
    tp = sum(1 for t, p in pairs if t == 1 and p == 1)
    tn = sum(1 for t, p in pairs if t == 0 and p == 0)
    fp = sum(1 for t, p in pairs if t == 0 and p == 1)
    fn = sum(1 for t, p in pairs if t == 1 and p == 0)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def compute_metrics(tp, tn, fp, fn):
    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )
    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    total = tp + tn + fp + fn
    accuracy = (
        (tp + tn) / total
        if total > 0
        else 0.0
    )
    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )
    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )
    fnr = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
    }


def metrics_row(threshold, y_true, predictions):
    counts = confusion_counts(y_true, predictions)
    metrics = compute_metrics(**counts)
    row = {"threshold": round(float(threshold), 6)}
    row.update(counts)
    row.update(metrics)
    row["balanced_accuracy"] = (
        metrics["recall"] + metrics["specificity"]
    ) / 2.0
    row["youden_j"] = (
        metrics["recall"] + metrics["specificity"] - 1.0
    )
    return row


def threshold_values():
    return np.round(
        np.arange(
            THRESHOLD_MIN,
            THRESHOLD_MAX + THRESHOLD_STEP / 2.0,
            THRESHOLD_STEP,
        ),
        6,
    )


def evaluate_threshold(y_true, scores, threshold):
    predictions = [
        1 if score >= threshold else 0
        for score in scores
    ]
    return metrics_row(
        threshold,
        y_true,
        predictions,
    )


def evaluate_policy_threshold(rows, threshold):
    y_true = [row["y_true"] for row in rows]
    predictions = []
    for row in rows:
        decision = policy_decide(
            row["unified_score"],
            threshold=float(threshold),
            schema_score=row["schema_score"],
            semantic_score=row["semantic_score"],
            statistical_score=row["statistical_score"],
        )
        predictions.append(
            1 if decision == POSITIVE else 0
        )
    return metrics_row(
        threshold,
        y_true,
        predictions,
    )


def threshold_sweep(y_true, scores):
    return [
        evaluate_threshold(
            y_true,
            scores,
            float(threshold),
        )
        for threshold in threshold_values()
    ]


def policy_threshold_sweep(rows):
    return [
        evaluate_policy_threshold(
            rows,
            float(threshold),
        )
        for threshold in threshold_values()
    ]


def best_thresholds(sweep_rows, key):
    best_value = max(row[key] for row in sweep_rows)
    best = [
        row["threshold"]
        for row in sweep_rows
        if row[key] == best_value
    ]
    return {
        "key": key,
        "value": best_value,
        "thresholds": best,
    }


def roc_auc(y_true, scores):
    labels = np.asarray(y_true, dtype=int)
    values = np.asarray(scores, dtype=float)
    positive = labels == 1
    negative = labels == 0
    n_pos = int(positive.sum())
    n_neg = int(negative.sum())
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values))

    index = 0
    while index < len(sorted_values):
        end = index
        while (
            end + 1 < len(sorted_values)
            and sorted_values[end + 1] == sorted_values[index]
        ):
            end += 1
        average_rank = (index + end) / 2.0 + 1.0
        ranks[order[index:end + 1]] = average_rank
        index = end + 1

    rank_sum_positive = float(ranks[positive].sum())
    u = (
        rank_sum_positive
        - n_pos * (n_pos + 1) / 2.0
    )
    return float(u / (n_pos * n_neg))


def average_precision(y_true, scores):
    labels = np.asarray(y_true, dtype=int)
    values = np.asarray(scores, dtype=float)
    if int(labels.sum()) == 0:
        return None

    order = np.argsort(-values, kind="mergesort")
    tp = 0
    fp = 0
    sum_precision = 0.0
    for index in order:
        if labels[index] == 1:
            tp += 1
            sum_precision += tp / (tp + fp)
        else:
            fp += 1
    return float(sum_precision / labels.sum())


def roc_points(y_true, scores):
    points = []
    for threshold in sorted(
        threshold_values(),
        reverse=True,
    ):
        predictions = [
            1 if score >= threshold else 0
            for score in scores
        ]
        counts = confusion_counts(
            y_true,
            predictions,
        )
        metrics = compute_metrics(**counts)
        points.append(
            {
                "threshold": float(threshold),
                "fpr": metrics["fpr"],
                "tpr": metrics["recall"],
            }
        )
    return points


def pr_points(y_true, scores):
    points = []
    for threshold in sorted(
        threshold_values(),
        reverse=True,
    ):
        predictions = [
            1 if score >= threshold else 0
            for score in scores
        ]
        counts = confusion_counts(
            y_true,
            predictions,
        )
        metrics = compute_metrics(**counts)
        points.append(
            {
                "threshold": float(threshold),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
            }
        )
    return points


def score_distribution_table(df):
    frames = []
    for column in [
        "schema_score",
        "statistical_score",
        "semantic_score",
        "unified_score",
    ]:
        grouped = (
            df
            .groupby("ground_truth")[column]
            .agg(["count", "min", "max", "mean", "median", "std"])
            .reset_index()
        )
        grouped.insert(0, "score", column)
        frames.append(grouped)
    return pd.concat(frames, ignore_index=True)


def per_drift_type_table(df):
    table = df.copy()
    table["detected"] = table["decision"] == POSITIVE
    rows = []
    for (drift_type, subtype), group in table.groupby(
        ["drift_type", "subtype"],
        sort=True,
    ):
        rows.append(
            {
                "drift_type": drift_type,
                "subtype": subtype,
                "windows": int(len(group)),
                "detected": int(group["detected"].sum()),
                "missed": int((~group["detected"]).sum()),
                "channels": "; ".join(
                    group["channel"].tolist()
                ),
                "event_ids": ", ".join(
                    group["event_id"].tolist()
                ),
            }
        )
    return pd.DataFrame(rows)


def detection_channel(series):
    fires = []
    if series["schema_score"] >= 1.0:
        fires.append("schema")
    if series["semantic_score"] >= 1.0:
        fires.append("semantic")
    if series["statistical_score"] >= 1.0:
        fires.append("statistical")
    if not fires:
        return "none"
    return "+".join(fires)


def per_column_table(df):
    rows = []
    for _, row in df.iterrows():
        details = load_json(row["statistical_details"])
        numeric = details.get("numeric", {})
        if not isinstance(numeric, dict):
            numeric = {}
        for column in NUMERIC_COLUMNS:
            metrics = numeric.get(column, {})
            if not isinstance(metrics, dict):
                metrics = {}
            rows.append(
                {
                    "event_id": row["event_id"],
                    "ground_truth": row["ground_truth"],
                    "subtype": row["subtype"],
                    "column": column,
                    "ks_statistic": metrics.get(
                        "ks_statistic"
                    ),
                    "ks_p_value": metrics.get(
                        "ks_p_value"
                    ),
                    "psi": metrics.get("psi"),
                }
            )
    return pd.DataFrame(rows)


def load_json(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return {}
    return {}


def variance_shift_analysis(df, per_column):
    controls = df[
        df["event_id"].isin(
            ["window_000", "window_001", "window_009"]
        )
    ]
    target = df[df["event_id"] == "window_005"]
    if len(target) == 0:
        return {}
    target_row = target.iloc[0]
    control_rows = controls
    control_temp = per_column[
        per_column["event_id"].isin(
            ["window_000", "window_001", "window_009"]
        )
        & (per_column["column"] == "temp")
    ]
    target_temp = per_column[
        (per_column["event_id"] == "window_005")
        & (per_column["column"] == "temp")
    ]
    result = {
        "target": {
            "event_id": "window_005",
            "statistical_score": float(
                target_row["statistical_score"]
            ),
            "unified_score": float(
                target_row["unified_score"]
            ),
            "decision": target_row["decision"],
            "temp_ks": float(
                target_temp["ks_statistic"].iloc[0]
            )
            if len(target_temp) > 0
            else None,
            "temp_psi": float(
                target_temp["psi"].iloc[0]
            )
            if len(target_temp) > 0
            else None,
        },
        "clean_controls": {
            "statistical_score_min": float(
                control_rows["statistical_score"].min()
            ),
            "statistical_score_max": float(
                control_rows["statistical_score"].max()
            ),
            "temp_ks_values": (
                [
                    float(value)
                    for value in control_temp[
                        "ks_statistic"
                    ].tolist()
                ]
            ),
            "temp_psi_values": (
                [
                    float(value)
                    for value in control_temp[
                        "psi"
                    ].tolist()
                ]
            ),
        },
        "separability": {},
    }
    target_stat = result["target"]["statistical_score"]
    clean_min = result["clean_controls"][
        "statistical_score_min"
    ]
    clean_max = result["clean_controls"][
        "statistical_score_max"
    ]
    result["separability"] = {
        "statistical_score_inside_clean_band": bool(
            clean_min <= target_stat <= clean_max
        ),
        "statistical_score_gap_to_clean_max": float(
            target_stat - clean_max
        ),
        "target_temp_ks_above_clean_max_temp_ks": bool(
            result["target"]["temp_ks"]
            > max(
                result["clean_controls"]["temp_ks_values"]
            )
        ),
    }
    return result


def load_dataframe():
    import psycopg2

    conn = psycopg2.connect(
        host=os.environ.get("PGHOST", "postgres"),
        port=int(os.environ.get("PGPORT", "5432")),
        dbname=os.environ.get("PGDATABASE", "drift_db"),
        user=os.environ.get("PGUSER", "drift"),
        password=os.environ.get("PGPASSWORD", "drift"),
    )
    query = """
        SELECT
            d.event_id,
            g.ground_truth,
            g.drift_type,
            g.subtype,
            g.feature,
            d.schema_drift_flag,
            d.schema_score,
            d.statistical_ks,
            d.statistical_psi,
            d.statistical_details,
            d.statistical_score,
            d.semantic_drift_flag,
            d.semantic_score,
            d.unified_score,
            d.decision
        FROM detection_events d
        JOIN ground_truth g ON d.event_id = g.event_id
        ORDER BY d.id
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def write_output(name, data):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    if isinstance(data, pd.DataFrame):
        data.to_csv(path, index=False)
    else:
        with open(path, "w") as file:
            json.dump(data, file, indent=2)
    print(f"  wrote {path}")
    return path


def make_plots(y_true, scores, score_table):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available; skipping plots")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    roc = roc_points(y_true, scores)
    plt.figure(figsize=(6, 6))
    plt.plot(
        [point["fpr"] for point in roc],
        [point["tpr"] for point in roc],
        marker="o",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (unified_score)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=120)
    plt.close()

    pr = pr_points(y_true, scores)
    plt.figure(figsize=(6, 6))
    plt.plot(
        [point["recall"] for point in pr],
        [point["precision"] for point in pr],
        marker="o",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (unified_score)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pr_curve.png", dpi=120)
    plt.close()

    scores_array = np.asarray(scores, dtype=float)
    labels = np.asarray(y_true, dtype=int)
    drift = scores_array[labels == 1]
    clean = scores_array[labels == 0]
    plt.figure(figsize=(8, 4))
    plt.hist(
        clean,
        bins=12,
        alpha=0.6,
        label="NO_DRIFT",
    )
    plt.hist(
        drift,
        bins=12,
        alpha=0.6,
        label="DRIFT",
    )
    plt.xlabel("unified_score")
    plt.ylabel("windows")
    plt.title("Unified Score Distribution by Ground Truth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "score_distribution.png",
        dpi=120,
    )
    plt.close()

    print("  wrote roc_curve.png / pr_curve.png / score_distribution.png")


def summarize_sweep(sweep_rows):
    summary = {}
    for key in ["f1", "youden_j", "balanced_accuracy"]:
        summary[key] = best_thresholds(
            sweep_rows,
            key,
        )
    return summary


def report(df):
    df = df.copy()
    df["y_true"] = df["ground_truth"].apply(
        ground_truth_binary
    )
    df["channel"] = df.apply(
        detection_channel,
        axis=1,
    )

    scores = df["unified_score"].astype(float).tolist()
    y_true = df["y_true"].tolist()

    rows_for_policy = df[[
        "unified_score",
        "schema_score",
        "semantic_score",
        "statistical_score",
        "y_true",
    ]].to_dict("records")

    unified_sweep = threshold_sweep(y_true, scores)
    policy_sweep = policy_threshold_sweep(
        rows_for_policy
    )

    distributions = score_distribution_table(df)
    drift_type_table = per_drift_type_table(df)
    column_table = per_column_table(df)

    roc = roc_auc(y_true, scores)
    pr = average_precision(y_true, scores)

    variance = variance_shift_analysis(
        df,
        column_table,
    )

    threshold_05_policy = next(
        row
        for row in policy_sweep
        if abs(row["threshold"] - 0.5) < 1e-9
    )
    threshold_05_unified = next(
        row
        for row in unified_sweep
        if abs(row["threshold"] - 0.5) < 1e-9
    )

    summary = {
        "dataset": (
            "controlled 10-window experimental ground-truth set"
        ),
        "windows": int(len(df)),
        "ground_truth_counts": (
            df["ground_truth"].value_counts().to_dict()
        ),
        "score_distributions": (
            distributions.to_dict("records")
        ),
        "unified_only": {
            "roc_auc": roc,
            "pr_auc": pr,
            "best": summarize_sweep(unified_sweep),
        },
        "final_policy": {
            "best": summarize_sweep(policy_sweep),
        },
        "threshold_0_5_unified_only": threshold_05_unified,
        "threshold_0_5_final_policy": threshold_05_policy,
        "per_drift_type": drift_type_table.to_dict(
            "records"
        ),
        "variance_shift_investigation": variance,
        "statistical_normalization": {
            "ks_floor": KS_FLOOR,
            "psi_scale": PSI_SCALE,
            "chi2_scale": CHI2_SCALE,
            "statistical_decision_threshold": (
                STATISTICAL_DECISION_THRESHOLD
            ),
            "note": (
                "constants frozen in configs/thresholds.yaml; "
                "clean statistical score band 0.12-0.25"
            ),
        },
        "limitations": [
            "10-window dataset: metrics and AUC are "
            "illustrative, not statistically reliable",
            "threshold selection is unstable across "
            "nearby thresholds",
            "variance-shift windows are expected false "
            "negatives under max-over-columns aggregation",
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sweep_frame = pd.DataFrame(unified_sweep)
    sweep_frame.insert(0, "mode", "unified_only")
    policy_frame = pd.DataFrame(policy_sweep)
    policy_frame.insert(0, "mode", "final_policy")
    combined = pd.concat(
        [sweep_frame, policy_frame],
        ignore_index=True,
    )
    combined.to_csv(
        OUTPUT_DIR / "threshold_sweep.csv",
        index=False,
    )
    print(f"  wrote {OUTPUT_DIR / 'threshold_sweep.csv'}")

    distributions.to_csv(
        OUTPUT_DIR / "score_distributions.csv",
        index=False,
    )
    print(
        "  wrote "
        f"{OUTPUT_DIR / 'score_distributions.csv'}"
    )

    drift_type_table.to_csv(
        OUTPUT_DIR / "per_drift_type.csv",
        index=False,
    )
    print(
        "  wrote "
        f"{OUTPUT_DIR / 'per_drift_type.csv'}"
    )

    column_table.to_csv(
        OUTPUT_DIR / "per_column.csv",
        index=False,
    )
    print(
        "  wrote "
        f"{OUTPUT_DIR / 'per_column.csv'}"
    )

    with open(
        OUTPUT_DIR / "metrics.json",
        "w",
    ) as file:
        json.dump(summary, file, indent=2)
    print(
        "  wrote "
        f"{OUTPUT_DIR / 'metrics.json'}"
    )

    make_plots(y_true, scores, distributions)

    return summary


def print_report(df, summary):
    print()
    print("=" * 90)
    print("PHASE 6 EVALUATION")
    print("=" * 90)

    print()
    print("PER-WINDOW EVALUATION ROWS")
    print()
    table = df[[
        "event_id",
        "ground_truth",
        "drift_type",
        "subtype",
        "schema_score",
        "statistical_score",
        "semantic_score",
        "unified_score",
        "decision",
    ]].copy()
    table["channel"] = table.apply(
        detection_channel,
        axis=1,
    )
    table["unified_score"] = table[
        "unified_score"
    ].round(4)
    print(table.to_string(index=False))

    print()
    print("SCORE DISTRIBUTIONS (by ground truth)")
    print()
    dist = pd.DataFrame(
        summary["score_distributions"]
    )
    print(dist.to_string(index=False))

    print()
    print("UNIFIED-ONLY SWEEP SUMMARY")
    print()
    print(
        "  ROC-AUC: "
        f"{summary['unified_only']['roc_auc']:.4f}"
    )
    print(
        "  PR-AUC (average precision): "
        f"{summary['unified_only']['pr_auc']:.4f}"
    )
    for key, best in summary["unified_only"][
        "best"
    ].items():
        print(
            f"  best by {key}: value="
            f"{best['value']:.4f} at thresholds="
            f"{best['thresholds']}"
        )

    print()
    print("FINAL-POLICY SWEEP SUMMARY (with overrides)")
    print()
    for key, best in summary["final_policy"][
        "best"
    ].items():
        print(
            f"  best by {key}: value="
            f"{best['value']:.4f} at thresholds="
            f"{best['thresholds']}"
        )

    print()
    print("METRICS AT THRESHOLD 0.5")
    print()
    print(
        "  unified-only: "
        f"{json.dumps(summary['threshold_0_5_unified_only'])}"
    )
    print(
        "  final policy: "
        f"{json.dumps(summary['threshold_0_5_final_policy'])}"
    )

    print()
    print("PER-DRIFT-TYPE")
    print()
    print(
        pd.DataFrame(
            summary["per_drift_type"]
        ).to_string(index=False)
    )

    print()
    print("VARIANCE-SHIFT INVESTIGATION (window_005)")
    print()
    print(
        json.dumps(
            summary["variance_shift_investigation"],
            indent=2,
        )
    )


def main():
    df = load_dataframe()
    summary = report(df)
    print_report(df, summary)
    return summary


if __name__ == "__main__":
    main()
