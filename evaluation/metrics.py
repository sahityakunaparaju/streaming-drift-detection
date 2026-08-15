"""
Generic drift-detection evaluation metrics (Phase 7, reused by
Phase 8 and Phase 9 unchanged).

Computes Precision / Recall / F1 / False Positive Rate / Accuracy /
Balanced Accuracy / Youden's J from ground-truth labels + detector
predictions, plus Detection Lag (mean/median/P95) from per-window
lag measurements.

Expected-false-negative handling:
    Ground-truth windows with `expected_false_negative` set are
    known, documented measurement limitations (variance shift,
    distribution shift under max-over-columns aggregation). They are
    reported two ways, never hidden:

    - raw:      computed over ALL windows (flagged misses count as
                ordinary false negatives).
    - adjusted: computed with flagged windows excluded from the
                evaluation entirely (they are expected misses, not
                ordinary detector failures).

    The flagged windows themselves are also reported separately
    (count, how many were detected, how many were missed) so the
    raw-vs-adjusted difference is visible and auditable.

This module is intentionally shape-agnostic: it operates on plain
sequences/lists and does not know about any specific experiment.
"""

import numpy as np

def confusion_counts(y_true, y_pred):
    """Standard 2x2 confusion counts from label/prediction sequences."""
    pairs = list(zip(y_true, y_pred))
    tp = sum(1 for t, p in pairs if t == 1 and p == 1)
    fp = sum(1 for t, p in pairs if t == 0 and p == 1)
    tn = sum(1 for t, p in pairs if t == 0 and p == 0)
    fn = sum(1 for t, p in pairs if t == 1 and p == 0)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

def compute_metrics(tp, fp, tn, fn):
    """Derive all scalar metrics from confusion counts (zero-safe)."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    balanced_accuracy = (recall + specificity) / 2.0
    youden_j = recall + specificity - 1.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "fpr": round(fpr, 6),
        "specificity": round(specificity, 6),
        "accuracy": round(accuracy, 6),
        "balanced_accuracy": round(balanced_accuracy, 6),
        "youden_j": round(youden_j, 6),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }

def _flagged_mask(expected_false_negative, n):
    if expected_false_negative is None:
        return np.zeros(n, dtype=bool)
    mask = np.asarray(
        [
            bool(value) and not pd_isna(value)
            for value in expected_false_negative
        ],
        dtype=bool,
    )
    return mask

def pd_isna(value):
    """NaN check that works for plain values without importing pandas."""
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return False

def evaluate(y_true, y_pred, expected_false_negative=None):
    """
    Raw + adjusted metrics from label/prediction sequences.

    Parameters
    ----------
    y_true : sequence of 0/1 ground-truth labels.
    y_pred : sequence of 0/1 detector predictions.
    expected_false_negative : optional sequence of flag values
        (non-empty flag string means "expected miss"). NaN/None/"" are
        treated as un-flagged.

    Returns a dict with "raw", "adjusted", and the flagged-window
    breakdown. Both raw and adjusted contain the full metric set.
    """
    y_true = [1 if bool(t) else 0 for t in y_true]
    y_pred = [1 if bool(p) else 0 for p in y_pred]
    n = len(y_true)

    flagged = _flagged_mask(expected_false_negative, n)

    raw = compute_metrics(**confusion_counts(y_true, y_pred))

    keep = ~flagged
    adjusted = compute_metrics(
        **confusion_counts(
            [y_true[i] for i in range(n) if keep[i]],
            [y_pred[i] for i in range(n) if keep[i]],
        )
    )

    n_flagged = int(flagged.sum())
    flagged_detected = 0
    flagged_missed = 0
    for i in range(n):
        if flagged[i]:
            if y_pred[i] == 1:
                flagged_detected += 1
            else:
                flagged_missed += 1

    return {
        "n": n,
        "n_positives": int(sum(y_true)),
        "n_negatives": int(n - sum(y_true)),
        "raw": raw,
        "adjusted": adjusted,
        "flagged": {
            "n_flagged": n_flagged,
            "n_flagged_detected": flagged_detected,
            "n_flagged_missed": flagged_missed,
        },
    }

def lag_summary(detection_lags, unit="windows"):
    """
    Summarize detection lags (per drift stream, in `unit`).

    detection_lags : sequence of numbers or None (never detected).
    Returns mean/median/p95 over streams that WERE detected, plus
    counts of detected / never-detected streams.
    """
    detected = [
        float(value)
        for value in detection_lags
        if value is not None
    ]
    never = sum(1 for value in detection_lags if value is None)
    if not detected:
        return {
            "unit": unit,
            "mean": None,
            "median": None,
            "p95": None,
            "n_detected": 0,
            "n_never_detected": never,
        }
    return {
        "unit": unit,
        "mean": round(float(np.mean(detected)), 4),
        "median": round(float(np.median(detected)), 4),
        "p95": round(float(np.percentile(detected, 95)), 4),
        "n_detected": len(detected),
        "n_never_detected": never,
    }

def detection_lag_windows(predictions, onset_position):
    """
    Lag (in windows) for one stream: first DRIFT prediction at or
    after `onset_position` minus `onset_position`. None if never
    detected. predictions is a sequence of 0/1 (or "DRIFT"/"NO_DRIFT").
    """
    binary = [
        1
        if str(p).strip().upper() == "DRIFT" or bool(p)
        else 0
        for p in predictions
    ]
    for position in range(onset_position, len(binary)):
        if binary[position] == 1:
            return position - onset_position
    return None
