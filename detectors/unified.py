"""Unified drift decision (Revision, Step 2d).

FINAL DECISION RULE — this docstring is the single source of truth:

    A window is DRIFT when ANY of the following channels fires:

    1. SCHEMA    any schema event (added / removed / renamed / type change
                 / nullability change). Binary by design: schema drift is
                 exact set comparison of columns, dtypes and nullability.
    2. SEMANTIC  any semantic contract event or value-contract violation.
    3. STATISTICAL  any single column is a trigger, i.e. its p-value passes
                 Benjamini-Hochberg correction across columns (alpha from
                 configs/thresholds.yaml) AND its effect size is at or above
                 that column's calibrated clean-data threshold.

    There is no separate "unified score threshold" path: the old
    `unified.threshold = 0.5` branch was mathematically unreachable
    (schema/semantic binary overrides forced it below 1/3), so it was
    removed. `unified_score` is now a continuous, reporting-only aggregate
    (mean of the three continuous channel scores). The decision uses the
    channel gates above.

    `statistical_score` is continuous and per-column: for each column
        column_score = min(1, effect_size / column_effect_threshold)
    and the channel score is the maximum over columns. A score >= 1 means
    the effect-size gate passed for at least one column (the p-value gate
    is evaluated in detectors/reference.py and reported per column).

The schema/semantic channel scores are severity-weighted sums of their
events (high = 1.0, medium = 0.6, low = 0.3, capped at 1.0), so they are
continuous rather than binary.
"""

import math

from detectors.thresholds import THRESHOLDS

STATISTICAL = THRESHOLDS.get("statistical", {})
UNIFIED = THRESHOLDS.get("unified", {})

# Calibrated effect-size score threshold. With the new per-column score
# (effect / threshold) a value of 1.0 means "effect-size gate passed".
STATISTICAL_DECISION_THRESHOLD = float(
    STATISTICAL.get("decision_threshold", 1.0)
)

SEVERITY_WEIGHTS = {
    str(key): float(value)
    for key, value in UNIFIED.get(
        "severity_weights",
        {"high": 1.0, "medium": 0.6, "low": 0.3},
    ).items()
}

DEFAULT_EFFECT_THRESHOLD = float(
    STATISTICAL.get("default_effect_threshold", 0.1)
)


def _legacy_effect_score(statistical_result):
    """Score legacy raw KS/PSI/chi2 details (streaming consumers)."""
    numeric = statistical_result.get("numeric", {}) or {}
    categorical = statistical_result.get("categorical", {}) or {}

    scores = []

    for metrics in numeric.values():
        if not isinstance(metrics, dict):
            continue
        if "effect_size" in metrics:
            effect = metrics.get("effect_size") or 0.0
        else:
            ks = metrics.get("ks_statistic")
            effect = abs(ks) if ks is not None and math.isfinite(ks) else 0.0
        threshold = metrics.get("threshold") or DEFAULT_EFFECT_THRESHOLD
        if threshold and threshold > 0:
            scores.append(effect / threshold)

    for metrics in categorical.values():
        if not isinstance(metrics, dict):
            continue
        if "effect_size" in metrics:
            effect = metrics.get("effect_size") or 0.0
        else:
            chi2 = metrics.get("chi2")
            effect = (
                math.sqrt(abs(chi2) / 1000.0)
                if chi2 is not None and math.isfinite(chi2)
                else 0.0
            )
        threshold = metrics.get("threshold") or DEFAULT_EFFECT_THRESHOLD
        if threshold and threshold > 0:
            scores.append(effect / threshold)

    return max(scores) if scores else 0.0


def statistical_score(statistical_result):
    """Continuous statistical channel score in [0, 1].

    Uses the per-column `statistical_score` produced by
    detectors/reference.py when present; otherwise falls back to scoring
    the raw per-column metrics against the configured effect thresholds
    (legacy callers such as the streaming consumers).
    """
    if not isinstance(statistical_result, dict):
        return 0.0

    explicit = statistical_result.get("statistical_score")
    if explicit is not None:
        return max(0.0, min(float(explicit), 1.0))

    score = _legacy_effect_score(statistical_result)
    return max(0.0, min(score, 1.0))


def channel_score(events):
    """Severity-weighted continuous score for a schema/semantic channel.

    `events` may be a bool (legacy callers) or a list of event dicts.
    """
    if isinstance(events, bool):
        return 1.0 if events else 0.0
    if events is None:
        return 0.0
    if isinstance(events, dict):
        return 1.0 if events else 0.0
    if not events:
        return 0.0
    total = 0.0
    for event in events:
        severity = "high"
        if isinstance(event, dict):
            severity = str(event.get("severity", "high")).lower()
        total += SEVERITY_WEIGHTS.get(severity, 1.0)
    return min(1.0, total)


def unified_score(
    schema_flag,
    statistical_result,
    semantic_flag,
):
    """Mean of the three continuous channel scores (reporting only)."""
    schema = channel_score(schema_flag)
    semantic = channel_score(semantic_flag)
    statistical = statistical_score(statistical_result)

    unified = (schema + statistical + semantic) / 3.0

    return {
        "schema_score": schema,
        "statistical_score": statistical,
        "semantic_score": semantic,
        "unified_score": unified,
    }


def decide(
    unified_score,
    threshold=None,
    schema_score=0.0,
    semantic_score=0.0,
    statistical_score=0.0,
    statistical_drift=None,
):
    """Apply the final OR-of-channels rule (see module docstring).

    `threshold` and the first positional argument are accepted for
    backwards compatibility with existing callers but no longer decide
    anything; the unified-score threshold path was removed.
    """
    if schema_score > 0.0:
        return "DRIFT"

    if semantic_score > 0.0:
        return "DRIFT"

    if statistical_drift is True:
        return "DRIFT"

    if statistical_score >= STATISTICAL_DECISION_THRESHOLD:
        return "DRIFT"

    return "NO_DRIFT"
