import math

from detectors.thresholds import THRESHOLDS

STATISTICAL = THRESHOLDS["statistical"]
UNIFIED = THRESHOLDS["unified"]

KS_FLOOR = float(STATISTICAL["ks_floor"])
PSI_SCALE = float(STATISTICAL["psi_scale"])
CHI2_SCALE = float(STATISTICAL["chi2_scale"])
STATISTICAL_DECISION_THRESHOLD = float(
    STATISTICAL["decision_threshold"]
)
UNIFIED_THRESHOLD = float(UNIFIED["threshold"])


def statistical_score(statistical_result):
    if not isinstance(statistical_result, dict):
        return 0.0

    numeric = statistical_result.get("numeric", {})
    categorical = statistical_result.get("categorical", {})

    max_ks = 0.0
    max_psi = 0.0

    for metrics in numeric.values():

        if not isinstance(metrics, dict):
            continue

        ks = metrics.get("ks_statistic")
        psi = metrics.get("psi")

        if ks is not None and math.isfinite(ks):
            max_ks = max(max_ks, abs(ks))

        if psi is not None and math.isfinite(psi):
            max_psi = max(max_psi, psi)

    max_chi2 = 0.0

    for metrics in categorical.values():

        if not isinstance(metrics, dict):
            continue

        chi2 = metrics.get("chi2")

        if chi2 is not None and math.isfinite(chi2):
            max_chi2 = max(max_chi2, chi2)

    ks_component = min(
        max(max_ks - KS_FLOOR, 0.0)
        / (1.0 - KS_FLOOR),
        1.0,
    )
    psi_component = min(
        max_psi / PSI_SCALE,
        1.0,
    )
    chi2_component = min(
        max_chi2 / CHI2_SCALE,
        1.0,
    )

    score = max(
        ks_component,
        psi_component,
        chi2_component,
    )

    return max(0.0, min(score, 1.0))


def unified_score(
    schema_flag,
    statistical_result,
    semantic_flag,
):
    schema = 1.0 if schema_flag else 0.0
    semantic = 1.0 if semantic_flag else 0.0
    statistical = statistical_score(
        statistical_result
    )

    unified = (
        schema + statistical + semantic
    ) / 3.0

    return {
        "schema_score": schema,
        "statistical_score": statistical,
        "semantic_score": semantic,
        "unified_score": unified,
    }


def decide(
    unified_score,
    threshold=UNIFIED_THRESHOLD,
    schema_score=0.0,
    semantic_score=0.0,
    statistical_score=0.0,
):
    if (
        schema_score >= UNIFIED["schema_max_score"]
        or semantic_score >= UNIFIED["semantic_max_score"]
        or statistical_score >= UNIFIED["statistical_max_score"]
    ):
        return "DRIFT"

    if statistical_score >= STATISTICAL_DECISION_THRESHOLD:
        return "DRIFT"

    if unified_score >= threshold:
        return "DRIFT"

    return "NO_DRIFT"
