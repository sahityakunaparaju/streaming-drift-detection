"""Per-column statistical drift measurement.

Revision (Step 2): scoring is per COLUMN, not max-over-columns.

For every numeric column we compute a set of distribution-comparison
statistics against the matching reference sample:

    ks_statistic       two-sample Kolmogorov-Smirnov statistic
    ks_p_value         its p-value
    psi                Population Stability Index (10 bins from the ref)
    smd                standardized mean difference  (mean_c - mean_r)/std_r
    wasserstein_norm   Wasserstein distance / std_r
    effect_size        max(|smd|, wasserstein_norm)

    The effect size deliberately does NOT include the KS statistic: the
    primary telemetry is quantized to a few discrete levels inside a
    1000-row window, so even a clean local window can have KS ~1 against
    a slightly different quantization level. KS and PSI are therefore
    reported as diagnostics, while the calibrated decision statistic is
    the (scale-free) mean/Wasserstein effect.
    p_value            ks_p_value (Bonferroni-corrected across the device
                       slices of the column when per-device referencing)

Categorical columns use a chi-square test:

    chi2, p_value, cramers_v = sqrt(chi2 / (n * (k - 1)))
    effect_size = cramers_v

A column is a drift trigger when BOTH hold:

    p_value (after Benjamini-Hochberg across columns) <= alpha
    AND effect_size >= the column's calibrated threshold

A window is statistically drifted when ANY column triggers. The
per-column thresholds are derived on CLEAN calibration windows only
(experiments/derive_thresholds.py) and stored in configs/thresholds.yaml.

The legacy `compare_numeric_fields` / `compare_distribution` helpers are
kept for backwards compatibility (they return raw KS/PSI only) and the
legacy `statistical_score` path in detectors/unified.py uses them.
"""

import math

import numpy as np
from scipy.stats import chi2_contingency, ks_2samp, wasserstein_distance

from detectors.thresholds import THRESHOLDS

PSI_BINS = int(THRESHOLDS["statistical"]["psi_bins"])

# Guard against division by a zero reference std (constant column).
MIN_STD = 1e-12


def calculate_psi(reference, current, bins=None):
    """Population Stability Index: sum((cur_pct - ref_pct) * ln(cur_pct / ref_pct))."""
    if bins is None:
        bins = PSI_BINS

    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    edges = np.quantile(
        reference,
        np.linspace(0, 1, bins + 1)
    )

    edges = np.unique(edges)

    if len(edges) < 2:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_counts, _ = np.histogram(reference, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)

    reference_pct = reference_counts / len(reference)
    current_pct = current_counts / len(current)

    epsilon = 1e-10

    reference_pct = np.maximum(reference_pct, epsilon)
    current_pct = np.maximum(current_pct, epsilon)

    psi = np.sum(
        (current_pct - reference_pct)
        * np.log(current_pct / reference_pct)
    )

    return float(psi)


def calculate_ks(reference, current):
    """Two-sample Kolmogorov-Smirnov statistic and p-value."""

    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    statistic, p_value = ks_2samp(
        reference,
        current
    )

    return {
        "ks_statistic": float(statistic),
        "ks_p_value": float(p_value),
    }


def compare_distribution(reference, current, bins=None):
    """Raw KS/PSI measurements; no thresholds or drift decisions here."""
    if bins is None:
        bins = PSI_BINS

    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    ks_result = calculate_ks(reference, current)

    psi = calculate_psi(
        reference,
        current,
        bins=bins
    )

    return {
        "ks_statistic": ks_result["ks_statistic"],
        "ks_p_value": ks_result["ks_p_value"],
        "psi": psi,
    }


def compare_numeric_fields(reference_df, current_df, columns):
    """Per-column raw KS and PSI measurements for the given fields."""

    results = {}

    for column in columns:

        if column not in reference_df.columns:
            raise ValueError(
                f"Column '{column}' missing from reference data"
            )

        if column not in current_df.columns:
            raise ValueError(
                f"Column '{column}' missing from current data"
            )

        reference = reference_df[column].dropna()
        current = current_df[column].dropna()

        results[column] = compare_distribution(
            reference,
            current
        )

    return results


def numeric_column_metrics(reference, current, bins=None):
    """Full per-column numeric statistics (KS/PSI/SMD/Wasserstein/effect)."""
    if bins is None:
        bins = PSI_BINS

    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    if len(reference) == 0 or len(current) == 0:
        return {
            "ks_statistic": 0.0,
            "ks_p_value": 1.0,
            "psi": 0.0,
            "smd": 0.0,
            "wasserstein_norm": 0.0,
            "effect_size": 0.0,
        }

    ks_result = calculate_ks(reference, current)
    psi = calculate_psi(reference, current, bins=bins)

    reference_std = float(np.std(reference))
    reference_mean = float(np.mean(reference))
    current_mean = float(np.mean(current))

    if reference_std <= MIN_STD:
        smd = 0.0
        wasserstein_norm = 0.0
    else:
        smd = (current_mean - reference_mean) / reference_std
        wasserstein_norm = (
            float(wasserstein_distance(reference, current))
            / reference_std
        )

    effect_size = max(abs(smd), abs(wasserstein_norm))

    return {
        "ks_statistic": ks_result["ks_statistic"],
        "ks_p_value": ks_result["ks_p_value"],
        "psi": psi,
        "smd": float(smd),
        "wasserstein_norm": float(wasserstein_norm),
        "effect_size": float(effect_size),
    }


def categorical_column_metrics(reference, current):
    """Chi-square test + Cramer's V (scale-free across category counts)."""
    reference = np.asarray(reference)
    current = np.asarray(current)

    reference = reference[~_is_null(reference)]
    current = current[~_is_null(current)]

    if len(reference) == 0 or len(current) == 0:
        return {
            "chi2": 0.0,
            "p_value": 1.0,
            "cramers_v": 0.0,
            "effect_size": 0.0,
            "n_categories": 0,
        }

    ref_counts = _value_counts(reference)
    cur_counts = _value_counts(current)

    categories = sorted(set(ref_counts) | set(cur_counts))

    observed = np.array(
        [
            [ref_counts.get(c, 0) for c in categories],
            [cur_counts.get(c, 0) for c in categories],
        ],
        dtype=float,
    )

    if observed.sum() == 0 or len(categories) < 2:
        return {
            "chi2": 0.0,
            "p_value": 1.0,
            "cramers_v": 0.0,
            "effect_size": 0.0,
            "n_categories": len(categories),
        }

    try:
        chi2, p_value, _dof, _expected = chi2_contingency(observed)
    except ValueError:
        # Degenerate table (e.g. a zero row).
        return {
            "chi2": 0.0,
            "p_value": 1.0,
            "cramers_v": 0.0,
            "effect_size": 0.0,
            "n_categories": len(categories),
        }

    total = observed.sum()
    cramers_v = math.sqrt(chi2 / (total * (len(categories) - 1)))

    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "cramers_v": float(cramers_v),
        "effect_size": float(cramers_v),
        "n_categories": len(categories),
    }


def benjamini_hochberg(p_values, alpha):
    """BH step-up: return a boolean list, True where the null is rejected.

    `p_values` is a sequence of (label, p_value) pairs or a plain list of
    p-values in the same order as the caller's labels.
    """
    values = [float(p) for p in p_values]
    n = len(values)
    rejected = [False] * n
    if n == 0:
        return rejected

    order = sorted(range(n), key=lambda i: values[i])
    threshold_rank = -1
    for rank, index in enumerate(order, start=1):
        if values[index] <= alpha * rank / n:
            threshold_rank = rank
    if threshold_rank == -1:
        return rejected
    for rank, index in enumerate(order, start=1):
        if rank <= threshold_rank:
            rejected[index] = True
    return rejected


def _value_counts(values):
    counts = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _is_null(values):
    """NaN/None mask that works for object and float arrays."""
    return np.array(
        [value is None or value != value for value in values],
        dtype=bool,
    )
