import numpy as np
from scipy.stats import ks_2samp

from detectors.thresholds import THRESHOLDS

PSI_BINS = int(THRESHOLDS["statistical"]["psi_bins"])

def calculate_psi(reference, current, bins=None):
    if bins is None:
        bins = PSI_BINS
    """
    Calculate Population Stability Index (PSI).

    PSI = sum((current_pct - reference_pct)
              * ln(current_pct / reference_pct))
    """

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
    """
    Calculate two-sample Kolmogorov-Smirnov test.
    """

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
    if bins is None:
        bins = PSI_BINS
    """
    Calculate raw statistical drift measurements.

    No thresholds or drift decisions are made here.
    """

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
    """
    Compare multiple numeric columns.

    Returns raw KS and PSI measurements for each field.
    """

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
