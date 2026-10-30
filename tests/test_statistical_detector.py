import numpy as np

from detectors.statistical_detector import (
    calculate_psi,
    calculate_ks,
    compare_distribution,
    numeric_column_metrics,
)


RNG = np.random.default_rng(42)


def test_identical_distributions():

    reference = RNG.normal(22, 2, 5000)
    current = RNG.normal(22, 2, 5000)

    result = compare_distribution(
        reference,
        current
    )

    assert result["ks_statistic"] < 0.05
    assert result["psi"] < 0.05


def test_mean_shift():

    reference = RNG.normal(22, 2, 5000)
    current = RNG.normal(26, 2, 5000)

    result = compare_distribution(
        reference,
        current
    )

    assert result["ks_statistic"] > 0.5
    assert result["ks_p_value"] < 0.001
    assert result["psi"] > 0.1


def test_variance_shift():

    reference = RNG.normal(22, 2, 5000)
    current = RNG.normal(22, 6, 5000)

    result = compare_distribution(
        reference,
        current
    )

    assert result["ks_statistic"] > 0.1
    assert result["ks_p_value"] < 0.001
    assert result["psi"] > 0.1


def test_identical_array_has_zero_psi():

    reference = np.array(
        [1, 2, 3, 4, 5],
        dtype=float
    )

    current = reference.copy()

    result = calculate_psi(
        reference,
        current
    )

    assert result == 0.0


def test_ks_identical_arrays():

    reference = np.array(
        [1, 2, 3, 4, 5],
        dtype=float
    )

    current = reference.copy()

    result = calculate_ks(
        reference,
        current
    )

    assert result["ks_statistic"] == 0.0
    assert result["ks_p_value"] == 1.0


def test_numeric_column_metrics_reports_effect_sizes():

    rng = np.random.default_rng(123)
    reference = rng.normal(22, 2, 3000)
    current = rng.normal(24, 2, 3000)

    metrics = numeric_column_metrics(reference, current)

    for key in [
        "ks_statistic",
        "ks_p_value",
        "psi",
        "smd",
        "wasserstein_norm",
        "effect_size",
    ]:
        assert key in metrics

    # 2 degrees of mean shift under a sigma=2 reference.
    assert abs(metrics["smd"] - 1.0) < 0.1
    # Effect excludes KS but still sees the shift.
    assert metrics["effect_size"] >= 0.8
    assert metrics["ks_p_value"] < 0.001


def test_numeric_column_metrics_clean_window_has_small_effect():

    rng = np.random.default_rng(7)
    reference = rng.normal(22, 2, 3000)
    current = rng.normal(22, 2, 1000)

    metrics = numeric_column_metrics(reference, current)

    assert metrics["effect_size"] < 0.2


def test_quantized_clean_window_effect_ignores_ks():
    """A different quantization level gives high KS but small effect."""
    reference = np.repeat([1.0, 1.1, 1.2, 1.3], 500)
    current = np.repeat([1.11, 1.21, 1.31, 1.41], 250)

    metrics = numeric_column_metrics(reference, current)

    assert metrics["ks_statistic"] > 0.2
    # A tiny mean shift keeps KS high but the effect size small.
    current = np.repeat([1.001, 1.101, 1.201, 1.301], 250)
    metrics = numeric_column_metrics(reference, current)
    assert metrics["ks_statistic"] > 0.0
    assert metrics["effect_size"] < 0.1