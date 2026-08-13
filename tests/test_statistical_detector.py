import numpy as np

from detectors.statistical_detector import (
    calculate_psi,
    calculate_ks,
    compare_distribution,
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