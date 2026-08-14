from drift_injection.flagging import (
    DISTRIBUTION_FN_REASON,
    EXPECTED_FN_AGGREGATION_LIMITATION,
    VARIANCE_FN_REASON,
    expected_false_negative_condition,
    expected_false_negative_reason,
)


def test_variance_conditions_flagged_as_expected_fn():

    for condition in [
        "variance_low",
        "variance_medium",
        "variance_high",
        "variance_shift",
    ]:
        assert expected_false_negative_condition(condition)
        assert (
            expected_false_negative_reason(condition)
            == VARIANCE_FN_REASON
        )


def test_distribution_conditions_flagged_as_expected_fn():

    # Phase 6 follow-up: distribution shift is the same known
    # aggregation limitation as variance shift and must receive the
    # identical expected-false-negative treatment.
    for condition in [
        "distribution_low",
        "distribution_medium",
        "distribution_high",
        "distribution_shift",
    ]:
        assert expected_false_negative_condition(condition)
        assert (
            expected_false_negative_reason(condition)
            == DISTRIBUTION_FN_REASON
        )


def test_shared_flag_string_but_distinct_reasons():

    # Phase 7 consumes the flag generically across drift types, so
    # both limitations must use the exact same flag value...
    assert EXPECTED_FN_AGGREGATION_LIMITATION == (
        "expected false negative — known aggregation limitation"
    )

    # ...while each keeps its own machine-readable reason.
    assert VARIANCE_FN_REASON != DISTRIBUTION_FN_REASON
    assert (
        expected_false_negative_reason("variance_low")
        == VARIANCE_FN_REASON
    )
    assert (
        expected_false_negative_reason("distribution_low")
        == DISTRIBUTION_FN_REASON
    )


def test_other_conditions_not_flagged():

    for condition in [
        "clean",
        "normal_noise",
        "mean_low",
        "mean_medium",
        "mean_high",
        "mean_shift",
        "category_low",
        "category_medium",
        "category_high",
        "category_shift",
        "added",
        "removed",
        "unit_change",
        "range_violation",
    ]:
        assert not expected_false_negative_condition(condition)
        assert expected_false_negative_reason(condition) is None
