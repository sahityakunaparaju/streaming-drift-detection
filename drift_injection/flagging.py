"""
Shared expected-false-negative flagging rule (Phase 6/7).

Known measurement limitations under max-over-columns statistical
aggregation (variance shift, distribution shift) are flagged as
"expected false negatives" so Phase 7+ precision/recall computation
can account for them instead of silently penalizing the detector.

This is the single source of truth. Both the Phase 6 tuning pass and
every window generator (stream_builder, phase7_generator) import the
rule from here; do not duplicate the logic elsewhere.

The rule is prefix-based, so it accepts both tuning-pass condition
names ("variance_low") and ground-truth subtype strings
("variance_shift") without redesign.
"""

EXPECTED_FN_AGGREGATION_LIMITATION = (
    "expected false negative — known aggregation limitation"
)

# Machine-readable flag attached to every variance_* condition.
# Reasons are prefixed with the canonical flag string so the value
# stored in ground truth is self-describing and consumers can match
# on the prefix (or simply on a non-empty value).
VARIANCE_FN_REASON = (
    "expected false negative — known aggregation limitation: "
    "post-shift KS (~0.26) sits below the sampling-noise floor of "
    "the other columns (~0.40-0.50) and PSI does not move under "
    "max-over-columns aggregation; documented in docs/phase6_notes.md, "
    "not fixed in Phase 6"
)

# Machine-readable flag attached to every distribution_* condition.
# Structurally the same limitation as variance shift (see notes).
DISTRIBUTION_FN_REASON = (
    "expected false negative — known aggregation limitation: "
    "temp KS elevation (0.448 -> 0.474) stays below the ks_floor "
    "(0.50) and PSI does not move under max-over-columns "
    "aggregation; structurally the same known limitation as "
    "variance shift; documented in docs/phase6_notes.md, not fixed "
    "in Phase 6"
)


def expected_false_negative_condition(condition):
    """
    Conditions/subtypes whose injected windows are expected false
    negatives under max-over-columns aggregation (variance shift,
    distribution shift). Phase 7 consumes the flag generically across
    drift types.
    """
    return (
        condition.startswith("variance_")
        or condition.startswith("distribution_")
    )


def expected_false_negative_reason(condition):
    """
    Machine-readable reason string for an expected-false-negative
    condition, or None when the condition is not flagged.
    """
    if condition.startswith("variance_"):
        return VARIANCE_FN_REASON
    if condition.startswith("distribution_"):
        return DISTRIBUTION_FN_REASON
    return None
