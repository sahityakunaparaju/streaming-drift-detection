"""
Shared expected-false-negative flagging rule (Phase 6/7).

Known measurement limitations under max-over-columns statistical
aggregation (variance shift, distribution shift, mean shift) are
flagged as "expected false negatives" so Phase 7+ precision/recall
computation can account for them instead of silently penalizing the
detector.

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

VARIANCE_FN_REASON = (
    "expected false negative — known aggregation limitation: "
    "post-shift KS (~0.26) sits below the sampling-noise floor of "
    "the other columns (~0.40-0.50) and PSI does not move under "
    "max-over-columns aggregation; documented in docs/phase6_notes.md, "
    "not fixed in Phase 6"
)

DISTRIBUTION_FN_REASON = (
    "expected false negative — known aggregation limitation: "
    "temp KS elevation (0.448 -> 0.474) stays below the ks_floor "
    "(0.50) and PSI does not move under max-over-columns "
    "aggregation; structurally the same known limitation as "
    "variance shift; documented in docs/phase6_notes.md, not fixed "
    "in Phase 6"
)

MEAN_FN_REASON = (
    "expected false negative — known aggregation limitation: "
    "mean-shift temp KS (0.27-0.48 at low/medium) sits below the "
    "ks_floor (0.50) and at or below the other columns' sampling-"
    "noise floor (~0.36-0.50), so max-over-columns aggregation masks "
    "the shifted column (co owns max_ks in 9/10 medium windows) and "
    "PSI does not move; high severity is marginal; diagnosed in "
    "experiments/diagnose_mean_shift.py, same known limitation as "
    "variance/distribution shift, not fixed in Phase 7"
)

def expected_false_negative_condition(condition):
    """
    Conditions/subtypes whose injected windows are expected false
    negatives under max-over-columns aggregation (variance shift,
    distribution shift, mean shift). Phase 7 consumes the flag
    generically across drift types.
    """
    return (
        condition.startswith("variance_")
        or condition.startswith("distribution_")
        or condition.startswith("mean_")
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
    if condition.startswith("mean_"):
        return MEAN_FN_REASON
    return None
