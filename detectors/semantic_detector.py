from datetime import datetime, timezone

import pandas as pd

from detectors.thresholds import THRESHOLDS

RANGE_TOLERANCE_FRACTION = float(
    THRESHOLDS["semantic"]["range_tolerance_fraction"]
)

def _event(column, change, old_value, new_value, severity="medium"):
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "drift_type": "semantic",
        "column": column,
        "change": change,
        "old_value": old_value,
        "new_value": new_value,
        "severity": severity,
    }

def compare_semantic_contracts(reference, current):
    """
    Compare structured semantic contracts.

    Level 1:
        - unit
        - valid range
        - allowed domain

    Level 2:
        - business meaning
        - computation/definition semantics (Phase 8: closes the
          Phase 6 definition-change coverage gap)
        - semantic metadata changes
    """

    events = []

    reference_fields = reference.get("fields", {})
    current_fields = current.get("fields", {})

    common_fields = (
        set(reference_fields.keys())
        & set(current_fields.keys())
    )

    for column in sorted(common_fields):

        ref = reference_fields[column]
        cur = current_fields[column]

        ref_unit = ref.get("unit")
        cur_unit = cur.get("unit")

        if ref_unit != cur_unit:
            events.append(
                _event(
                    column,
                    "unit_change",
                    ref_unit,
                    cur_unit,
                    "high",
                )
            )

        ref_range = ref.get("valid_range")
        cur_range = cur.get("valid_range")

        if ref_range != cur_range:
            events.append(
                _event(
                    column,
                    "range_change",
                    ref_range,
                    cur_range,
                    "medium",
                )
            )

        ref_domain = ref.get("allowed_domain")
        cur_domain = cur.get("allowed_domain")

        if ref_domain != cur_domain:
            events.append(
                _event(
                    column,
                    "allowed_domain_change",
                    ref_domain,
                    cur_domain,
                    "medium",
                )
            )

        ref_meaning = ref.get("business_meaning")
        cur_meaning = cur.get("business_meaning")

        if ref_meaning != cur_meaning:
            events.append(
                _event(
                    column,
                    "business_meaning_change",
                    ref_meaning,
                    cur_meaning,
                    "high",
                )
            )

        ref_definition = ref.get("definition")
        cur_definition = cur.get("definition")

        if ref_definition != cur_definition:
            events.append(
                _event(
                    column,
                    "definition_change",
                    ref_definition,
                    cur_definition,
                    "high",
                )
            )

    return events

def validate_values_against_contract(df, contract):
    """
    Level 1 value validation.

    Checks:
        - numerical range
        - allowed domain

    This detects semantic violations even when
    the schema remains unchanged.
    """

    events = []

    fields = contract.get("fields", {})

    for column, specification in fields.items():

        if column not in df.columns:
            continue

        series = df[column]

        valid_range = specification.get("valid_range")

        if not pd.api.types.is_numeric_dtype(series):
            valid_range = None

        if valid_range and len(valid_range) == 2:

            minimum = valid_range["min"]
            maximum = valid_range["max"]

            tolerance = (
                RANGE_TOLERANCE_FRACTION
                * (maximum - minimum)
            )

            lower = minimum - tolerance
            upper = maximum + tolerance

            violations = series[
                (series < lower)
                | (series > upper)
            ]

            if len(violations) > 0:

                events.append(
                    _event(
                        column,
                        "range_violation",
                        {
                            "min": minimum,
                            "max": maximum,
                        },
                        {
                            "violations": int(len(violations)),
                            "observed_min": float(series.min()),
                            "observed_max": float(series.max()),
                        },
                        "high",
                    )
                )

        allowed_domain = specification.get("allowed_domain")

        if allowed_domain:

            invalid = ~series.isin(allowed_domain)

            if invalid.any():

                invalid_values = (
                    series[invalid]
                    .drop_duplicates()
                    .tolist()
                )

                events.append(
                    _event(
                        column,
                        "allowed_domain_violation",
                        allowed_domain,
                        invalid_values,
                        "high",
                    )
                )

    return events
