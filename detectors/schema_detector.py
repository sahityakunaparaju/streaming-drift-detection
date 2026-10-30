from datetime import datetime, timezone

from detectors.thresholds import THRESHOLDS

# Rate-aware nullability tolerance: a nullability_change is flagged
# only when the absolute difference in missing-rate between reference
# and current exceeds this threshold (in fraction, not percentage).
# 0.05 = 5 percentage points.
#
# Design rationale:
# - Primary IoT dataset: zero nulls everywhere. Tolerance is irrelevant
#   because the reference schema has no missing_rate fields (backward
#   compat path uses boolean comparison).
# - Secondary (Beijing) dataset: columns have 0.08-4.92% natural
#   nulls.  Individual 1K windows vary by a few pp from the reference.
#   5pp tolerance absorbs this natural variation while still catching
#   genuinely injected nullability changes (e.g., 0% -> 20%).
NULLABILITY_RATE_TOLERANCE = 0.05

SEVERITY_LABELS = THRESHOLDS.get("schema", {}).get(
    "severity_labels",
    {
        "added": "high",
        "removed": "high",
        "renamed": "high",
        "type_change": "medium",
        "nullability_change": "low",
    },
)

def compare_schemas(reference_schema, current_schema):
    """
    Compare reference and current schemas.

    Returns one drift event for every detected schema difference.
    """

    events = []

    reference_columns = reference_schema.get("columns", {})
    current_columns = current_schema.get("columns", {})

    timestamp = datetime.now(timezone.utc).isoformat()

    reference_names = set(reference_columns.keys())
    current_names = set(current_columns.keys())

    added_columns = current_names - reference_names
    removed_columns = reference_names - current_names

    for column in sorted(added_columns):
        current = current_columns[column]

        events.append({
            "timestamp": timestamp,
            "drift_type": "schema",
            "column": column,
            "change": "added",
            "old_value": None,
            "new_value": current.get("dtype"),
            "severity": SEVERITY_LABELS["added"],
        })

    for column in sorted(removed_columns):
        reference = reference_columns[column]

        events.append({
            "timestamp": timestamp,
            "drift_type": "schema",
            "column": column,
            "change": "removed",
            "old_value": reference.get("dtype"),
            "new_value": None,
            "severity": SEVERITY_LABELS["removed"],
        })

    common_columns = reference_names & current_names

    for column in sorted(common_columns):

        reference = reference_columns[column]
        current = current_columns[column]

        reference_dtype = reference.get("dtype")
        current_dtype = current.get("dtype")

        if reference_dtype != current_dtype:

            events.append({
                "timestamp": timestamp,
                "drift_type": "schema",
                "column": column,
                "change": "type_change",
                "old_value": reference_dtype,
                "new_value": current_dtype,
                "severity": SEVERITY_LABELS["type_change"],
            })

        # ---- nullability check (rate-aware when missing_rate present) ----
        ref_rate = reference.get("missing_rate")
        cur_rate = current.get("missing_rate")

        if ref_rate is not None and cur_rate is not None:
            # Rate-aware path: compare missing rates with tolerance.
            # Zero-baseline columns use zero tolerance (any null is drift).
            if ref_rate == 0.0:
                flagged = cur_rate > 0.0
            else:
                flagged = abs(ref_rate - cur_rate) > NULLABILITY_RATE_TOLERANCE
        else:
            # Backward-compat path: boolean nullable comparison.
            # Used when the reference schema is hand-crafted without
            # missing_rate (e.g., configs/schema_reference.yaml).
            reference_nullable = reference.get("nullable")
            current_nullable = current.get("nullable")
            flagged = reference_nullable != current_nullable

        if flagged:
            old_val = ref_rate if ref_rate is not None else reference.get("nullable")
            new_val = cur_rate if cur_rate is not None else current.get("nullable")

            events.append({
                "timestamp": timestamp,
                "drift_type": "schema",
                "column": column,
                "change": "nullability_change",
                "old_value": old_val,
                "new_value": new_val,
                "severity": SEVERITY_LABELS["nullability_change"],
            })

    return events

def detect_renames(reference_schema, current_schema, rename_map):
    """
    Detect explicitly declared column renames.

    Example:
        {"temperature": "temp"}

    means reference column 'temperature' became 'temp'.
    """

    events = []

    timestamp = datetime.now(timezone.utc).isoformat()

    reference_columns = reference_schema.get("columns", {})
    current_columns = current_schema.get("columns", {})

    for old_name, new_name in rename_map.items():

        if old_name not in reference_columns:
            continue

        if new_name not in current_columns:
            continue

        events.append({
            "timestamp": timestamp,
            "drift_type": "schema",
            "column": new_name,
            "change": "renamed",
            "old_value": old_name,
            "new_value": new_name,
            "severity": SEVERITY_LABELS["renamed"],
        })

    return events
