from datetime import datetime, timezone


def compare_schemas(reference_schema, current_schema):
    """
    Compare reference and current schemas.

    Returns one drift event for every detected schema difference.
    """

    events = []

    reference_columns = reference_schema.get("columns", {})
    current_columns = current_schema.get("columns", {})

    timestamp = datetime.now(timezone.utc).isoformat()

    # --------------------------------------------------
    # Added / removed columns
    # --------------------------------------------------

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
            "severity": "high",
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
            "severity": "high",
        })

    # --------------------------------------------------
    # Existing columns
    # --------------------------------------------------

    common_columns = reference_names & current_names

    for column in sorted(common_columns):

        reference = reference_columns[column]
        current = current_columns[column]

        # -----------------------------
        # Type change
        # -----------------------------

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
                "severity": "high",
            })

        # -----------------------------
        # Nullability change
        # -----------------------------

        reference_nullable = reference.get("nullable")
        current_nullable = current.get("nullable")

        if reference_nullable != current_nullable:

            events.append({
                "timestamp": timestamp,
                "drift_type": "schema",
                "column": column,
                "change": "nullability_change",
                "old_value": reference_nullable,
                "new_value": current_nullable,
                "severity": "medium",
            })

    # --------------------------------------------------
    # Renamed columns
    # --------------------------------------------------
    #
    # A pure schema dictionary cannot reliably determine
    # whether a removed column was renamed or simply removed.
    #
    # Therefore rename detection requires an explicit
    # rename mapping.
    # --------------------------------------------------

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
            "severity": "high",
        })

    return events