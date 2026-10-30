from detectors.schema_detector import (
    compare_schemas,
    detect_renames,
    NULLABILITY_RATE_TOLERANCE,
)


def make_reference_schema():
    return {
        "columns": {
            "temp": {
                "dtype": "float64",
                "nullable": False,
            },
            "motion": {
                "dtype": "bool",
                "nullable": False,
            },
            "device": {
                "dtype": "string",
                "nullable": False,
            },
        }
    }


def test_added_column():

    reference = make_reference_schema()

    current = make_reference_schema()
    current["columns"]["battery_level"] = {
        "dtype": "float64",
        "nullable": False,
    }

    events = compare_schemas(reference, current)

    assert len(events) == 1
    assert events[0]["change"] == "added"
    assert events[0]["column"] == "battery_level"


def test_removed_column():

    reference = make_reference_schema()

    current = make_reference_schema()
    del current["columns"]["motion"]

    events = compare_schemas(reference, current)

    assert len(events) == 1
    assert events[0]["change"] == "removed"
    assert events[0]["column"] == "motion"


def test_type_change():

    reference = make_reference_schema()

    current = make_reference_schema()
    current["columns"]["temp"]["dtype"] = "string"

    events = compare_schemas(reference, current)

    assert len(events) == 1
    assert events[0]["change"] == "type_change"
    assert events[0]["column"] == "temp"
    assert events[0]["old_value"] == "float64"
    assert events[0]["new_value"] == "string"


def test_nullability_change():

    reference = make_reference_schema()

    current = make_reference_schema()
    current["columns"]["temp"]["nullable"] = True

    events = compare_schemas(reference, current)

    assert len(events) == 1
    assert events[0]["change"] == "nullability_change"
    assert events[0]["column"] == "temp"


def test_identical_schemas_are_silent():

    reference = make_reference_schema()
    current = make_reference_schema()

    events = compare_schemas(reference, current)

    assert events == []


def test_explicit_rename():

    reference = {
        "columns": {
            "temp": {
                "dtype": "float64",
                "nullable": False,
            }
        }
    }

    current = {
        "columns": {
            "temperature": {
                "dtype": "float64",
                "nullable": False,
            }
        }
    }

    events = detect_renames(
        reference,
        current,
        {"temp": "temperature"},
    )

    assert len(events) == 1
    assert events[0]["change"] == "renamed"
    assert events[0]["old_value"] == "temp"
    assert events[0]["new_value"] == "temperature"


# ------------------------------------------------------------------
# Rate-aware nullability tests (Phase 13 fix)
# ------------------------------------------------------------------


def make_rate_schema(**rates):
    """Build a schema dict with missing_rate per column."""
    columns = {}
    for col, rate in rates.items():
        columns[col] = {
            "dtype": "float64",
            "nullable": rate > 0,
            "missing_rate": rate,
        }
    return {"columns": columns}


def test_rate_aware_within_tolerance_is_silent():
    """A small missing-rate difference within tolerance fires no event."""
    reference = make_rate_schema(pm25=0.02, co=0.05)
    current = make_rate_schema(pm25=0.04, co=0.07)
    # Both within NULLABILITY_RATE_TOLERANCE (0.05)
    events = compare_schemas(reference, current)
    nullability_events = [e for e in events if e["change"] == "nullability_change"]
    assert nullability_events == []


def test_rate_aware_beyond_tolerance_fires():
    """A large missing-rate difference beyond tolerance fires an event."""
    reference = make_rate_schema(pm25=0.02)
    current = make_rate_schema(pm25=0.20)
    # 18pp difference > 5pp tolerance
    events = compare_schemas(reference, current)
    nullability_events = [e for e in events if e["change"] == "nullability_change"]
    assert len(nullability_events) == 1
    assert nullability_events[0]["column"] == "pm25"


def test_rate_aware_zero_baseline_any_null_fires():
    """When reference rate is 0.0, any non-zero current rate fires."""
    reference = make_rate_schema(pm25=0.0)
    current = make_rate_schema(pm25=0.001)
    events = compare_schemas(reference, current)
    nullability_events = [e for e in events if e["change"] == "nullability_change"]
    assert len(nullability_events) == 1


def test_rate_aware_zero_baseline_zero_current_silent():
    """When both rates are 0.0, no event fires."""
    reference = make_rate_schema(pm25=0.0)
    current = make_rate_schema(pm25=0.0)
    events = compare_schemas(reference, current)
    nullability_events = [e for e in events if e["change"] == "nullability_change"]
    assert nullability_events == []


def test_backward_compat_boolean_only():
    """Schemas without missing_rate fall back to boolean comparison."""
    reference = {"columns": {"temp": {"dtype": "float64", "nullable": False}}}
    current = {"columns": {"temp": {"dtype": "float64", "nullable": True}}}
    events = compare_schemas(reference, current)
    nullability_events = [e for e in events if e["change"] == "nullability_change"]
    assert len(nullability_events) == 1


def test_tolerance_constant_is_five_percent():
    """Verify the tolerance constant is 0.05 (5 percentage points)."""
    assert NULLABILITY_RATE_TOLERANCE == 0.05