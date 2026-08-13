from detectors.schema_detector import (
    compare_schemas,
    detect_renames,
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