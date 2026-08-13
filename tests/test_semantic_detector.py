import pandas as pd

from detectors.semantic_detector import (
    compare_semantic_contracts,
    validate_values_against_contract,
    RANGE_TOLERANCE_FRACTION,
)


def make_temp_contract(unit="celsius"):
    return {
        "fields": {
            "temp": {
                "business_meaning": "device_temperature",
                "data_role": "sensor_measurement",
                "unit": unit,
                "valid_range": {
                    "min": 0.0,
                    "max": 30.6,
                },
            }
        }
    }


def make_device_contract(meaning="iot_device_identifier"):
    return {
        "fields": {
            "device": {
                "business_meaning": meaning,
                "data_role": "identifier",
                "unit": "device_id",
            }
        }
    }


def test_unit_change_detected():

    reference = make_temp_contract("celsius")
    current = make_temp_contract("fahrenheit")

    events = compare_semantic_contracts(
        reference,
        current,
    )

    assert len(events) == 1
    assert events[0]["change"] == "unit_change"
    assert events[0]["column"] == "temp"
    assert events[0]["old_value"] == "celsius"
    assert events[0]["new_value"] == "fahrenheit"


def test_business_meaning_change_detected():

    reference = make_device_contract(
        "iot_device_identifier"
    )

    current = make_device_contract(
        "building_zone_identifier"
    )

    events = compare_semantic_contracts(
        reference,
        current,
    )

    assert len(events) == 1
    assert events[0]["change"] == "business_meaning_change"
    assert events[0]["column"] == "device"


def test_no_contract_change_is_silent():

    reference = make_temp_contract("celsius")
    current = make_temp_contract("celsius")

    events = compare_semantic_contracts(
        reference,
        current,
    )

    assert events == []


def test_range_violation_detected():

    contract = make_temp_contract("celsius")

    df = pd.DataFrame({
        "temp": [20.0, 22.0, 25.0, 35.0]
    })

    events = validate_values_against_contract(
        df,
        contract,
    )

    assert len(events) == 1
    assert events[0]["change"] == "range_violation"
    assert events[0]["column"] == "temp"


def test_allowed_domain_violation_detected():

    contract = {
        "fields": {
            "device": {
                "business_meaning": "iot_device_identifier",
                "unit": "device_id",
                "allowed_domain": [
                    "device_a",
                    "device_b",
                    "device_c",
                ],
            }
        }
    }

    df = pd.DataFrame({
        "device": [
            "device_a",
            "device_b",
            "unknown_device",
        ]
    })

    events = validate_values_against_contract(
        df,
        contract,
    )

    assert len(events) == 1
    assert events[0]["change"] == "allowed_domain_violation"


def test_range_value_within_tolerance_not_detected():

    contract = make_temp_contract("celsius")

    width = 30.6 - 0.0
    overrun = 0.5 * RANGE_TOLERANCE_FRACTION * width

    df = pd.DataFrame({
        "temp": [20.0, 22.0, 30.6 + overrun]
    })

    events = validate_values_against_contract(
        df,
        contract,
    )

    assert events == []


def test_range_value_beyond_tolerance_detected():

    contract = make_temp_contract("celsius")

    width = 30.6 - 0.0
    overrun = 2.0 * RANGE_TOLERANCE_FRACTION * width

    df = pd.DataFrame({
        "temp": [20.0, 22.0, 30.6 + overrun]
    })

    events = validate_values_against_contract(
        df,
        contract,
    )

    assert len(events) == 1
    assert events[0]["change"] == "range_violation"
    assert events[0]["column"] == "temp"


def test_domain_violation_fires_on_real_contract():

    import json

    with open(
        "data/primary/reference_profile.json",
        "r",
    ) as file:
        profile = json.load(file)

    contract = profile["semantic_contract"]

    allowed = contract["fields"]["device"]["allowed_domain"]

    assert sorted(allowed) == [
        "00:0f:00:70:91:0a",
        "1c:bf:ce:15:ec:4d",
        "b8:27:eb:bf:9d:51",
    ]

    clean_df = pd.DataFrame({
        "device": allowed
    })

    assert (
        validate_values_against_contract(
            clean_df,
            contract,
        )
        == []
    )

    invalid_df = pd.DataFrame({
        "device": [allowed[0], "unknown_device_ff:ff"]
    })

    events = validate_values_against_contract(
        invalid_df,
        contract,
    )

    assert len(events) == 1
    assert events[0]["change"] == "allowed_domain_violation"
    assert events[0]["column"] == "device"