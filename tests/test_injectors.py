import numpy as np
import pandas as pd

from drift_injection.injectors import (
    inject_add_column,
    inject_remove_column,
    inject_rename_column,
    inject_type_change,
    inject_mean_shift,
    inject_variance_shift,
    inject_distribution_shift,
    inject_category_shift,
    inject_unit_change,
    inject_definition_change,
    inject_domain_change,
    inject_business_meaning_change,
    inject_normal_noise,
)

def make_sample_data():
    return pd.DataFrame({
        "temp": np.array([
            18.0, 20.0, 21.0, 22.0, 23.0,
            24.0, 25.0, 26.0, 27.0, 28.0
        ]),
        "humidity": np.array([
            40.0, 45.0, 50.0, 55.0, 60.0,
            65.0, 70.0, 75.0, 80.0, 85.0
        ]),
        "device": [
            "device_a",
            "device_b",
            "device_c",
            "device_a",
            "device_b",
            "device_c",
            "device_a",
            "device_b",
            "device_c",
            "device_a",
        ],
        "motion": [
            True, False, True, False, True,
            False, True, False, True, False
        ],
    })

def test_add_column():

    df = make_sample_data()

    result = inject_add_column(
        df,
        "battery_level",
        100.0,
    )

    assert "battery_level" in result.columns
    assert len(result) == len(df)

def test_remove_column():

    df = make_sample_data()

    result = inject_remove_column(
        df,
        "motion",
    )

    assert "motion" not in result.columns

def test_rename_column():

    df = make_sample_data()

    result = inject_rename_column(
        df,
        {"temp": "temperature"},
    )

    assert "temp" not in result.columns
    assert "temperature" in result.columns

def test_type_change():

    df = make_sample_data()

    result = inject_type_change(
        df,
        "temp",
        "string",
    )

    assert result["temp"].dtype == object

def test_mean_shift_medium():

    df = make_sample_data()

    original_mean = df["temp"].mean()
    original_std = df["temp"].std()

    result = inject_mean_shift(
        df,
        "temp",
        "medium",
    )

    new_mean = result["temp"].mean()

    expected_shift = 0.75 * original_std

    assert np.isclose(
        new_mean - original_mean,
        expected_shift,
        atol=1e-10,
    )

def test_variance_shift():

    df = make_sample_data()

    original_variance = df["temp"].var()

    result = inject_variance_shift(
        df,
        "temp",
        "medium",
    )

    new_variance = result["temp"].var()

    assert new_variance > original_variance

def test_distribution_shift():

    df = make_sample_data()

    result = inject_distribution_shift(
        df,
        "temp",
        "medium",
    )

    assert not np.allclose(
        df["temp"].values,
        result["temp"].values,
    )

def test_category_shift():

    df = make_sample_data()

    original_frequency = (
        df["device"] == "device_a"
    ).mean()

    result = inject_category_shift(
        df,
        "device",
        "device_a",
        "medium",
    )

    new_frequency = (
        result["device"] == "device_a"
    ).mean()

    assert new_frequency > original_frequency

def test_unit_change_metadata_only():

    df = make_sample_data()

    result, metadata = inject_unit_change(
        df,
        "temp",
        "celsius",
        "fahrenheit",
        metadata_only=True,
    )

    assert np.array_equal(
        df["temp"].values,
        result["temp"].values,
    )

    assert metadata["old_unit"] == "celsius"
    assert metadata["new_unit"] == "fahrenheit"
    assert metadata["metadata_only"] is True

def test_unit_change_actual_conversion():

    df = make_sample_data()

    result, metadata = inject_unit_change(
        df,
        "temp",
        "celsius",
        "fahrenheit",
        metadata_only=False,
    )

    expected = (
        df["temp"] * 9 / 5
    ) + 32

    assert np.allclose(
        result["temp"],
        expected,
    )

    assert not np.array_equal(
        df["temp"].values,
        result["temp"].values,
    )

def test_definition_change():

    df = make_sample_data()

    result, metadata = inject_definition_change(
        df,
        "temp",
        "device_temperature",
        "ambient_temperature",
    )

    assert np.array_equal(
        df["temp"].values,
        result["temp"].values,
    )

    assert (
        metadata["old_definition"]
        == "device_temperature"
    )

    assert (
        metadata["new_definition"]
        == "ambient_temperature"
    )

def test_domain_change():

    df = make_sample_data()

    result, metadata = inject_domain_change(
        df,
        "device",
        [
            "device_a",
            "device_b",
            "device_c",
            "device_d",
        ],
    )

    assert np.array_equal(
        df["device"].values,
        result["device"].values,
    )

    assert metadata["change"] == "domain_change"

def test_business_meaning_change():

    df = make_sample_data()

    result, metadata = inject_business_meaning_change(
        df,
        "device",
        "iot_device_identifier",
        "building_zone_identifier",
    )

    assert np.array_equal(
        df["device"].values,
        result["device"].values,
    )

    assert (
        metadata["change"]
        == "business_meaning_change"
    )

def test_normal_noise_changes_values_but_stays_close():

    df = make_sample_data()

    result = inject_normal_noise(
        df,
        columns=["temp"],
        noise_fraction=0.01,
        random_state=42,
    )

    assert not np.array_equal(
        df["temp"].values,
        result["temp"].values,
    )

    original_mean = df["temp"].mean()
    new_mean = result["temp"].mean()

    assert abs(new_mean - original_mean) < 0.5
