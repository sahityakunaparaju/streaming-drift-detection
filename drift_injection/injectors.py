import copy

import numpy as np
import pandas as pd


# ============================================================
# Helper
# ============================================================

def _validate_column(df, column):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")


def _severity_shift(df, column, severity):
    """
    Calculate the target mean shift using the reference
    column's standard deviation.

    low    = 0.25 * sigma
    medium = 0.75 * sigma
    high   = 1.50 * sigma
    """

    if severity not in {"low", "medium", "high"}:
        raise ValueError(
            "severity must be one of: low, medium, high"
        )

    sigma = df[column].std()

    multipliers = {
        "low": 0.25,
        "medium": 0.75,
        "high": 1.50,
    }

    return multipliers[severity] * sigma


# ============================================================
# SCHEMA INJECTORS
# ============================================================

def inject_add_column(
    df,
    column="battery_level",
    value=100.0,
):
    """
    Add a new column to the batch.
    """

    result = df.copy()

    if column in result.columns:
        raise ValueError(
            f"Column '{column}' already exists"
        )

    result[column] = value

    return result


def inject_remove_column(
    df,
    column="motion",
):
    """
    Remove an existing column.
    """

    _validate_column(df, column)

    result = df.copy()
    result = result.drop(columns=[column])

    return result


def inject_rename_column(
    df,
    rename_map,
):
    """
    Explicitly rename columns.

    Example:
        {"temp": "temperature"}
    """

    result = df.copy()

    for old_name, new_name in rename_map.items():

        if old_name not in result.columns:
            raise ValueError(
                f"Column '{old_name}' not found"
            )

        if new_name in result.columns:
            raise ValueError(
                f"Column '{new_name}' already exists"
            )

    return result.rename(columns=rename_map)


def inject_type_change(
    df,
    column="temp",
    target_type="string",
):
    """
    Change the datatype of a column.

    Supported target types:
        string
        float64
        int64
        bool
    """

    _validate_column(df, column)

    result = df.copy()

    if target_type == "string":
        result[column] = result[column].astype(str)

    elif target_type == "float64":
        result[column] = result[column].astype(float)

    elif target_type == "int64":
        result[column] = result[column].astype(int)

    elif target_type == "bool":
        result[column] = result[column].astype(bool)

    else:
        raise ValueError(
            f"Unsupported target_type: {target_type}"
        )

    return result


# ============================================================
# STATISTICAL INJECTORS
# ============================================================

def inject_mean_shift(
    df,
    column="temp",
    severity="medium",
):
    """
    Shift the mean by:

        low    = 0.25 sigma
        medium = 0.75 sigma
        high   = 1.50 sigma

    The same offset is added to every value so that the
    intended mean shift is deterministic.
    """

    _validate_column(df, column)

    result = df.copy()

    shift = _severity_shift(
        result,
        column,
        severity,
    )

    result[column] = result[column] + shift

    return result


def inject_variance_shift(
    df,
    column="temp",
    severity="medium",
):
    """
    Increase variance while preserving approximately
    the original mean.

    Severity controls the scale multiplier.
    """

    _validate_column(df, column)

    result = df.copy()

    original = result[column].astype(float)

    mean = original.mean()
    std = original.std()

    multipliers = {
        "low": 1.25,
        "medium": 1.75,
        "high": 2.50,
    }

    multiplier = multipliers.get(severity)

    if multiplier is None:
        raise ValueError(
            "severity must be one of: low, medium, high"
        )

    result[column] = (
        (original - mean) * multiplier
    ) + mean

    return result


def inject_distribution_shift(
    df,
    column="temp",
    severity="medium",
    random_state=42,
):
    """
    Apply a distributional transformation that changes
    the shape of the distribution.

    The transformation uses a nonlinear power operation
    around the column mean.
    """

    _validate_column(df, column)

    result = df.copy()

    values = result[column].astype(float).to_numpy()

    mean = values.mean()
    centered = values - mean

    exponents = {
        "low": 1.10,
        "medium": 1.25,
        "high": 1.50,
    }

    exponent = exponents.get(severity)

    if exponent is None:
        raise ValueError(
            "severity must be one of: low, medium, high"
        )

    transformed = (
        np.sign(centered)
        * (np.abs(centered) ** exponent)
    )

    scale = (
        np.std(centered)
        / (np.std(transformed) + 1e-12)
    )

    result[column] = mean + transformed * scale

    return result


def inject_category_shift(
    df,
    column="device",
    category=None,
    severity="medium",
    random_state=42,
):
    """
    Increase the frequency of one categorical value.

    The selected category replaces a proportion of
    randomly selected records.

    This changes the categorical distribution while
    preserving the allowed category domain.
    """

    _validate_column(df, column)

    result = df.copy()

    categories = result[column].dropna().unique().tolist()

    if len(categories) < 2:
        raise ValueError(
            "Category shift requires at least two categories"
        )

    if category is None:
        category = categories[0]

    if category not in categories:
        raise ValueError(
            f"Category '{category}' not found"
        )

    proportions = {
        "low": 0.10,
        "medium": 0.25,
        "high": 0.50,
    }

    proportion = proportions.get(severity)

    if proportion is None:
        raise ValueError(
            "severity must be one of: low, medium, high"
        )

    rng = np.random.default_rng(random_state)

    candidate_indices = result.index[
        result[column] != category
    ]

    n_change = int(
        len(candidate_indices) * proportion
    )

    if n_change > 0:
        selected = rng.choice(
            candidate_indices,
            size=n_change,
            replace=False,
        )

        result.loc[selected, column] = category

    return result


# ============================================================
# SEMANTIC INJECTORS
# ============================================================

def inject_unit_change(
    df,
    column="temp",
    reference_unit="celsius",
    new_unit="fahrenheit",
    metadata_only=True,
):
    """
    Inject a unit change.

    metadata_only=True:
        Values remain completely unchanged.
        Only metadata describing the unit changes.

    metadata_only=False:
        Numerical values are actually converted.

    Currently supports Celsius <-> Fahrenheit.
    """

    _validate_column(df, column)

    result = df.copy()

    metadata = {
        "column": column,
        "old_unit": reference_unit,
        "new_unit": new_unit,
        "metadata_only": metadata_only,
    }

    if metadata_only:
        return result, metadata

    if (
        reference_unit == "celsius"
        and new_unit == "fahrenheit"
    ):
        result[column] = (
            result[column] * 9 / 5
        ) + 32

    elif (
        reference_unit == "fahrenheit"
        and new_unit == "celsius"
    ):
        result[column] = (
            result[column] - 32
        ) * 5 / 9

    else:
        raise ValueError(
            "Only Celsius/Fahrenheit conversion is "
            "currently supported"
        )

    return result, metadata


def inject_definition_change(
    df,
    column,
    reference_definition,
    new_definition,
):
    """
    Change a structured semantic definition while
    leaving the dataframe values unchanged.
    """

    _validate_column(df, column)

    result = df.copy()

    metadata = {
        "column": column,
        "change": "definition_change",
        "old_definition": reference_definition,
        "new_definition": new_definition,
    }

    return result, metadata


def inject_domain_change(
    df,
    column="device",
    new_domain=None,
):
    """
    Change the declared semantic domain.

    Values themselves remain unchanged.
    """

    _validate_column(df, column)

    if new_domain is None:
        raise ValueError(
            "new_domain must be provided"
        )

    result = df.copy()

    metadata = {
        "column": column,
        "change": "domain_change",
        "new_domain": new_domain,
    }

    return result, metadata


def inject_business_meaning_change(
    df,
    column,
    reference_meaning,
    new_meaning,
):
    """
    Change the declared business meaning without
    changing the raw values.
    """

    _validate_column(df, column)

    result = df.copy()

    metadata = {
        "column": column,
        "change": "business_meaning_change",
        "old_meaning": reference_meaning,
        "new_meaning": new_meaning,
    }

    return result, metadata


# ============================================================
# NORMAL VARIATION
# ============================================================

def inject_normal_noise(
    df,
    columns=None,
    noise_fraction=0.01,
    random_state=42,
):
    """
    Add small random noise to numerical columns.

    The noise is intentionally kept small and within
    normal reference bounds.

    This represents normal variation rather than drift.
    """

    result = df.copy()

    if columns is None:
        columns = [
            column
            for column in result.columns
            if pd.api.types.is_numeric_dtype(
                result[column]
            )
            and not pd.api.types.is_bool_dtype(
                result[column]
            )
        ]

    rng = np.random.default_rng(random_state)

    for column in columns:

        _validate_column(result, column)

        series = result[column].astype(float)

        std = series.std()

        noise = rng.normal(
            loc=0.0,
            scale=std * noise_fraction,
            size=len(series),
        )

        result[column] = series + noise

    return result