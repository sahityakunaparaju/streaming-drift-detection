import json
from pathlib import Path

import pandas as pd
import yaml


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "primary" / "iot_telemetry_data.csv"
SCHEMA_PATH = PROJECT_ROOT / "configs" / "schema_reference.yaml"
SEMANTIC_PATH = PROJECT_ROOT / "configs" / "semantic_contract.yaml"

OUTPUT_PATH = PROJECT_ROOT / "data" / "primary" / "reference_profile.json"


# --------------------------------------------------
# Load configuration
# --------------------------------------------------

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


# --------------------------------------------------
# Build statistical profile
# --------------------------------------------------

def build_statistical_profile(df):
    profile = {}

    for column in df.columns:

        series = df[column]

        column_profile = {
            "dtype": str(series.dtype),
            "count": int(series.count()),
            "missing_count": int(series.isna().sum()),
            "missing_percentage": float(series.isna().mean() * 100),
            "unique_count": int(series.nunique()),
        }

        # Numerical columns
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):

            column_profile["statistics"] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "q25": float(series.quantile(0.25)),
                "median": float(series.median()),
                "q75": float(series.quantile(0.75)),
                "max": float(series.max()),
            }

        # Categorical / string columns
        elif pd.api.types.is_object_dtype(series):

            frequencies = series.value_counts(normalize=True)

            column_profile["value_frequencies"] = {
                str(key): float(value)
                for key, value in frequencies.items()
            }

        # Boolean columns
        elif pd.api.types.is_bool_dtype(series):

            frequencies = series.value_counts(normalize=True)

            column_profile["value_frequencies"] = {
                str(key): float(value)
                for key, value in frequencies.items()
            }

        profile[column] = column_profile

    return profile


# --------------------------------------------------
# Timestamp profile
# --------------------------------------------------

def build_timestamp_profile(df):

    ts = df["ts"]

    return {
        "min": float(ts.min()),
        "max": float(ts.max()),
        "duration_seconds": float(ts.max() - ts.min()),
        "duration_days": float((ts.max() - ts.min()) / 86400),
    }


# --------------------------------------------------
# Main profiler
# --------------------------------------------------

def main():

    print("=" * 60)
    print("REFERENCE PROFILER")
    print("=" * 60)

    print(f"\nLoading dataset:")
    print(DATA_PATH)

    df = pd.read_csv(DATA_PATH)

    print(f"\nDataset shape: {df.shape}")

    schema = load_yaml(SCHEMA_PATH)
    semantic_contract = load_yaml(SEMANTIC_PATH)

    statistical_profile = build_statistical_profile(df)
    timestamp_profile = build_timestamp_profile(df)

    reference_profile = {
        "profile_version": "1.0",

        "dataset": {
            "name": "iot_telemetry_data",
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
        },

        "schema": schema,

        "statistics": statistical_profile,

        "timestamp": timestamp_profile,

        "semantic_contract": semantic_contract,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(reference_profile, file, indent=2)

    print("\nReference profile created:")
    print(OUTPUT_PATH)

    print("\nTimestamp range:")
    print(f"  Start: {timestamp_profile['min']}")
    print(f"  End:   {timestamp_profile['max']}")
    print(f"  Days:  {timestamp_profile['duration_days']:.2f}")

    print("\nProfiler completed successfully.")


if __name__ == "__main__":
    main()