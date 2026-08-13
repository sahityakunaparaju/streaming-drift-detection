import pandas as pd

from drift_injection.injectors import (
    inject_mean_shift,
    inject_normal_noise,
    inject_unit_change,
)

from drift_injection.ground_truth import (
    log_injection,
    log_no_drift,
)


INPUT = "data/primary/iot_telemetry_data.csv"
OUTPUT = "results/raw/ground_truth.csv"


df = pd.read_csv(INPUT)

# Use a small static batch for Phase 3.2 validation.
batch = df.iloc[:1000].copy()


# ------------------------------------------------------------
# 1. Statistical drift
# ------------------------------------------------------------

modified = inject_mean_shift(
    batch,
    column="temp",
    severity="medium",
)

log_injection(
    drift_type="statistical",
    subtype="mean_shift",
    feature="temp",
    severity="medium",
)


# ------------------------------------------------------------
# 2. Semantic Experiment A
#    Metadata changes, values DO NOT change
# ------------------------------------------------------------

modified, metadata = inject_unit_change(
    batch,
    column="temp",
    reference_unit="celsius",
    new_unit="fahrenheit",
    metadata_only=True,
)

log_injection(
    drift_type="semantic",
    subtype="unit_change",
    feature="temp",
    severity="high",
)


# ------------------------------------------------------------
# 3. Normal variation
# ------------------------------------------------------------

normal_batch = inject_normal_noise(
    batch,
    columns=["temp"],
    noise_fraction=0.01,
)

log_no_drift(
    subtype="normal_noise",
    feature="temp",
)


print("=" * 60)
print("GROUND TRUTH SAMPLE CREATED")
print("=" * 60)
print(f"Output: {OUTPUT}")

ground_truth = pd.read_csv(OUTPUT)

print()
print(ground_truth.to_string(index=False))

print()
print("LABEL COUNTS")
print(ground_truth["ground_truth"].value_counts())