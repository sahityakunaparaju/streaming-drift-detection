# Primary Dataset — IoT Telemetry

## Dataset Overview

This project uses an IoT telemetry dataset containing sensor readings collected from multiple IoT devices.

The dataset contains:

- 405,184 records
- 9 columns
- 3 unique devices
- No missing values
- Unix epoch timestamps
- Continuous numerical sensor measurements
- Boolean device-state measurements

## Columns

| Column | Data Type | Description |
|---|---|---|
| `ts` | float64 | Unix epoch timestamp |
| `device` | string | IoT device identifier |
| `co` | float64 | Carbon monoxide sensor reading |
| `humidity` | float64 | Relative humidity reading |
| `light` | bool | Light-state indicator |
| `lpg` | float64 | LPG sensor reading |
| `motion` | bool | Motion-state indicator |
| `smoke` | float64 | Smoke sensor reading |
| `temp` | float64 | Device temperature |

## Dataset Statistics

| Field | Min | Mean | Std | Median | Max |
|---|---:|---:|---:|---:|---:|
| `co` | 0.001171 | 0.004639 | 0.001250 | 0.004812 | 0.014420 |
| `humidity` | 1.1 | 60.512 | 11.366 | 54.9 | 99.9 |
| `lpg` | 0.002693 | 0.007237 | 0.001444 | 0.007489 | 0.016567 |
| `smoke` | 0.006692 | 0.019264 | 0.004086 | 0.019950 | 0.046590 |
| `temp` | 0.0 | 22.454 | 2.698 | 22.2 | 30.6 |

## Data Quality

All nine columns contain zero missing values in the original dataset.

The dataset contains three unique device identifiers.

The `light` and `motion` fields are Boolean variables with two possible states.

## Temperature Unit Verification

The published dataset metadata describes the `temp` field as Fahrenheit.

However, independent verification of the actual values shows a range of 0.0–30.6 with a mean of approximately 22.45. These values are consistent with indoor temperature measurements expressed in Celsius rather than Fahrenheit.

Therefore, this project uses:

`temp` → **Celsius**

This correction is explicitly documented so that the semantic reference contract is based on the observed data rather than an unverified metadata label.

## Role in the Research

This dataset serves as the primary streaming dataset for evaluating schema, statistical, and semantic drift detection.

The original dataset is treated as the **reference / normal distribution**. Controlled drift will later be injected into copies of this data, with the original reference profile remaining unchanged.