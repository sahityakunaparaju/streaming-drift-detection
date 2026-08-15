"""
Phase 7 controlled window generator.

Builds two artifacts from the primary reference dataset:

1. MATRIX — one independent window per (drift_type, severity) cell of
   the Section R controlled experiment matrix, plus clean /
   normal_noise controls. Used for precision/recall/F1/FPR, broken
   down per detector and per severity level.

2. LAG STREAMS — one 5-window stream per drift type (3 clean windows
   followed by 2 windows with the drift persisting). Used for
   detection-lag measurement (first-detection window minus onset).

Every ground-truth event carries `expected_false_negative` derived
from the shared rule (drift_injection/flagging.py) — no window type
is silently unflagged. Variance-shift and distribution-shift windows
are flagged end-to-end here for the first time in a real window set.

Seed-aware window placement (Section Y): window START POSITIONS are
sampled per seed inside fixed bounding regions (matrix within
[0, MATRIX_REGION), lag streams within [LAG_REGION_BASE,
LAG_REGION_END)), so different seeds draw different, non-overlapping
windows from the dataset. Window counts, subtype/severity
distribution, and expected-false-negative flagging are identical
across seeds; only WHERE each window is placed varies.

Reproducible: pass the same --seed and get the same windows.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from drift_injection.flagging import (
    expected_false_negative_reason,
)
from drift_injection.ground_truth import (
    create_event,
    append_event,
)
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
    inject_normal_noise,
)

WINDOW_SIZE = 1000

MATRIX_REGION = 100000
LAG_REGION_BASE = 120000
LAG_REGION_END = 250000
LAG_STREAM_STRIDE = 5 * WINDOW_SIZE

def sample_non_overlapping_starts(
    rng,
    n,
    window_size,
    region_start,
    region_end,
):
    """Sample n non-overlapping window start rows inside a region.

    Places n windows of `window_size` rows so that no two windows
    share any rows, all within [region_start, region_end). Uses the
    standard sorted-offsets construction: offsets are n distinct
    values in [0, region_size - n*window_size], and

        start_i = region_start + offset_i + i * window_size

    which guarantees start_{i+1} >= start_i + window_size (no
    overlap) and max(start_n) + window_size <= region_end.
    """
    region_size = region_end - region_start
    span = region_size - n * window_size
    if span < 0:
        raise ValueError(
            f"Region [{region_start}, {region_end}) too small for "
            f"{n} windows of {window_size} rows (need "
            f"{n * window_size} rows, have {region_size})."
        )
    offsets = np.sort(
        rng.choice(
            np.arange(0, span + 1),
            size=n,
            replace=False,
        )
    )
    starts = region_start + offsets + np.arange(n) * window_size
    return starts

RANGE_VIOLATION_FRACTION = 0.02
RANGE_MAGNITUDES = {
    "low": 0.02,
    "medium": 0.08,
    "high": 0.25,
}
DOMAIN_INVALID_COUNTS = {
    "low": 5,
    "medium": 50,
    "high": 200,
}

SCHEMA_SEVERITY = {
    "added": "high",
    "removed": "high",
    "renamed": "high",
    "type_change": "medium",
    "nullability_change": "low",
}

STATISTICAL_TYPES = ["mean_shift", "variance_shift",
                     "distribution_shift", "category_shift"]
STATISTICAL_SEVERITIES = ["low", "medium", "high"]

SCHEMA_TYPES = ["added", "removed", "renamed",
                "type_change", "nullability_change"]

SEMANTIC_SINGLE = [
    ("unit_change", "temp", "high"),
    ("unit_conversion", "temp", "high"),
    ("business_meaning_change", "temp", "high"),
    ("range_change", "temp", "medium"),
    ("allowed_domain_change", "device", "medium"),
]

CONTROLS = [
    ("clean", None, "none"),
    ("normal_noise", "temp", "none"),
]

def build_matrix_spec():
    spec = []
    for severity in STATISTICAL_SEVERITIES:
        for subtype in STATISTICAL_TYPES:
            feature = "temp" if subtype != "category_shift" else "device"
            spec.append(("statistical", subtype, feature, severity))
    for subtype in SCHEMA_TYPES:
        feature = {
            "added": "battery_level",
            "removed": "motion",
            "renamed": "temp",
            "type_change": "temp",
            "nullability_change": "temp",
        }[subtype]
        spec.append(("schema", subtype, feature,
                     SCHEMA_SEVERITY[subtype]))
    for subtype, feature, severity in SEMANTIC_SINGLE:
        spec.append(("semantic", subtype, feature, severity))
    for severity in STATISTICAL_SEVERITIES:
        spec.append(("semantic", "range_violation", "temp", severity))
        spec.append(("semantic", "domain_violation", "device", severity))
    for subtype, feature, severity in CONTROLS:
        spec.append(("normal", subtype, feature, severity))
    return spec

def build_lag_spec():
    """One 5-window stream per drift type at medium severity.

    Returns list of (stream_id, drift_type, subtype, feature).
    """
    streams = []
    for subtype in STATISTICAL_TYPES:
        feature = "temp" if subtype != "category_shift" else "device"
        streams.append((f"lag-{subtype}", "statistical",
                        subtype, feature))
    for subtype in SCHEMA_TYPES:
        feature = {
            "added": "battery_level",
            "removed": "motion",
            "renamed": "temp",
            "type_change": "temp",
            "nullability_change": "temp",
        }[subtype]
        streams.append((f"lag-{subtype}", "schema", subtype, feature))
    for subtype, feature, _severity in SEMANTIC_SINGLE:
        streams.append((f"lag-{subtype}", "semantic", subtype, feature))
    for subtype in ["range_violation", "domain_violation"]:
        streams.append((f"lag-{subtype}", "semantic", subtype, "temp"
                        if subtype == "range_violation" else "device"))
    return streams

def inject_range_violation(batch, magnitude_fraction, random_state):
    """Push a fraction of temp values above the contract max."""
    contract = {"valid_range": {"min": 0.0, "max": 30.6}}
    valid_range = contract["valid_range"]
    width = valid_range["max"] - valid_range["min"]
    overrun = magnitude_fraction * width
    result = batch.copy()
    n = int(len(result) * RANGE_VIOLATION_FRACTION)
    rng = np.random.default_rng(random_state)
    selected = rng.choice(result.index, size=n, replace=False)
    result.loc[selected, "temp"] = valid_range["max"] + overrun
    return result

def inject_domain_violation(batch, n_invalid, random_state):
    result = batch.copy()
    rng = np.random.default_rng(random_state)
    selected = rng.choice(
        result.index,
        size=min(n_invalid, len(result)),
        replace=False,
    )
    result.loc[selected, "device"] = "unknown_device"
    return result

def inject_nullability_change(batch, column="temp", n_nulls=2):
    """Make a column nullable by injecting a few nulls."""
    result = batch.copy()
    rng = np.random.default_rng(3)
    selected = rng.choice(
        result.index,
        size=min(n_nulls, len(result)),
        replace=False,
    )
    result.loc[selected, column] = np.nan
    return result

def semantic_metadata_for(subtype):
    if subtype == "unit_change":
        return {"change": "unit_change", "column": "temp",
                "new_unit": "fahrenheit", "converted": False}
    if subtype == "unit_conversion":
        return {"change": "unit_change", "column": "temp",
                "new_unit": "fahrenheit", "converted": True}
    if subtype == "business_meaning_change":
        return {"change": "business_meaning_change", "column": "temp",
                "new_meaning": "case_temperature"}
    if subtype == "range_change":
        return {"change": "range_change", "column": "temp",
                "valid_range": {"min": 0.0, "max": 40.0}}
    if subtype == "allowed_domain_change":
        return {"change": "allowed_domain_change", "column": "device",
                "allowed_domain": [
                    "1c:bf:ce:15:ec:4d",
                    "00:0f:00:70:91:0a",
                ]}
    return None

def inject_window(batch, drift_type, subtype, severity, random_state):
    """
    Apply the injection for one (drift_type, subtype, severity) cell.

    Returns (modified_df, metadata) where metadata is None for
    non-semantic-contract windows.
    """
    if drift_type == "normal":
        if subtype == "normal_noise":
            return inject_normal_noise(
                batch,
                columns=["temp"],
                noise_fraction=0.01,
                random_state=random_state,
            ), None
        return batch, None

    if drift_type == "schema":
        if subtype == "added":
            return inject_add_column(
                batch,
                column="battery_level",
                value=100.0,
            ), None
        if subtype == "removed":
            return inject_remove_column(
                batch,
                column="motion",
            ), None
        if subtype == "renamed":
            return inject_rename_column(
                batch,
                {"temp": "temperature"},
            ), None
        if subtype == "type_change":
            return inject_type_change(
                batch,
                column="temp",
                target_type="string",
            ), None
        if subtype == "nullability_change":
            return inject_nullability_change(batch), None

    if drift_type == "statistical":
        if subtype == "mean_shift":
            return inject_mean_shift(
                batch, column="temp", severity=severity,
            ), None
        if subtype == "variance_shift":
            return inject_variance_shift(
                batch, column="temp", severity=severity,
            ), None
        if subtype == "distribution_shift":
            return inject_distribution_shift(
                batch, column="temp", severity=severity,
                random_state=random_state,
            ), None
        if subtype == "category_shift":
            return inject_category_shift(
                batch,
                column="device",
                category=batch["device"].iloc[0],
                severity=severity,
                random_state=random_state,
            ), None

    if drift_type == "semantic":
        metadata = semantic_metadata_for(subtype)
        if metadata is not None:
            if subtype == "unit_conversion":
                converted, _meta = inject_unit_change(
                    batch,
                    column="temp",
                    reference_unit="celsius",
                    new_unit="fahrenheit",
                    metadata_only=False,
                )
                return converted, metadata
            return batch, metadata
        if subtype == "range_violation":
            return inject_range_violation(
                batch,
                RANGE_MAGNITUDES[severity],
                random_state,
            ), None
        if subtype == "domain_violation":
            return inject_domain_violation(
                batch,
                DOMAIN_INVALID_COUNTS[severity],
                random_state,
            ), None

    raise ValueError(
        f"Unknown matrix cell: {drift_type}/{subtype}/{severity}"
    )

def apply_semantic_metadata(contract, metadata):
    """
    Apply generator-format semantic metadata to a deep copy of the
    reference contract (mirrors the streaming consumer's
    make_semantic_contract, generalized to all contract changes).
    """
    if not metadata:
        return contract
    current = json.loads(json.dumps(contract))
    column = metadata.get("column")
    if column and column in current.get("fields", {}):
        field = current["fields"][column]
        if "new_unit" in metadata:
            field["unit"] = metadata["new_unit"]
        if "new_meaning" in metadata:
            field["business_meaning"] = metadata["new_meaning"]
        if "valid_range" in metadata:
            field["valid_range"] = metadata["valid_range"]
        if "allowed_domain" in metadata:
            field["allowed_domain"] = metadata["allowed_domain"]
    return current

def _event_for(batch, drift_type, subtype, feature, severity,
               injected, ground_truth, random_state):
    return create_event(
        drift_type=drift_type,
        subtype=subtype,
        feature=feature,
        severity=severity,
        injected=injected,
        ground_truth=ground_truth,
        timestamp=str(batch["ts"].iloc[0]),
        expected_false_negative=(
            expected_false_negative_reason(subtype) or ""
        ),
    )

def build_phase7_set(
    input_path,
    output_dir="results/phase7/windows",
    ground_truth_path="results/phase7/ground_truth.csv",
    manifest_path="results/phase7/manifest.json",
    window_size=WINDOW_SIZE,
    seed=42,
):
    """
    Generate the full Phase 7 test set (matrix + lag streams).

    Returns the manifest dict; also returns windows in memory via the
    manifest["_windows"] key (event_id -> DataFrame) for the harness.
    """
    output_dir = Path(output_dir)
    ground_truth_path = Path(ground_truth_path)
    manifest_path = Path(manifest_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_path.parent.mkdir(parents=True, exist_ok=True)
    if ground_truth_path.exists():
        ground_truth_path.unlink()

    df = pd.read_csv(input_path)

    if LAG_REGION_END > len(df):
        raise ValueError(
            f"LAG_REGION_END ({LAG_REGION_END}) exceeds dataset "
            f"length ({len(df)})."
        )

    rng = np.random.default_rng(seed)
    windows_in_memory = {}
    manifest_windows = []

    matrix_spec = build_matrix_spec()
    starts = sample_non_overlapping_starts(
        rng,
        len(matrix_spec),
        window_size,
        0,
        MATRIX_REGION,
    )

    for cell_index, (drift_type, subtype, feature, severity) in enumerate(
        matrix_spec
    ):
        start = starts[cell_index]
        batch = df.iloc[start:start + window_size].copy()
        injected = drift_type != "normal"
        ground_truth = "DRIFT" if injected else "NO_DRIFT"

        modified, metadata = inject_window(
            batch,
            drift_type,
            subtype,
            severity,
            seed,
        )

        event_id = (
            f"matrix-{drift_type}-{subtype}-{severity}"
        )
        event = _event_for(
            batch,
            drift_type,
            subtype,
            feature,
            severity,
            injected,
            ground_truth,
            seed,
        )
        event["event_id"] = event_id

        window_path = output_dir / f"{event_id}.csv"
        modified.to_csv(window_path, index=False)

        metadata_path = None
        if metadata is not None:
            metadata_path = output_dir / f"{event_id}.metadata.json"
            with open(metadata_path, "w") as file:
                json.dump(metadata, file, indent=2)

        append_event(event, ground_truth_path)

        windows_in_memory[event_id] = modified
        manifest_windows.append({
            "event_id": event_id,
            "file": str(window_path),
            "metadata_file": (
                str(metadata_path) if metadata_path else None
            ),
            "set": "matrix",
            "stream_id": None,
            "stream_position": None,
            "drift_type": drift_type,
            "subtype": subtype,
            "feature": feature,
            "severity": severity,
            "injected": injected,
            "ground_truth": ground_truth,
            "expected_false_negative": event["expected_false_negative"],
            "start_row": int(start),
        })

    lag_spec = build_lag_spec()
    stream_starts = sample_non_overlapping_starts(
        rng,
        len(lag_spec),
        LAG_STREAM_STRIDE,
        LAG_REGION_BASE,
        LAG_REGION_END,
    )
    for stream_index, (stream_id, drift_type, subtype, feature) in enumerate(
        lag_spec
    ):
        stream_base = int(stream_starts[stream_index])
        onset_position = 3
        for position in range(5):
            start = stream_base + position * window_size
            batch = df.iloc[start:start + window_size].copy()
            drifted = position >= onset_position
            injected = drifted
            ground_truth = "DRIFT" if injected else "NO_DRIFT"

            if drifted:
                modified, metadata = inject_window(
                    batch,
                    drift_type,
                    subtype,
                    "medium",
                    seed,
                )
            else:
                modified, metadata = batch, None

            event_id = f"{stream_id}-{position:02d}"
            event = _event_for(
                batch,
                drift_type,
                subtype,
                feature,
                "medium" if drifted else "none",
                injected,
                ground_truth,
                seed,
            )
            event["event_id"] = event_id

            window_path = output_dir / f"{event_id}.csv"
            modified.to_csv(window_path, index=False)

            metadata_path = None
            if metadata is not None:
                metadata_path = (
                    output_dir / f"{event_id}.metadata.json"
                )
                with open(metadata_path, "w") as file:
                    json.dump(metadata, file, indent=2)

            append_event(event, ground_truth_path)

            windows_in_memory[event_id] = modified
            manifest_windows.append({
                "event_id": event_id,
                "file": str(window_path),
                "metadata_file": (
                    str(metadata_path) if metadata_path else None
                ),
                "set": "lag",
                "stream_id": stream_id,
                "stream_position": position,
                "drift_type": drift_type,
                "subtype": subtype,
                "feature": feature,
                "severity": "medium" if drifted else "none",
                "injected": injected,
                "ground_truth": ground_truth,
                "expected_false_negative": (
                    event["expected_false_negative"]
                ),
                "start_row": int(start),
            })

    manifest = {
        "seed": seed,
        "window_size": window_size,
        "reference_rows": int(len(df)),
        "n_windows": len(manifest_windows),
        "n_matrix": sum(
            1 for w in manifest_windows if w["set"] == "matrix"
        ),
        "n_lag": sum(
            1 for w in manifest_windows if w["set"] == "lag"
        ),
        "n_expected_false_negative": sum(
            1
            for w in manifest_windows
            if w["expected_false_negative"]
        ),
        "windows": manifest_windows,
        "_windows": windows_in_memory,
    }

    with open(manifest_path, "w") as file:
        json.dump(
            {
                key: value
                for key, value in manifest.items()
                if key != "_windows"
            },
            file,
            indent=2,
        )

    return manifest

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate the Phase 7 controlled test set."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--input",
        default="data/primary/iot_telemetry_data.csv",
    )
    args = parser.parse_args()

    manifest = build_phase7_set(
        args.input,
        seed=args.seed,
    )
    print(
        f"Generated {manifest['n_windows']} windows "
        f"({manifest['n_matrix']} matrix + {manifest['n_lag']} lag); "
        f"{manifest['n_expected_false_negative']} flagged "
        f"expected-false-negative."
    )
