"""
Phase 8 (Section T) — semantic blind-spot test-set generator.

Builds one dedicated, ISOLATED test set per seed for the Semantic
Blind-Spot experiment:

    5 semantic blind-spot cases (values untouched, metadata-only
    contract change, schema untouched):
        - unit_change            (temp: celsius -> fahrenheit)
        - business_meaning_change (temp: device_temperature ->
                                   case_temperature)
        - range_change           (temp: valid_range 0-30.6 -> 0-40)
        - allowed_domain_change  (device: 3-ID domain -> 2-ID domain)
        - definition_change      (temp: instantaneous_raw_sensor_value
                                   -> exponentially_weighted_moving_average)
    plus 2 controls (clean, normal_noise).

Design guarantees for isolation:

- Schema is untouched: every blind-spot window is a raw, unmodified
  slice of the primary dataset, so the reference schema and the window
  schema are IDENTICAL by construction. The evaluation harness still
  re-verifies this per case with compare_schemas().
- Statistical signal is absent by construction: no value is modified,
  so KS/PSI/chi2 are pure sampling noise between the 1000-row window
  and the 405k-row reference. The harness re-verifies per case that
  the measured values fall below the frozen thresholds (ks_floor 0.50,
  decision_threshold 0.25) and EXCLUDES any case where they do not.
- Windows are drawn from rows [320000, 405184), a region disjoint
  from the Phase 7 matrix/lag regions, where a full scan shows no
  1000-row window reaches the statistical decision threshold
  (max score 0.215 < 0.25, max KS 0.371 < 0.50).

Seed-aware window placement (Section Y): window START POSITIONS are
sampled per seed with sample_non_overlapping_starts() (the same
construction as Phase 7), so each seed draws different, non-overlapping
windows. Window counts, subtype list, severity, and ground truth are
identical across seeds; only WHERE each window is placed varies.

Reproducible: pass the same --seed and get the same windows.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from drift_injection.ground_truth import (
    create_event,
    append_event,
)
from drift_injection.injectors import inject_normal_noise
from drift_injection.phase7_generator import (
    sample_non_overlapping_starts,
    semantic_metadata_for,
)

WINDOW_SIZE = 1000

# Disjoint from Phase 7 regions ([0, 100000) matrix, [120000, 250000)
# lag). Full-scan probe: no 1000-row window in [320000, 405184) has a
# statistical score >= 0.25 (max 0.215) or KS >= 0.50 (max 0.371).
PHASE8_REGION_START = 320000
PHASE8_REGION_END = 405184

# (subtype, feature, severity) — metadata-only semantic contract
# changes. Ground truth is DRIFT for every one.
BLIND_SPOT_CASES = [
    ("unit_change", "temp", "high"),
    ("business_meaning_change", "temp", "high"),
    ("range_change", "temp", "medium"),
    ("allowed_domain_change", "device", "medium"),
    ("definition_change", "temp", "high"),
]

CONTROLS = [
    ("clean", None, "none"),
    ("normal_noise", "temp", "none"),
]


def _event_for(batch, drift_type, subtype, feature, severity,
               injected, ground_truth):
    return create_event(
        drift_type=drift_type,
        subtype=subtype,
        feature=feature,
        severity=severity,
        injected=injected,
        ground_truth=ground_truth,
        timestamp=str(batch["ts"].iloc[0]),
        expected_false_negative="",
    )


def build_phase8_set(
    input_path,
    output_dir="results/phase8/windows",
    ground_truth_path="results/phase8/ground_truth.csv",
    manifest_path="results/phase8/manifest.json",
    window_size=WINDOW_SIZE,
    seed=42,
):
    """
    Generate the Phase 8 blind-spot test set for one seed.

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

    if PHASE8_REGION_END > len(df):
        raise ValueError(
            f"PHASE8_REGION_END ({PHASE8_REGION_END}) exceeds dataset "
            f"length ({len(df)})."
        )

    rng = np.random.default_rng(seed)
    windows_in_memory = {}
    manifest_windows = []

    spec = [
        ("semantic", subtype, feature, severity)
        for subtype, feature, severity in BLIND_SPOT_CASES
    ] + [
        ("normal", subtype, feature, severity)
        for subtype, feature, severity in CONTROLS
    ]

    starts = sample_non_overlapping_starts(
        rng,
        len(spec),
        window_size,
        PHASE8_REGION_START,
        PHASE8_REGION_END,
    )

    for cell_index, (drift_type, subtype, feature, severity) in enumerate(
        spec
    ):
        start = int(starts[cell_index])
        batch = df.iloc[start:start + window_size].copy()
        injected = drift_type != "normal"
        ground_truth = "DRIFT" if injected else "NO_DRIFT"

        metadata = None
        modified = batch

        if drift_type == "semantic":
            # Metadata-only: values are UNCHANGED. The contract change
            # is carried in the metadata sidecar and applied to the
            # reference contract by the evaluation harness
            # (apply_semantic_metadata), mirroring Phase 7 semantics.
            metadata = semantic_metadata_for(subtype)

        elif subtype == "normal_noise":
            modified = inject_normal_noise(
                batch,
                columns=["temp"],
                noise_fraction=0.01,
                random_state=seed,
            )

        event_id = f"bs-{subtype}"
        event = _event_for(
            batch,
            drift_type,
            subtype,
            feature,
            severity,
            injected,
            ground_truth,
        )
        event["event_id"] = event_id

        window_path = output_dir / f"{event_id}.csv"
        modified.to_csv(window_path, index=False)

        metadata_path = None
        if metadata is not None:
            metadata_path = output_dir / f"{event_id}.metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as file:
                json.dump(metadata, file, indent=2)

        append_event(event, ground_truth_path)

        windows_in_memory[event_id] = modified
        manifest_windows.append({
            "event_id": event_id,
            "file": str(window_path),
            "metadata_file": (
                str(metadata_path) if metadata_path else None
            ),
            "set": "blind_spot" if drift_type == "semantic" else "control",
            "drift_type": drift_type,
            "subtype": subtype,
            "feature": feature,
            "severity": severity,
            "injected": injected,
            "ground_truth": ground_truth,
            "expected_false_negative": "",
            "start_row": start,
        })

    manifest = {
        "seed": seed,
        "window_size": window_size,
        "reference_rows": int(len(df)),
        "n_windows": len(manifest_windows),
        "n_blind_spot": sum(
            1 for w in manifest_windows if w["set"] == "blind_spot"
        ),
        "n_control": sum(
            1 for w in manifest_windows if w["set"] == "control"
        ),
        "windows": manifest_windows,
        "_windows": windows_in_memory,
    }

    with open(manifest_path, "w", encoding="utf-8") as file:
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
        description="Generate the Phase 8 blind-spot test set."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--input",
        default="data/primary/iot_telemetry_data.csv",
    )
    args = parser.parse_args()

    manifest = build_phase8_set(
        args.input,
        seed=args.seed,
    )
    print(
        f"Generated {manifest['n_windows']} windows "
        f"({manifest['n_blind_spot']} blind-spot + "
        f"{manifest['n_control']} control); seed={args.seed}."
    )
