"""
Phase 7 orchestrator — baseline ladder (B0-B5) + controlled matrix.

Usage:
    PYTHONPATH=. python3 experiments/run_phase7.py [--seed 42]

Writes to results/phase7/:
    windows/          generated window CSVs + metadata sidecars
    ground_truth.csv  ground truth for every window
    manifest.json     per-window metadata (incl. flagging)
    results.json      full structured results
    tables/           ladder.csv, severity_matrix.csv, lag.csv,
                      subtype_recall.csv

Each detector config is run ONCE over every window; all summaries
(ladder, severity matrix, subtype recall, detection lag) are derived
from the cached rows. The canonical run uses --seed 42 (Section Y
multi-seed repetition is a later phase; every run is reproducible
with --seed).
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluation.baseline import (
    run_b0_null_monitor,
    run_b1_schema,
    run_b2_statistical,
    run_b3_schema_statistical,
    run_b4_unified,
    run_semantic_only,
    run_b5_evidently,
)
from evaluation.metrics import (
    evaluate,
    lag_summary,
    detection_lag_windows,
)
from drift_injection.phase7_generator import build_phase7_set

DEFAULT_SEED = 42
OUTPUT_ROOT = Path("results/phase7")

ALL_CONFIGS = [
    "B0_null",
    "B1_schema",
    "B2_statistical",
    "B3_schema_statistical",
    "B4_unified",
    "B5_evidently",
    "semantic",
]

LADDER_ORDER = [
    "B0_null",
    "B1_schema",
    "B2_statistical",
    "B3_schema_statistical",
    "B4_unified",
    "B5_evidently",
]

MATRIX_DETECTORS = [
    ("B1_schema", "schema"),
    ("B2_statistical", "statistical"),
    ("semantic", "semantic"),
    ("B4_unified", "combined"),
]

def load_reference():
    """Load the primary reference dataframe and its profile JSON."""
    reference_df = pd.read_csv(
        "data/primary/iot_telemetry_data.csv"
    )
    with open(
        "data/primary/reference_profile.json",
        "r",
    ) as file:
        profile = json.load(file)
    return reference_df, profile

def run_config(name, reference_df, profile, window_df, metadata):
    """Dispatch one detector config by name; returns (decision, details)."""
    if name == "B0_null":
        return run_b0_null_monitor(window_df)
    if name == "B1_schema":
        return run_b1_schema(profile["schema"], window_df)
    if name == "B2_statistical":
        return run_b2_statistical(reference_df, window_df)
    if name == "B3_schema_statistical":
        return run_b3_schema_statistical(
            profile["schema"],
            reference_df,
            window_df,
        )
    if name == "B4_unified":
        return run_b4_unified(
            profile["schema"],
            profile["semantic_contract"],
            reference_df,
            window_df,
            metadata,
        )
    if name == "B5_evidently":
        return run_b5_evidently(reference_df, window_df)
    if name == "semantic":
        return run_semantic_only(
            profile["semantic_contract"],
            window_df,
            metadata,
        )
    raise ValueError(f"Unknown config: {name}")

def load_metadata(metadata_file):
    """Load a metadata sidecar JSON, or None if no file given."""
    if not metadata_file:
        return None
    with open(metadata_file, "r") as file:
        return json.load(file)

def run_detector_over_manifest(
    name,
    reference_df,
    profile,
    manifest,
):
    """Run one detector over every window; return rows of results."""
    rows = []
    for entry in manifest["windows"]:
        window_df = manifest["_windows"][entry["event_id"]]
        metadata = load_metadata(entry.get("metadata_file"))
        decision, details = run_config(
            name,
            reference_df,
            profile,
            window_df,
            metadata,
        )
        rows.append({
            "event_id": entry["event_id"],
            "set": entry["set"],
            "stream_id": entry.get("stream_id"),
            "stream_position": entry.get("stream_position"),
            "drift_type": entry["drift_type"],
            "subtype": entry["subtype"],
            "severity": entry["severity"],
            "ground_truth": entry["ground_truth"],
            "expected_false_negative": entry[
                "expected_false_negative"
            ],
            "decision": decision,
        })
    return rows

def summarize_rows(rows):
    """Metrics over a set of rows, with raw + adjusted variants."""
    available = [
        row for row in rows
        if row["decision"] != "UNAVAILABLE"
    ]
    y_true = [
        1 if row["ground_truth"] == "DRIFT" else 0
        for row in available
    ]
    y_pred = [
        1 if row["decision"] == "DRIFT" else 0
        for row in available
    ]
    expected_fn = [
        row["expected_false_negative"] for row in available
    ]
    return evaluate(y_true, y_pred, expected_fn)

def lag_for_rows(rows, onset_position=3):
    """Per-stream detection lag in windows."""
    by_stream = {}
    for row in rows:
        if row["set"] != "lag" or not row["stream_id"]:
            continue
        by_stream.setdefault(row["stream_id"], []).append(row)

    lags = []
    for stream_id in sorted(by_stream):
        stream_rows = sorted(
            by_stream[stream_id],
            key=lambda r: r["stream_position"] or 0,
        )
        predictions = [
            1 if row["decision"] == "DRIFT" else 0
            for row in stream_rows
        ]
        lags.append(
            detection_lag_windows(
                predictions,
                onset_position=onset_position,
            )
        )
    return lags

def main():
    """Run the Phase 7 baseline ladder and controlled matrix."""
    parser = argparse.ArgumentParser(
        description="Phase 7 baseline ladder + controlled matrix."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="RNG seed (default 42; canonical result).",
    )
    parser.add_argument(
        "--input",
        default="data/primary/iot_telemetry_data.csv",
    )
    args = parser.parse_args()
    seed = args.seed

    print("=" * 90)
    print("PHASE 7 - BASELINE LADDER (B0-B5) + CONTROLLED MATRIX")
    print(f"seed={seed}")
    print("=" * 90)

    reference_df, profile = load_reference()

    manifest = build_phase7_set(
        args.input,
        output_dir=str(OUTPUT_ROOT / "windows"),
        ground_truth_path=str(OUTPUT_ROOT / "ground_truth.csv"),
        manifest_path=str(OUTPUT_ROOT / "manifest.json"),
        seed=seed,
    )
    print(
        f"\nTest set: {manifest['n_windows']} windows "
        f"({manifest['n_matrix']} matrix + {manifest['n_lag']} lag); "
        f"{manifest['n_expected_false_negative']} flagged "
        "expected-false-negative."
    )

    rows_by_config = {}
    for name in ALL_CONFIGS:
        print(f"\nRunning {name} over {manifest['n_windows']} windows ...")
        rows_by_config[name] = run_detector_over_manifest(
            name,
            reference_df,
            profile,
            manifest,
        )

    results = {
        "seed": seed,
        "window_size": manifest["window_size"],
        "test_set": {
            "n_windows": manifest["n_windows"],
            "n_matrix": manifest["n_matrix"],
            "n_lag": manifest["n_lag"],
            "n_expected_false_negative": (
                manifest["n_expected_false_negative"]
            ),
        },
        "ladder": {},
        "severity_matrix": {},
        "subtype_recall": {},
    }

    print("\n" + "=" * 90)
    print("BASELINE LADDER - full test set (matrix + lag streams)")
    print("=" * 90)

    ladder_rows = []
    for name in LADDER_ORDER:
        rows = rows_by_config[name]
        summary = summarize_rows(rows)
        lags = lag_for_rows(rows)
        lag = lag_summary(lags)

        results["ladder"][name] = {
            "metrics": summary,
            "lag": lag,
        }
        ladder_rows.append({
            "config": name,
            **{f"raw_{k}": v for k, v in summary["raw"].items()},
            **{
                f"adj_{k}": v
                for k, v in summary["adjusted"].items()
            },
            "flagged": summary["flagged"]["n_flagged"],
            "flagged_detected": summary["flagged"][
                "n_flagged_detected"
            ],
            "flagged_missed": summary["flagged"]["n_flagged_missed"],
            "lag_mean": lag["mean"],
            "lag_median": lag["median"],
            "lag_p95": lag["p95"],
            "lag_n_detected": lag["n_detected"],
            "lag_n_never": lag["n_never_detected"],
        })
        print(
            f"\n{name}: "
            f"raw P={summary['raw']['precision']:.3f} "
            f"R={summary['raw']['recall']:.3f} "
            f"F1={summary['raw']['f1']:.3f} "
            f"FPR={summary['raw']['fpr']:.3f} | "
            f"adj P={summary['adjusted']['precision']:.3f} "
            f"R={summary['adjusted']['recall']:.3f} "
            f"F1={summary['adjusted']['f1']:.3f} | "
            f"lag mean/med/p95 = "
            f"{lag['mean']}/{lag['median']}/{lag['p95']} windows "
            f"(detected {lag['n_detected']}, never "
            f"{lag['n_never_detected']})"
        )

    print("\n" + "=" * 90)
    print("CONTROLLED MATRIX - by severity level (matrix windows)")
    print("=" * 90)

    matrix_window_ids = {
        entry["event_id"]
        for entry in manifest["windows"]
        if entry["set"] == "matrix"
    }

    severity_rows = []
    for name, label in MATRIX_DETECTORS:
        rows = rows_by_config[name]
        matrix_rows = [
            row for row in rows
            if row["event_id"] in matrix_window_ids
        ]
        for severity in ["none", "low", "medium", "high"]:
            subset = [
                row for row in matrix_rows
                if row["severity"] == severity
            ]
            if not subset:
                continue
            summary = summarize_rows(subset)
            results["severity_matrix"].setdefault(
                severity, {}
            )[label] = summary
            severity_rows.append({
                "severity": severity,
                "detector": label,
                "raw_precision": summary["raw"]["precision"],
                "raw_recall": summary["raw"]["recall"],
                "raw_f1": summary["raw"]["f1"],
                "raw_fpr": summary["raw"]["fpr"],
                "adj_precision": summary["adjusted"]["precision"],
                "adj_recall": summary["adjusted"]["recall"],
                "adj_f1": summary["adjusted"]["f1"],
                "adj_fpr": summary["adjusted"]["fpr"],
                "n": summary["n"],
            })
            print(
                f"  severity={severity:<7} {label:<12} "
                f"raw R={summary['raw']['recall']:.3f} "
                f"F1={summary['raw']['f1']:.3f} "
                f"FPR={summary['raw']['fpr']:.3f} | "
                f"adj R={summary['adjusted']['recall']:.3f} "
                f"F1={summary['adjusted']['f1']:.3f} "
                f"(n={summary['n']})"
            )

    print("\n" + "=" * 90)
    print("PER-SUBTYPE RECALL (raw / adjusted)")
    print("=" * 90)

    subtype_rows = []
    for name, label in MATRIX_DETECTORS:
        rows = rows_by_config[name]
        for subtype in sorted({
            row["subtype"] for row in rows
            if row["ground_truth"] == "DRIFT"
        }):
            subset = [
                row for row in rows
                if row["subtype"] == subtype
                and row["ground_truth"] == "DRIFT"
            ]
            if not subset:
                continue
            summary = summarize_rows(subset)
            results["subtype_recall"].setdefault(
                subtype, {}
            )[label] = {
                "raw_recall": summary["raw"]["recall"],
                "adj_recall": summary["adjusted"]["recall"],
                "n": summary["n"],
                "flagged": summary["flagged"]["n_flagged"],
            }
            subtype_rows.append({
                "subtype": subtype,
                "detector": label,
                "raw_recall": summary["raw"]["recall"],
                "adj_recall": summary["adjusted"]["recall"],
                "n": summary["n"],
                "flagged": summary["flagged"]["n_flagged"],
            })

    tables_dir = OUTPUT_ROOT / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(ladder_rows).to_csv(
        tables_dir / "ladder.csv", index=False
    )
    pd.DataFrame(severity_rows).to_csv(
        tables_dir / "severity_matrix.csv", index=False
    )
    pd.DataFrame(subtype_rows).to_csv(
        tables_dir / "subtype_recall.csv", index=False
    )

    with open(OUTPUT_ROOT / "results.json", "w") as file:
        json.dump(results, file, indent=2)

    print(f"\nWrote results to {OUTPUT_ROOT / 'results.json'}")
    print(f"Tables to {tables_dir}")

if __name__ == "__main__":
    main()
