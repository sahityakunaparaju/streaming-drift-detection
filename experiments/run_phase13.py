"""
Phase 13 — Secondary Dataset Sanity Check (Generalization).

Merges the 12 Beijing Air-Quality station CSVs, builds a reference
profile, injects representative drift windows (including a
category-shift event on wind direction), and runs the B0-B4 baseline
ladder with the SAME frozen thresholds from configs/thresholds.yaml.

No threshold tuning is performed. This confirms whether frozen
thresholds generalize to a genuinely different dataset.

Usage:
    PYTHONPATH=. python3 experiments/run_phase13.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from drift_injection.injectors import (
    inject_add_column,
    inject_mean_shift,
    inject_category_shift,
    inject_unit_change,
    inject_normal_noise,
)
from drift_injection.phase7_generator import (
    inject_nullability_change,
)
from drift_injection.ground_truth import create_event, append_event
from evaluation.baseline import (
    build_current_schema,
    run_b0_null_monitor,
    run_b1_schema,
    run_b2_statistical,
    run_b3_schema_statistical,
    run_b4_unified,
)
from evaluation.metrics import evaluate

RAW_DIR = Path("data/secondary/raw")
OUTPUT_DIR = Path("results/phase13")
WINDOWS_DIR = OUTPUT_DIR / "windows"
PROFILE_PATH = OUTPUT_DIR / "reference_profile.json"
SCHEMA_PATH = OUTPUT_DIR / "schema_reference.yaml"
GROUND_TRUTH_PATH = OUTPUT_DIR / "ground_truth.csv"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
RESULTS_PATH = OUTPUT_DIR / "results.json"
REF_DATA_PATH = Path("data/secondary/reference_data.csv")

WINDOW_SIZE = 1000
SEED = 42


# ============================================================
# Step 1: Merge & Clean
# ============================================================

def merge_and_clean():
    """Merge all 12 station CSVs into a single working dataframe."""
    print("Step 1: Merging and cleaning 12 station CSVs ...")

    csv_files = sorted(RAW_DIR.glob("PRSA_Data_*.csv"))
    assert len(csv_files) == 12, f"Expected 12 CSVs, found {len(csv_files)}"

    frames = []
    for path in csv_files:
        df = pd.read_csv(path, na_values=["NA"])
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)

    merged = merged.drop(columns=["No"])

    merged["ts"] = pd.to_datetime(
        merged[["year", "month", "day", "hour"]]
        .rename(columns={
            "year": "year", "month": "month",
            "day": "day", "hour": "hour",
        })
    ).astype(np.int64) / 10**9  # unix epoch seconds

    merged = merged.drop(columns=["year", "month", "day", "hour"])

    cols = list(merged.columns)
    cols.remove("ts")
    cols.remove("station")
    merged = merged[["ts", "station"] + cols]

    missing = merged.isna().mean()
    print(f"  Shape: {merged.shape}")
    print(f"  Missing value rates:")
    for col in merged.columns:
        rate = missing[col]
        if rate > 0:
            print(f"    {col}: {rate:.4f} ({rate*100:.2f}%)")

    # Save merged data
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(REF_DATA_PATH, index=False)
    print(f"  Saved merged data to {REF_DATA_PATH}")

    return merged


def discover_categorical_columns(contract):
    """Extract categorical columns from the semantic contract.

    Looks for fields with data_role containing 'categorical' or with
    an explicit 'allowed_values' list.  Excludes identifier fields
    (data_role == 'identifier' or 'timestamp').
    """
    fields = contract.get("fields", {})
    cat_cols = []
    for col, spec in fields.items():
        role = spec.get("data_role", "")
        if "categorical" in role and "identifier" not in role:
            cat_cols.append(col)
        elif "allowed_values" in spec and "identifier" not in role:
            # allowed_values implies categorical even if role not explicit
            if col not in cat_cols:
                cat_cols.append(col)
    return cat_cols


# ============================================================
# Step 2: Build Reference Profile & Schema
# ============================================================

def build_reference_profile(df, reference_slice):
    """Build reference profile and schema for the secondary dataset.

    The schema reference is built from the FULL merged dataset using
    build_current_schema() (same logic as baseline.py) so that nullable
    captures all columns that CAN have nulls anywhere. This prevents
    spurious nullability_change events when a small clean window
    happens to have zero nulls in a column that has nulls elsewhere.
    """
    print("\nStep 2: Building reference profile ...")

    schema_ref = build_current_schema(df)  # rate-aware nullability: ~0.1% natural nulls won't false-fire on clean windows
    schema_ref["schema_version"] = "1.0"
    schema_ref["dataset"] = {
        "name": "beijing_air_quality",
        "description": "Reference schema for Beijing Multi-Site Air-Quality dataset",
    }

    with open(SCHEMA_PATH, "w", encoding="utf-8") as f:
        yaml.dump(schema_ref, f, default_flow_style=False, sort_keys=False)
    print(f"  Schema written to {SCHEMA_PATH}")
    print(f"  Schema built from full merged dataset ({len(df)} rows) using build_current_schema()")

    profile_stats = {}
    for col in df.columns:
        series = df[col]
        col_profile = {
            "dtype": str(series.dtype),
            "count": int(series.count()),
            "missing_count": int(series.isna().sum()),
            "missing_percentage": float(series.isna().mean() * 100),
            "unique_count": int(series.nunique()),
        }
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            col_profile["statistics"] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "q25": float(series.quantile(0.25)),
                "median": float(series.median()),
                "q75": float(series.quantile(0.75)),
                "max": float(series.max()),
            }
        elif pd.api.types.is_object_dtype(series):
            frequencies = series.value_counts(normalize=True)
            col_profile["value_frequencies"] = {
                str(k): float(v) for k, v in frequencies.items()
            }
        profile_stats[col] = col_profile

    with open("configs/semantic_contract_secondary.yaml",
              "r", encoding="utf-8") as f:
        semantic_contract = yaml.safe_load(f)

    reference_profile = {
        "profile_version": "1.0",
        "dataset": {
            "name": "beijing_air_quality",
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
        },
        "schema": schema_ref,
        "statistics": profile_stats,
        "semantic_contract": semantic_contract,
    }

    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(reference_profile, f, indent=2)
    print(f"  Reference profile written to {PROFILE_PATH}")

    return reference_profile


# ============================================================
# Step 3: Inject Drift Windows
# ============================================================

def inject_windows(df, reference_profile):
    """Create representative drift windows + clean controls."""
    print("\nStep 3: Injecting drift windows ...")

    windows = []
    rng = np.random.default_rng(SEED)

    # --- Clean control window 1 ---
    start = 0
    clean1 = df.iloc[start:start + WINDOW_SIZE].copy()
    event_id = "clean-00"
    event = create_event(
        drift_type="normal", subtype="clean", feature=None,
        severity="none", injected=False, ground_truth="NO_DRIFT",
        timestamp=str(clean1["ts"].iloc[0]),
    )
    event["event_id"] = event_id
    windows.append({
        "event_id": event_id, "drift_type": "normal",
        "subtype": "clean", "severity": "none",
        "injected": False, "ground_truth": "NO_DRIFT",
        "expected_false_negative": "", "metadata_file": None,
        "df": clean1, "event": event,
    })

    # --- Clean control window 2 (normal noise) ---
    start = WINDOW_SIZE
    clean2_base = df.iloc[start:start + WINDOW_SIZE].copy()
    clean2 = inject_normal_noise(
        clean2_base, columns=["PM2.5", "TEMP"],
        noise_fraction=0.01, random_state=SEED,
    )
    event_id = "clean-01"
    event = create_event(
        drift_type="normal", subtype="normal_noise", feature="PM2.5",
        severity="none", injected=False, ground_truth="NO_DRIFT",
        timestamp=str(clean2_base["ts"].iloc[0]),
    )
    event["event_id"] = event_id
    windows.append({
        "event_id": event_id, "drift_type": "normal",
        "subtype": "normal_noise", "severity": "none",
        "injected": False, "ground_truth": "NO_DRIFT",
        "expected_false_negative": "", "metadata_file": None,
        "df": clean2, "event": event,
    })

    # --- Schema drift: added column ---
    start = 2 * WINDOW_SIZE
    schema_base = df.iloc[start:start + WINDOW_SIZE].copy()
    schema_win = inject_add_column(
        schema_base, column="AQI_index", value=50.0,
    )
    event_id = "schema-added"
    event = create_event(
        drift_type="schema", subtype="added", feature="AQI_index",
        severity="high", injected=True, ground_truth="DRIFT",
        timestamp=str(schema_base["ts"].iloc[0]),
    )
    event["event_id"] = event_id
    windows.append({
        "event_id": event_id, "drift_type": "schema",
        "subtype": "added", "severity": "high",
        "injected": True, "ground_truth": "DRIFT",
        "expected_false_negative": "", "metadata_file": None,
        "df": schema_win, "event": event,
    })

    # --- Statistical drift: mean shift on PM2.5 ---
    start = 3 * WINDOW_SIZE
    stat_base = df.iloc[start:start + WINDOW_SIZE].copy()
    stat_win = inject_mean_shift(
        stat_base, column="PM2.5", severity="high",
    )
    event_id = "stat-mean-shift"
    event = create_event(
        drift_type="statistical", subtype="mean_shift",
        feature="PM2.5", severity="high",
        injected=True, ground_truth="DRIFT",
        timestamp=str(stat_base["ts"].iloc[0]),
    )
    event["event_id"] = event_id
    windows.append({
        "event_id": event_id, "drift_type": "statistical",
        "subtype": "mean_shift", "severity": "high",
        "injected": True, "ground_truth": "DRIFT",
        "expected_false_negative": "", "metadata_file": None,
        "df": stat_win, "event": event,
    })

    # --- Category shift on wd (wind direction) ---
    start = 4 * WINDOW_SIZE
    cat_base = df.iloc[start:start + WINDOW_SIZE].copy()
    top_category = cat_base["wd"].value_counts().index[0]
    cat_win = inject_category_shift(
        cat_base, column="wd", category=top_category,
        severity="high", random_state=SEED,
    )
    event_id = "cat-wd-shift"
    event = create_event(
        drift_type="statistical", subtype="category_shift",
        feature="wd", severity="high",
        injected=True, ground_truth="DRIFT",
        timestamp=str(cat_base["ts"].iloc[0]),
    )
    event["event_id"] = event_id
    windows.append({
        "event_id": event_id, "drift_type": "statistical",
        "subtype": "category_shift", "severity": "high",
        "injected": True, "ground_truth": "DRIFT",
        "expected_false_negative": "", "metadata_file": None,
        "df": cat_win, "event": event,
    })

    # --- Semantic drift: unit change on TEMP ---
    start = 5 * WINDOW_SIZE
    sem_base = df.iloc[start:start + WINDOW_SIZE].copy()
    sem_win, sem_meta = inject_unit_change(
        sem_base, column="TEMP",
        reference_unit="celsius", new_unit="fahrenheit",
        metadata_only=True,
    )
    event_id = "sem-unit-change"
    event = create_event(
        drift_type="semantic", subtype="unit_change",
        feature="TEMP", severity="high",
        injected=True, ground_truth="DRIFT",
        timestamp=str(sem_base["ts"].iloc[0]),
    )
    event["event_id"] = event_id
    windows.append({
        "event_id": event_id, "drift_type": "semantic",
        "subtype": "unit_change", "severity": "high",
        "injected": True, "ground_truth": "DRIFT",
        "expected_false_negative": "", "metadata_file": sem_meta,
        "df": sem_win, "event": event,
    })

    # --- Schema drift: nullability change on PM2.5 ---
    start = 6 * WINDOW_SIZE
    null_base = df.iloc[start:start + WINDOW_SIZE].copy()
    # 20% nulls vs ~2% baseline: dramatic enough to flag as drift
    null_win = inject_nullability_change(
        null_base, column="PM2.5", n_nulls=200,
    )
    event_id = "schema-nullability"
    event = create_event(
        drift_type="schema", subtype="nullability_change",
        feature="PM2.5", severity="low",
        injected=True, ground_truth="DRIFT",
        timestamp=str(null_base["ts"].iloc[0]),
    )
    event["event_id"] = event_id
    windows.append({
        "event_id": event_id, "drift_type": "schema",
        "subtype": "nullability_change", "severity": "low",
        "injected": True, "ground_truth": "DRIFT",
        "expected_false_negative": "", "metadata_file": None,
        "df": null_win, "event": event,
    })

    # Save all windows and ground truth
    WINDOWS_DIR.mkdir(parents=True, exist_ok=True)
    if GROUND_TRUTH_PATH.exists():
        GROUND_TRUTH_PATH.unlink()

    manifest_windows = []
    for w in windows:
        df_win = w["df"]
        win_path = WINDOWS_DIR / f"{w['event_id']}.csv"
        df_win.to_csv(win_path, index=False)

        meta_path = None
        if w["metadata_file"] is not None:
            meta_path = WINDOWS_DIR / f"{w['event_id']}.metadata.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(w["metadata_file"], f, indent=2)

        event = w["event"]
        event["expected_false_negative"] = w["expected_false_negative"]
        append_event(event, GROUND_TRUTH_PATH)

        manifest_windows.append({
            "event_id": w["event_id"],
            "file": str(win_path),
            "metadata_file": str(meta_path) if meta_path else None,
            "drift_type": w["drift_type"],
            "subtype": w["subtype"],
            "severity": w["severity"],
            "injected": w["injected"],
            "ground_truth": w["ground_truth"],
            "expected_false_negative": w["expected_false_negative"],
        })

    manifest = {
        "seed": SEED,
        "window_size": WINDOW_SIZE,
        "n_windows": len(manifest_windows),
        "windows": manifest_windows,
        "_windows": {w["event_id"]: w["df"] for w in windows},
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {k: v for k, v in manifest.items() if k != "_windows"},
            f, indent=2,
        )

    print(f"  Created {len(windows)} windows:")
    for w in windows:
        print(f"    {w['event_id']}: {w['drift_type']}/{w['subtype']} "
              f"({w['ground_truth']})")

    return manifest


# ============================================================
# Step 4: Run B0-B4
# ============================================================

def run_b0b4(manifest, reference_df, reference_profile,
             categorical_columns=None):
    """Run B0-B4 baseline ladder on secondary dataset windows."""
    print("\nStep 4: Running B0-B4 baseline ladder ...")
    if categorical_columns:
        print(f"  Categorical columns for chi2: {categorical_columns}")

    schema = reference_profile["schema"]
    contract = reference_profile["semantic_contract"]

    configs = [
        ("B0_null", "null monitor"),
        ("B1_schema", "schema detector"),
        ("B2_statistical", "statistical detector"),
        ("B3_schema_statistical", "schema + statistical"),
        ("B4_unified", "unified (schema + stat + semantic)"),
    ]

    results = {}

    for name, description in configs:
        print(f"\n  Running {name} ({description}) ...")
        rows = []

        for entry in manifest["windows"]:
            win_df = manifest["_windows"][entry["event_id"]]
            meta_file = entry.get("metadata_file")
            metadata = None
            if meta_file:
                with open(meta_file, "r") as f:
                    metadata = json.load(f)

            if name == "B0_null":
                decision, details = run_b0_null_monitor(win_df)
            elif name == "B1_schema":
                decision, details = run_b1_schema(schema, win_df)
            elif name == "B2_statistical":
                decision, details = run_b2_statistical(
                    reference_df, win_df,
                    categorical_columns=categorical_columns,
                )
            elif name == "B3_schema_statistical":
                decision, details = run_b3_schema_statistical(
                    schema, reference_df, win_df,
                    categorical_columns=categorical_columns,
                )
            elif name == "B4_unified":
                decision, details = run_b4_unified(
                    schema, contract, reference_df, win_df, metadata,
                    categorical_columns=categorical_columns,
                )

            rows.append({
                "event_id": entry["event_id"],
                "drift_type": entry["drift_type"],
                "subtype": entry["subtype"],
                "ground_truth": entry["ground_truth"],
                "decision": decision,
            })

        y_true = [1 if r["ground_truth"] == "DRIFT" else 0 for r in rows]
        y_pred = [1 if r["decision"] == "DRIFT" else 0 for r in rows]
        metrics = evaluate(y_true, y_pred)

        results[name] = {
            "raw": metrics["raw"],
            "adjusted": metrics["adjusted"],
            "n": metrics["n"],
            "n_positives": metrics["n_positives"],
            "n_negatives": metrics["n_negatives"],
            "rows": rows,
        }

        print(f"    Raw:    P={metrics['raw']['precision']:.3f} "
              f"R={metrics['raw']['recall']:.3f} "
              f"F1={metrics['raw']['f1']:.3f} "
              f"FPR={metrics['raw']['fpr']:.3f}")
        print(f"    Adj:    P={metrics['adjusted']['precision']:.3f} "
              f"R={metrics['adjusted']['recall']:.3f} "
              f"F1={metrics['adjusted']['f1']:.3f} "
              f"FPR={metrics['adjusted']['fpr']:.3f}")

        for r in rows:
            marker = "OK" if r["decision"] == r["ground_truth"] else "FAIL"
            print(f"      {marker} {r['event_id']}: "
                  f"truth={r['ground_truth']}, "
                  f"decision={r['decision']}")

    return results


# ============================================================
# Main
# ============================================================

def main():
    """Run the Phase 13 secondary-dataset generalization check end to end."""
    print("=" * 90)
    print("PHASE 13 — SECONDARY DATASET SANITY CHECK (Generalization)")
    print("=" * 90)

    merged_df = merge_and_clean()

    # Reference slice must not overlap any window start position
    ref_start = 6 * WINDOW_SIZE
    reference_slice = merged_df.iloc[ref_start:ref_start + 10000].copy()
    print(f"\nReference slice: rows {ref_start}-{ref_start + 10000} "
          f"({len(reference_slice)} rows)")

    reference_profile = build_reference_profile(merged_df, reference_slice)

    contract = reference_profile["semantic_contract"]
    categorical_columns = discover_categorical_columns(contract)
    print(f"\nCategorical columns from contract: {categorical_columns}")

    manifest = inject_windows(merged_df, reference_profile)

    results = run_b0b4(manifest, reference_slice, reference_profile,
                       categorical_columns=categorical_columns)

    serializable = {
        name: {
            "raw": r["raw"],
            "adjusted": r["adjusted"],
            "n": r["n"],
            "n_positives": r["n_positives"],
            "n_negatives": r["n_negatives"],
        }
        for name, r in results.items()
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Config':<30} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FPR':>10}")
    print("-" * 70)
    for name in ["B0_null", "B1_schema", "B2_statistical",
                  "B3_schema_statistical", "B4_unified"]:
        r = results[name]
        print(f"{name:<30} "
              f"{r['raw']['precision']:>10.3f} "
              f"{r['raw']['recall']:>10.3f} "
              f"{r['raw']['f1']:>10.3f} "
              f"{r['raw']['fpr']:>10.3f}")
    print("-" * 70)
    print(f"(n={results['B0_null']['n']} windows: "
          f"{results['B0_null']['n_positives']} drift, "
          f"{results['B0_null']['n_negatives']} clean)")

    return results


if __name__ == "__main__":
    main()
