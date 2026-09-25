"""
Diagnose the mean_shift blind spot (B4 statistical channel).

For every mean_shift window across the 10 multi-seed runs, compute:
  - injected magnitude (severity sigma multiplier)
  - per-column KS / PSI (via the real detector helpers)
  - temp (shifted column) KS/PSI vs the max over ALL other columns
  - the statistical_score and whether decide() fires

Compares against frozen thresholds (ks_floor=0.50, decision=0.25,
psi_scale=72.0) and against the clean-window noise floor, to
determine whether mean_shift fails for the same aggregation-masking
reason as variance/distribution shift or for a distinct reason.

Usage:
    PYTHONPATH=. python experiments/diagnose_mean_shift.py
"""
import json
from pathlib import Path

import pandas as pd

from detectors.unified import statistical_score, decide
from evaluation.baseline import (
    statistical_details_for,
    NUMERIC_COLUMNS_EXCLUDE,
)

SEEDS = [42, 123, 7, 11, 2024, 314, 2718, 999, 555, 777]
MULTISEED = Path("results/phase7/multiseed")

def analyze_window(reference_df, window_df, ground_truth, subtype, severity, seed):
    """Measure statistical response of one mean-shift window."""
    details = statistical_details_for(reference_df, window_df)
    numeric = details.get("numeric", {})
    score = statistical_score(details)
    decision = decide(
        score / 3.0,
        schema_score=0.0,
        semantic_score=0.0,
        statistical_score=score,
    )

    per_col = {}
    for col, m in numeric.items():
        if not isinstance(m, dict):
            continue
        per_col[col] = {
            "ks": m.get("ks_statistic"),
            "psi": m.get("psi"),
        }

    shifted = "temp"
    shifted_ks = per_col.get(shifted, {}).get("ks", 0.0) or 0.0
    shifted_psi = per_col.get(shifted, {}).get("psi", 0.0) or 0.0

    other_ks = max(
        (v["ks"] for c, v in per_col.items() if c != shifted and v.get("ks") is not None),
        default=0.0,
    )
    other_psi = max(
        (v["psi"] for c, v in per_col.items() if c != shifted and v.get("psi") is not None),
        default=0.0,
    )
    max_ks = max(v["ks"] for v in per_col.values() if v.get("ks") is not None)
    max_psi = max(v["psi"] for v in per_col.values() if v.get("psi") is not None)

    shift_sigma = abs(
        window_df[shifted].mean() - reference_df[shifted].mean()
    ) / reference_df[shifted].std()

    return {
        "seed": seed,
        "event_id": "mean_shift",
        "severity": severity,
        "ground_truth": ground_truth,
        "shift_sigma": shift_sigma,
        "shifted_col": shifted,
        "shifted_ks": shifted_ks,
        "shifted_psi": shifted_psi,
        "other_max_ks": other_ks,
        "other_max_psi": other_psi,
        "max_ks": max_ks,
        "max_psi": max_psi,
        "statistical_score": score,
        "decision": decision,
        "per_column": per_col,
    }

def main():
    """Diagnose the mean-shift severity sweep across all seeds."""
    reference_df = pd.read_csv("data/primary/iot_telemetry_data.csv")

    rows = []
    for seed in SEEDS:
        seed_dir = MULTISEED / f"seed-{seed}"
        manifest = json.load(open(seed_dir / "manifest.json"))
        windows = manifest.get("_windows")
        if windows is None:

            windows = {}
            for entry in manifest["windows"]:
                windows[entry["event_id"]] = pd.read_csv(entry["file"])
        for entry in manifest["windows"]:
            if entry["subtype"] == "mean_shift":
                row = analyze_window(
                    reference_df,
                    windows[entry["event_id"]],
                    entry["ground_truth"],
                    entry["subtype"],
                    entry["severity"],
                    seed,
                )
                row["event_id"] = entry["event_id"]
                row["set"] = entry["set"]
                rows.append(row)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)

    print("=" * 110)
    print("MEAN_SHIFT WINDOWS: per-window KS/PSI vs thresholds (10 seeds)")
    print("=" * 110)

    matrix = df[df["set"] == "matrix"].sort_values(["seed", "severity"])
    print("\n--- MATRIX windows (controlled severities) ---")
    for _, r in matrix.iterrows():
        print(
            f"seed {r['seed']:<5} sev={r['severity']:<7} "
            f"shift={r['shift_sigma']:.2f} sigma  "
            f"temp_ks={r['shifted_ks']:.3f}  temp_psi={r['shifted_psi']:.2f}  "
            f"other_max_ks={r['other_max_ks']:.3f}  other_max_psi={r['other_max_psi']:.2f}  "
            f"max_ks={r['max_ks']:.3f}  score={r['statistical_score']:.3f}  "
            f"-> {r['decision']}"
        )

    print("\n--- summary by severity (matrix) ---")
    for sev in ["low", "medium", "high"]:
        sub = matrix[matrix["severity"] == sev]
        if sub.empty:
            continue
        print(
            f"{sev:<7} n={len(sub)}  "
            f"temp_ks mean={sub['shifted_ks'].mean():.3f} "
            f"(min {sub['shifted_ks'].min():.3f}, max {sub['shifted_ks'].max():.3f})  "
            f"other_max_ks mean={sub['other_max_ks'].mean():.3f}  "
            f"temp_psi mean={sub['shifted_psi'].mean():.2f}  "
            f"score mean={sub['statistical_score'].mean():.3f}  "
            f"detected={int((sub['decision']=='DRIFT').sum())}/{len(sub)}"
        )

    lag = df[df["set"] == "lag"]
    print("\n--- lag streams (medium severity) ---")
    if not lag.empty:
        print(
            f"n={len(lag)}  temp_ks mean={lag['shifted_ks'].mean():.3f}  "
            f"other_max_ks mean={lag['other_max_ks'].mean():.3f}  "
            f"score mean={lag['statistical_score'].mean():.3f}  "
            f"detected={int((lag['decision']=='DRIFT').sum())}/{len(lag)}"
        )

    print("\n--- clean-window noise floor (matrix clean + normal_noise, all seeds) ---")
    clean_rows = []
    for seed in SEEDS:
        seed_dir = MULTISEED / f"seed-{seed}"
        manifest = json.load(open(seed_dir / "manifest.json"))
        windows = manifest.get("_windows")
        if windows is None:
            windows = {}
            for entry in manifest["windows"]:
                windows[entry["event_id"]] = pd.read_csv(entry["file"])
        for entry in manifest["windows"]:
            if entry["subtype"] in ("clean", "normal_noise"):
                details = statistical_details_for(reference_df, windows[entry["event_id"]])
                numeric = details.get("numeric", {})
                max_ks = max(
                    (m.get("ks_statistic", 0.0) for m in numeric.values() if isinstance(m, dict)),
                    default=0.0,
                )
                max_psi = max(
                    (m.get("psi", 0.0) for m in numeric.values() if isinstance(m, dict)),
                    default=0.0,
                )
                clean_rows.append({
                    "subtype": entry["subtype"],
                    "max_ks": max_ks,
                    "max_psi": max_psi,
                    "temp_ks": numeric.get("temp", {}).get("ks_statistic", 0.0),
                })
    clean = pd.DataFrame(clean_rows)
    print(
        f"n={len(clean)}  max_ks mean={clean['max_ks'].mean():.3f} "
        f"(p50 {clean['max_ks'].median():.3f}, max {clean['max_ks'].max():.3f})  "
        f"temp_ks mean={clean['temp_ks'].mean():.3f} (max {clean['temp_ks'].max():.3f})  "
        f"max_psi mean={clean['max_psi'].mean():.2f}"
    )

    out = Path("results/phase7/diagnose_mean_shift.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {"rows": rows, "clean_floor": clean.to_dict("records")},
        open(out, "w"),
        indent=2,
        default=str,
    )
    print(f"\nFull per-column detail written to {out}")

if __name__ == "__main__":
    main()
