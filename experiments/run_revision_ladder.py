"""Re-run the B0-B4 ladder on the revised detectors (Step 4b/4c).

Differences from the old Phase 7 sweep:

* Uses the new reference-driven per-column statistical detector
  (configs/reference.yaml + configs/thresholds.yaml v2).
* Reports RAW metrics only. `expected_false_negative` is still recorded
  as an analysis column, but it never removes a window from a metric.
* Adds a per-subtype x severity recall matrix (mean +/- std over the ten
  seeds), including the clean-window false-positive rate.

Outputs (results/revision/):
    seed-<N>/rows.csv                    per-window decisions
    seed-<N>/seed_results.json           per-seed raw metrics + lag
    ladder_raw.csv                       mean/std per config x metric
    subtype_severity_recall.csv          mean/std recall per subtype x severity
    summary.json                         machine-readable aggregate
    REPORT.md                            human-readable report

Usage:
    PYTHONPATH=. python3 experiments/run_revision_ladder.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from drift_injection.phase7_generator import build_phase7_set
from evaluation.metrics import evaluate, lag_summary
from experiments.run_phase7 import (
    load_reference,
    run_detector_over_manifest,
    lag_for_rows,
)

SEEDS = [42, 123, 7, 11, 2024, 314, 2718, 999, 555, 777]
CONFIGS = [
    "B0_null",
    "B1_schema",
    "B2_statistical",
    "B3_schema_statistical",
    "B4_unified",
]
STATISTICAL_SUBTYPES = [
    "mean_shift",
    "variance_shift",
    "distribution_shift",
    "category_shift",
]
SEVERITIES = ["low", "medium", "high"]
OUTPUT_ROOT = Path("results/revision")
METRICS = ["precision", "recall", "f1", "fpr", "accuracy"]


def raw_summary(rows):
    """Raw metrics only (the flag is reported, never used in a denominator)."""
    available = [row for row in rows if row["decision"] != "UNAVAILABLE"]
    y_true = [1 if row["ground_truth"] == "DRIFT" else 0 for row in available]
    y_pred = [1 if row["decision"] == "DRIFT" else 0 for row in available]
    flagged = [row.get("expected_false_negative", "") for row in available]
    result = evaluate(y_true, y_pred, flagged)
    return {
        "n": result["n"],
        "n_positives": result["n_positives"],
        "n_negatives": result["n_negatives"],
        "raw": result["raw"],
        "flagged_analysis_only": result["flagged"],
    }


def run_seed(seed, reference_df, profile, reference_model, raw_df, seed_dir):
    manifest = build_phase7_set(
        "data/primary/iot_telemetry_data.csv",
        output_dir=str(seed_dir / "windows"),
        ground_truth_path=str(seed_dir / "ground_truth.csv"),
        manifest_path=str(seed_dir / "manifest.json"),
        seed=seed,
    )
    rows_by_config = {}
    for name in CONFIGS:
        print(f"  seed {seed}: {name} ...")
        rows_by_config[name] = run_detector_over_manifest(
            name,
            reference_df,
            profile,
            manifest,
            reference_model=reference_model,
            raw_df=raw_df,
        )

    all_rows = []
    for name, rows in rows_by_config.items():
        for row in rows:
            row = dict(row)
            row["config"] = name
            all_rows.append(row)
    pd.DataFrame(all_rows).to_csv(seed_dir / "rows.csv", index=False)

    ladder = {}
    for name in CONFIGS:
        ladder[name] = {
            "metrics": raw_summary(rows_by_config[name]),
            "lag": lag_summary(lag_for_rows(rows_by_config[name])),
        }

    matrix_rows = [
        {**row, "config": name}
        for name in CONFIGS
        for row in rows_by_config[name]
        if row["set"] == "matrix"
    ]
    clean = [row for row in rows_by_config["B4_unified"]
             if row["ground_truth"] == "NO_DRIFT"]
    clean_fp = sum(1 for row in clean if row["decision"] == "DRIFT")
    seed_results = {
        "seed": seed,
        "test_set": {
            "n_windows": manifest["n_windows"],
            "n_matrix": manifest["n_matrix"],
            "n_lag": manifest["n_lag"],
        },
        "ladder": ladder,
        "matrix_rows": [
            {
                "config": row["config"],
                "subtype": row["subtype"],
                "severity": row["severity"],
                "ground_truth": row["ground_truth"],
                "decision": row["decision"],
            }
            for row in matrix_rows
        ],
        "clean_windows": len(clean),
        "clean_false_positives": clean_fp,
        "clean_fpr": (
            clean_fp / len(clean) if clean else None
        ),
    }
    with open(seed_dir / "seed_results.json", "w") as file:
        json.dump(seed_results, file, indent=2)
    return rows_by_config, seed_results


def mean_std(values):
    values = [float(value) for value in values]
    return float(np.mean(values)), float(np.std(values))


def aggregate_ladder(seed_results):
    rows = []
    for config in CONFIGS:
        for metric in METRICS:
            per_seed = [
                seed["ladder"][config]["metrics"]["raw"][metric]
                for seed in seed_results
            ]
            mean, std = mean_std(per_seed)
            rows.append({
                "config": config,
                "metric": metric,
                "mean": mean,
                "std": std,
                "min": min(per_seed),
                "max": max(per_seed),
                "per_seed": per_seed,
            })
        for metric in ["mean", "median", "p95"]:
            per_seed = [
                seed["ladder"][config]["lag"][metric]
                for seed in seed_results
            ]
            numeric = [value for value in per_seed if value is not None]
            if not numeric:
                continue
            mean, std = mean_std(numeric)
            rows.append({
                "config": config,
                "metric": f"lag_{metric}",
                "mean": mean,
                "std": std,
                "min": min(numeric),
                "max": max(numeric),
                "per_seed": per_seed,
            })
    return rows


def aggregate_subtype_severity(seed_results):
    """Recall per subtype x severity x config, mean/std over seeds."""
    rows = []
    for subtype in STATISTICAL_SUBTYPES:
        for severity in SEVERITIES:
            for config in [
                "B2_statistical",
                "B3_schema_statistical",
                "B4_unified",
            ]:
                per_seed = []
                for seed in seed_results:
                    subset = [
                        row for row in seed["matrix_rows"]
                        if row["subtype"] == subtype
                        and row["severity"] == severity
                        and row["config"] == config
                    ]
                    if not subset:
                        continue
                    per_seed.append(
                        1.0 if subset[0]["decision"] == "DRIFT" else 0.0
                    )
                if not per_seed:
                    continue
                mean, std = mean_std(per_seed)
                rows.append({
                    "subtype": subtype,
                    "severity": severity,
                    "config": config,
                    "n_seeds": len(per_seed),
                    "recall_mean": mean,
                    "recall_std": std,
                    "per_seed": per_seed,
                })
    return rows


def format_table(rows, index, columns, value):
    frame = pd.DataFrame(rows)
    if frame.empty:
        return "(no rows)"
    pivot = frame.pivot(index=index, columns=columns, values=value)
    return pivot.round(4).to_string()


def old_before_table():
    """Old v1 raw means from the frozen Phase 7 multi-seed output."""
    path = Path("results/phase7/multiseed/aggregate_ladder.csv")
    if not path.exists():
        return pd.DataFrame()
    old = pd.read_csv(path)
    old = old[(old["variant"] == "raw") & (
        old["metric"].isin(METRICS)
    )][["config", "metric", "mean", "std"]]
    return old.rename(columns={"mean": "mean_v1", "std": "std_v1"})


def build_report(ladder_rows, subtype_rows, seed_results, new_ladder):
    lines = ["# Revision - B0-B4 ladder, raw metrics (10 seeds)", ""]
    lines.append(
        "All metrics are RAW: every window counts, including the windows "
        "tagged `expected_false_negative`. That flag is kept only as an "
        "analysis column and never removes a window from a denominator."
    )
    lines.append("")
    lines.append("## Ladder (mean +/- std over 10 seeds, raw)")
    lines.append("")
    frame = pd.DataFrame(ladder_rows)
    for config in CONFIGS:
        subset = frame[
            (frame["config"] == config)
            & (frame["metric"].isin(METRICS))
        ]
        if subset.empty:
            continue
        pieces = [
            f"{row['metric']}={row['mean']:.3f}+/-{row['std']:.3f}"
            for _, row in subset.iterrows()
        ]
        lines.append(f"- **{config}**: " + ", ".join(pieces))
    lines.append("")
    lines.append("## Clean-window false-positive rate (B4, raw)")
    lines.append("")
    fprs = [seed["clean_fpr"] for seed in seed_results]
    mean, std = mean_std(fprs)
    lines.append(
        f"- mean over seeds: {mean:.4f} +/- {std:.4f} "
        f"(clean windows per seed: {seed_results[0]['clean_windows']})"
    )
    lines.append("")
    lines.append("## Statistical subtype recall by severity (B2, raw)")
    lines.append("")
    subset = [
        row for row in subtype_rows if row["config"] == "B2_statistical"
    ]
    pivot = pd.DataFrame(subset).pivot(
        index="subtype", columns="severity", values="recall_mean"
    )
    lines.append(pivot.round(3).to_string())
    lines.append("")
    lines.append("Counts of detected windows (mean over seeds, out of 10):")
    lines.append("")
    counts = pd.DataFrame(subset).pivot(
        index="subtype", columns="severity", values="recall_mean"
    ) * 10
    lines.append(counts.round(1).to_string())
    lines.append("")
    lines.append("## Before / after (raw means, 10 seeds)")
    lines.append("")
    old = old_before_table()
    if not old.empty:
        new = pd.DataFrame(new_ladder)[
            ["config", "metric", "mean", "std"]
        ].rename(columns={"mean": "mean_v2", "std": "std_v2"})
        merged = old.merge(
            new,
            on=["config", "metric"],
            how="outer",
        )
        merged = merged[merged["config"].isin(CONFIGS)]
        merged["delta"] = merged["mean_v2"] - merged["mean_v1"]
        lines.append(merged.round(4).to_string(index=False))
    else:
        lines.append("(v1 aggregate not found)")
    lines.append("")
    lines.append(
        "Old vs new numbers are not a like-for-like tuning comparison: the "
        "reference sample, reference mode, decision rule and thresholds all "
        "changed. They are reported to show the honest direction of change."
    )
    return "\n".join(lines)


def main():
    global OUTPUT_ROOT

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument(
        "--output",
        default=str(OUTPUT_ROOT),
        help="Output directory (default: results/revision).",
    )
    args = parser.parse_args()
    OUTPUT_ROOT = Path(args.output)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    reference_df, profile, reference_model, raw_df = load_reference()
    seed_results = []
    for seed in args.seeds:
        seed_dir = OUTPUT_ROOT / f"seed-{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== seed {seed} ===")
        _rows, result = run_seed(
            seed,
            reference_df,
            profile,
            reference_model,
            raw_df,
            seed_dir,
        )
        seed_results.append(result)

    ladder_rows = aggregate_ladder(seed_results)
    subtype_rows = aggregate_subtype_severity(seed_results)

    def _dump(rows, path):
        frame = pd.DataFrame(rows)
        if not frame.empty and "per_seed" in frame:
            frame["per_seed"] = frame["per_seed"].apply(json.dumps)
        frame.to_csv(path, index=False)
        return frame

    _dump(ladder_rows, OUTPUT_ROOT / "ladder_raw.csv")
    _dump(subtype_rows, OUTPUT_ROOT / "subtype_severity_recall.csv")

    summary = {
        "seeds": args.seeds,
        "configs": CONFIGS,
        "metrics": METRICS,
        "note": (
            "raw metrics only; expected_false_negative is an analysis "
            "column and never a denominator"
        ),
        "ladder": ladder_rows,
        "subtype_severity_recall": subtype_rows,
        "clean_fpr": [
            {
                "seed": seed["seed"],
                "clean_windows": seed["clean_windows"],
                "false_positives": seed["clean_false_positives"],
                "fpr": seed["clean_fpr"],
            }
            for seed in seed_results
        ],
    }
    with open(
        OUTPUT_ROOT / "summary.json", "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2, default=str)

    report = build_report(ladder_rows, subtype_rows, seed_results, ladder_rows)
    with open(
        OUTPUT_ROOT / "REPORT.md", "w", encoding="utf-8"
    ) as file:
        file.write(report)

    print("\nWrote results to results/revision/")
    print(report)


if __name__ == "__main__":
    main()
