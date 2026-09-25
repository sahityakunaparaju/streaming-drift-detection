"""
Refresh the multi-seed Phase 7 results after the flagging rule changed
(mean_shift added to expected-false-negative).

The window generator is deterministic per seed (placement + injection
both seeded), and detectors do not read the flag — so only the
`expected_false_negative` column and the summaries derived from it
change. This script:

  1. regenerates each seed's windows / ground_truth.csv / manifest.json
     via build_phase7_set (fast; no detector runs),
  2. recomputes per-seed ladder + subtype summaries from the EXISTING
     rows.csv decisions joined to the NEW flags,
  3. rewrites seed_results.json,
  4. re-runs the standard aggregation (mean +/- std, high-variance
     flags) into results/phase7/multiseed/.

Usage:
    PYTHONPATH=. python experiments/refresh_multiseed_flags.py
"""
import json
from pathlib import Path

import pandas as pd

from drift_injection.phase7_generator import build_phase7_set
from evaluation.metrics import evaluate, lag_summary
from experiments.run_phase7 import (
    ALL_CONFIGS,
    LADDER_ORDER,
    lag_for_rows,
    summarize_rows,
)
from experiments.run_phase7_multiseed import (
    DEFAULT_SEEDS,
    SUBTYPE_DETECTORS,
    aggregate_ladder,
    aggregate_subtype_recall,
    flag_high_variance,
)

OUTPUT_ROOT = Path("results/phase7/multiseed")
INPUT = "data/primary/iot_telemetry_data.csv"

def rebuild_seed(seed):
    """Regenerate one seed's ground truth/manifest and recompute metrics."""
    seed_dir = OUTPUT_ROOT / f"seed-{seed}"
    print(f"  regenerating seed {seed} ground truth/manifest ...")
    manifest = build_phase7_set(
        INPUT,
        output_dir=str(seed_dir / "windows"),
        ground_truth_path=str(seed_dir / "ground_truth.csv"),
        manifest_path=str(seed_dir / "manifest.json"),
        seed=seed,
    )

    rows = pd.read_csv(seed_dir / "rows.csv")
    flag_by_id = {
        entry["event_id"]: entry["expected_false_negative"]
        for entry in manifest["windows"]
    }
    rows["expected_false_negative"] = rows["event_id"].map(flag_by_id)

    ladder = {}
    for name in LADDER_ORDER:
        sub = rows[rows["config"] == name]
        ladder[name] = {
            "metrics": summarize_rows(sub.to_dict("records")),
            "lag": lag_summary(
                lag_for_rows(sub.to_dict("records"))
            ),
        }
    sem = rows[rows["config"] == "semantic"]
    ladder["semantic"] = {
        "metrics": summarize_rows(sem.to_dict("records")),
        "lag": lag_summary(lag_for_rows(sem.to_dict("records"))),
    }

    subtype_recall = {}
    drift_rows = rows[rows["ground_truth"] == "DRIFT"]
    for subtype in sorted(drift_rows["subtype"].unique()):
        subtype_recall[str(subtype)] = {}
        for name, label in SUBTYPE_DETECTORS:
            subset = drift_rows[
                (drift_rows["subtype"] == subtype)
                & (drift_rows["config"] == name)
            ]
            if subset.empty:
                continue
            summary = summarize_rows(subset.to_dict("records"))
            subtype_recall[str(subtype)][label] = {
                "raw_recall": summary["raw"]["recall"],
                "adj_recall": summary["adjusted"]["recall"],
                "n": summary["n"],
                "flagged": summary["flagged"]["n_flagged"],
            }

    seed_results = {
        "seed": seed,
        "test_set": {
            "n_windows": manifest["n_windows"],
            "n_matrix": manifest["n_matrix"],
            "n_lag": manifest["n_lag"],
            "n_expected_false_negative": (
                manifest["n_expected_false_negative"]
            ),
        },
        "ladder": ladder,
        "subtype_recall": subtype_recall,
    }
    with open(seed_dir / "seed_results.json", "w") as file:
        json.dump(seed_results, file, indent=2)
    return seed_results

def main():
    """Rebuild every seed's results with corrected expected-FN flags."""
    seed_results_list = []
    for seed in DEFAULT_SEEDS:
        seed_results_list.append(rebuild_seed(seed))

    ladder_rows = aggregate_ladder(seed_results_list)
    subtype_rows = aggregate_subtype_recall(seed_results_list)
    ladder_flags = flag_high_variance(ladder_rows, "ladder")
    subtype_flags = flag_high_variance(subtype_rows, "subtype")

    def _serializable(row):
        out = dict(row)
        out["per_seed"] = json.dumps(row["per_seed"])
        return out

    pd.DataFrame(
        [_serializable(r) for r in ladder_rows]
    ).to_csv(OUTPUT_ROOT / "aggregate_ladder.csv", index=False)
    pd.DataFrame(
        [_serializable(r) for r in subtype_rows]
    ).to_csv(OUTPUT_ROOT / "aggregate_subtype_recall.csv", index=False)

    flag_rows = ladder_flags + subtype_flags
    flag_columns = [
        "config", "subtype", "detector", "variant", "metric",
        "mean", "std", "rel_std", "min", "max", "seed42",
        "per_seed", "flag",
    ]
    if flag_rows:
        pd.DataFrame(
            [_serializable(r) for r in flag_rows]
        ).to_csv(OUTPUT_ROOT / "high_variance.csv", index=False)
    else:
        pd.DataFrame(columns=flag_columns).to_csv(
            OUTPUT_ROOT / "high_variance.csv", index=False
        )

    summary = {
        "seeds": DEFAULT_SEEDS,
        "n_seeds": len(DEFAULT_SEEDS),
        "canonical_seed": 42,
        "note": (
            "refreshed after mean_shift added to expected-false-"
            "negative flagging; decisions unchanged (deterministic "
            "generator), only flag-derived summaries recomputed"
        ),
        "ladder": ladder_rows,
        "subtype_recall": subtype_rows,
        "high_variance": {
            "ladder": ladder_flags,
            "subtype_recall": subtype_flags,
        },
    }
    with open(OUTPUT_ROOT / "summary.json", "w") as file:
        json.dump(summary, file, indent=2, default=str)

    print("\nRefreshed aggregate tables in", OUTPUT_ROOT)
    print("\nB4 headline:")
    for row in ladder_rows:
        if row["config"] == "B4_unified" and row["variant"] in (
            "raw", "adjusted"
        ) and row["metric"] in ("precision", "recall", "f1", "fpr"):
            print(
                f"  B4 {row['variant']:<8} {row['metric']:<10} "
                f"mean={row['mean']:.4f} +/- {row['std']:.4f}"
            )

if __name__ == "__main__":
    main()
