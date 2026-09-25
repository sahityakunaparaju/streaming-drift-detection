"""
Phase 7 multi-seed robustness sweep (Section Y).

Runs the exact Phase 7 harness (build_phase7_set -> run_config ->
evaluate / lag) under multiple RNG seeds, with NO changes to frozen
thresholds or injection logic. Produces per-seed raw results and
aggregated mean +/- std tables for the baseline ladder and the
subtype-recall breakdown, plus a high-variance flag list.

Outputs under results/phase7/multiseed/:
    seed-<N>/seed_results.json   full per-seed metrics + lag
    seed-<N>/rows.csv            per-window decisions for every config
    seed-<N>/manifest.json       per-seed manifest (incl. window count)
    aggregate_ladder.csv         mean/std per config x metric (raw+adj)
    aggregate_subtype_recall.csv mean/std per subtype x detector
    high_variance.csv            cells with std > 20% of mean
    summary.json                 machine-readable aggregate
    REPORT.md                    human-readable report

Usage:
    PYTHONPATH=. python3 experiments/run_phase7_multiseed.py \
        --seeds 42 123 7 11 2024 314 2718 999 555 777

The default seed list includes the canonical seed 42.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from drift_injection.phase7_generator import build_phase7_set
from evaluation.metrics import lag_summary
from experiments.run_phase7 import (
    ALL_CONFIGS,
    LADDER_ORDER,
    load_reference,
    run_config,
    run_detector_over_manifest,
    summarize_rows,
    lag_for_rows,
)

DEFAULT_SEEDS = [42, 123, 7, 11, 2024, 314, 2718, 999, 555, 777]
OUTPUT_ROOT = Path("results/phase7/multiseed")

LADDER_METRICS = ["precision", "recall", "f1", "fpr"]
LAG_METRICS = ["mean", "median", "p95"]

SUBTYPE_DETECTORS = [
    ("B1_schema", "schema"),
    ("B2_statistical", "statistical"),
    ("semantic", "semantic"),
    ("B4_unified", "combined"),
]

VARIANCE_RATIO_THRESHOLD = 0.2

def run_seed(seed, reference_df, profile, input_path, seed_dir):
    """Run the full harness for one seed; return (rows_by_config,
    manifest, per-seed summary dict)."""
    seed_dir = Path(seed_dir)
    seed_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_phase7_set(
        input_path,
        output_dir=str(seed_dir / "windows"),
        ground_truth_path=str(seed_dir / "ground_truth.csv"),
        manifest_path=str(seed_dir / "manifest.json"),
        seed=seed,
    )

    rows_by_config = {}
    for name in ALL_CONFIGS:
        print(f"  seed {seed}: running {name} ...")
        rows_by_config[name] = run_detector_over_manifest(
            name,
            reference_df,
            profile,
            manifest,
        )

    all_rows = []
    for name, rows in rows_by_config.items():
        for row in rows:
            row = dict(row)
            row["config"] = name
            all_rows.append(row)
    pd.DataFrame(all_rows).to_csv(
        seed_dir / "rows.csv",
        index=False,
    )

    ladder = {}
    for name in LADDER_ORDER:
        rows = rows_by_config[name]
        summary = summarize_rows(rows)
        lags = lag_for_rows(rows)
        ladder[name] = {
            "metrics": summary,
            "lag": lag_summary(lags),
        }

    ladder["semantic"] = {
        "metrics": summarize_rows(rows_by_config["semantic"]),
        "lag": lag_summary(
            lag_for_rows(rows_by_config["semantic"])
        ),
    }

    subtype_recall = {}
    for subtype in sorted({
        row["subtype"]
        for row in rows_by_config["B4_unified"]
        if row["ground_truth"] == "DRIFT"
    }):
        subtype_recall[subtype] = {}
        for name, label in SUBTYPE_DETECTORS:
            rows = rows_by_config[name]
            subset = [
                row for row in rows
                if row["subtype"] == subtype
                and row["ground_truth"] == "DRIFT"
            ]
            if not subset:
                continue
            summary = summarize_rows(subset)
            subtype_recall[subtype][label] = {
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

    return rows_by_config, manifest, seed_results

def mean_std(values):
    """Arithmetic mean and population std of a numeric list."""
    values = [float(v) for v in values]
    return float(np.mean(values)), float(np.std(values))

def aggregate_ladder(seed_results_list):
    """Per config x metric: mean/std across seeds (raw + adjusted),
    plus per-seed value list. Lag mean/median/p95 aggregated too."""
    configs = LADDER_ORDER + ["semantic"]
    rows = []
    for config in configs:
        for variant in ["raw", "adjusted"]:
            for metric in LADDER_METRICS:
                per_seed = [
                    seed_res["ladder"][config]["metrics"][variant][metric]
                    for seed_res in seed_results_list
                ]
                mean, std = mean_std(per_seed)
                rows.append({
                    "config": config,
                    "variant": variant,
                    "metric": metric,
                    "mean": mean,
                    "std": std,
                    "rel_std": (std / mean) if mean != 0 else None,
                    "min": min(per_seed),
                    "max": max(per_seed),
                    "seed42": per_seed[
                        next(i for i, s in enumerate(
                            seed_results_list
                        ) if s["seed"] == 42)
                    ] if any(
                        s["seed"] == 42
                        for s in seed_results_list
                    ) else None,
                    "per_seed": per_seed,
                })
        for metric in LAG_METRICS:
            per_seed = [
                seed_res["ladder"][config]["lag"][metric]
                for seed_res in seed_results_list
            ]
            if all(v is None for v in per_seed):
                continue
            numeric = [v for v in per_seed if v is not None]
            mean, std = mean_std(numeric)
            rows.append({
                "config": config,
                "variant": "lag",
                "metric": f"lag_{metric}",
                "mean": mean,
                "std": std,
                "rel_std": (std / mean) if mean != 0 else None,
                "min": min(numeric),
                "max": max(numeric),
                "seed42": per_seed[
                    next(i for i, s in enumerate(
                        seed_results_list
                    ) if s["seed"] == 42)
                ] if any(
                    s["seed"] == 42
                    for s in seed_results_list
                ) else None,
                "per_seed": per_seed,
            })
    return rows

def aggregate_subtype_recall(seed_results_list):
    """Per subtype x detector: mean/std of raw and adjusted recall."""
    rows = []
    subtypes = sorted({
        subtype
        for seed_res in seed_results_list
        for subtype in seed_res["subtype_recall"]
    })
    for subtype in subtypes:
        for name, label in SUBTYPE_DETECTORS:
            for variant in ["raw_recall", "adj_recall"]:
                per_seed = []
                for seed_res in seed_results_list:
                    cell = seed_res["subtype_recall"].get(subtype, {})
                    entry = cell.get(label)
                    if entry is None:
                        per_seed.append(None)
                    else:
                        per_seed.append(entry[variant])
                numeric = [v for v in per_seed if v is not None]
                if not numeric:
                    continue
                mean, std = mean_std(numeric)
                rows.append({
                    "subtype": subtype,
                    "detector": label,
                    "variant": variant,
                    "mean": mean,
                    "std": std,
                    "rel_std": (std / mean) if mean != 0 else None,
                    "min": min(numeric),
                    "max": max(numeric),
                    "per_seed": per_seed,
                })
    return rows

def flag_high_variance(aggregate_rows, metric_kind):
    """Flag cells where std > 20% of the (positive) mean, or where
    std > 0 on a zero mean (unstable from zero)."""
    flagged = []
    for row in aggregate_rows:
        mean, std = row["mean"], row["std"]
        rel = row["rel_std"]
        if mean != 0 and rel is not None and rel > VARIANCE_RATIO_THRESHOLD:
            flagged.append({
                **{k: row[k] for k in
                   ("config" if metric_kind == "ladder" else "subtype",
                    ) if k in row},
                **{k: row[k] for k in
                   ("detector", "variant", "metric", "mean", "std",
                    "rel_std", "min", "max", "seed42", "per_seed")
                   if k in row},
                "flag": "std > 20% of mean",
            })
        elif mean == 0 and std > 0:
            flagged.append({
                **{k: row[k] for k in
                   ("config" if metric_kind == "ladder" else "subtype",
                    ) if k in row},
                **{k: row[k] for k in
                   ("detector", "variant", "metric", "mean", "std",
                    "rel_std", "min", "max", "seed42", "per_seed")
                   if k in row},
                "flag": "nonzero std on zero mean",
            })
    return flagged

def main():
    """Run the Phase 7 multi-seed robustness sweep (Section Y)."""
    parser = argparse.ArgumentParser(
        description="Phase 7 multi-seed robustness sweep (Section Y)."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="List of seeds (default: %(default)s).",
    )
    parser.add_argument(
        "--input",
        default="data/primary/iot_telemetry_data.csv",
    )
    parser.add_argument(
        "--out",
        default=str(OUTPUT_ROOT),
    )
    args = parser.parse_args()

    seeds = args.seeds
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("PHASE 7 MULTI-SEED ROBUSTNESS SWEEP (Section Y)")
    print(f"seeds: {seeds}")
    print("=" * 90)

    reference_df, profile = load_reference()

    seed_results_list = []
    for seed in seeds:
        print(f"\n--- seed {seed} ---")
        seed_dir = out_root / f"seed-{seed}"
        if (seed_dir / "seed_results.json").exists():
            with open(seed_dir / "seed_results.json") as file:
                seed_results_list.append(json.load(file))
            print(f"  cached results found; reusing seed-{seed}")
            continue
        _rows, _manifest, seed_results = run_seed(
            seed,
            reference_df,
            profile,
            args.input,
            seed_dir,
        )
        seed_results_list.append(seed_results)
        print(
            f"  seed {seed}: {seed_results['test_set']['n_windows']} "
            "windows done."
        )

    ladder_rows = aggregate_ladder(seed_results_list)
    subtype_rows = aggregate_subtype_recall(seed_results_list)

    ladder_flag_rows = flag_high_variance(ladder_rows, "ladder")
    subtype_flag_rows = flag_high_variance(subtype_rows, "subtype")

    def _serializable(row):
        out = dict(row)
        out["per_seed"] = json.dumps(row["per_seed"])
        return out

    pd.DataFrame(
        [_serializable(r) for r in ladder_rows]
    ).to_csv(out_root / "aggregate_ladder.csv", index=False)
    pd.DataFrame(
        [_serializable(r) for r in subtype_rows]
    ).to_csv(out_root / "aggregate_subtype_recall.csv", index=False)
    flag_rows = ladder_flag_rows + subtype_flag_rows
    flag_columns = [
        "config", "subtype", "detector", "variant", "metric",
        "mean", "std", "rel_std", "min", "max", "seed42",
        "per_seed", "flag",
    ]
    if flag_rows:
        pd.DataFrame(
            [_serializable(r) for r in flag_rows]
        ).to_csv(
            out_root / "high_variance.csv",
            index=False,
        )
    else:
        pd.DataFrame(columns=flag_columns).to_csv(
            out_root / "high_variance.csv",
            index=False,
        )

    summary = {
        "seeds": seeds,
        "n_seeds": len(seeds),
        "canonical_seed": 42 if 42 in seeds else None,
        "variance_ratio_threshold": VARIANCE_RATIO_THRESHOLD,
        "ladder": ladder_rows,
        "subtype_recall": subtype_rows,
        "high_variance": {
            "ladder": ladder_flag_rows,
            "subtype_recall": subtype_flag_rows,
        },
    }
    with open(out_root / "summary.json", "w") as file:
        json.dump(summary, file, indent=2, default=str)

    print(f"\nWrote per-seed + aggregate outputs to {out_root}")

    print("\n" + "=" * 90)
    print("B4 (unified) HEADLINE ACROSS SEEDS")
    print("=" * 90)
    for row in ladder_rows:
        if row["config"] == "B4_unified" and (
            row["variant"] == "raw" or row["variant"] == "adjusted"
        ) and row["metric"] in ("precision", "recall", "f1", "fpr"):
            print(
                f"  B4 {row['variant']:<8} {row['metric']:<10} "
                f"mean={row['mean']:.4f} +/- {row['std']:.4f} "
                f"(seed42={row['seed42']}) "
                f"min={row['min']:.4f} max={row['max']:.4f}"
            )

    print("\n" + "=" * 90)
    print("HIGH-VARIANCE CELLS (std > 20% of mean)")
    print("=" * 90)
    if not (ladder_flag_rows + subtype_flag_rows):
        print("  none")
    for row in ladder_flag_rows:
        print(
            f"  LADDER {row['config']} {row['variant']} "
            f"{row['metric']}: mean={row['mean']:.4f} "
            f"std={row['std']:.4f} rel={row['rel_std']:.2f} "
            f"[{row['flag']}]"
        )
    for row in subtype_flag_rows:
        print(
            f"  SUBTYPE {row['subtype']}/{row['detector']} "
            f"{row['variant']}: mean={row['mean']:.4f} "
            f"std={row['std']:.4f} rel={row['rel_std']:.2f} "
            f"[{row['flag']}]"
        )

if __name__ == "__main__":
    main()
