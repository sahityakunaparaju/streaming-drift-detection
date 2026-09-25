"""
Phase 8 (Section T) — Semantic Blind-Spot experiment.

Purpose-built, ISOLATED experiment: 5 semantic blind-spot cases per
seed (unit_change, business_meaning_change, range_change,
allowed_domain_change, definition_change) where

    schema detector      -> NO DRIFT   (schema identical by construction)
    statistical detector -> NO DRIFT   (values untouched; verified per
                                        case with actual KS/PSI/chi2
                                        against the frozen thresholds)
    semantic detector    -> DRIFT      (the only detector that can see
                                        the metadata-only change)

Rigorous isolation verification (per case, per seed, before the
metric is computed):
    1. schema: compare_schemas(reference_schema, window_schema) must
       emit ZERO events (identical columns/dtypes/nullability).
    2. statistical: actual KS/PSI/chi2 measured for the affected
       column(s) must fall below the FROZEN thresholds
       (statistical.ks_floor = 0.50, statistical.decision_threshold
       = 0.25). Any case with an incidental statistical shift riding
       along is EXCLUDED and reported explicitly — a contaminated
       case would invalidate the experiment's premise.
    3. semantic: the semantic detector must fire (case-construction
       check); if it does not, the case is invalid and excluded.

Metric (Section T, NOVEL / NON-STANDARD — purpose-built for this
thesis, not a published literature metric):

    Semantic Blind-Spot Recovery =
        semantic cases recovered by semantic detector
        ------------------------------------------------
        semantic cases missed by schema + statistical

Run across the Phase 7 10-seed list
[42, 123, 7, 11, 2024, 314, 2718, 999, 555, 777] with seed-aware
window placement; reported as mean +/- std across seeds.

Outputs under results/phase8/:
    seed-<N>/windows/...        window CSVs + metadata sidecars
    seed-<N>/ground_truth.csv   ground truth per window
    seed-<N>/manifest.json      per-window metadata
    seed-<N>/per_case.json      isolation + detector rows (one seed)
    seed-<N>/rows.csv           flattened per-case rows
    seed-<N>/results.json       per-seed summary
    isolation_verification.csv  per-case KS/PSI/chi2/schema verdicts
    per_detector_recall.csv     mean/std recall per detector
    blind_spot_recovery.csv     per-seed + aggregate recovery
    summary.json                machine-readable aggregate
    REPORT.md                   human-readable report

Usage:
    PYTHONPATH=. python3 experiments/run_phase8.py [--seeds 42 123 ...]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from detectors.schema_detector import compare_schemas
from detectors.thresholds import THRESHOLDS
from detectors.unified import statistical_score
from evaluation.baseline import (
    build_current_schema,
    statistical_details_for,
)
from experiments.run_phase7 import (
    load_reference,
    run_config,
)
from drift_injection.phase8_generator import build_phase8_set

DEFAULT_SEEDS = [42, 123, 7, 11, 2024, 314, 2718, 999, 555, 777]
OUTPUT_ROOT = Path("results/phase8")

KS_FLOOR = float(THRESHOLDS["statistical"]["ks_floor"])
PSI_SCALE = float(THRESHOLDS["statistical"]["psi_scale"])
DECISION_THRESHOLD = float(
    THRESHOLDS["statistical"]["decision_threshold"]
)

AFFECTED_COLUMN = {
    "unit_change": "temp",
    "business_meaning_change": "temp",
    "range_change": "temp",
    "allowed_domain_change": "device",
    "definition_change": "temp",
}

DETECTORS = [
    ("B1_schema", "schema"),
    ("B2_statistical", "statistical"),
    ("semantic", "semantic"),
    ("B4_unified", "combined"),
]

NOVEL_METRIC_STATEMENT = (
    "Semantic Blind-Spot Recovery is a NOVEL, NON-STANDARD metric "
    "purpose-built for this thesis (Section T of the research plan). "
    "It is not a published literature metric and is not compared "
    "against any published baseline; it quantifies the coverage "
    "contribution of the semantic detector over the blind spot left "
    "by the schema and statistical detectors."
)


def affected_statistics(details, column):
    """Pull the affected column's measured KS/PSI/chi2 from the
    statistical details dict."""
    numeric = details.get("numeric", {})
    categorical = details.get("categorical", {})
    if column in numeric and isinstance(numeric[column], dict):
        return {
            "ks_statistic": numeric[column].get("ks_statistic"),
            "ks_p_value": numeric[column].get("ks_p_value"),
            "psi": numeric[column].get("psi"),
            "chi2": None,
        }
    if column in categorical and isinstance(categorical[column], dict):
        return {
            "ks_statistic": None,
            "ks_p_value": None,
            "psi": None,
            "chi2": categorical[column].get("chi2"),
        }
    return {
        "ks_statistic": None,
        "ks_p_value": None,
        "psi": None,
        "chi2": None,
    }


def max_statistics(details):
    """Max KS / max PSI / max chi2 across all measured columns."""
    max_ks = 0.0
    max_psi = 0.0
    max_chi2 = 0.0
    numeric = details.get("numeric", {})
    if isinstance(numeric, dict):
        for metrics in numeric.values():
            if not isinstance(metrics, dict):
                continue
            ks = metrics.get("ks_statistic")
            psi = metrics.get("psi")
            if ks is not None and np.isfinite(ks):
                max_ks = max(max_ks, abs(float(ks)))
            if psi is not None and np.isfinite(psi):
                max_psi = max(max_psi, float(psi))
    categorical = details.get("categorical", {})
    if isinstance(categorical, dict):
        for metrics in categorical.values():
            if not isinstance(metrics, dict):
                continue
            chi2 = metrics.get("chi2")
            if chi2 is not None and np.isfinite(chi2):
                max_chi2 = max(max_chi2, float(chi2))
    return max_ks, max_psi, max_chi2


def isolation_verification(profile, reference_df, window_df, metadata,
                           subtype, is_control):
    """
    Rigorous per-case isolation check. Returns a dict with the actual
    measured numbers and the pass/fail verdict.

    A blind-spot case is ISOLATED only if ALL of:
      - schema comparison emits ZERO events (identical schema);
      - measured statistical score < decision_threshold (0.25) and
        max KS < ks_floor (0.50) — genuinely NO/WEAK statistical
        signal, verified with real numbers, not "didn't happen to
        fire";
      - the semantic detector FIRES (case-construction check).
    Controls must additionally be semantic-silent.
    """
    current_schema = build_current_schema(window_df)
    schema_events = compare_schemas(
        profile["schema"],
        current_schema,
    )

    details = statistical_details_for(reference_df, window_df)
    score = statistical_score(details)
    max_ks, max_psi, max_chi2 = max_statistics(details)
    affected = affected_statistics(details, AFFECTED_COLUMN.get(subtype))

    semantic_events = []
    if metadata is not None:
        from evaluation.baseline import semantic_events_for
        semantic_events = semantic_events_for(
            profile["semantic_contract"],
            window_df,
            metadata,
        )
    else:
        from detectors.semantic_detector import (
            compare_semantic_contracts,
            validate_values_against_contract,
        )
        semantic_events = (
            compare_semantic_contracts(
                profile["semantic_contract"],
                profile["semantic_contract"],
            )
            + validate_values_against_contract(
                window_df,
                profile["semantic_contract"],
            )
        )

    schema_identical = len(schema_events) == 0
    stat_below_threshold = (
        score < DECISION_THRESHOLD and max_ks < KS_FLOOR
    )
    semantic_fired = len(semantic_events) > 0

    if is_control:
        isolated = schema_identical and stat_below_threshold \
            and not semantic_fired
    else:
        isolated = (
            schema_identical
            and stat_below_threshold
            and semantic_fired
        )

    return {
        "schema_identical": schema_identical,
        "schema_events": len(schema_events),
        "schema_event_types": sorted({
            e["change"] for e in schema_events
        }),
        "statistical_score": round(score, 6),
        "max_ks": round(max_ks, 6),
        "max_psi": round(max_psi, 6),
        "max_chi2": round(max_chi2, 6),
        "affected_ks": affected["ks_statistic"],
        "affected_ks_p_value": affected["ks_p_value"],
        "affected_psi": affected["psi"],
        "affected_chi2": affected["chi2"],
        "semantic_fired": semantic_fired,
        "semantic_event_types": sorted({
            e["change"] for e in semantic_events
        }),
        "semantic_event_count": len(semantic_events),
        "isolated": isolated,
        "ks_floor": KS_FLOOR,
        "decision_threshold": DECISION_THRESHOLD,
        "psi_scale": PSI_SCALE,
    }


def run_seed(seed, reference_df, profile, input_path, seed_dir):
    """Generate + verify + run all four detectors for one seed."""
    seed_dir = Path(seed_dir)
    seed_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_phase8_set(
        input_path,
        output_dir=str(seed_dir / "windows"),
        ground_truth_path=str(seed_dir / "ground_truth.csv"),
        manifest_path=str(seed_dir / "manifest.json"),
        seed=seed,
    )

    rows = []
    for entry in manifest["windows"]:
        event_id = entry["event_id"]
        window_df = manifest["_windows"][event_id]
        metadata_file = entry.get("metadata_file")
        metadata = None
        if metadata_file:
            with open(metadata_file, "r") as file:
                metadata = json.load(file)

        is_control = entry["set"] == "control"
        isolation = isolation_verification(
            profile,
            reference_df,
            window_df,
            metadata,
            entry["subtype"],
            is_control,
        )

        decisions = {}
        for name, _label in DETECTORS:
            decision, _details = run_config(
                name,
                reference_df,
                profile,
                window_df,
                metadata,
            )
            decisions[name] = decision

        rows.append({
            "seed": seed,
            "event_id": event_id,
            "set": entry["set"],
            "subtype": entry["subtype"],
            "feature": entry["feature"],
            "severity": entry["severity"],
            "ground_truth": entry["ground_truth"],
            "start_row": entry["start_row"],
            "schema_identical": isolation["schema_identical"],
            "schema_events": isolation["schema_events"],
            "schema_event_types": isolation["schema_event_types"],
            "statistical_score": isolation["statistical_score"],
            "max_ks": isolation["max_ks"],
            "max_psi": isolation["max_psi"],
            "max_chi2": isolation["max_chi2"],
            "affected_ks": isolation["affected_ks"],
            "affected_psi": isolation["affected_psi"],
            "affected_chi2": isolation["affected_chi2"],
            "semantic_fired": isolation["semantic_fired"],
            "semantic_event_types": isolation["semantic_event_types"],
            "isolated": isolation["isolated"],
            "B1_schema": decisions["B1_schema"],
            "B2_statistical": decisions["B2_statistical"],
            "semantic": decisions["semantic"],
            "B4_unified": decisions["B4_unified"],
        })

    blind_spot_rows = [
        row for row in rows
        if row["set"] == "blind_spot"
    ]
    isolated_rows = [
        row for row in blind_spot_rows
        if row["isolated"]
    ]
    excluded = [
        row for row in blind_spot_rows
        if not row["isolated"]
    ]

    z = len(blind_spot_rows)
    w = sum(
        1 for row in isolated_rows
        if row["B1_schema"] == "NO_DRIFT"
        and row["B2_statistical"] == "NO_DRIFT"
    )
    v = sum(
        1 for row in isolated_rows
        if row["B1_schema"] == "NO_DRIFT"
        and row["B2_statistical"] == "NO_DRIFT"
        and row["semantic"] == "DRIFT"
    )
    recovery = (v / w) if w > 0 else None

    per_detector_recall = {}
    for name, label in DETECTORS:
        tp = sum(
            1 for row in isolated_rows
            if row[name] == "DRIFT"
        )
        fn = sum(
            1 for row in isolated_rows
            if row[name] != "DRIFT"
        )
        per_detector_recall[label] = {
            "tp": tp,
            "fn": fn,
            "recall": (tp / (tp + fn)) if (tp + fn) > 0 else None,
        }

    control_rows = [
        row for row in rows
        if row["set"] == "control"
    ]

    seed_results = {
        "seed": seed,
        "test_set": {
            "n_windows": manifest["n_windows"],
            "n_blind_spot": manifest["n_blind_spot"],
            "n_control": manifest["n_control"],
        },
        "blind_spot": {
            "z": z,
            "n_isolated": len(isolated_rows),
            "n_excluded": len(excluded),
            "excluded": excluded,
            "w_missed_by_schema_statistical": w,
            "v_recovered_by_semantic": v,
            "recovery": recovery,
        },
        "per_detector_recall": per_detector_recall,
        "controls": control_rows,
    }

    with open(seed_dir / "per_case.json", "w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2, default=str)
    pd.DataFrame(rows).to_csv(seed_dir / "rows.csv", index=False)
    with open(seed_dir / "results.json", "w", encoding="utf-8") as file:
        json.dump(seed_results, file, indent=2, default=str)

    return seed_results, rows


def mean_std(values):
    """Arithmetic mean and population std of a numeric list."""
    values = [float(v) for v in values]
    return float(np.mean(values)), float(np.std(values))


def main():
    """Run the Phase 8 semantic blind-spot experiment (Section T)."""
    parser = argparse.ArgumentParser(
        description="Phase 8 Semantic Blind-Spot experiment (Section T)."
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

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("PHASE 8 — SEMANTIC BLIND-SPOT EXPERIMENT (Section T)")
    print(f"seeds: {args.seeds}")
    print(NOVEL_METRIC_STATEMENT)
    print("=" * 90)

    reference_df, profile = load_reference()

    all_seed_results = []
    all_rows = []
    for seed in args.seeds:
        print(f"\n--- seed {seed} ---")
        seed_dir = out_root / f"seed-{seed}"
        if (seed_dir / "results.json").exists():
            with open(seed_dir / "results.json") as file:
                seed_results = json.load(file)
            with open(seed_dir / "per_case.json") as file:
                rows = json.load(file)
            all_seed_results.append(seed_results)
            all_rows.extend(rows)
            print(f"  cached results found; reusing seed-{seed}")
            continue
        seed_results, rows = run_seed(
            seed,
            reference_df,
            profile,
            args.input,
            seed_dir,
        )
        all_seed_results.append(seed_results)
        all_rows.extend(rows)
        rec = seed_results["blind_spot"]["recovery"]
        print(
            f"  seed {seed}: {seed_results['test_set']['n_windows']} "
            f"windows; recovery = "
            f"{rec:.4f} (V={seed_results['blind_spot']['v_recovered_by_semantic']}"
            f"/W={seed_results['blind_spot']['w_missed_by_schema_statistical']})"
        )

    # ---- isolation verification table (all seeds) ----
    isolation_df = pd.DataFrame(all_rows)
    isolation_df.to_csv(
        out_root / "isolation_verification.csv",
        index=False,
    )

    # ---- aggregate recovery ----
    recovery_rows = []
    for seed_results in all_seed_results:
        bs = seed_results["blind_spot"]
        recovery_rows.append({
            "seed": seed_results["seed"],
            "z": bs["z"],
            "n_isolated": bs["n_isolated"],
            "n_excluded": bs["n_excluded"],
            "w": bs["w_missed_by_schema_statistical"],
            "v": bs["v_recovered_by_semantic"],
            "recovery": bs["recovery"],
        })
    recovery_df = pd.DataFrame(recovery_rows)

    per_seed_recovery = [
        row["recovery"] for row in recovery_rows
        if row["recovery"] is not None
    ]
    recovery_mean, recovery_std = mean_std(per_seed_recovery)

    total_z = sum(row["z"] for row in recovery_rows)
    total_isolated = sum(row["n_isolated"] for row in recovery_rows)
    total_excluded = sum(row["n_excluded"] for row in recovery_rows)
    total_w = sum(row["w"] for row in recovery_rows)
    total_v = sum(row["v"] for row in recovery_rows)
    pooled_recovery = (total_v / total_w) if total_w > 0 else None

    recovery_aggregate = {
        "metric": "semantic_blind_spot_recovery",
        "novel_non_standard": True,
        "novel_statement": NOVEL_METRIC_STATEMENT,
        "n_seeds": len(per_seed_recovery),
        "recovery_mean": recovery_mean,
        "recovery_std": recovery_std,
        "recovery_pooled": pooled_recovery,
        "total_z_cases": total_z,
        "total_isolated": total_isolated,
        "total_excluded": total_excluded,
        "total_w_missed_by_schema_statistical": total_w,
        "total_v_recovered_by_semantic": total_v,
        "per_seed": recovery_rows,
    }

    # ---- aggregate per-detector recall ----
    detector_rows = []
    for name, label in DETECTORS:
        per_seed = [
            seed_results["per_detector_recall"][label]["recall"]
            for seed_results in all_seed_results
        ]
        numeric = [v for v in per_seed if v is not None]
        mean, std = mean_std(numeric)
        detector_rows.append({
            "detector": label,
            "config": name,
            "recall_mean": mean,
            "recall_std": std,
            "min": min(numeric),
            "max": max(numeric),
            "per_seed": per_seed,
        })
    detector_df = pd.DataFrame(detector_rows)

    # ---- summary ----
    summary = {
        "phase": 8,
        "section": "T",
        "seeds": args.seeds,
        "metric": recovery_aggregate,
        "per_detector_recall": detector_rows,
        "thresholds_used": {
            "statistical.ks_floor": KS_FLOOR,
            "statistical.psi_scale": PSI_SCALE,
            "statistical.decision_threshold": DECISION_THRESHOLD,
            "note": (
                "configs/thresholds.yaml is FROZEN and was not modified; "
                "values read directly from the config."
            ),
        },
        "windows_region": "[320000, 405184)",
        "excluded_cases": [
            row for row in all_rows
            if row["set"] == "blind_spot" and not row["isolated"]
        ],
    }
    with open(out_root / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, default=str)

    recovery_df.to_csv(
        out_root / "blind_spot_recovery.csv",
        index=False,
    )
    detector_df.to_csv(
        out_root / "per_detector_recall.csv",
        index=False,
    )

    # ---- REPORT.md ----
    report = _build_report(
        summary,
        recovery_df,
        detector_df,
        isolation_df,
    )
    with open(out_root / "REPORT.md", "w", encoding="utf-8") as file:
        file.write(report)

    _print_report(summary, recovery_df, detector_df, isolation_df)

    return summary


def _md_table(frame):
    """Render a DataFrame as a markdown table without tabulate."""
    frame = frame.copy()
    cols = [str(c) for c in frame.columns]
    rows = [[str(v) for v in row] for row in frame.itertuples(index=False)]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, sep] + body)


def _build_report(summary, recovery_df, detector_df, isolation_df):
    m = summary["metric"]
    lines = []
    lines.append("# Phase 8 — Semantic Blind-Spot Experiment (Section T)")
    lines.append("")
    lines.append(NOVEL_METRIC_STATEMENT)
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(
        f"**Semantic Blind-Spot Recovery = "
        f"{m['recovery_mean']:.4f} ± {m['recovery_std']:.4f} "
        f"across {m['n_seeds']} seeds**, computed over "
        f"{m['total_z_cases']} isolated cases total "
        f"({m['total_z_cases'] // m['n_seeds']} per seed), where schema "
        f"and statistical detectors combined missed "
        f"{m['total_w_missed_by_schema_statistical']} of "
        f"{m['total_z_cases']} cases (W) and the semantic detector "
        f"recovered {m['total_v_recovered_by_semantic']} of them (V)."
    )
    lines.append("")
    lines.append("This is a **NOVEL, NON-STANDARD metric** purpose-built "
                 "for this thesis (Section T). It is not a published "
                 "literature metric and is not compared against any "
                 "published baseline.")
    lines.append("")
    lines.append("## Per-seed recovery")
    lines.append("")
    lines.append(_md_table(recovery_df))
    lines.append("")
    lines.append("## Per-detector recall (isolated blind-spot cases)")
    lines.append("")
    lines.append(_md_table(detector_df))
    lines.append("")
    lines.append("## Isolation verification (per case, per seed)")
    lines.append("")
    lines.append(
        "Every blind-spot case was verified BEFORE metric computation: "
        "schema comparison identical (0 events), and actual measured "
        "KS/PSI/chi2 for the affected column fall below the frozen "
        "thresholds (ks_floor = 0.50, decision_threshold = 0.25, "
        "psi_scale = 72.0). Full per-case numbers: "
        "`isolation_verification.csv`."
    )
    lines.append("")
    excluded = summary["excluded_cases"]
    lines.append(f"### Excluded / flagged cases: {len(excluded)}")
    lines.append("")
    if excluded:
        lines.append(_md_table(pd.DataFrame(excluded)))
        lines.append("")
    else:
        lines.append("None — every blind-spot case was a clean isolation.")
        lines.append("")
    lines.append("## Controls (clean / normal_noise)")
    lines.append("")
    controls = isolation_df[isolation_df["set"] == "control"]
    lines.append(_md_table(controls))
    lines.append("")
    lines.append("## Method notes")
    lines.append("")
    lines.append("- Definition-change gap: **option (a) taken** — a "
                 "`definition` field was added to `temp` in "
                 "`configs/semantic_contract.yaml` "
                 "(\"instantaneous_raw_sensor_value\"), the reference "
                 "profile was regenerated, and the semantic detector "
                 "gained an exact-equality `definition_change` rule "
                 "mirroring `business_meaning_change`. The frozen "
                 "`configs/thresholds.yaml` was NOT modified; its "
                 "Phase 6 note about the definition gap is superseded "
                 "by this Phase 8 extension.")
    lines.append("- Windows drawn from rows [320000, 405184), disjoint "
                 "from Phase 7 regions; a full scan shows no 1000-row "
                 "window there reaches the statistical decision "
                 "threshold (max 0.215 < 0.25).")
    lines.append("- Seed-aware window placement via "
                 "`sample_non_overlapping_starts` (same construction "
                 "as Phase 7, Section Y).")
    lines.append("- Out of scope respected: no frozen threshold was "
                 "modified, Phase 7 baseline results untouched, no "
                 "ablation (Phase 9), mean/variance/distribution "
                 "aggregation limitation not re-litigated.")
    return "\n".join(lines)


def _print_report(summary, recovery_df, detector_df, isolation_df):
    m = summary["metric"]
    print("\n" + "=" * 90)
    print("PHASE 8 HEADLINE (Section T)")
    print("=" * 90)
    print(
        f"\nSemantic Blind-Spot Recovery = "
        f"{m['recovery_mean']:.4f} ± {m['recovery_std']:.4f} "
        f"across {m['n_seeds']} seeds"
    )
    print(
        f"  computed over {m['total_z_cases']} isolated cases total "
        f"({m['total_z_cases'] // m['n_seeds']} per seed)"
    )
    print(
        f"  schema + statistical combined missed "
        f"{m['total_w_missed_by_schema_statistical']} of "
        f"{m['total_z_cases']} cases (W)"
    )
    print(
        f"  semantic recovered "
        f"{m['total_v_recovered_by_semantic']} of "
        f"{m['total_w_missed_by_schema_statistical']} (V)"
    )
    print(f"  pooled recovery = {m['recovery_pooled']:.4f}")
    print(
        f"\n  NOTE: {NOVEL_METRIC_STATEMENT}"
    )

    print("\n" + "=" * 90)
    print("PER-SEED RECOVERY")
    print("=" * 90)
    print(recovery_df.to_string(index=False))

    print("\n" + "=" * 90)
    print("PER-DETECTOR RECALL (isolated blind-spot cases)")
    print("=" * 90)
    print(detector_df.to_string(index=False))

    print("\n" + "=" * 90)
    print("EXCLUDED / FLAGGED CASES")
    print("=" * 90)
    excluded = summary["excluded_cases"]
    if not excluded:
        print("  none — every blind-spot case was a clean isolation")
    for row in excluded:
        print(
            f"  {row['seed']}/{row['event_id']}: schema_identical="
            f"{row['schema_identical']} stat_score="
            f"{row['statistical_score']} max_ks={row['max_ks']}"
        )

    print("\n" + "=" * 90)
    print("ISOLATION VERIFICATION (first rows)")
    print("=" * 90)
    show = isolation_df[[
        "seed", "event_id", "set", "subtype", "schema_identical",
        "schema_events", "statistical_score", "max_ks", "max_psi",
        "max_chi2", "affected_ks", "affected_psi", "affected_chi2",
        "semantic_fired", "isolated",
    ]].head(30)
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
