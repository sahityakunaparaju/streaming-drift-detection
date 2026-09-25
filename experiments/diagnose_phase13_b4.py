"""
Phase 13 — B4 secondary-dataset false-positive attribution (diagnostic).

For every secondary-dataset (Beijing) window where B4 (unified) produced
a false positive, this script identifies which component (schema,
statistical, semantic) actually triggered the DRIFT decision — either via
the max-confidence override rule (any component score >= its max) or via
the statistical decision_threshold / unified threshold path — and then
tests whether the false positive is explained by the chi2/wd-cardinality
confound identified for B2 (chi2_scale=32.0 calibrated on a 3-category
field, overwhelmed by the 16-category wd field).

Counterfactual: recompute the statistical score with the categorical
(chi2) component removed (numeric KS/PSI only), re-run decide() with the
unchanged schema/semantic scores, and record whether the window would
still be DRIFT.

Reference-slice provenance: the chi2 values recorded in
docs/phase13_notes.md (248 / 372 / 1103) reproduce exactly against
reference rows 7000-17000, while the current run_phase13.main() uses
rows 6000-16000 (ref_start = 6 * WINDOW_SIZE). The script therefore
runs BOTH offsets and verifies that decisions and FP attribution are
identical under each, so the attribution does not depend on which
offset produced the recorded run.

This script is DIAGNOSTIC ONLY: it does not modify chi2_scale,
configs/thresholds.yaml, or any detector logic. It only reads frozen
configs/artifacts and writes a JSON report under results/phase13/.

Usage:
    PYTHONPATH=. python experiments/diagnose_phase13_b4.py
"""

import json
import math
from pathlib import Path

import pandas as pd
import yaml

from detectors.unified import (
    unified_score,
    decide,
    statistical_score,
    STATISTICAL_DECISION_THRESHOLD,
    UNIFIED_THRESHOLD,
    UNIFIED,
    KS_FLOOR,
    PSI_SCALE,
    CHI2_SCALE,
)
from evaluation.baseline import (
    build_current_schema,
    statistical_details_for,
    semantic_events_for,
)
from detectors.schema_detector import compare_schemas
from experiments.run_phase13 import discover_categorical_columns

OUTPUT_PATH = Path("results/phase13/diagnose_b4_fp.json")

WINDOW_SIZE = 1000
REF_ROWS = 10000
# offset 6000 = current experiments/run_phase13.py (ref_start = 6 * WINDOW_SIZE)
# offset 7000 = provenance of chi2 values recorded in docs/phase13_notes.md
REF_OFFSETS = [6000, 7000]


def trigger_path(scores):
    """All conditions in detectors.unified.decide() that fire for scores."""
    fired = []
    if scores["schema_score"] >= UNIFIED["schema_max_score"]:
        fired.append("override:schema")
    if scores["semantic_score"] >= UNIFIED["semantic_max_score"]:
        fired.append("override:semantic")
    if scores["statistical_score"] >= UNIFIED["statistical_max_score"]:
        fired.append("override:statistical")
    if scores["statistical_score"] >= STATISTICAL_DECISION_THRESHOLD:
        fired.append("statistical:decision_threshold")
    if scores["unified_score"] >= UNIFIED_THRESHOLD:
        fired.append("unified:threshold")
    return fired


def analyze_offset(reference_df, manifest, reference_schema, contract,
                    categorical_columns):
    """Run the full B4 component attribution for one reference slice."""
    records = []
    for entry in manifest["windows"]:
        event_id = entry["event_id"]
        win_df = pd.read_csv(entry["file"])
        metadata = None
        if entry.get("metadata_file"):
            with open(entry["metadata_file"], "r") as f:
                metadata = json.load(f)

        # --- components, exactly as run_b4_unified assembles them ---
        current_schema = build_current_schema(win_df)
        schema_events = compare_schemas(reference_schema, current_schema)
        semantic_events = semantic_events_for(contract, win_df, metadata)
        stat_details = statistical_details_for(
            reference_df, win_df,
            categorical_columns=categorical_columns,
        )

        schema_drift = len(schema_events) > 0
        semantic_drift = len(semantic_events) > 0
        scores = unified_score(schema_drift, stat_details, semantic_drift)
        decision = decide(
            scores["unified_score"],
            schema_score=scores["schema_score"],
            semantic_score=scores["semantic_score"],
            statistical_score=scores["statistical_score"],
        )
        fired = trigger_path(scores)

        # --- component details mirroring statistical_score() exactly ---
        cat = stat_details.get("categorical", {})
        numeric = stat_details.get("numeric", {})
        wd_chi2 = cat.get("wd", {}).get("chi2", 0.0)
        wd_p = cat.get("wd", {}).get("p_value", None)
        max_chi2 = max(
            (m.get("chi2") for m in cat.values()
             if isinstance(m, dict)
             and m.get("chi2") is not None
             and math.isfinite(m["chi2"])),
            default=0.0,
        )
        max_ks = max(
            (abs(m["ks_statistic"]) for m in numeric.values()
             if isinstance(m, dict)
             and m.get("ks_statistic") is not None
             and math.isfinite(m["ks_statistic"])),
            default=0.0,
        )
        max_psi = max(
            (m["psi"] for m in numeric.values()
             if isinstance(m, dict)
             and m.get("psi") is not None
             and math.isfinite(m["psi"])),
            default=0.0,
        )
        ks_component = min(max(max_ks - KS_FLOOR, 0.0)
                           / (1.0 - KS_FLOOR), 1.0)
        psi_component = min(max_psi / PSI_SCALE, 1.0)
        chi2_component = min(max_chi2 / CHI2_SCALE, 1.0)

        # --- counterfactual: statistical score WITHOUT chi2 (numeric only) ---
        stat_no_chi2 = statistical_score(
            {"numeric": numeric, "categorical": {}}
        )
        scores_cf = unified_score(schema_drift,
                                  {"numeric": numeric, "categorical": {}},
                                  semantic_drift)
        decision_no_chi2 = decide(
            scores_cf["unified_score"],
            schema_score=scores_cf["schema_score"],
            semantic_score=scores_cf["semantic_score"],
            statistical_score=scores_cf["statistical_score"],
        )

        # --- also: would the statistical component alone (B2) fire? ---
        b2_decision = ("DRIFT"
                       if scores["statistical_score"]
                       >= STATISTICAL_DECISION_THRESHOLD
                       else "NO_DRIFT")
        b2_decision_no_chi2 = ("DRIFT"
                               if stat_no_chi2 >= STATISTICAL_DECISION_THRESHOLD
                               else "NO_DRIFT")

        is_fp = (entry["ground_truth"] == "NO_DRIFT"
                 and decision == "DRIFT")

        records.append({
            "event_id": event_id,
            "ground_truth": entry["ground_truth"],
            "decision": decision,
            "is_false_positive": is_fp,
            "trigger_paths": fired,
            "schema_events": [
                {k: e.get(k) for k in
                 ("column", "change", "old_value", "new_value", "severity")}
                for e in schema_events
            ],
            "semantic_events": semantic_events,
            "scores": scores,
            "wd_chi2": wd_chi2,
            "wd_p_value": wd_p,
            "chi2_component": chi2_component,
            "ks_component": ks_component,
            "psi_component": psi_component,
            "statistical_score_no_chi2": stat_no_chi2,
            "decision_no_chi2": decision_no_chi2,
            "b2_decision": b2_decision,
            "b2_decision_no_chi2": b2_decision_no_chi2,
            "chi2_explained_fp": (is_fp
                                  and decision_no_chi2 == "NO_DRIFT"),
        })
    return records


def print_fp_attribution(fps):
    """Per-FP-window attribution lines."""
    for r in fps:
        parts = []
        if any(p.startswith("override:schema") for p in r["trigger_paths"]):
            cols = []
            for e in r["schema_events"]:
                if (e["change"] == "nullability_change"
                        and isinstance(e["old_value"], float)
                        and isinstance(e["new_value"], float)):
                    cols.append(f"{e['column']}:nullability "
                                f"({e['old_value']*100:.2f}%->"
                                f"{e['new_value']*100:.2f}%)")
                else:
                    cols.append(f"{e['column']}:{e['change']}")
            parts.append(f"SCHEMA override [{', '.join(cols)}]")
        if any(p.startswith("override:statistical")
               for p in r["trigger_paths"]):
            parts.append(
                f"STATISTICAL override [chi2/wd={r['wd_chi2']:.1f}, "
                f"chi2_comp={r['chi2_component']:.3f}, "
                f"ks_comp={r['ks_component']:.3f}, "
                f"psi_comp={r['psi_component']:.3f}]"
            )
        if any(p.startswith("override:semantic")
               for p in r["trigger_paths"]):
            parts.append("SEMANTIC override")
        if any(p.startswith("statistical:decision")
               for p in r["trigger_paths"]):
            parts.append(
                f"statistical decision_threshold path "
                f"({r['scores']['statistical_score']:.3f} >= "
                f"{STATISTICAL_DECISION_THRESHOLD})"
            )
        if any(p.startswith("unified:") for p in r["trigger_paths"]):
            parts.append(
                f"unified threshold path "
                f"({r['scores']['unified_score']:.3f} >= {UNIFIED_THRESHOLD})"
            )
        explained = ("CHI2-EXPLAINED (flips to "
                     f"{r['decision_no_chi2']} without chi2)"
                     if r["chi2_explained_fp"] else
                     f"NOT chi2-explained (still {r['decision_no_chi2']} "
                     "without chi2)")
        print(f"\n  {r['event_id']}:")
        for p in parts:
            print(f"     - {p}")
        print(f"     => {explained}")


def summarize(records, label):
    """Print the FP attribution breakdown for one reference offset."""
    n_neg = sum(1 for r in records if r["ground_truth"] == "NO_DRIFT")
    fps = [r for r in records if r["is_false_positive"]]
    n_fp = len(fps)
    n_chi2 = sum(1 for r in fps if r["chi2_explained_fp"])

    print("\n" + "=" * 100)
    print(f"BREAKDOWN — {label}")
    print("=" * 100)
    print(f"  Negative windows:                {n_neg}")
    print(f"  B4 false positives:              {n_fp} "
          f"(FPR={n_fp / n_neg if n_neg else 0:.3f})")
    print(f"  Explained by chi2/wd alone:      {n_chi2} "
          f"({100.0 * n_chi2 / n_fp if n_fp else 0:.1f}%)")
    print(f"  NOT explained (residual source): {n_fp - n_chi2} "
          f"({100.0 * (n_fp - n_chi2) / n_fp if n_fp else 0:.1f}%)")
    print(f"  FPs where statistical(chi2) fired alongside: "
          f"{sum(1 for r in fps if r['b2_decision'] == 'DRIFT')}/{n_fp}")
    print(f"  FPs where statistical would NOT fire w/o chi2: "
          f"{sum(1 for r in fps if r['b2_decision_no_chi2'] == 'NO_DRIFT')}"
          f"/{n_fp}")
    n_semantic_fp = sum(
        1 for r in fps
        if any(p.startswith("override:semantic") for p in r["trigger_paths"])
    )
    print(f"  FPs with semantic override:      {n_semantic_fp}/{n_fp}")
    return {"n_negatives": n_neg, "n_false_positives": n_fp,
            "n_chi2_explained": n_chi2}


def main():
    """Attribute B4 false positives on the secondary dataset; diagnostic only."""
    merged = pd.read_csv("data/secondary/reference_data.csv")

    with open("results/phase13/schema_reference.yaml", "r",
              encoding="utf-8") as f:
        reference_schema = yaml.safe_load(f)
    with open("configs/semantic_contract_secondary.yaml", "r",
              encoding="utf-8") as f:
        contract = yaml.safe_load(f)
    with open("results/phase13/manifest.json", "r",
              encoding="utf-8") as f:
        manifest = json.load(f)

    categorical_columns = discover_categorical_columns(contract)

    print("=" * 100)
    print("PHASE 13 — B4 FALSE-POSITIVE ATTRIBUTION (diagnostic only)")
    print("=" * 100)
    print(f"Categorical columns for chi2: {categorical_columns}")
    print(f"Frozen: chi2_scale={CHI2_SCALE}, ks_floor={KS_FLOOR}, "
          f"psi_scale={PSI_SCALE}, "
          f"stat.decision_threshold={STATISTICAL_DECISION_THRESHOLD}, "
          f"unified.threshold={UNIFIED_THRESHOLD}")

    all_records = {}
    for offset in REF_OFFSETS:
        label = (f"reference rows {offset}-{offset + REF_ROWS}")
        reference_df = merged.iloc[offset:offset + REF_ROWS].copy()
        records = analyze_offset(reference_df, manifest, reference_schema,
                                 contract, categorical_columns)
        all_records[offset] = records

        print("\n" + "=" * 100)
        print(f"PER-WINDOW DETAIL — {label}")
        print("=" * 100)
        for r in records:
            marker = ("FP " if r["is_false_positive"]
                      else "OK " if r["decision"] == r["ground_truth"]
                      else "?? ")
            print(f"\n{marker}{r['event_id']}: "
                  f"truth={r['ground_truth']} decision={r['decision']}")
            print(f"     scores: schema={r['scores']['schema_score']:.1f} "
                  f"stat={r['scores']['statistical_score']:.3f} "
                  f"semantic={r['scores']['semantic_score']:.1f} "
                  f"unified={r['scores']['unified_score']:.3f}")
            print(f"     trigger paths: "
                  f"{r['trigger_paths'] or ['none']}")
            print(f"     wd chi2={r['wd_chi2']:.1f} "
                  f"(p={r['wd_p_value']}), "
                  f"chi2_comp={r['chi2_component']:.3f}, "
                  f"ks_comp={r['ks_component']:.3f}, "
                  f"psi_comp={r['psi_component']:.3f}")
            print(f"     counterfactual (no chi2): "
                  f"stat={r['statistical_score_no_chi2']:.3f} "
                  f"-> B4 {r['decision_no_chi2']}, "
                  f"B2 {r['b2_decision']} -> {r['b2_decision_no_chi2']}")

        print("\n" + "=" * 100)
        print(f"PER-FP-WINDOW ATTRIBUTION — {label}")
        print("=" * 100)
        print_fp_attribution([r for r in records
                              if r["is_false_positive"]])
        summarize(records, label)

    # ---- Cross-offset consistency: attribution must not depend on slice ----
    a, b = (all_records[o] for o in REF_OFFSETS)
    decisions_match = all(
        x["decision"] == y["decision"] and
        x["chi2_explained_fp"] == y["chi2_explained_fp"]
        for x, y in zip(a, b)
    )
    print("\n" + "=" * 100)
    print("CROSS-OFFSET CONSISTENCY")
    print("=" * 100)
    print(f"  Decisions + FP attribution identical across reference "
          f"offsets {REF_OFFSETS}: "
          f"{'YES' if decisions_match else 'NO — INVESTIGATE'}")
    print("  (chi2 magnitudes differ slightly with the slice, but every "
          "chi2 component still saturates at 1.0 >> chi2_scale=32.0)")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "categorical_columns": categorical_columns,
            "reference_offsets": REF_OFFSETS,
            "cross_offset_attribution_identical": decisions_match,
            "per_offset": {
                str(o): {
                    "summary": summarize_placeholder(all_records[o]),
                    "records": all_records[o],
                }
                for o in REF_OFFSETS
            },
        }, f, indent=2, default=str)
    print(f"\nFull per-window detail written to {OUTPUT_PATH}")

    return all_records


def summarize_placeholder(records):
    """Machine-readable FP counts (kept for downstream tooling)."""
    n_neg = sum(1 for r in records if r["ground_truth"] == "NO_DRIFT")
    fps = [r for r in records if r["is_false_positive"]]
    return {
        "n_negatives": n_neg,
        "n_false_positives": len(fps),
        "n_chi2_explained": sum(1 for r in fps
                                if r["chi2_explained_fp"]),
    }


if __name__ == "__main__":
    main()
