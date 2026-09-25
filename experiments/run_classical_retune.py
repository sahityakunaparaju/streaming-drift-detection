"""
Phase 10 Retune — Good-faith single-adjustment retuning of DDM and KSWIN.

Runs the same evaluation protocol as run_classical_comparison.py but with
one retuned parameter per detector, side-by-side with the defaults:

  DDM:    k=2.0 (default) vs k=3.0 (retuned: widens deviation threshold
          to avoid p_min=0 degeneracy on zero-baseline warm-ups)
  KSWIN:  alpha=0.005 (default) vs alpha=0.0001 (retuned: Bonferroni-
          corrected for ~500 total tests per window, targeting ~5%
          compound OR-over-5-features false-alarm rate)
  ADWIN:  unchanged (already well-calibrated)

Outputs (results/phase10/retune/):
    retune_comparison.csv/md    side-by-side default vs retuned metrics
    retune_summary.json         machine-readable aggregate
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from detectors.classical_detectors import (
    DDM_DEVIATION_K,
    DDM_DEVIATION_K_RETUNED,
    DETECTOR_NAMES,
    KSWIN_ALPHA_DEFAULT,
    KSWIN_ALPHA_RETUNED,
    detector_configs,
    evaluate_window,
)
from evaluation.metrics import evaluate

PHASE7_ROOT = Path("results/phase7/multiseed")
PHASE8_ROOT = Path("results/phase8")
OUTPUT_ROOT = Path("results/phase10/retune")

DEFAULT_SEEDS = [42, 123, 7, 11, 2024, 314, 2718, 999, 555, 777]
WARMUP_SIZE = 2000

B4_CONFIG = "B4_unified"
LADDER_METRICS = ["precision", "recall", "f1", "fpr"]


def load_reference_df():
    """Load the primary IoT telemetry dataset."""
    return pd.read_csv("data/primary/iot_telemetry_data.csv")


def local_warmup(reference_df, start_row):
    """WARMUP_SIZE rows immediately preceding a window start."""
    start = max(0, int(start_row) - WARMUP_SIZE)
    return reference_df.iloc[start:int(start_row)].reset_index(drop=True)


def load_phase7_windows(seed):
    """Load one seed's Phase 7 manifest and window CSVs."""
    manifest = json.load(open(
        PHASE7_ROOT / f"seed-{seed}" / "manifest.json"
    ))
    windows = {}
    for entry in manifest["windows"]:
        windows[entry["event_id"]] = pd.read_csv(entry["file"])
    return manifest, windows


# Configurations to compare per detector
# Each entry: (label, ddm_k, kswin_alpha)
VARIANTS = {
    "adwin": [
        ("default", None, None),
    ],
    "ddm": [
        ("default", DDM_DEVIATION_K, None),
        ("retuned_k3", DDM_DEVIATION_K_RETUNED, None),
    ],
    "kswin": [
        ("default", None, KSWIN_ALPHA_DEFAULT),
        ("retuned_alpha0001", None, KSWIN_ALPHA_RETUNED),
    ],
}


def run_seed(seed, reference_df):
    """Evaluate all variant configs for one seed. Returns per-variant rows."""
    manifest, windows = load_phase7_windows(seed)

    all_rows = []  # rows for every variant
    for entry in manifest["windows"]:
        event_id = entry["event_id"]
        window_df = windows[event_id]
        warmup_df = local_warmup(reference_df, entry["start_row"])
        base = {
            "seed": seed,
            "event_id": event_id,
            "set": entry["set"],
            "stream_id": entry.get("stream_id"),
            "stream_position": entry.get("stream_position"),
            "drift_type": entry["drift_type"],
            "subtype": entry["subtype"],
            "severity": entry["severity"],
            "ground_truth": entry["ground_truth"],
            "expected_false_negative": entry["expected_false_negative"],
        }
        for det_name in DETECTOR_NAMES:
            for variant_label, ddm_k, kswin_alpha in VARIANTS[det_name]:
                decision = (
                    "DRIFT"
                    if evaluate_window(
                        window_df, warmup_df, det_name,
                        ddm_k=ddm_k, kswin_alpha=kswin_alpha,
                    )
                    else "NO_DRIFT"
                )
                all_rows.append({
                    **base,
                    "detector": det_name,
                    "variant": variant_label,
                    "decision": decision,
                })

    return all_rows


def summarize_metrics(rows):
    """Raw + adjusted P/R/F1/FPR over per-window decision rows."""
    y_true = [1 if r["ground_truth"] == "DRIFT" else 0 for r in rows]
    y_pred = [1 if r["decision"] == "DRIFT" else 0 for r in rows]
    expected_fn = [r["expected_false_negative"] for r in rows]
    return evaluate(y_true, y_pred, expected_fn)


def mean_std(values):
    """Arithmetic mean and population std of a numeric list."""
    values = [float(v) for v in values]
    return float(np.mean(values)), float(np.std(values))


def main():
    """Run the Phase 10 good-faith retune comparison and write the report."""
    parser = argparse.ArgumentParser(
        description="Phase 10 retune comparison."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--out", default=str(OUTPUT_ROOT))
    args = parser.parse_args()
    seeds = args.seeds
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("PHASE 10 — RETUNE COMPARISON: DDM k=3.0, KSWIN alpha=0.0001")
    print(f"seeds: {seeds}")
    print("=" * 90)

    reference_df = load_reference_df()
    ladder_df = pd.read_csv(PHASE7_ROOT / "aggregate_ladder.csv")

    all_rows = []
    for seed in seeds:
        print(f"\n--- seed {seed} ---")
        rows = run_seed(seed, reference_df)
        all_rows.extend(rows)
        seed_dir = out_root / f"seed-{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(seed_dir / "rows.csv", index=False)
        print(f"  {len(rows)} window-decisions")

    rows_df = pd.DataFrame(all_rows)

    # ---------- Build comparison table ----------
    # Structure: for each detector × variant, compute metrics across seeds
    config_rows = []
    for det_name in DETECTOR_NAMES:
        for variant_label, _, _ in VARIANTS[det_name]:
            subset = rows_df[
                (rows_df["detector"] == det_name)
                & (rows_df["variant"] == variant_label)
            ]
            per_seed = {
                v: {m: [] for m in ["precision", "recall", "f1", "fpr"]}
                for v in ["raw", "adjusted"]
            }
            for seed in seeds:
                seed_rows = subset[subset["seed"] == seed].to_dict("records")
                summary = summarize_metrics(seed_rows)
                for v in ["raw", "adjusted"]:
                    for m in per_seed[v]:
                        per_seed[v][m].append(summary[v][m])
            row = {
                "detector": det_name.upper(),
                "variant": variant_label,
            }
            for v in ["raw", "adjusted"]:
                for m in ["f1", "precision", "recall", "fpr"]:
                    ms, ss = mean_std(per_seed[v][m])
                    row[f"{v}_{m}"] = f"{ms:.4f} +/- {ss:.4f}"
            config_rows.append(row)

    # B4 row
    b4_row = {"detector": "B4", "variant": "existing_validated"}
    for metric in LADDER_METRICS:
        cells = ladder_df[
            (ladder_df["config"] == B4_CONFIG)
            & (ladder_df["variant"] == "raw")
            & (ladder_df["metric"] == metric)
        ]
        if not cells.empty:
            m, s = float(cells.iloc[0]["mean"]), float(cells.iloc[0]["std"])
            b4_row[f"raw_{metric}"] = f"{m:.4f} +/- {s:.4f}"
        cells = ladder_df[
            (ladder_df["config"] == B4_CONFIG)
            & (ladder_df["variant"] == "adjusted")
            & (ladder_df["metric"] == metric)
        ]
        if not cells.empty:
            m, s = float(cells.iloc[0]["mean"]), float(cells.iloc[0]["std"])
            b4_row[f"adjusted_{metric}"] = f"{m:.4f} +/- {s:.4f}"
    config_rows.append(b4_row)

    comp_df = pd.DataFrame(config_rows)
    comp_df.to_csv(out_root / "retune_comparison.csv", index=False)

    # ---------- Summary JSON ----------
    summary = {
        "phase": "10-retune",
        "seeds": seeds,
        "ddm": {
            "default": {"k": DDM_DEVIATION_K},
            "retuned": {"k": DDM_DEVIATION_K_RETUNED},
            "rationale": (
                "Default k=2.0 produces zero surprise bits on warm-up "
                "for low-variance features (co, lpg, smoke), causing "
                "p_min=0 and threshold=0 degeneracy. Widening to k=3.0 "
                "ensures nonzero baseline surprise rate."
            ),
        },
        "kswin": {
            "default": {"alpha": KSWIN_ALPHA_DEFAULT},
            "retuned": {"alpha": KSWIN_ALPHA_RETUNED},
            "rationale": (
                "Default alpha=0.005 compounds across ~100 tests/window "
                "× 5 features to ~92% compound false-alarm rate. "
                "Bonferroni correction: alpha=0.05/(100×5)=0.0001 "
                "targets ~5% compound rate."
            ),
        },
        "comparison": config_rows,
    }
    with open(out_root / "retune_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # ---------- Markdown report ----------
    report = _build_report(comp_df, seeds)
    with open(out_root / "retune_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    _print_report(comp_df)
    return summary


def _md_table(frame):
    frame = frame.copy()
    cols = [str(c) for c in frame.columns]
    rows_data = [[str(v) for v in row]
                  for row in frame.itertuples(index=False)]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows_data]
    return "\n".join([header, sep] + body)


def _build_report(comp_df, seeds):
    lines = []
    lines.append("# Phase 10 Retune — Good-Faith Single-Adjustment Comparison")
    lines.append("")
    lines.append(
        "One documented adjustment per detector, no iteration. "
        "DDM: k=2.0→3.0 (widen deviation threshold). "
        "KSWIN: alpha=0.005→0.0001 (Bonferroni correction for "
        "~500 tests/window targeting ~5% compound FPR)."
    )
    lines.append("")
    lines.append("## Side-by-side comparison")
    lines.append("")
    lines.append(_md_table(comp_df))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(_interpretation(comp_df, seeds))
    return "\n".join(lines)


def _interpretation(comp_df, seeds):
    """Parse the comparison table to derive the outcome statement."""
    lines = []

    # Extract DDM default vs retuned FPR
    for _, row in comp_df.iterrows():
        if row.get("detector") == "DDM" and row.get("variant") == "default":
            ddm_def_fpr = row.get("raw_fpr", "")
        if row.get("detector") == "DDM" and row.get("variant") == "retuned_k3":
            ddm_ret_fpr = row.get("raw_fpr", "")
        if row.get("detector") == "KSWIN" and row.get("variant") == "default":
            ksw_def_fpr = row.get("raw_fpr", "")
        if row.get("detector") == "KSWIN" and row.get("variant") == "retuned_alpha0001":
            ksw_ret_fpr = row.get("raw_fpr", "")
        if row.get("detector") == "B4":
            b4_fpr = row.get("raw_fpr", "")

    lines.append("### DDM (k=2.0 → k=3.0)")
    lines.append(f"- Default FPR: {ddm_def_fpr}")
    lines.append(f"- Retuned FPR: {ddm_ret_fpr}")
    lines.append(
        "- Outcome: see overall table for full metrics. "
        "One isolated k adjustment."
    )
    lines.append("")
    lines.append("### KSWIN (alpha=0.005 → alpha=0.0001)")
    lines.append(f"- Default FPR: {ksw_def_fpr}")
    lines.append(f"- Retuned FPR: {ksw_ret_fpr}")
    lines.append(
        "- Outcome: see overall table for full metrics. "
        "One isolated alpha adjustment."
    )
    lines.append("")
    lines.append("### B4 (reference)")
    lines.append(f"- FPR: {b4_fpr}")
    lines.append("")
    return "\n".join(lines)


def _print_report(comp_df):
    print("\n" + "=" * 90)
    print("RETUNE COMPARISON")
    print("=" * 90)
    print(comp_df.to_string(index=False))


if __name__ == "__main__":
    main()
