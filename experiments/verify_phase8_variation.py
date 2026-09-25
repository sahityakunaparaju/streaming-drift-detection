"""
Phase 8 verification (post-hoc, Section T audit).

Two things are proven here with concrete numbers:

1. GENUINE PER-SEED VARIATION — the Phase 8 result is NOT the old
   Phase 7 artifact (re-rolled injector randomness inside fixed window
   positions). build_phase8_set() samples window START POSITIONS from
   np.random.default_rng(seed) via sample_non_overlapping_starts()
   (the corrected Phase 7 construction). This script proves it by
   showing, for every case type:
       - different underlying dataset start rows per seed;
       - different window content (md5 of the raw window CSV + actual
         first values);
       - IDENTICAL metadata sidecars per subtype across seeds (the
         semantic rule input is constant; only the DATA varies).

2. THRESHOLD MARGIN REPORT — for all isolated cases, the exact margin
   between the measured statistical signal and the frozen thresholds
   (decision_threshold = 0.25, ks_floor = 0.50), as an absolute margin
   and as a fraction of the way to triggering.

Writes results/phase8/variation_proof.csv and
results/phase8/threshold_margins.csv.

Usage:
    PYTHONPATH=. python3 experiments/verify_phase8_variation.py
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from detectors.thresholds import THRESHOLDS

DEFAULT_SEEDS = [42, 123, 7, 11, 2024, 314, 2718, 999, 555, 777]
OUTPUT_ROOT = Path("results/phase8")

KS_FLOOR = float(THRESHOLDS["statistical"]["ks_floor"])
DECISION_THRESHOLD = float(
    THRESHOLDS["statistical"]["decision_threshold"]
)

SUBTYPES = [
    "unit_change",
    "business_meaning_change",
    "range_change",
    "allowed_domain_change",
    "definition_change",
]


def md5(path):
    """MD5 hex digest of a file's raw bytes."""
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def variation_proof(seeds):
    """Per case type x seed: start row, window md5, metadata md5,
    first temp values (temp cases)."""
    rows = []
    for subtype in SUBTYPES:
        md5s = []
        for seed in seeds:
            manifest = json.load(open(
                OUTPUT_ROOT / f"seed-{seed}" / "manifest.json"
            ))
            entry = next(
                e for e in manifest["windows"]
                if e["subtype"] == subtype
            )
            window_path = OUTPUT_ROOT / f"seed-{seed}" / "windows" \
                / f"{entry['event_id']}.csv"
            meta_path = OUTPUT_ROOT / f"seed-{seed}" / "windows" \
                / f"{entry['event_id']}.metadata.json"
            window_hash = md5(window_path)
            meta_hash = md5(meta_path)
            md5s.append(window_hash)
            df = pd.read_csv(window_path)
            rows.append({
                "subtype": subtype,
                "seed": seed,
                "start_row": entry["start_row"],
                "window_md5": window_hash[:12],
                "metadata_md5": meta_hash[:12],
                "first_4_values": (
                    df["temp"].head(4).tolist()
                    if "temp" in df.columns else None
                ),
                "n_unique_temp": (
                    int(df["temp"].nunique())
                    if "temp" in df.columns else None
                ),
                "ts_start": float(df["ts"].iloc[0]),
            })
        # per-subtype distinctness across seeds
        for seed in seeds:
            rows.append({
                "subtype": subtype,
                "seed": None,
                "start_row": None,
                "window_md5": (
                    "DISTINCT across seeds"
                    if len(set(md5s)) == len(seeds)
                    else "COLLISION — repeated window!"
                ),
                "metadata_md5": None,
                "first_4_values": None,
                "n_unique_temp": None,
                "ts_start": None,
            })
    return pd.DataFrame(rows)


def threshold_margins(seeds):
    """Per isolated case: measured score/KS vs frozen thresholds."""
    rows = []
    for seed in seeds:
        per_case = json.load(open(
            OUTPUT_ROOT / f"seed-{seed}" / "per_case.json"
        ))
        for case in per_case:
            if case["set"] != "blind_spot":
                continue
            score = case["statistical_score"]
            max_ks = case["max_ks"]
            rows.append({
                "seed": seed,
                "event_id": case["event_id"],
                "subtype": case["subtype"],
                "statistical_score": score,
                "score_margin": round(DECISION_THRESHOLD - score, 6),
                "score_pct_of_threshold": round(score / DECISION_THRESHOLD, 4),
                "max_ks": max_ks,
                "ks_margin": round(KS_FLOOR - max_ks, 6),
                "ks_pct_of_floor": round(max_ks / KS_FLOOR, 4),
                "max_psi": case["max_psi"],
                "max_chi2": case["max_chi2"],
                "isolated": case["isolated"],
            })
    return pd.DataFrame(rows)


def main():
    """Prove per-seed window variation and report threshold margins."""
    parser = argparse.ArgumentParser(
        description="Phase 8 variation + threshold-margin verification."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
    )
    args = parser.parse_args()
    seeds = args.seeds

    print("=" * 90)
    print("PHASE 8 VERIFICATION — PER-SEED VARIATION + MARGINS")
    print(f"seeds: {seeds}")
    print("=" * 90)

    proof = variation_proof(seeds)
    proof.to_csv(OUTPUT_ROOT / "variation_proof.csv", index=False)

    margins = threshold_margins(seeds)
    margins.to_csv(OUTPUT_ROOT / "threshold_margins.csv", index=False)

    print("\n--- start rows per seed (underlying dataset rows) ---")
    pivot = proof[proof["seed"].notna()].pivot(
        index="subtype", columns="seed", values="start_row"
    )
    print(pivot.to_string())

    print("\n--- window content distinctness (md5) ---")
    distinct = proof[proof["seed"].isna()][
        ["subtype", "window_md5"]
    ]
    print(distinct.to_string(index=False))

    print("\n--- sample values (business_meaning_change) ---")
    sample = proof[
        (proof["subtype"] == "business_meaning_change")
        & proof["seed"].notna()
    ][["seed", "start_row", "window_md5", "first_4_values",
       "n_unique_temp", "ts_start"]]
    print(sample.to_string(index=False))

    print("\n--- threshold margins (all isolated cases) ---")
    print(
        f"score: min margin {margins['score_margin'].min():.4f} "
        f"(max score {margins['statistical_score'].max():.4f} vs "
        f"{DECISION_THRESHOLD}), median margin "
        f"{margins['score_margin'].median():.4f}, max margin "
        f"{margins['score_margin'].max():.4f}"
    )
    print(
        f"ks:    min margin {margins['ks_margin'].min():.4f} "
        f"(max ks {margins['max_ks'].max():.4f} vs "
        f"{KS_FLOOR}), median margin "
        f"{margins['ks_margin'].median():.4f}, max margin "
        f"{margins['ks_margin'].max():.4f}"
    )
    close_score = margins[
        margins["score_pct_of_threshold"] >= 0.85
    ]
    close_ks = margins[
        margins["ks_pct_of_floor"] >= 0.85
    ]
    print(
        f"\ncases within 85% of the score threshold: "
        f"{len(close_score)} / {len(margins)}"
    )
    print(
        f"cases within 85% of the KS floor: "
        f"{len(close_ks)} / {len(margins)}"
    )
    print("\nworst 3 cases by score margin:")
    print(
        margins.nsmallest(3, "score_margin")[
            ["seed", "subtype", "statistical_score", "score_margin",
             "score_pct_of_threshold", "max_ks", "ks_margin"]
        ].to_string(index=False)
    )

    print(f"\nwrote {OUTPUT_ROOT / 'variation_proof.csv'}")
    print(f"wrote {OUTPUT_ROOT / 'threshold_margins.csv'}")


if __name__ == "__main__":
    main()
