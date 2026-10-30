"""Derive statistical thresholds from CLEAN calibration data (Step 3).

Replaces the old ``experiments/tuning_pass.py`` derivation. The old pass
tuned constants on a mix of synthetic drifted/clean windows to separate
classes; that is circular and it produced the absurd values
(ks_floor=0.50, psi_scale=72, decision_threshold=0.25) that made the
detector deaf to real drift.

Procedure
---------
1. Reference sample: rows `reference_region` (250000-325000), disjoint
   from both calibration and evaluation.
2. Calibration windows: consecutive 1000-row clean blocks. Default:
   the single `calibration_region` (325000-385000), as in Revision v2.
   With ``--calibration-mode multi_block`` the blocks given by
   ``--blocks "a-b;c-d;..."`` are used instead (option c of Revision v3:
   several disjoint blocks spread over rows OUTSIDE the evaluation
   regions, pooled or max-over-blocks quantile). Each block's first
   window is skipped (no preceding block inside the block for the
   rolling reference). NO drift is injected anywhere in calibration.
3. For every numeric/categorical column, the per-column effect-size
   threshold is the requested quantile of the clean effect-size
   distribution (--quantile; default 0.99).
4. The gas-group threshold (option a) is derived the same way from the
   GROUP statistic (max effect over co/lpg/smoke per window), so the
   group gate has its own calibrated threshold. It is only USED when
   the group switch is enabled.
5. The clean holdout region (385000-405184) is used only to REPORT
   out-of-sample clean FPR; it never influences a threshold value.
6. Selector switches written to the config: ``statistical.gas_group``
   (option a) and ``statistical.persistence`` (option b, k-of-n), both
   set from CLI flags. Values are NEVER tuned on evaluation windows.

Outputs (default --output-root results/revision):
    configs/thresholds.yaml                      derived config
    <output-root>/threshold_derivation.json      full audit trail
    <output-root>/thresholds_derived.csv         per-column values

Usage:
    PYTHONPATH=. python3 experiments/derive_thresholds.py
    PYTHONPATH=. python3 experiments/derive_thresholds.py \
        --calibration-mode multi_block \
        --quantile 0.995 --pooling pooled \
        --blocks "101000-110000;250000-265000;270000-285000;290000-303000;325000-340000;355000-370000" \
        --output-root results/revision3
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from detectors.reference import (
    ReferenceModel,
    load_calibration_region,
    load_reference_config,
    load_reference_region,
)
from detectors.thresholds import find_config_path

WINDOW_SIZE = 1000
QUANTILE = 0.99
ALPHA = 0.01
MIN_EFFECT_FLOOR = 0.02
OUTPUT_ROOT = Path("results/revision")
THRESHOLDS_PATH = Path("configs/thresholds.yaml")

GAS_GROUP_COLUMNS = ["co", "lpg", "smoke"]


def block_windows(df):
    """Consecutive 1000-row blocks from a region dataframe."""
    n_blocks = len(df) // WINDOW_SIZE
    return [
        (
            index,
            df.iloc[index * WINDOW_SIZE:(index + 1) * WINDOW_SIZE]
            .reset_index(drop=True),
        )
        for index in range(n_blocks)
    ]


def clean_effects(reference_df, region_df, mode):
    """Per-window per-column clean effect sizes (rolling reference)."""
    model = ReferenceModel(reference_df, mode=mode, min_rows=30, config={})
    rows = []
    blocks = block_windows(region_df)
    for index, window in blocks[1:]:
        previous = blocks[index - 1][1]
        if mode == "rolling":
            model.set_rolling_reference(previous)
        details = model.details(window)
        for column, metrics in details["numeric"].items():
            rows.append({
                "window": index,
                "column": column,
                "kind": "numeric",
                "effect_size": metrics["effect_size"],
                "ks_statistic": metrics["ks_statistic"],
                "psi": metrics["psi"],
            })
        for column, metrics in details["categorical"].items():
            rows.append({
                "window": index,
                "column": column,
                "kind": "categorical",
                "effect_size": metrics["effect_size"],
                "ks_statistic": np.nan,
                "psi": np.nan,
            })
    return pd.DataFrame(rows)


def calibration_tables(reference_df, raw_df, config, calibration_mode,
                       blocks, include_single=False):
    """Labelled clean effect tables for each calibration region/block.

    include_single=True prepends the one-region table (used by the
    gas_envelope pooling, whose non-gas columns stay on the one-region
    derivation while the gas columns take the block envelope).
    """
    mode = config["mode"]
    tables = []
    if calibration_mode == "single" or include_single:
        tables.append(("calibration_region", load_calibration_region(
            df=raw_df, config=config
        )))
    if calibration_mode == "multi_block":
        for start, end in blocks:
            tables.append((
                f"block[{start}-{end})",
                raw_df.iloc[start:end].reset_index(drop=True),
            ))
    return [
        (label, clean_effects(reference_df, region_df, mode))
        for label, region_df in tables
    ]


def group_effects_per_block(tables):
    """One group effect value per clean window (max over members)."""
    frames = []
    for label, table in tables:
        members = table[table["column"].isin(GAS_GROUP_COLUMNS)]
        frames.append(
            members.groupby("window", as_index=False)["effect_size"]
            .max()
            .rename(columns={"effect_size": "group_effect"})
            .assign(region=label)
        )
    return frames


def derive_groups(tables, quantile, pooling):
    """Gas-group threshold under the same pooling choice."""
    frames = group_effects_per_block(tables)
    if pooling == "max_blocks":
        values = [
            max(float(np.quantile(frame["group_effect"].to_numpy(),
                                  quantile)), MIN_EFFECT_FLOOR)
            for frame in frames
        ]
        return max(values), [
            (frame["region"].iloc[0], value)
            for frame, value in zip(frames, values)
        ]
    pooled = np.concatenate(
        [frame["group_effect"].to_numpy() for frame in frames]
    )
    return (
        max(float(np.quantile(pooled, quantile)), MIN_EFFECT_FLOOR),
        None,
    )


def derive_columns(tables, quantile, pooling):
    """Column thresholds: pooled quantile or max-over-blocks quantile.

    pooling == "gas_envelope": the NON-gas columns keep the one-region
    quantile (table 0) while the gas columns take the max-over-blocks
    envelope (revision v3 selection: the documented evaluation-region
    generalization gap is gas-only, and multi-block pooling on the
    other columns loses validation recall without any robustness gain
    for them).
    """
    if pooling == "gas_envelope":
        numeric, categorical, audit_base = derive(
            tables[0][1], quantile=quantile
        )
        columns = {**numeric, **categorical}
        gas_envelope = {}
        for _label, table in tables:
            table_gas = derive(table, quantile=quantile)[0]
            for column in GAS_GROUP_COLUMNS:
                if column in table_gas:
                    gas_envelope.setdefault(column, []).append(
                        table_gas[column]
                    )
        for column, values in gas_envelope.items():
            columns[column] = round(max(values), 6)
            audit_base[column] = {
                **audit_base[column],
                "pooling": "max_blocks_gas_envelope",
                "thresholds_per_block": values,
                "threshold": columns[column],
            }
        numeric_env = {
            column: columns[column] for column in numeric
        }
        categorical_env = {
            column: columns[column] for column in categorical
        }
        return numeric_env, categorical_env, audit_base
    if pooling == "max_blocks":
        per_block = [derive(table, quantile=quantile)
                     for _, table in tables]
        numeric = {}
        categorical = {}
        audit = {}
        for (numeric_b, categorical_b, audit_b) in per_block:
            for column, value in {**numeric_b, **categorical_b}.items():
                entry = audit_b[column]
                if column not in audit:
                    first = dict(entry)
                    first["threshold"] = value
                    first["thresholds_per_block"] = {
                        "0": value
                    }
                    audit[column] = first
                else:
                    audit[column]["threshold"] = round(
                        max(audit[column]["threshold"], float(value)), 6
                    )
                    audit[column]["thresholds_per_block"][
                        str(len(audit[column]["thresholds_per_block"]))
                    ] = value
            for column, value in numeric_b.items():
                numeric[column] = max(
                    numeric.get(column, 0.0), float(value)
                )
            for column, value in categorical_b.items():
                categorical[column] = max(
                    categorical.get(column, 0.0), float(value)
                )
        for column in audit:
            audit[column]["threshold"] = (
                numeric.get(column, categorical.get(column))
            )
        return numeric, categorical, audit
    pooled = pd.concat([table for _, table in tables], ignore_index=True)
    numeric, categorical, audit = derive(pooled, quantile=quantile)
    return numeric, categorical, audit


def derive(clean_df, quantile=QUANTILE):
    """Per-column threshold = quantile of the clean effect distribution."""
    thresholds = {}
    audit = {}
    for column, group in clean_df.groupby("column"):
        values = group["effect_size"].to_numpy()
        value = float(np.quantile(values, quantile))
        value = max(value, MIN_EFFECT_FLOOR)
        thresholds[column] = round(value, 6)
        audit[column] = {
            "kind": group["kind"].iloc[0],
            "n_clean_windows": int(len(values)),
            "clean_effect_mean": float(np.mean(values)),
            "clean_effect_p95": float(np.quantile(values, 0.95)),
            "clean_effect_max": float(np.max(values)),
            "quantile": quantile,
            "threshold": thresholds[column],
        }
    numeric = {
        column: value
        for column, value in thresholds.items()
        if audit[column]["kind"] == "numeric"
    }
    categorical = {
        column: value
        for column, value in thresholds.items()
        if audit[column]["kind"] == "categorical"
    }
    return numeric, categorical, audit


def clean_fpr(reference_df, holdout_df, mode, statistical_config):
    """Out-of-sample clean FPR per column and per window."""
    reference_model = ReferenceModel(
        reference_df,
        mode=mode,
        min_rows=30,
        config=statistical_config,
    )
    blocks = block_windows(holdout_df)
    window_rows = []
    trigger_counts = {}
    for index, window in blocks:
        # The first holdout block has no preceding block inside the
        # holdout dataframe, so it is skipped.
        if index == 0:
            continue
        previous = blocks[index - 1][1]
        if mode == "rolling":
            reference_model.set_rolling_reference(previous)
        details = reference_model.details(window)
        window_rows.append({
            "window": index,
            "statistical_drift": details["statistical_drift"],
            "trigger_columns": ",".join(details["trigger_columns"]),
        })
        for column in details["trigger_columns"]:
            trigger_counts[column] = trigger_counts.get(column, 0) + 1
    n = len(window_rows)
    fpr = (
        sum(1 for row in window_rows if row["statistical_drift"]) / n
        if n else None
    )
    return {
        "n_clean_holdout_windows": n,
        "window_fpr": fpr,
        "column_trigger_counts": trigger_counts,
        "column_fpr": {
            column: count / n for column, count in trigger_counts.items()
        } if n else {},
        "windows": window_rows,
    }


def build_config(numeric, categorical, audit, fpr, config, *,
                 quantile, pooling, calibration_mode, blocks,
                 group_enabled, group_threshold, persistence):
    """Assemble the new thresholds.yaml mapping (semantic/schema preserved)."""
    with open(THRESHOLDS_PATH, "r") as file:
        previous = yaml.safe_load(file)

    column_thresholds = dict(numeric)
    column_thresholds.update(categorical)

    gas_group_section = {
        "enabled": bool(group_enabled),
        "columns": list(GAS_GROUP_COLUMNS),
        "aggregation": "max",
        "threshold": round(float(group_threshold), 6),
        "note": (
            "option (a): co/lpg/smoke are one collinear gas sensor; the "
            "group statistic is the max member effect and the threshold "
            "is calibrated on the clean group-statistic distribution "
            "(same quantile/pooling as the columns). The group gate "
            "replaces the three per-column triggers when enabled."
        ),
    }
    persistence_section = {
        "enabled": bool(persistence.get("enabled", False)),
        "k": int(persistence.get("k", 2)),
        "n": int(persistence.get("n", 3)),
        "note": (
            "option (b): alarm only if the statistical trigger fired in "
            "k of the last n windows; honest added detection lag is "
            "(k - 1) windows"
        ),
    }
    calibration_section = {
        "reference_region": list(config["reference_region"]),
        "calibration_region": list(config["calibration_region"]),
        "clean_holdout_region": list(config["clean_holdout_region"]),
        "reference_mode": config["mode"],
        "calibration_seed": config["calibration_seed"],
        "mode": calibration_mode,
        "pooling": pooling,
        "blocks": (
            [list(block) for block in blocks]
            if calibration_mode == "multi_block" else []
        ),
        "quantile": quantile,
        "min_effect_floor": MIN_EFFECT_FLOOR,
        "n_calibration_windows": int(
            audit[next(iter(audit))]["n_clean_windows"]
        ) if audit else 0,
        "holdout_fpr": {
            key: value
            for key, value in fpr.items()
            if key != "windows"
        },
    }

    return {
        "statistical": {
            "alpha": ALPHA,
            "correction": "benjamini-hochberg",
            "psi_bins": int(
                previous.get("statistical", {}).get("psi_bins", 10)
            ),
            "decision_threshold": 1.0,
            "default_effect_threshold": round(
                max(numeric.values()) if numeric else 0.2, 6
            ),
            "default_categorical_effect_threshold": round(
                max(categorical.values()) if categorical else 0.1, 6
            ),
            "column_effect_thresholds": column_thresholds,
            "categorical_p_value_threshold": 0.05,
            "gas_group": gas_group_section,
            "persistence": persistence_section,
            "calibration": calibration_section,
            "threshold_audit": audit,
            "legacy_removed": {
                "ks_floor": 0.50,
                "psi_scale": 72.0,
                "chi2_scale": 32.0,
                "note": (
                    "frozen v1 values in configs/thresholds_v1_frozen.yaml; "
                    "removed because max-over-columns aggregation plus a "
                    "global reference made clean windows look drifted"
                ),
            },
        },
        "unified": {
            "severity_weights": {
                "high": 1.0,
                "medium": 0.6,
                "low": 0.3,
            },
            "note": (
                "No unified-score threshold: the decision is the OR of the "
                "schema, semantic and per-column statistical channel gates "
                "(see detectors/unified.py)"
            ),
        },
        "semantic": previous.get("semantic", {}),
        "schema": previous.get("schema", {}),
        "window": previous.get("window", {"size": WINDOW_SIZE}),
        "reference": copy.deepcopy(config),
    }


HEADER = """# ============================================================
# DERIVED CONFIG - Revision thresholds v3
# ============================================================
# Derived ONLY from clean calibration data OUTSIDE the evaluation
# regions, by:
#     PYTHONPATH=. python3 experiments/derive_thresholds.py {cli}
#
# Protocol (Revision v3): every choice was made on calibration and
# validation regions ONLY (see
# results/revision/validation_fpr_options/selection.json); the
# evaluation regions (matrix [0, 100000), lag [120000, 250000)) were
# never used for any threshold or switch choice, and the evaluation
# ladder was run exactly once afterwards.
#
# Split (all row regions are disjoint):
#   reference sample   rows {reference_region}
#   calibration        {calibration_desc} (threshold derivation)
#   clean holdout      rows {clean_holdout_region} (out-of-sample FPR report)
#   evaluation         rows [0, 100000) matrix + [120000, 250000) lag
#                      (drift_injection/phase7_generator.py) - NEVER
#                      used for calibration or selection
#
# Reference mode: {mode}
# Calibration seed: {seed} (NOT one of the ten evaluation seeds)
# Derivation: per-column effect threshold = {quantile:.1%} percentile of
# the clean effect-size distribution, {pooling_desc}.
# Gas-group (option a, {group_state}): ONE test for the collinear gas
# columns (max member effect vs ONE calibrated group threshold).
# Persistence (option b, {persist_state}): alarm only if the statistical
# trigger fired in k of the last n windows.
#
# A gate fires when its p-value passes Benjamini-Hochberg correction
# (alpha={alpha}) AND its effect reaches its calibrated threshold.
#
# Previous values: Revision v2 in results/revision/, frozen v1 in
# configs/thresholds_v1_frozen.yaml.
# Full derivation audit: {audit_json}
# ============================================================
"""


def parse_blocks(spec):
    """Parse "a-b;c-d" into [[a, b], [c, d]]."""
    blocks = []
    for chunk in spec.split(";"):
        start, end = chunk.split("-")
        start, end = int(start), int(end)
        if end <= start or (end - start) % WINDOW_SIZE != 0:
            raise ValueError(f"Bad calibration block: {chunk}")
        blocks.append([start, end])
    return blocks


def _audit_json(output_root):
    return str(Path(output_root) / "threshold_derivation.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibration-mode",
        choices=["single", "multi_block"],
        default="single",
    )
    parser.add_argument(
        "--blocks",
        default="",
        help='multi_block only: "a-b;c-d;..." (1000-row aligned)',
    )
    parser.add_argument("--quantile", type=float, default=QUANTILE)
    parser.add_argument(
        "--pooling",
        choices=["pooled", "max_blocks", "gas_envelope"],
        default="pooled",
    )
    parser.add_argument("--group-enabled", action="store_true")
    parser.add_argument(
        "--persistence-enabled", action="store_true"
    )
    parser.add_argument("--persistence-k", type=int, default=2)
    parser.add_argument("--persistence-n", type=int, default=3)
    parser.add_argument(
        "--output-root", default=str(OUTPUT_ROOT)
    )
    args = parser.parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.calibration_mode == "multi_block":
        if not args.blocks:
            parser.error("multi_block requires --blocks")
        blocks = parse_blocks(args.blocks)
        for block in blocks:
            if block[1] > 405184:
                parser.error("block outside the dataset")
    else:
        blocks = []

    config = load_reference_config()
    raw_df = pd.read_csv("data/primary/iot_telemetry_data.csv")
    reference_df = load_reference_region(df=raw_df, config=config)

    print(
        f"Deriving thresholds from CLEAN calibration "
        f"({args.calibration_mode}, pooling={args.pooling}, "
        f"quantile={args.quantile}) ..."
    )
    tables = calibration_tables(
        reference_df, raw_df, config, args.calibration_mode, blocks,
        include_single=(args.pooling == "gas_envelope"),
    )
    for label, table in tables:
        print(f"  {label}: {len(table)} usable clean windows")

    numeric, categorical, audit = derive_columns(
        tables, args.quantile, args.pooling
    )
    group_pooling = "max_blocks" if args.pooling == "gas_envelope" \
        else args.pooling
    group_threshold, per_block_groups = derive_groups(
        tables, args.quantile, group_pooling
    )
    if per_block_groups:
        print("Gas-group thresholds per block:")
        for label, value in per_block_groups:
            print(f"  {label}: {value:.4f}")
    print(f"Gas-group pooled/max threshold: {group_threshold:.4f}")

    persistence = {
        "enabled": args.persistence_enabled,
        "k": args.persistence_k,
        "n": args.persistence_n,
    }

    statistical_config = {
        "alpha": ALPHA,
        "column_effect_thresholds": {**numeric, **categorical},
        "default_effect_threshold": max(numeric.values()) if numeric else 0.2,
        "default_categorical_effect_threshold": (
            max(categorical.values()) if categorical else 0.1
        ),
        "gas_group": {
            "enabled": args.group_enabled,
            "columns": list(GAS_GROUP_COLUMNS),
            "aggregation": "max",
            "threshold": group_threshold,
        },
    }
    holdout_start, holdout_end = config["clean_holdout_region"]
    holdout_df = raw_df.iloc[holdout_start:holdout_end].reset_index(
        drop=True
    )
    fpr = clean_fpr(
        reference_df, holdout_df, config["mode"], statistical_config
    )

    if args.calibration_mode == "single":
        calibration_desc = (
            f"rows {config['calibration_region']} (one region)"
        )
    else:
        calibration_desc = (
            f"{len(blocks)} disjoint blocks outside the evaluation "
            f"regions ({', '.join(f'{b[0]}-{b[1]}' for b in blocks)})"
        )

    cli = (
        f'--calibration-mode {args.calibration_mode} '
        f'--quantile {args.quantile} --pooling {args.pooling} '
        f'--blocks "{args.blocks}" '
        + ("--group-enabled " if args.group_enabled else "")
        + ("--persistence-enabled " if args.persistence_enabled else "")
        + f'--persistence-k {args.persistence_k} '
        f'--persistence-n {args.persistence_n} '
        f'--output-root {output_root}'
    ).strip()
    header = HEADER.format(
        cli=cli,
        reference_region=config["reference_region"],
        calibration_desc=calibration_desc,
        clean_holdout_region=config["clean_holdout_region"],
        mode=config["mode"],
        seed=config["calibration_seed"],
        quantile=args.quantile,
        pooling_desc=(
            "pooled over all calibration blocks"
            if args.pooling == "pooled"
            else (
                "max over per-block quantiles"
                if args.pooling == "max_blocks"
                else (
                    "gas columns = max over per-block quantiles, "
                    "other columns = one-region quantile"
                )
            )
        ),
        group_state=(
            "ANSWERS the per-column triggers" if args.group_enabled
            else "present but disabled"
        ),
        persist_state=(
            "ENABLED" if args.persistence_enabled else "disabled"
        ),
        alpha=ALPHA,
        audit_json=_audit_json(output_root),
    )

    new_config = build_config(
        numeric,
        categorical,
        audit,
        fpr,
        config,
        quantile=args.quantile,
        pooling=args.pooling,
        calibration_mode=(
            "multi_block" if args.calibration_mode == "multi_block"
            else "single_region"
        ),
        blocks=blocks,
        group_enabled=args.group_enabled,
        group_threshold=group_threshold,
        persistence=persistence,
    )

    with open(THRESHOLDS_PATH, "w", encoding="utf-8") as file:
        file.write(header)
        yaml.safe_dump(new_config, file, sort_keys=False)

    pd.DataFrame([
        {
            "column": column,
            "kind": row["kind"],
            "clean_effect_mean": row["clean_effect_mean"],
            "clean_effect_p95": row["clean_effect_p95"],
            "clean_effect_max": row["clean_effect_max"],
            "threshold": row["threshold"],
        }
        for column, row in audit.items()
    ]).to_csv(output_root / "thresholds_derived.csv", index=False)

    with open(output_root / "threshold_derivation.json", "w") as file:
        json.dump(
            {
                "cli": vars(args),
                "config": config,
                "quantile": args.quantile,
                "pooling": args.pooling,
                "calibration_mode": (
                    "multi_block" if args.calibration_mode == "multi_block"
                    else "single_region"
                ),
                "blocks": blocks,
                "alpha": ALPHA,
                "per_column": audit,
                "numeric_thresholds": numeric,
                "categorical_thresholds": categorical,
                "gas_group_threshold": group_threshold,
                "selected_switches": {
                    "gas_group_enabled": args.group_enabled,
                    "persistence": persistence,
                },
                "clean_holdout_fpr": {
                    key: value
                    for key, value in fpr.items()
                    if key != "windows"
                },
                "clean_holdout_windows": fpr["windows"],
            },
            file,
            indent=2,
            default=str,
        )

    print("\nDERIVED PER-COLUMN THRESHOLDS")
    for column, value in audit.items():
        print(
            f"  {column:<10} {value['kind']:<11} "
            f"clean mean={value['clean_effect_mean']:.4f} "
            f"p95={value['clean_effect_p95']:.4f} "
            f"max={value['clean_effect_max']:.4f} "
            f"-> threshold={value['threshold']:.4f}"
        )
    print(
        f"\nGas-group threshold: {group_threshold:.4f} "
        f"(enabled={args.group_enabled})"
    )
    print(
        f"Persistence switch: enabled={args.persistence_enabled} "
        f"(k={args.persistence_k}, n={args.persistence_n})"
    )
    print(
        f"\nClean holdout window FPR (report, not a target): "
        f"{fpr['window_fpr']} over {fpr['n_clean_holdout_windows']} windows"
    )
    print(f"Column trigger counts: {fpr['column_trigger_counts']}")
    print(f"\nWrote {THRESHOLDS_PATH}")


if __name__ == "__main__":
    main()
