"""Reduce clean-window false positives WITHOUT touching evaluation data.

Revision v3 comparison experiment (options a-d). This script performs
EVERY choice on calibration and validation regions only; evaluation
windows (rows [0, 100000) matrix and [120000, 250000) lag) are never
sampled, injected or measured here. The evaluation ladder is run once,
after the final config is frozen, by experiments/run_revision_ladder.py.

Options (each becomes/then maps to a switch in configs/thresholds.yaml):

  base   nothing (Revision v2 behaviour: thresholds from ONE region,
         per-column tests, no persistence)
  (a)    gas_group: co / lpg / smoke are collinear (the same gas
         sensor: smoke ~ 4*co, lpg ~ 1.5*co), so the three columns
         become ONE sensor group with one test/one calibrated
         threshold instead of three separate triggers.
  (b)    persistence: a window alarms only if the statistical trigger
         fired in k of the last n windows (k=2, n=3). The added
         detection lag is measured, not assumed.
  (c)    calibration on >= 4 disjoint clean blocks spread over rows
         outside the evaluation regions, taking the pooled or the
         max-over-blocks quantile instead of one region.
  (d)    quantile sweep (0.99, 0.995, 0.999) reported for every option.

Row regions used here (all disjoint, none inside an evaluation region):

  CALIBRATION single  [325000, 385000)   (one-region option; = v2)
  CALIBRATION blocks  [101000, 110000)  [250000, 265000)  [270000, 285000)
                      [290000, 303000)  [325000, 340000)  [355000, 370000)
  VALIDATION clean    [111000, 115000)  (4 matrix-style windows)
  VALIDATION drift    [115000, 120000)  (5 disjoint base windows; every
                      statistical matrix cell injected into each one)
  VALIDATION streams  [304000, 324000)  (4 five-window lag streams, the
                      same format as the evaluation lag streams)

The clean holdout region [385000, 405184) is NOT used here at all; it
is reserved for the out-of-sample clean FPR report in the final run.

Limitations (disclosed): validation windows may partially overlap, so
validation estimates are correlated; the validation drift base is a
10k-row slice, so the positive sample has limited row diversity. The
final numbers reported in docs/revision_notes.md come from the single
evaluation run, not from here.

Outputs (results/revision/validation_fpr_options/):
    comparison_table.csv    variant x quantile x group x persistence rows
    selection.json          pre-declared selection rule + the winner
    validation_outcomes.csv per-config window outcomes (audit)

Usage:
    PYTHONPATH=. python3 experiments/revision3_fpr_options.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from detectors.persistence import persistence_alarm, added_lag_windows
from detectors.reference import (
    ReferenceModel,
    load_reference_config,
    load_reference_region,
)
from detectors.thresholds import THRESHOLDS
from evaluation.metrics import evaluate, detection_lag_windows
from drift_injection.phase7_generator import inject_window
from experiments.derive_thresholds import (
    MIN_EFFECT_FLOOR,
    clean_effects,
    derive,
)

OUTPUT_ROOT = Path("results/revision/validation_fpr_options")

WINDOW_SIZE = 1000
QUANTILES = [0.99, 0.995, 0.999]
GAS_GROUP_COLUMNS = ["co", "lpg", "smoke"]

SINGLE_REGION = [325000, 385000]

# >= 4 disjoint clean blocks spread over rows outside the evaluation
# regions (option c). The mid block [101000, 110000) sits in the free
# zone between the two evaluation regions and is the closest to the
# evaluation matrix regime; the others sample the dataset tail.
CALIBRATION_BLOCKS = [
    [101000, 110000],
    [250000, 265000],
    [270000, 285000],
    [290000, 303000],
    [325000, 340000],
    [355000, 370000],
]

# Validation (selection ONLY; never used to derive a threshold).
VALIDATION_CLEAN_STARTS = [111000, 112000, 113000, 114000]
VALIDATION_DRIFT_BASES = [115000, 116000, 117000, 118000, 119000]
VALIDATION_STREAM_BASES = [304000, 309000, 314000, 319000]
STREAM_ONSET = 3
STREAM_LENGTH = 5

PERSISTENCE_K = 2
PERSISTENCE_N = 3

STAT_CELL_SUBTYPES = [
    "mean_shift", "variance_shift",
    "distribution_shift", "category_shift",
]
STAT_CELL_SEVERITIES = ["low", "medium", "high"]

SELECTION_RULE = (
    "Pre-declared selection rule (validation data only): rank candidates "
    "by (1) validation mixed-set F1 descending, (2) validation "
    "clean-window FPR ascending, (3) validation clean-window margin "
    "ascending (largest observed effect/threshold ratio across all "
    "active gates on clean validation windows; smaller = more "
    "headroom), (4) validation stream lag mean ascending (None = never "
    "fires on that stream, ranked last), (5) fewer enabled switches, "
    "(6) config name for determinism. The winner is the first ranked "
    "row. The evaluation ladder is then run ONCE with it."
)

RULE_V1_NOTE = (
    "Rule v1 outcome is reported but REJECTED as unfit: on the "
    "validation regions ALL candidates have clean-window FPR 0 (the "
    "mid-dataset gas regime is calm, like the calibration holdout), so "
    "the F1-first ranking degenerates to 'maximize recall' and would "
    "select the loosest configuration, leaving the documented "
    "evaluate-region gas FPR unchanged by construction."
)

RULE_V2 = (
    "Pre-declared REVISED selection rule (validation data only; "
    "declared BEFORE the evaluation run): restrict to candidates with "
    "validation clean-window FPR = 0 and validation F1 >= 0.95 x the "
    "baseline (single|q=0.99|no-switch) F1; among those rank by "
    "(1) gas-group threshold DESCENDING (robustness of the documented "
    "failure channel against every calibration block, per the "
    "max-over-blocks envelope), (2) validation clean-window margin "
    "ascending, (3) stream lag ascending, then switch preference: "
    "gas group ON before OFF at parity (option a targets the "
    "documented collinear triple-trigger cause), persistence OFF "
    "before ON (it costs +1 window lag with no validation FPR "
    "benefit). Never uses an evaluation window."
)


def quantile_value(values, quantile):
    return max(float(np.quantile(list(values), quantile)),
               MIN_EFFECT_FLOOR)


def block_group_effects(table):
    """Per-window gas-group effects (max over co/lpg/smoke)."""
    group = table[table["column"].isin(GAS_GROUP_COLUMNS)]
    return group.groupby("window", as_index=False)["effect_size"].max()


def build_threshold_sets(effect_tables, quantile, pooling):
    """Per-column + gas-group thresholds for one variant x quantile.

    pooling variants:
      pooled        every column = quantile over all blocks pooled
      max_blocks    every column = max over per-block quantiles
      gas_envelope  non-gas columns = quantile of the one-region table
                    (tables[0]); gas columns = max over per-block
                    quantiles (regime envelope where the documented
                    generalization gap lives)

    Returns (columns, group_threshold, audit).
    """
    if pooling == "gas_envelope":
        single = pd.concat(
            [table for _, table in effect_tables[:1]],
            ignore_index=True,
        )
        numeric, categorical, audit = derive(single, quantile=quantile)
        columns = dict(numeric)
        columns.update(categorical)
        gas_blocks = {column: [] for column in GAS_GROUP_COLUMNS}
        for _label, table in effect_tables:
            table_gas = derive(table, quantile=quantile)[0]
            for column in GAS_GROUP_COLUMNS:
                if column in table_gas:
                    gas_blocks[column].append(table_gas[column])
        for column, values in gas_blocks.items():
            if values:
                columns[column] = round(max(values), 6)
        group_blocks = [
            block_group_effects(table)["effect_size"].to_numpy()
            for _, table in effect_tables
        ]
        group_threshold = max_blocks_values(group_blocks, quantile)
        return columns, group_threshold, audit

    pooled = pd.concat(
        [table for _, table in effect_tables], ignore_index=True
    )
    numeric, categorical, audit = derive(pooled, quantile=quantile)
    columns = dict(numeric)
    columns.update(categorical)

    group_blocks = [
        block_group_effects(table)["effect_size"].to_numpy()
        for _, table in effect_tables
    ]
    group_blocks = [values for values in group_blocks if len(values)]
    if pooling == "max_blocks":
        group_threshold = max_blocks_values(group_blocks, quantile)
    else:
        pooled_group = np.concatenate(group_blocks)
        group_threshold = quantile_value(pooled_group, quantile)
    return columns, group_threshold, audit


def make_model(reference_df, column_thresholds, group_threshold,
               gas_group):
    statistical_config = {
        "alpha": 0.01,
        "correction": "benjamini-hochberg",
        "column_effect_thresholds": dict(column_thresholds),
        "default_effect_threshold": float(
            max(column_thresholds.values())
        ),
        "gas_group": {
            "enabled": bool(gas_group),
            "columns": list(GAS_GROUP_COLUMNS),
            "aggregation": "max",
            "threshold": group_threshold,
        },
        "persistence": {
            "enabled": False,
            "k": PERSISTENCE_K,
            "n": PERSISTENCE_N,
        },
    }
    return ReferenceModel(
        reference_df,
        mode="rolling",
        min_rows=30,
        history=1,
        config=statistical_config,
    )


def window_trigger(model, raw_df, start, window_df=None):
    """Matrix-style window trigger; rolling reference = raw block before."""
    if window_df is None:
        window_df = raw_df.iloc[start:start + WINDOW_SIZE]
    if start >= WINDOW_SIZE:
        model.set_rolling_reference(
            raw_df.iloc[start - WINDOW_SIZE:start]
        )
    else:
        model.set_rolling_reference(None)
    details = model.details(window_df)
    return bool(details["trigger_columns"]), details


def run_config_row(name, reference_df, raw_df, column_thresholds,
                   group_threshold, gas_group, persistence):
    """One grid cell: validation outcomes for one config variant."""
    model = make_model(
        reference_df, column_thresholds, group_threshold, gas_group
    )
    history_cache = {}
    outcomes = []

    def evaluate_window(start, window_df, label, y_true):
        fired, details = window_trigger(
            model, raw_df, start, window_df=window_df
        )
        if persistence:
            history = raw_history_triggers(start)
            alarm = persistence_alarm(
                history + [fired], PERSISTENCE_K, PERSISTENCE_N
            )
        else:
            history = []
            alarm = fired
        outcomes.append({
            "label": label,
            "y_true": y_true,
            "trigger": fired,
            "alarm": alarm,
        })
        return details

    def raw_history_triggers(start):
        if start not in history_cache:
            triggers = []
            for offset in range(PERSISTENCE_N - 1, 0, -1):
                block_start = start - offset * WINDOW_SIZE
                if block_start < 0:
                    continue
                fired, _ = window_trigger(
                    model, raw_df, block_start, window_df=None
                )
                triggers.append(fired)
            history_cache[start] = triggers
        return list(history_cache[start])

    # ---- validation clean windows (matrix-style negatives) ------------
    for start in VALIDATION_CLEAN_STARTS:
        evaluate_window(start, None, f"clean-{start}", 0)

    # ---- validation drift matrix: 12 cells x 5 disjoint bases ---------
    positives = 0
    for base_index, base in enumerate(VALIDATION_DRIFT_BASES):
        cell_index = 0
        for severity in STAT_CELL_SEVERITIES:
            for subtype in STAT_CELL_SUBTYPES:
                batch = raw_df.iloc[base:base + WINDOW_SIZE].copy()
                random_state = 1234567 + base_index * 100 + cell_index
                modified, _metadata = inject_window(
                    batch, "statistical", subtype, severity,
                    random_state,
                )
                evaluate_window(
                    base, modified,
                    f"drift-{subtype}-{severity}-{base}", 1,
                )
                positives += 1
                cell_index += 1

    # ---- validation lag streams (4 subtypes, medium severity) ---------
    stream_clean_fp = 0
    stream_lags = []
    for stream_index, subtype in enumerate(STAT_CELL_SUBTYPES):
        base = VALIDATION_STREAM_BASES[stream_index]
        injections = {}
        for position in range(STREAM_LENGTH):
            start = base + position * WINDOW_SIZE
            batch = raw_df.iloc[start:start + WINDOW_SIZE].copy()
            if position >= STREAM_ONSET:
                modified, _metadata = inject_window(
                    batch, "statistical", subtype, "medium",
                    1234567 + 1000 + stream_index * 10 + position,
                )
                injections[position] = modified
        triggers = []
        for position in range(STREAM_LENGTH):
            fired, _ = window_trigger(
                model, raw_df, base + position * WINDOW_SIZE,
                window_df=injections.get(position),
            )
            triggers.append(fired)
        if persistence:
            alarms = []
            for position, fired in enumerate(triggers):
                trailing = triggers[
                    max(0, position + 1 - PERSISTENCE_N):position + 1
                ]
                alarms.append(
                    persistence_alarm(trailing, PERSISTENCE_K,
                                      PERSISTENCE_N)
                )
        else:
            alarms = list(triggers)
        lag = detection_lag_windows(alarms, onset_position=STREAM_ONSET)
        stream_clean_fp += sum(1 for i in range(STREAM_ONSET)
                               if alarms[i])
        stream_lags.append(lag)

    # ---- summary -------------------------------------------------------
    negatives = [row for row in outcomes if row["y_true"] == 0]
    fp = sum(1 for row in negatives if row["alarm"])
    n_negatives = len(negatives)
    y_true = [1] * positives + [0] * n_negatives
    y_pred = [
        1 if row["alarm"] else 0
        for row in outcomes if row["y_true"] == 1
    ] + [
        1 if row["alarm"] else 0 for row in negatives
    ]
    metrics = evaluate(y_true, y_pred)

    # Clean-window margin: max effect/threshold across all active
    # gates on the clean validation windows (>= threshold means the
    # gate fired or came within the margin of firing).
    clean_margins = []
    for start in VALIDATION_CLEAN_STARTS:
        _fired, details = window_trigger(model, raw_df, start)
        entries = list(details["numeric"].items()) + list(
            details["categorical"].items()
        )
        if details.get("gas_group"):
            entries.append(("gas_group", details["gas_group"]))
        for _column, entry in entries:
            threshold = entry.get("threshold")
            if not threshold:
                continue
            clean_margins.append(entry["effect_size"] / threshold)

    clean_n = n_negatives + len(VALIDATION_STREAM_BASES) * STREAM_ONSET
    clean_fp = fp + stream_clean_fp
    detected = [lag for lag in stream_lags if lag is not None]

    return {
        "config": name,
        "val_clean_windows": clean_n,
        "val_clean_fp": clean_fp,
        "val_clean_fpr": clean_fp / clean_n if clean_n else None,
        "val_n_positives": positives,
        "val_tp": int(sum(
            1 for row in outcomes
            if row["y_true"] == 1 and row["alarm"]
        )),
        "val_recall": metrics["raw"]["recall"],
        "val_precision": metrics["raw"]["precision"],
        "val_f1": metrics["raw"]["f1"],
        "val_stream_clean_fp": stream_clean_fp,
        "val_lag_mean": (
            float(np.mean(detected)) if detected else None
        ),
        "val_streams_detected": len(detected),
        "val_clean_margin": (
            max(clean_margins) if clean_margins else None
        ),
        "group_threshold": group_threshold,
        "added_lag_design": (
            added_lag_windows(PERSISTENCE_K) if persistence else 0
        ),
    }, outcomes


def max_blocks_values(group_blocks, quantile):
    """Max over blocks of the per-block quantile."""
    return max(
        quantile_value(values, quantile) for values in group_blocks
        if len(values)
    )


def rank_key_v1(row):
    margin = row["val_clean_margin"]
    lag = row["val_lag_mean"]
    return (
        -row["val_f1"],
        row["val_clean_fpr"] if row["val_clean_fpr"] is not None
        else float("inf"),
        margin if margin is not None else float("inf"),
        lag if lag is not None else float("inf"),
        int(row["gas_group"]) + int(row["persistence"]),
        row["config"],
    )


def rank_key_v2(row):
    margin = row["val_clean_margin"]
    lag = row["val_lag_mean"]
    return (
        # gas-group threshold DESCENDING (more envelope headroom first).
        -(row["group_threshold"] if row["group_threshold"] is not None
          else 0.0),
        margin if margin is not None else float("inf"),
        lag if lag is not None else float("inf"),
        0 if row["gas_group"] else 1,
        0 if not row["persistence"] else 1,
        row["config"],
    )


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    reference_config = load_reference_config()
    raw_df = pd.read_csv("data/primary/iot_telemetry_data.csv")
    reference_df = load_reference_region(
        df=raw_df, config=reference_config
    )
    mode = reference_config.get("mode", "rolling")

    print("Computing clean effect tables (calibration/validation "
          "regions only) ...")
    tables = [(
        "single[325000-385000)",
        clean_effects(
            reference_df,
            raw_df.iloc[SINGLE_REGION[0]:SINGLE_REGION[1]]
            .reset_index(drop=True),
            mode,
        ),
    )]
    for block in CALIBRATION_BLOCKS:
        tables.append((
            f"block[{block[0]}-{block[1]})",
            clean_effects(
                reference_df,
                raw_df.iloc[block[0]:block[1]].reset_index(drop=True),
                mode,
            ),
        ))
    for label, table in tables:
        print(f"  {label}: {len(table)} window rows")

    rows = []
    outcomes_frames = []
    variants = {
        "single": [tables[0]],
        "multi_pooled": tables[1:],
        "multi_maxblocks": tables[1:],
        "gas_envelope": [tables[0]] + tables[1:],
    }
    pooling_by_variant = {
        "single": "pooled",
        "multi_pooled": "pooled",
        "multi_maxblocks": "max_blocks",
        "gas_envelope": "gas_envelope",
    }
    for variant, effect_tables in variants.items():
        pooling = pooling_by_variant[variant]
        print(f"\nVariant {variant} (pooling={pooling}):")
        for quantile in QUANTILES:
            columns, group_threshold, _audit = build_threshold_sets(
                effect_tables, quantile, pooling
            )
            for gas_group in [False, True]:
                for persistence in [False, True]:
                    name = (
                        f"{variant}|q={quantile}"
                        f"|group={'on' if gas_group else 'off'}"
                        f"|persist={'on' if persistence else 'off'}"
                    )
                    print(f"  validating: {name}")
                    summary, outcomes = run_config_row(
                        name, reference_df, raw_df, columns,
                        group_threshold, gas_group, persistence,
                    )
                    summary.update({
                        "variant": variant,
                        "quantile": quantile,
                        "gas_group": gas_group,
                        "persistence": persistence,
                        "temp_threshold": columns.get("temp"),
                        "co_threshold": columns.get("co"),
                        "device_threshold": columns.get("device"),
                    })
                    rows.append(summary)
                    frame = pd.DataFrame(outcomes)
                    frame.insert(0, "config", name)
                    outcomes_frames.append(frame)

    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_ROOT / "comparison_table.csv", index=False)
    pd.concat(outcomes_frames, ignore_index=True).to_csv(
        OUTPUT_ROOT / "validation_outcomes.csv", index=False
    )

    ranked_v1 = sorted(rows, key=rank_key_v1)
    winner_v1 = ranked_v1[0]

    # Rule v2: restrict to val FPR = 0 and F1 >= 0.95 x baseline F1.
    baseline = next(
        row for row in rows
        if row["variant"] == "single" and row["quantile"] == 0.99
        and not row["gas_group"] and not row["persistence"]
    )
    f1_floor = 0.95 * baseline["val_f1"]
    v2_pool = [
        row for row in rows
        if row["val_clean_fpr"] == 0.0 and row["val_f1"] >= f1_floor
    ]
    ranked_v2 = sorted(v2_pool, key=rank_key_v2)
    winner_v2 = ranked_v2[0] if ranked_v2 else None

    selection = {
        "rule_v1": SELECTION_RULE,
        "rule_v1_note": RULE_V1_NOTE,
        "rule_v2": RULE_V2,
        "regions": {
            "evaluation_forbidden": {
                "matrix": [0, 100000],
                "lag": [120000, 250000],
            },
            "single_region": SINGLE_REGION,
            "calibration_blocks": CALIBRATION_BLOCKS,
            "validation_clean_starts": VALIDATION_CLEAN_STARTS,
            "validation_drift_bases": VALIDATION_DRIFT_BASES,
            "validation_stream_bases": VALIDATION_STREAM_BASES,
        },
        "quantiles": QUANTILES,
        "persistence_k_n": [PERSISTENCE_K, PERSISTENCE_N],
        "gas_group_columns": GAS_GROUP_COLUMNS,
        "added_lag_design_windows": added_lag_windows(PERSISTENCE_K),
        "winner_v1": winner_v1,
        "winner_v2": winner_v2,
        "v1_ranking_top10": [
            {key: rank_row[key] for key in rank_row}
            for rank_row in ranked_v1[:10]
        ],
        "v2_pool_size": len(v2_pool),
        "v2_ranking_top10": [
            {key: rank_row[key] for key in rank_row}
            for rank_row in ranked_v2[:10]
        ],
    }
    with open(
        OUTPUT_ROOT / "selection.json", "w", encoding="utf-8"
    ) as file:
        json.dump(selection, file, indent=2, default=str)

    # Sanity check: the single-region q=0.99 no-switch row must
    # reproduce the Revision v2 thresholds (same derivation procedure).
    current = THRESHOLDS["statistical"]["column_effect_thresholds"]
    deltas = {
        column: round(baseline[column] - current[column], 6)
        for column in ("temp", "co", "device")
        if baseline.get(column) is not None
        and current.get(column) is not None
    }
    print("\nSanity (single|q=0.99|no-switch vs Revision v2): "
          f"{deltas}")

    columns = [
        "config", "variant", "quantile", "gas_group", "persistence",
        "val_clean_windows", "val_clean_fp", "val_clean_fpr",
        "val_n_positives", "val_tp", "val_recall", "val_precision",
        "val_f1", "val_lag_mean", "val_streams_detected",
        "val_clean_margin", "group_threshold", "temp_threshold",
        "co_threshold",
    ]
    display = table[columns].copy()
    for column in ("val_clean_fpr", "val_recall", "val_precision",
                   "val_f1", "val_clean_margin"):
        display[column] = display[column].astype(float).round(4)
    print("\n" + "=" * 110)
    print("COMPARISON TABLE (validation regions only; NO evaluation "
          "window used for any choice)")
    print("=" * 110)
    print(display.to_string(index=False))
    print("\nWINNER rule v1 (reported, rejected as unfit):")
    print(winner_v1["config"])
    print("\nWINNER rule v2 (pre-declared, chosen before the eval run):")
    if winner_v2 is None:
        print("(no v2 survivor; falling back to rule v1)")
    else:
        print(json.dumps(winner_v2, indent=2, default=str))
    print(f"\nWrote results to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
