"""
Baseline ladder detector configurations (Phase 7, Section Q).

B0  null-value monitoring only
B1  schema detector only
B2  KS+PSI (statistical) only
B3  schema + statistical
B4  schema + statistical + semantic (full unified detector)
B5  off-the-shelf tool (Evidently AI, default config) — qualitative

Each `run_*` function returns (decision, details) where decision is
"DRIFT"/"NO_DRIFT" and details is a JSON-serializable dict.
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from detectors.schema_detector import compare_schemas
from detectors.statistical_detector import compare_numeric_fields
from detectors.semantic_detector import (
    compare_semantic_contracts,
    validate_values_against_contract,
)
from detectors.unified import (
    unified_score,
    decide,
    STATISTICAL_DECISION_THRESHOLD,
)
from detectors.thresholds import THRESHOLDS

from drift_injection.phase7_generator import (
    apply_semantic_metadata,
)

CATEGORICAL_P_VALUE_THRESHOLD = float(
    THRESHOLDS["statistical"]["categorical_p_value_threshold"]
)

NUMERIC_COLUMNS_EXCLUDE = {"ts"}

def build_current_schema(df):
    columns = {}
    for column in df.columns:
        dtype = str(df[column].dtype)
        if dtype.startswith("float"):
            dtype_name = "float64"
        elif dtype.startswith("int"):
            dtype_name = "int64"
        elif dtype == "bool":
            dtype_name = "bool"
        else:
            dtype_name = "string"
        columns[column] = {
            "dtype": dtype_name,
            "nullable": bool(df[column].isna().any()),
        }
    return {"schema_version": "stream", "columns": columns}

def categorical_drift(reference_series, current_series):
    ref_counts = reference_series.value_counts()
    cur_counts = current_series.value_counts()
    categories = sorted(
        set(ref_counts.index.astype(str))
        | set(cur_counts.index.astype(str))
    )
    if not categories:
        return {"drift": False, "chi2": 0.0, "p_value": 1.0}
    ref_values = [int(ref_counts.get(c, 0)) for c in categories]
    cur_values = [int(cur_counts.get(c, 0)) for c in categories]
    observed = np.array([ref_values, cur_values], dtype=float)
    if observed.sum() == 0:
        return {"drift": False, "chi2": 0.0, "p_value": 1.0}
    chi2, p_value, _, _ = chi2_contingency(observed)
    return {
        "drift": bool(p_value < CATEGORICAL_P_VALUE_THRESHOLD),
        "chi2": float(chi2),
        "p_value": float(p_value),
    }

def statistical_details_for(reference_df, current_df):
    """Numeric KS/PSI per column + categorical chi2, as a dict."""
    numeric_columns = [
        column
        for column in reference_df.columns
        if column in current_df.columns
        and pd.api.types.is_numeric_dtype(reference_df[column])
        and pd.api.types.is_numeric_dtype(current_df[column])
        and column not in NUMERIC_COLUMNS_EXCLUDE
    ]
    numeric = {}
    if numeric_columns:
        try:
            numeric = compare_numeric_fields(
                reference_df,
                current_df,
                numeric_columns,
            )
        except Exception as exc:
            numeric = {"error": str(exc)}

    categorical = {}
    if (
        "device" in reference_df.columns
        and "device" in current_df.columns
    ):
        categorical["device"] = categorical_drift(
            reference_df["device"],
            current_df["device"],
        )
    return {"numeric": numeric, "categorical": categorical}

def semantic_events_for(reference_contract, current_df, metadata):
    current_contract = apply_semantic_metadata(
        reference_contract,
        metadata,
    )
    contract_events = compare_semantic_contracts(
        reference_contract,
        current_contract,
    )
    value_events = validate_values_against_contract(
        current_df,
        current_contract,
    )
    return contract_events + value_events

def run_b0_null_monitor(current_df):
    null_rates = current_df.isna().mean()
    max_null_rate = float(null_rates.max()) if len(null_rates) else 0.0
    decision = "DRIFT" if max_null_rate > 0.0 else "NO_DRIFT"
    return decision, {
        "max_null_rate": max_null_rate,
        "columns_with_nulls": [
            str(column)
            for column in null_rates.index
            if null_rates[column] > 0
        ],
    }

def run_b1_schema(reference_schema, current_df):
    current_schema = build_current_schema(current_df)
    events = compare_schemas(reference_schema, current_schema)
    return ("DRIFT" if events else "NO_DRIFT"), {
        "events": events,
        "current_schema": current_schema,
    }

def run_b2_statistical(reference_df, current_df):
    details = statistical_details_for(reference_df, current_df)
    score = _statistical_score(details)
    decision = (
        "DRIFT"
        if score >= STATISTICAL_DECISION_THRESHOLD
        else "NO_DRIFT"
    )
    return decision, {
        "statistical_details": details,
        "statistical_score": score,
    }

def _statistical_score(details):
    from detectors.unified import statistical_score

    return statistical_score(details)

def run_b3_schema_statistical(reference_schema, reference_df,
                              current_df):
    schema_decision, schema_details = run_b1_schema(
        reference_schema,
        current_df,
    )
    stat_decision, stat_details = run_b2_statistical(
        reference_df,
        current_df,
    )
    decision = (
        "DRIFT"
        if schema_decision == "DRIFT" or stat_decision == "DRIFT"
        else "NO_DRIFT"
    )
    return decision, {
        "schema": schema_details,
        "statistical": stat_details,
    }

def run_b4_unified(reference_schema, reference_contract,
                   reference_df, current_df, metadata):
    schema_decision, schema_details = run_b1_schema(
        reference_schema,
        current_df,
    )
    stat_details = statistical_details_for(reference_df, current_df)
    semantic_events = semantic_events_for(
        reference_contract,
        current_df,
        metadata,
    )

    schema_drift = schema_decision == "DRIFT"
    semantic_drift = len(semantic_events) > 0

    scores = unified_score(
        schema_drift,
        stat_details,
        semantic_drift,
    )
    decision = decide(
        scores["unified_score"],
        schema_score=scores["schema_score"],
        semantic_score=scores["semantic_score"],
        statistical_score=scores["statistical_score"],
    )
    return decision, {
        "schema_events": schema_details["events"],
        "semantic_events": semantic_events,
        "scores": scores,
    }

def run_semantic_only(reference_contract, current_df, metadata):
    events = semantic_events_for(
        reference_contract,
        current_df,
        metadata,
    )
    return ("DRIFT" if events else "NO_DRIFT"), {
        "events": events,
    }

def run_b5_evidently(reference_df, current_df):
    """
    Qualitative drift comparison with Evidently's DataDrift (default
    config). Returns (decision, details). If evidently is not
    installed, returns a structured "not available" marker instead of
    failing the whole run.
    """
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except ImportError:
        return "UNAVAILABLE", {
            "error": (
                "evidently not installed; B5 qualitative comparison "
                "skipped (pip install evidently)"
            ),
        }

    numeric_columns = [
        column
        for column in reference_df.columns
        if pd.api.types.is_numeric_dtype(reference_df[column])
        and not pd.api.types.is_bool_dtype(reference_df[column])
        and column != "ts"
        and column in current_df.columns
    ]

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_df[numeric_columns],
        current_data=current_df[numeric_columns],
    )
    summary = json.loads(report.json())

    drifted_columns = []
    try:
        for metric in summary.get("metrics", []):
            results = metric.get("result", {})
            if "drift_by_columns" not in results:
                continue
            for column, info in results["drift_by_columns"].items():
                if info.get("drift_detected"):
                    drifted_columns.append(column)
            break
    except (KeyError, IndexError, TypeError):
        drifted_columns = []

    decision = "DRIFT" if drifted_columns else "NO_DRIFT"
    return decision, {
        "drifted_columns": drifted_columns,
        "summary": summary,
    }

def baseline_configs():
    """name -> callable(reference, window_ctx) -> (decision, details)."""
    return {
        "B0_null": run_b0_null_monitor,
        "B1_schema": run_b1_schema,
        "B2_statistical": run_b2_statistical,
        "B3_schema_statistical": run_b3_schema_statistical,
        "B4_unified": run_b4_unified,
        "B5_evidently": run_b5_evidently,
    }
