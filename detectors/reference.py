"""Reference builders for the statistical detector (Revision, Step 1).

The primary dataset is time-ordered and mixes three devices whose
distributions differ substantially. Comparing every window to ONE global
reference therefore makes even clean windows look drifted (KS ~0.44 on
co/lpg/smoke/humidity), which forced the old thresholds to absurd values
and made the detector deaf to real drift. This module supports three
reference modes, selectable in configs/reference.yaml:

    global      one global reference sample (legacy behaviour)
    per_device  a separate reference sample per device; a window is split
                by device and each device slice is compared to its own
                device reference
    rolling     compare each window to the immediately preceding clean
                window(s), with a fallback to `global` before any clean
                history exists

Categorical columns (e.g. `device`) are ALWAYS compared globally: a
device slice contains only one device value, so a per-device chi-square
test on `device` is meaningless (and category shift is injected into
`device`).

Device slices with fewer than `min_rows_per_device` rows are skipped and
recorded in `details["skipped_slices"]`; they are never silently dropped.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from detectors.statistical_detector import (
    benjamini_hochberg,
    categorical_column_metrics,
    numeric_column_metrics,
)
from detectors.thresholds import THRESHOLDS

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REFERENCE_CONFIG_PATH = PROJECT_ROOT / "configs" / "reference.yaml"

MODES = ("global", "per_device", "rolling")

# Columns never treated as drift candidates by the statistical channel.
EXCLUDED_NUMERIC_COLUMNS = {"ts"}

# Revision v3, option (a): collinear gas-sensor columns can be tested as
# ONE sensor group instead of three separate triggers.
GAS_GROUP_LABEL = "gas_group"
GAS_GROUP_DEFAULT_COLUMNS = ["co", "lpg", "smoke"]


def gas_group_config(config):
    """Parse the optional gas-group switch from a statistical config dict.

    Expected shape (all keys optional except the switch):

        gas_group:
          enabled: true
          columns: [co, lpg, smoke]
          aggregation: max        # or mean
          threshold: 0.43         # ONE group-level calibrated threshold

    Returns None when the group gate is disabled (legacy behaviour).
    """
    group = (config or {}).get("gas_group") or {}
    if not isinstance(group, dict) or not group.get("enabled"):
        return None
    columns = [
        str(column) for column in group.get(
            "columns", GAS_GROUP_DEFAULT_COLUMNS
        )
    ]
    aggregation = str(group.get("aggregation", "max"))
    if aggregation not in ("max", "mean"):
        raise ValueError(
            f"Unknown gas_group aggregation '{aggregation}'; "
            "expected 'max' or 'mean'"
        )
    threshold = group.get("threshold")
    return {
        "columns": columns,
        "aggregation": aggregation,
        "threshold": float(threshold) if threshold is not None else None,
    }


def load_reference_config(path=None):
    """Load configs/reference.yaml (mode, min rows, row regions)."""
    path = Path(path) if path else REFERENCE_CONFIG_PATH
    with open(path, "r") as file:
        return yaml.safe_load(file)


def numeric_columns_of(df):
    """Numeric candidate columns (bool included; `ts` excluded)."""
    return [
        column
        for column in df.columns
        if column not in EXCLUDED_NUMERIC_COLUMNS
        and pd.api.types.is_numeric_dtype(df[column])
        and not pd.api.types.is_timedelta64_dtype(df[column])
        and column not in categorical_columns_of(df)
    ]


def categorical_columns_of(df):
    """Object/string candidate columns (e.g. `device`)."""
    return [
        column
        for column in df.columns
        if pd.api.types.is_object_dtype(df[column])
        or (
            pd.api.types.is_string_dtype(df[column])
            and not pd.api.types.is_numeric_dtype(df[column])
        )
    ]


def _column_threshold(config, column, kind):
    """Calibrated effect-size threshold for one column (None if absent)."""
    thresholds = (config or {}).get("column_effect_thresholds", {}) or {}
    if column in thresholds:
        return float(thresholds[column])
    if kind == "categorical":
        default = (config or {}).get("default_categorical_effect_threshold")
    else:
        default = (config or {}).get("default_effect_threshold")
    if default is None:
        return None
    return float(default)


def _alpha(config):
    return float((config or {}).get("alpha", 0.01))


class ReferenceModel:
    """Builds per-column comparisons of a window against a reference.

    Parameters
    ----------
    reference_df : pd.DataFrame
        The (clean) reference sample.
    mode : str
        One of `global`, `per_device`, `rolling`.
    min_rows : int
        Minimum rows for a device slice / reference sample to be compared.
    history : int
        Number of preceding clean windows pooled in `rolling` mode.
    """

    def __init__(
        self,
        reference_df,
        mode="per_device",
        min_rows=30,
        history=1,
        config=None,
    ):
        if mode not in MODES:
            raise ValueError(
                f"Unknown reference mode '{mode}'; expected one of {MODES}"
            )
        self.reference_df = reference_df
        self.mode = mode
        self.min_rows = int(min_rows)
        self.history = int(history)
        self.config = config if config is not None else THRESHOLDS.get(
            "statistical", {}
        )
        self.gas_group = gas_group_config(self.config)
        self.numeric_columns = numeric_columns_of(reference_df)
        self.categorical_columns = categorical_columns_of(reference_df)
        self._history = []
        self._rolling_reference = None
        self._device_references = None
        self._skipped = []
        if mode == "per_device":
            if "device" not in reference_df.columns:
                raise ValueError(
                    "per_device reference mode requires a `device` column"
                )
            self._device_references = {
                str(device): group  # noqa: E501
                for device, group in reference_df.groupby("device")
                if len(group) >= self.min_rows
            }

    # ------------------------------------------------------------------
    # rolling history
    # ------------------------------------------------------------------
    def reset(self):
        """Clear rolling history (call before each stream / run)."""
        self._history = []

    def update_clean(self, window_df):
        """Push a window known to be clean onto the rolling history."""
        self._history.append(window_df.copy())
        if len(self._history) > self.history:
            self._history = self._history[-self.history:]

    def set_rolling_reference(self, reference_df):
        """Set an explicit preceding clean reference block (offline replay).

        The Phase 7 harness passes the raw row block immediately before the
        window; no injection is ever written back to the raw dataset, so
        that block is clean by construction.
        """
        self._rolling_reference = reference_df

    # ------------------------------------------------------------------
    # slices
    # ------------------------------------------------------------------
    def numeric_slices(self, window_df):
        """(label, reference_df, window_df) triples for the numeric channel.

        Appends skipped slices to self._skipped.
        """
        if self.mode == "rolling" and self._rolling_reference is not None:
            reference = self._rolling_reference
            if len(reference) >= self.min_rows:
                return [("rolling", reference, window_df)]
            self._skipped.append(
                {
                    "label": "rolling",
                    "reason": "explicit reference below min_rows",
                    "rows": int(len(reference)),
                }
            )
            return [("global_fallback", self.reference_df, window_df)]

        if self.mode == "rolling" and self._history:
            if len(self._history) == 1:
                history = self._history[0]
            else:
                history = pd.concat(self._history, ignore_index=True)
            if len(history) >= self.min_rows:
                return [("rolling", history, window_df)]
            self._skipped.append(
                {
                    "label": "rolling",
                    "reason": "history below min_rows",
                    "rows": int(len(history)),
                }
            )
            return [("global_fallback", self.reference_df, window_df)]

        if self.mode == "per_device" and self._device_references:
            slices = []
            for device, group in window_df.groupby("device"):
                reference = self._device_references.get(str(device))
                if reference is None:
                    self._skipped.append(
                        {
                            "label": str(device),
                            "reason": "no reference sample for device",
                            "rows": int(len(group)),
                        }
                    )
                    continue
                if len(group) < self.min_rows:
                    self._skipped.append(
                        {
                            "label": str(device),
                            "reason": "window slice below min_rows",
                            "rows": int(len(group)),
                        }
                    )
                    continue
                slices.append((str(device), reference, group))
            if not slices:
                self._skipped.append(
                    {
                        "label": "window",
                        "reason": "no device slice usable; global fallback",
                        "rows": int(len(window_df)),
                    }
                )
                return [("global_fallback", self.reference_df, window_df)]
            return slices

        return [("global", self.reference_df, window_df)]

    # ------------------------------------------------------------------
    # details
    # ------------------------------------------------------------------
    def numeric_details(self, window_df):
        """Per-column numeric metrics, aggregated over device slices.

        For each column we keep the slice with the largest effect size and
        Bonferroni-correct its p-value across all device slices of the
        window (a column is tested once per slice).
        """
        details = {}
        slices = self.numeric_slices(window_df)
        n_slices = max(1, len(slices))
        for column in self.numeric_columns:
            best = None
            min_p = 1.0
            for _label, reference_df, window_df_slice in slices:
                if column not in reference_df.columns or (
                    column not in window_df_slice.columns
                ):
                    continue
                reference_values = pd.to_numeric(
                    reference_df[column], errors="coerce"
                ).dropna()
                current_values = pd.to_numeric(
                    window_df_slice[column], errors="coerce"
                ).dropna()
                if (
                    len(reference_values) < self.min_rows
                    or len(current_values) < self.min_rows
                ):
                    self._skipped.append(
                        {
                            "label": column,
                            "reason": "column values below min_rows",
                            "rows": int(len(current_values)),
                        }
                    )
                    continue
                metrics = numeric_column_metrics(
                    reference_values.to_numpy(),
                    current_values.to_numpy(),
                )
                if (
                    best is None
                    or metrics["effect_size"] > best["effect_size"]
                ):
                    best = dict(metrics)
                    best["slice"] = _label
                min_p = min(min_p, metrics["ks_p_value"])
            if best is None:
                continue
            best["ks_p_value"] = min(1.0, min_p * n_slices)
            best["p_value"] = best["ks_p_value"]
            self._finalize(best, column, "numeric")
            details[column] = best
        return details

    def categorical_details(self, window_df):
        """Per-column categorical metrics, always against the global reference."""
        details = {}
        for column in self.categorical_columns:
            if (
                column not in self.reference_df.columns
                or column not in window_df.columns
            ):
                continue
            reference_values = self.reference_df[column].dropna()
            current_values = window_df[column].dropna()
            if len(current_values) < 1:
                continue
            metrics = categorical_column_metrics(
                reference_values.to_numpy(),
                current_values.to_numpy(),
            )
            self._finalize(metrics, column, "categorical")
            details[column] = metrics
        return details

    def _finalize(self, metrics, column, kind):
        threshold = _column_threshold(self.config, column, kind)
        effect = metrics["effect_size"]
        metrics["threshold"] = threshold
        if threshold and threshold > 0:
            metrics["score"] = max(0.0, min(1.0, effect / threshold))
        else:
            metrics["score"] = 0.0
        metrics["kind"] = kind

    def _gas_group_entry(self, numeric, group):
        """Aggregate the group members into ONE composite test entry.

        The group effect is `max` (or `mean`) of the member effect sizes,
        and the group p-value is the Bonferroni-style minimum across
        members (the group is rejected if at least one member is). The
        group is compared against ONE group-level calibrated threshold,
        so a window can fire the gas channel at most once instead of
        once per gas column.
        """
        members = [
            column for column in group["columns"] if column in numeric
        ]
        if not members:
            return None, set()

        effects = [float(numeric[c]["effect_size"]) for c in members]
        p_values = [float(numeric[c].get("p_value", 1.0)) for c in members]
        if group["aggregation"] == "mean":
            effect = float(np.mean(effects))
        else:
            effect = float(max(effects))

        threshold = group["threshold"]
        if threshold is None:
            member_thresholds = [
                numeric[c].get("threshold") for c in members
                if numeric[c].get("threshold") is not None
            ]
            threshold = max(member_thresholds) if member_thresholds else None

        entry = {
            "kind": "gas_group",
            "members": members,
            "aggregation": group["aggregation"],
            "effect_size": effect,
            "p_value": min(p_values),
            "member_effects": {
                c: float(numeric[c]["effect_size"]) for c in members
            },
            "threshold": (
                float(threshold) if threshold is not None else None
            ),
        }
        if threshold is not None and threshold > 0:
            entry["score"] = max(0.0, min(1.0, effect / threshold))
        else:
            entry["score"] = 0.0
        return entry, set(members)

    def details(self, window_df):
        """Full per-column comparison + drift decision for one window."""
        self._skipped = []
        numeric = self.numeric_details(window_df)
        categorical = self.categorical_details(window_df)

        group_entry = None
        members = set()
        if self.gas_group is not None:
            group_entry, members = self._gas_group_entry(
                numeric, self.gas_group
            )

        # Active gate entries: every column not swallowed by the group
        # gate, plus the composite group entry itself. Members keep
        # their metrics for diagnosis but cannot trigger on their own.
        entries = [
            (column, metrics)
            for column, metrics in numeric.items()
            if column not in members
        ] + list(categorical.items())
        if group_entry is not None:
            entries.append((GAS_GROUP_LABEL, group_entry))

        rejected = benjamini_hochberg(
            [metrics.get("p_value", 1.0) for _, metrics in entries],
            _alpha(self.config),
        )

        flags = {}
        triggers = []
        for (column, metrics), significant in zip(entries, rejected):
            threshold = metrics.get("threshold")
            effect_ok = (
                threshold is not None
                and metrics["effect_size"] >= threshold
            )
            drift = bool(significant and effect_ok)
            metrics["bh_significant"] = bool(significant)
            metrics["effect_ok"] = bool(effect_ok)
            metrics["drift"] = drift
            flags[column] = drift
            if drift:
                triggers.append(column)

        for member in sorted(members):
            metrics = numeric[member]
            metrics["gated_by_group"] = True
            metrics["drift"] = False
            metrics["bh_significant"] = False
            metrics["effect_ok"] = False
            flags[member] = False

        scores = [
            metrics.get("score", 0.0) for _, metrics in entries
        ]
        statistical_score = max(scores) if scores else 0.0

        return {
            "reference_mode": self.mode,
            "min_rows": self.min_rows,
            "numeric": numeric,
            "categorical": categorical,
            "gas_group": group_entry,
            "column_flags": flags,
            "trigger_columns": triggers,
            "statistical_drift": bool(triggers),
            "statistical_score": float(min(1.0, statistical_score)),
            "skipped_slices": list(self._skipped),
        }


def build_reference_model(reference_df, config=None):
    """Build a ReferenceModel from configs/reference.yaml + thresholds."""
    config = config or load_reference_config()
    statistical = THRESHOLDS.get("statistical", {})
    return ReferenceModel(
        reference_df,
        mode=config.get("mode", "per_device"),
        min_rows=int(config.get("min_rows_per_device", 30)),
        history=int(config.get("rolling_history", 1)),
        config=statistical,
    )


def load_reference_region(df=None, config=None):
    """Reference sample rows defined by configs/reference.yaml.

    Never overlaps the calibration region or the evaluation regions.
    """
    config = config or load_reference_config()
    if df is None:
        df = pd.read_csv(
            PROJECT_ROOT / "data" / "primary" / "iot_telemetry_data.csv"
        )
    start, end = config["reference_region"]
    return df.iloc[int(start):int(end)].reset_index(drop=True)


def load_calibration_region(df=None, config=None):
    """Calibration-only rows (threshold derivation); disjoint from the rest."""
    config = config or load_reference_config()
    if df is None:
        df = pd.read_csv(
            PROJECT_ROOT / "data" / "primary" / "iot_telemetry_data.csv"
        )
    start, end = config["calibration_region"]
    return df.iloc[int(start):int(end)].reset_index(drop=True)


def save_json(payload, path):
    """Write a JSON payload, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        json.dump(payload, file, indent=2, default=str)
    return path
