"""
Classical streaming drift-detector adapters (Phase 10 literature
comparison).

Wraps river 0.21.0's standard implementations of ADWIN, DDM, and
KSWIN behind one `update(observation) -> drift_detected` interface,
and defines the observation stream supplied to them.

OBSERVATION STREAM PROTOCOL (documented; see docs/phase10_notes.md)
-------------------------------------------------------------------
* One evaluation unit is one Phase 7/8 window (1000 rows), labeled
  DRIFT/NO_DRIFT by the existing ground truth.
* The classical detectors are UNIVARIATE NUMERIC stream detectors.
  They do NOT see the schema channel, the semantic contract, or the
  reference profile that B4 receives. Each window is evaluated with:
      (a) a 2000-row reference warm-up = the PRIMARY dataset's rows
          immediately preceding the window (rows [start-2000, start),
          read from the reference dataset, never from other injected
          windows) — the natural "recent past" a streaming detector
          sees, so a value-unchanged window is a continuation of its
          warm-up and carries no spurious drift signal, then
      (b) the window's 1000 observations, in dataset row order.
  A window is flagged DRIFT iff a detector signals during the WINDOW
  portion (signals during warm-up are adaptation, ignored).
* Five per-feature numeric streams are watched in parallel,
  mirroring B4's max-over-columns statistical aggregation (OR over
  streams): temp, co, humidity, lpg, smoke. `ts` is excluded, exactly
  as in B4's statistical channel. Categorical and semantic
  information is NOT part of the observation stream: these are
  univariate NUMERICAL detectors, so the categorical `device` column
  and the semantic contract are outside their observational scope
  (documented protocol limitation, like schema-structural drift).
* Reference context (warm-up): the 2000 rows of the PRIMARY dataset
  immediately preceding each window (rows [start-2000, start), read
  from the reference dataset, never from other injected windows). A
  value-unchanged window is therefore a continuation of its warm-up
  (no drift signal), while a value-changed window differs from it.
* ADWIN and DDM observe every row. DDM's documented input contract is
  a BINARY error stream (river drift.binary.DDM: "x is an entry in a
  stream of bits, where 1 indicates error/failure"), so DDM observes
  a derived binary "surprise" stream: bit = 1 iff |x - mu_ref| >
  k * sigma_ref, with mu_ref/sigma_ref fit on the warm-up (k = 2.0,
  documented adapter choice). This is the standard unsupervised use
  of DDM and does not alter the detector itself.
* KSWIN (river 0.21) is a pure-Python detector whose per-observation
  cost is ~100x ADWIN's (a scipy KS test on every update). Two
  documented adapter choices bound the cost without changing KSWIN's
  per-test power (always stat_size=30 vs 30):
      - stride=10: KSWIN observes every 10th row, so its 100-value
        sliding window spans exactly one 1000-row window;
      - cold start: the sliding window is pre-filled via river's
        documented `window=` parameter with the last 100 decimated
        warm-up values, so the warm-up context is present without
        being re-fed.
* Columns that are absent (removed/renamed) or non-numeric (type
  change) in a window are SKIPPED for that window (the stream cannot
  be observed); missing values (nullability change) are dropped. This
  is documented as a protocol limitation: schema-structural drift is
  outside the numerical-stream representation.

PARAMETERS (library defaults unless noted)
------------------------------------------
    ADWIN  delta=0.002, clock=32            (river defaults)
    KSWIN  alpha=0.005, window_size=100,
           stat_size=30                     (river defaults)
           + adapter: stride=10, cold-start prefill, seed=42
    DDM    warm_start=30, warning_threshold=2.0,
           drift_threshold=3.0              (river defaults)
           + adapter: deviation k=2.0 sigma, reference fit on warm-up

All detectors are deterministic given the same observations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from river import drift
from river.drift.binary import DDM as _RiverDDM

DETECTOR_NAMES = ["adwin", "ddm", "kswin"]

# Numeric sensor columns (mirrors B4 statistical channel; ts
# excluded). Categorical columns (device) are NOT included: these are
# univariate numerical detectors, and categorical/semantic
# information is outside their observational scope (documented).
NUMERIC_FEATURES = ["temp", "co", "humidity", "lpg", "smoke"]
FEATURES = NUMERIC_FEATURES

# Reference warm-up length: the 2000 primary-dataset rows immediately
# preceding each window (read from the reference dataset).
WARMUP_SIZE = 2000

# DDM adapter: deviation multiplier for the binary surprise stream.
DDM_DEVIATION_K = 2.0

# DDM retuned adapter (Phase 10 retune): wider deviation threshold
# avoids p_min=0 degeneracy when warm-up surprise rate is zero.
DDM_DEVIATION_K_RETUNED = 3.0

# KSWIN adapter: observation decimation factor (see module docstring).
KSWIN_STRIDE = 10

# KSWIN default alpha (river default).
KSWIN_ALPHA_DEFAULT = 0.005

# KSWIN retuned alpha (Phase 10 retune): Bonferroni-corrected for
# ~500 total tests per window (100 per feature × 5 features),
# targeting ~5% compound OR-over-5-features false-alarm rate.
KSWIN_ALPHA_RETUNED = 0.0001

# Reproducibility seed for river KSWIN's internal uniform sampling.
KSWIN_SEED = 42


class DetectorConfig:
    """One detector's configuration, for reporting (Section 9-A)."""

    def __init__(self, name, input_stream, parameters, library):
        self.name = name
        self.input_stream = input_stream
        self.parameters = parameters
        self.library = library

    def as_dict(self):
        """Config fields as a plain dict for the report tables."""
        return {
            "detector": self.name,
            "input_stream": self.input_stream,
            "parameters": self.parameters,
            "library": self.library,
        }


def detector_configs():
    """The detector configurations (name, stream, parameters, library) reported in the paper."""
    return [
        DetectorConfig(
            "adwin",
            "raw numeric values, every row; 5 per-feature streams "
            "(temp, co, humidity, lpg, smoke) against local 2000-row "
            "warm-up",
            "delta=0.002, clock=32 (river defaults)",
            "river 0.21.0 (river.drift.ADWIN)",
        ),
        DetectorConfig(
            "ddm",
            "binary surprise stream, every row "
            "(|x-mu|>2*sigma vs local warm-up)",
            "warm_start=30, warning_threshold=2.0, "
            "drift_threshold=3.0 (river defaults); k=2.0 (adapter)",
            "river 0.21.0 (river.drift.binary.DDM)",
        ),
        DetectorConfig(
            "kswin",
            "raw numeric values, every 10th row; 5 per-feature streams "
            "(sliding window spans 1000 rows)",
            "alpha=0.005, window_size=100, stat_size=30 "
            "(river defaults); stride=10 + cold-start prefill "
            "(adapter), seed=42",
            "river 0.21.0 (river.drift.KSWIN)",
        ),
    ]


def new_detector(name, window=None, kswin_alpha=None):
    """Return a fresh detector instance exposing update(x) and
    .drift_detected (resets on each call).

    `window` is KSWIN's documented cold-start parameter: pre-filled
    sliding-window values so the reference context need not be
    re-fed. Ignored for ADWIN/DDM (they have no cold-start API).
    `kswin_alpha` overrides the default alpha for KSWIN.
    """
    if name == "adwin":
        return drift.ADWIN()
    if name == "ddm":
        return _RiverDDM()
    if name == "kswin":
        alpha = kswin_alpha if kswin_alpha is not None else KSWIN_ALPHA_DEFAULT
        return drift.KSWIN(alpha=alpha, seed=KSWIN_SEED, window=window)
    raise ValueError(f"Unknown detector: {name}")


def feature_values(window_df, feature, reference_df=None):
    """Extract the numeric observation vector for one feature from a
    window, handling absent columns and nulls.

    Returns (values, available): values is a float array (possibly
    empty) or None when the feature cannot be observed in this window
    (column absent or non-numeric — schema-structural drift).
    `reference_df` is accepted for interface symmetry (device encoding
    was removed when categorical columns were excluded from the
    stream) and is unused.
    """
    del reference_df  # interface symmetry only
    if feature not in window_df.columns:
        return None, False
    series = window_df[feature]
    if not pd.api.types.is_numeric_dtype(series):
        return None, False  # type_change: stream not observable
    return series.dropna().astype(float).to_numpy(), True


def surprise_bits(values, reference_values, k=DDM_DEVIATION_K):
    """Binary DDM stream: bit = 1 iff the observation deviates from
    the reference distribution by more than k sigma."""
    ref = np.asarray(reference_values, dtype=float)
    if len(ref) == 0:
        return np.zeros(len(values), dtype=int)
    mu = float(ref.mean())
    sigma = float(ref.std())
    if sigma == 0 or not np.isfinite(sigma):
        return np.zeros(len(values), dtype=int)
    vals = np.asarray(values, dtype=float)
    return (np.abs(vals - mu) > k * sigma).astype(int)


def _feed(detector, values, start_index=0, report_from=0):
    """Feed observations; return True iff the detector signals at an
    index >= report_from (indices are in the fed stream)."""
    for i in range(start_index, len(values)):
        detector.update(float(values[i]))
        if detector.drift_detected and i >= report_from:
            return True
    return False


def evaluate_window(window_df, warmup_df, detector_name,
                    features=FEATURES, ddm_k=None, kswin_alpha=None):
    """
    Per-window evaluation: warm-up observations then the window's
    observations, per feature. Returns True (DRIFT) iff the detector
    signals during the WINDOW portion of ANY observable feature.

    ADWIN/DDM: full-rate stream, warm-up fed first (its signals are
    adaptation, ignored; window signals count).
    KSWIN: stride-decimated stream; the sliding window is pre-filled
    (cold start) with the last 100 decimated warm-up values, so every
    fed observation is a window observation and any signal counts.

    Optional overrides for Phase 10 retune:
        ddm_k: deviation multiplier (default DDM_DEVIATION_K).
        kswin_alpha: per-test alpha (default KSWIN_ALPHA_DEFAULT).
    """
    _k = ddm_k if ddm_k is not None else DDM_DEVIATION_K
    _alpha = kswin_alpha if kswin_alpha is not None else KSWIN_ALPHA_DEFAULT
    stride = KSWIN_STRIDE if detector_name == "kswin" else 1
    for feature in features:
        warm_vals, warm_ok = feature_values(warmup_df, feature)
        win_vals, win_ok = feature_values(
            window_df, feature, reference_df=warmup_df
        )
        if not warm_ok or not win_ok or len(win_vals) == 0:
            continue

        if detector_name == "ddm":
            detector = new_detector("ddm")
            obs = surprise_bits(
                np.concatenate([warm_vals, win_vals]),
                warm_vals, k=_k,
            )
            if _feed(detector, obs, report_from=len(warm_vals)):
                return True
        elif detector_name == "kswin":
            warm_dec = warm_vals[::stride]
            win_dec = win_vals[::stride]
            prefill = (
                warm_dec[-100:].tolist() if len(warm_dec) else None
            )
            detector = new_detector("kswin", window=prefill,
                                    kswin_alpha=_alpha)
            if _feed(detector, win_dec, report_from=0):
                return True
        else:  # adwin: full rate, warm-up first
            detector = new_detector("adwin")
            obs = np.concatenate([warm_vals, win_vals])
            if _feed(detector, obs, report_from=len(warm_vals)):
                return True
    return False

