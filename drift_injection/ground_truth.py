from datetime import datetime, timezone
from pathlib import Path
import uuid

import pandas as pd


GROUND_TRUTH_COLUMNS = [
    "event_id",
    "timestamp",
    "drift_type",
    "subtype",
    "feature",
    "severity",
    "injected",
    "ground_truth",
    # Phase 6: machine-readable flag for windows whose drift is
    # expected to be missed under a documented measurement
    # limitation (e.g. variance shift under max-over-columns
    # aggregation). Empty string means "not flagged". Phase 7's
    # precision/recall must account for these windows instead of
    # silently penalizing the detector.
    "expected_false_negative",
]


def create_event(
    drift_type,
    subtype,
    feature,
    severity,
    injected=True,
    ground_truth=None,
    timestamp=None,
    event_id=None,
    expected_false_negative="",
):
    """
    Create one ground-truth event.

    Every event is explicitly labelled as DRIFT or NO_DRIFT.

    expected_false_negative: non-empty flag string for windows whose
    drift is expected to be missed under a documented measurement
    limitation (Phase 6 variance-shift aggregation caveat). Phase 7
    reads this to avoid penalizing the detector for known issues.
    """

    if ground_truth is None:
        ground_truth = "DRIFT" if injected else "NO_DRIFT"

    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    if event_id is None:
        event_id = str(uuid.uuid4())

    return {
        "event_id": event_id,
        "timestamp": timestamp,
        "drift_type": drift_type,
        "subtype": subtype,
        "feature": feature,
        "severity": severity,
        "injected": bool(injected),
        "ground_truth": ground_truth,
        "expected_false_negative": expected_false_negative,
    }


def append_event(
    event,
    output_path="results/raw/ground_truth.csv",
):
    """
    Append one event to the ground-truth CSV.
    Creates directories/file if they do not exist.
    """

    output = Path(output_path)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    row = pd.DataFrame(
        [event],
        columns=GROUND_TRUTH_COLUMNS,
    )

    file_exists = output.exists()

    row.to_csv(
        output,
        mode="a",
        header=not file_exists,
        index=False,
    )

    return output


def log_injection(
    drift_type,
    subtype,
    feature,
    severity,
    output_path="results/raw/ground_truth.csv",
    timestamp=None,
    expected_false_negative="",
):
    """
    Log a positive drift event.
    """

    event = create_event(
        drift_type=drift_type,
        subtype=subtype,
        feature=feature,
        severity=severity,
        injected=True,
        ground_truth="DRIFT",
        timestamp=timestamp,
        expected_false_negative=expected_false_negative,
    )

    append_event(
        event,
        output_path,
    )

    return event


def log_no_drift(
    subtype="normal_noise",
    feature=None,
    output_path="results/raw/ground_truth.csv",
    timestamp=None,
    expected_false_negative="",
):
    """
    Log an explicit negative/control event.

    This is essential for later FPR, precision,
    recall and F1 calculations.
    """

    event = create_event(
        drift_type="normal",
        subtype=subtype,
        feature=feature,
        severity="none",
        injected=False,
        ground_truth="NO_DRIFT",
        timestamp=timestamp,
        expected_false_negative=expected_false_negative,
    )

    append_event(
        event,
        output_path,
    )

    return event


def load_ground_truth(
    input_path="results/raw/ground_truth.csv",
):
    """
    Load the complete ground-truth file.
    """

    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Ground-truth file not found: {path}"
        )

    return pd.read_csv(path)