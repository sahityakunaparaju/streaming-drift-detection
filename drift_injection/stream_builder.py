import json
from pathlib import Path

import pandas as pd

from drift_injection.injectors import (
    inject_add_column,
    inject_remove_column,
    inject_mean_shift,
    inject_variance_shift,
    inject_category_shift,
    inject_unit_change,
    inject_normal_noise,
)

from drift_injection.flagging import (
    expected_false_negative_reason,
)
from drift_injection.ground_truth import (
    create_event,
    append_event,
)


# Phase 6/7: variance-shift and distribution-shift detection under
# max-over-columns statistical aggregation are known, documented
# measurement limitations (docs/phase6_notes.md). Windows injecting
# them are flagged via the shared rule in drift_injection/flagging.py
# so Phase 7's precision/recall computation can account for the
# expected false negative instead of silently penalizing the
# detector. The controlled stream below contains a variance-shift
# window (window_005), flagged by the rule; distribution-shift
# windows are exercised end-to-end by the Phase 7 matrix generator
# (drift_injection/phase7_generator.py), which applies the same rule
# at every event-creation site.


def build_stream(
    input_path,
    output_dir="results/raw/stream",
    ground_truth_path="results/raw/ground_truth.csv",
    window_size=1000,
    max_windows=10,
):
    """
    Build a controlled static stream from the primary dataset.

    Each window is either:
        - clean
        - normal noise
        - schema drift
        - statistical drift
        - semantic drift

    No Kafka/Spark is involved here.
    """

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    ground_truth_path = Path(ground_truth_path)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # build_stream generates the complete controlled set from
    # scratch: start with a fresh ground-truth file so reruns never
    # mix old rows (e.g. a pre-Phase-6 schema) with new ones.
    if ground_truth_path.exists():
        ground_truth_path.unlink()

    df = pd.read_csv(input_path)

    available_windows = len(df) // window_size

    num_windows = min(
        available_windows,
        max_windows,
    )

    if num_windows == 0:
        raise ValueError(
            "Dataset is smaller than window_size"
        )

    generated_windows = []

    for window_id in range(num_windows):

        start = window_id * window_size
        end = start + window_size

        batch = df.iloc[start:end].copy()

        # --------------------------------------------------
        # Schedule
        # --------------------------------------------------

        # Semantic metadata produced by this window's
        # injection (None unless an injector returns it).
        metadata = None

        # Window 0: clean
        if window_id == 0:

            modified = batch

            event = create_event(
                drift_type="normal",
                subtype="clean",
                feature=None,
                severity="none",
                injected=False,
                ground_truth="NO_DRIFT",
                timestamp=str(batch["ts"].iloc[0]),
            )

        # Window 1: normal noise
        elif window_id == 1:

            modified = inject_normal_noise(
                batch,
                columns=["temp"],
                noise_fraction=0.01,
                random_state=42,
            )

            event = create_event(
                drift_type="normal",
                subtype="normal_noise",
                feature="temp",
                severity="none",
                injected=False,
                ground_truth="NO_DRIFT",
                timestamp=str(batch["ts"].iloc[0]),
            )

        # Window 2: add column
        elif window_id == 2:

            modified = inject_add_column(
                batch,
                column="battery_level",
                value=100.0,
            )

            event = create_event(
                drift_type="schema",
                subtype="added",
                feature="battery_level",
                severity="high",
                injected=True,
                ground_truth="DRIFT",
                timestamp=str(batch["ts"].iloc[0]),
            )

        # Window 3: remove column
        elif window_id == 3:

            modified = inject_remove_column(
                batch,
                column="motion",
            )

            event = create_event(
                drift_type="schema",
                subtype="removed",
                feature="motion",
                severity="high",
                injected=True,
                ground_truth="DRIFT",
                timestamp=str(batch["ts"].iloc[0]),
            )

        # Window 4: mean shift
        elif window_id == 4:

            modified = inject_mean_shift(
                batch,
                column="temp",
                severity="medium",
            )

            event = create_event(
                drift_type="statistical",
                subtype="mean_shift",
                feature="temp",
                severity="medium",
                injected=True,
                ground_truth="DRIFT",
                timestamp=str(batch["ts"].iloc[0]),
            )

        # Window 5: variance shift
        elif window_id == 5:

            modified = inject_variance_shift(
                batch,
                column="temp",
                severity="medium",
            )

            event = create_event(
                drift_type="statistical",
                subtype="variance_shift",
                feature="temp",
                severity="medium",
                injected=True,
                ground_truth="DRIFT",
                timestamp=str(batch["ts"].iloc[0]),
                expected_false_negative=(
                    expected_false_negative_reason(
                        "variance_shift"
                    )
                    or ""
                ),
            )

        # Window 6: category shift
        elif window_id == 6:

            modified = inject_category_shift(
                batch,
                column="device",
                category=batch["device"].iloc[0],
                severity="medium",
            )

            event = create_event(
                drift_type="statistical",
                subtype="category_shift",
                feature="device",
                severity="medium",
                injected=True,
                ground_truth="DRIFT",
                timestamp=str(batch["ts"].iloc[0]),
            )

        # Window 7: semantic metadata-only unit change
        elif window_id == 7:

            modified, metadata = inject_unit_change(
                batch,
                column="temp",
                reference_unit="celsius",
                new_unit="fahrenheit",
                metadata_only=True,
            )

            event = create_event(
                drift_type="semantic",
                subtype="unit_change",
                feature="temp",
                severity="high",
                injected=True,
                ground_truth="DRIFT",
                timestamp=str(batch["ts"].iloc[0]),
            )

        # Window 8: actual Celsius → Fahrenheit conversion
        elif window_id == 8:

            modified, metadata = inject_unit_change(
                batch,
                column="temp",
                reference_unit="celsius",
                new_unit="fahrenheit",
                metadata_only=False,
            )

            event = create_event(
                drift_type="semantic",
                subtype="unit_conversion",
                feature="temp",
                severity="high",
                injected=True,
                ground_truth="DRIFT",
                timestamp=str(batch["ts"].iloc[0]),
            )

        # Window 9: clean control
        else:

            modified = batch

            event = create_event(
                drift_type="normal",
                subtype="clean",
                feature=None,
                severity="none",
                injected=False,
                ground_truth="NO_DRIFT",
                timestamp=str(batch["ts"].iloc[0]),
            )

        # --------------------------------------------------
        # Save window
        # --------------------------------------------------

        window_path = (
            output_dir
            / f"window_{window_id:03d}.csv"
        )

        modified.to_csv(
            window_path,
            index=False,
        )

        # --------------------------------------------------
        # Persist semantic metadata as a sidecar file
        # --------------------------------------------------
        #
        # Windows whose injector returns metadata (e.g. the
        # semantic unit-change windows) write a small JSON
        # sidecar so downstream consumers do not need to
        # hardcode the schedule.
        # --------------------------------------------------

        if metadata is not None:

            metadata_path = (
                output_dir
                / f"{window_path.stem}.metadata.json"
            )

            with open(metadata_path, "w") as file:
                json.dump(
                    metadata,
                    file,
                    indent=2,
                )

        generated_windows.append(
            {
                "window_id": window_id,
                "start_row": start,
                "end_row": end,
                "file": str(window_path),
            }
        )

        # --------------------------------------------------
        # Save ground truth
        # --------------------------------------------------

        event["event_id"] = (
            f"window-{window_id:03d}"
        )

        append_event(
            event,
            ground_truth_path,
        )

    return generated_windows