"""Revision reporting tests: raw metrics only, trigger columns recorded."""

from experiments.run_revision_ladder import raw_summary
from experiments.run_phase7 import _trigger_columns


def test_raw_summary_never_excludes_flagged_windows():
    rows = [
        {
            "decision": "DRIFT",
            "ground_truth": "DRIFT",
            "expected_false_negative": "expected false negative",
        },
        {
            "decision": "NO_DRIFT",
            "ground_truth": "DRIFT",
            "expected_false_negative": "",
        },
        {
            "decision": "NO_DRIFT",
            "ground_truth": "NO_DRIFT",
            "expected_false_negative": "",
        },
    ]
    result = raw_summary(rows)

    assert result["n"] == 3
    assert result["raw"]["tp"] == 1
    assert result["raw"]["fn"] == 1
    assert result["raw"]["recall"] == 0.5
    # The flag is reported, but it did not remove the window.
    assert result["flagged_analysis_only"]["n_flagged"] == 1


def test_trigger_columns_extraction():
    assert _trigger_columns({"trigger_columns": ["temp"]}) == ["temp"]
    assert _trigger_columns(
        {"statistical": {"trigger_columns": ["device"]}}
    ) == ["device"]
    assert _trigger_columns({"scores": {}}) == []
    assert _trigger_columns(None) == []
