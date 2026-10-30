"""DEPRECATED entry point — use experiments/derive_thresholds.py.

The original tuning pass (1873 lines) tuned normalization constants on a
mix of synthetic clean and drifted windows until the classes separated.
That procedure was circular and produced the values that made the
statistical detector deaf to real drift (ks_floor=0.50, psi_scale=72,
chi2_scale=32, decision_threshold=0.25); it also flagged the drift
subtypes it missed as "expected false negatives" and excluded them from
the reported metrics.

The replacement derivation (experiments/derive_thresholds.py) calibrates
per-column effect-size thresholds on CLEAN calibration windows only, on a
row region disjoint from the reference and evaluation splits, and writes
configs/thresholds.yaml together with a full audit trail.

This module is kept so older commands still run; it simply delegates.
"""

import sys

from experiments.derive_thresholds import main


def run():  # pragma: no cover - thin CLI wrapper
    """Delegate to the clean threshold derivation."""
    return main()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run())
