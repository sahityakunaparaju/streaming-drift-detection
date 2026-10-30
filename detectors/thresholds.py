import os
from pathlib import Path

import yaml


def _candidate_paths():
    env_path = os.environ.get("THRESHOLDS_CONFIG")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    here = Path(__file__).resolve()
    candidates.append(
        here.parent.parent / "configs" / "thresholds.yaml"
    )
    candidates.append(Path("configs/thresholds.yaml"))
    candidates.append(Path("/tmp/configs/thresholds.yaml"))
    candidates.append(Path("/tmp/thresholds.yaml"))
    return candidates


def find_config_path():
    """First existing thresholds.yaml from THRESHOLDS_CONFIG or the default locations."""
    for candidate in _candidate_paths():
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "configs/thresholds.yaml not found; "
        "set THRESHOLDS_CONFIG to its location"
    )


def load_thresholds():
    """Load and return the frozen thresholds.yaml mapping."""
    config_path = find_config_path()
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


THRESHOLDS = load_thresholds()
