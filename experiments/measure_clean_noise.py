"""Measure clean-window noise for each reference mode (Revision, Step 1b).

Compares CLEAN windows only (no drift injected) against the reference under
the three reference modes:

    global      one global reference sample
    per_device  each device slice against its own device reference
    rolling     each window against the immediately preceding clean window

Windows are drawn from the CALIBRATION region (rows 325000-405184), which is
disjoint from both the reference region (250000-325000) and the Phase 7
evaluation regions ([0,100000) and [120000,250000)). No evaluation window is
ever used here.

Outputs:
    results/revision/clean_baseline_noise.csv          per-window per-column
    results/revision/clean_baseline_noise_summary.csv  per-mode per-column
    results/revision/clean_baseline_noise.png          KS distribution plot

Usage:
    PYTHONPATH=. python3 experiments/measure_clean_noise.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from detectors.reference import (
    ReferenceModel,
    load_calibration_region,
    load_reference_config,
    load_reference_region,
)

WINDOW_SIZE = 1000
N_WINDOWS = 60
# Deliberately different from the ten evaluation seeds.
SEEDS = [1234567, 1234568, 1234569]
OUTPUT_ROOT = Path("results/revision")


def clean_windows(calibration_df, seed):
    """N_WINDOWS consecutive non-overlapping clean windows (seed-offset)."""
    n_blocks = len(calibration_df) // WINDOW_SIZE
    max_offset = max(0, n_blocks - N_WINDOWS)
    offset = seed % (max_offset + 1) if max_offset else 0
    windows = []
    for index in range(N_WINDOWS):
        start = (offset + index) * WINDOW_SIZE
        windows.append(
            calibration_df.iloc[start:start + WINDOW_SIZE].reset_index(
                drop=True
            )
        )
    return windows


def measure_mode(mode, reference_df, calibration_df, seed):
    """Rows of per-column noise measurements for one mode/seed."""
    model = ReferenceModel(
        reference_df,
        mode=mode,
        min_rows=30,
        history=1,
    )
    rows = []
    for index, window in enumerate(clean_windows(calibration_df, seed)):
        details = model.details(window)
        if mode == "rolling":
            model.update_clean(window)
        for column, metrics in details["numeric"].items():
            rows.append({
                "mode": mode,
                "seed": seed,
                "window": index,
                "column": column,
                "kind": "numeric",
                "ks_statistic": metrics["ks_statistic"],
                "ks_p_value": metrics["ks_p_value"],
                "psi": metrics["psi"],
                "smd": metrics["smd"],
                "wasserstein_norm": metrics["wasserstein_norm"],
                "effect_size": metrics["effect_size"],
                "slice": metrics.get("slice"),
            })
        for column, metrics in details["categorical"].items():
            rows.append({
                "mode": mode,
                "seed": seed,
                "window": index,
                "column": column,
                "kind": "categorical",
                "ks_statistic": np.nan,
                "ks_p_value": metrics["p_value"],
                "psi": np.nan,
                "smd": np.nan,
                "wasserstein_norm": np.nan,
                "effect_size": metrics["effect_size"],
                "slice": "global",
            })
    return rows


def summarize(df):
    """Per-mode per-column KS/PSI summary (mean / p95 / max)."""
    rows = []
    for (mode, column), group in df.groupby(["mode", "column"]):
        rows.append({
            "mode": mode,
            "column": column,
            "kind": group["kind"].iloc[0],
            "n_windows": int(len(group)),
            "ks_mean": float(group["ks_statistic"].mean()),
            "ks_p95": float(group["ks_statistic"].quantile(0.95)),
            "ks_max": float(group["ks_statistic"].max()),
            "psi_mean": float(group["psi"].mean()),
            "psi_p95": float(group["psi"].quantile(0.95)),
            "psi_max": float(group["psi"].max()),
            "effect_mean": float(group["effect_size"].mean()),
            "effect_p95": float(group["effect_size"].quantile(0.95)),
            "effect_max": float(group["effect_size"].max()),
        })
    return pd.DataFrame(rows)


def plot(summary, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available; skipping plot")
        return None

    numeric = summary[summary["kind"] == "numeric"]
    if numeric.empty:
        return None
    columns = sorted(numeric["column"].unique())
    modes = ["global", "per_device", "rolling"]

    fig, axes = plt.subplots(
        1, len(columns), figsize=(2.4 * len(columns), 4), sharey=True
    )
    if len(columns) == 1:
        axes = [axes]
    for axis, column in zip(axes, columns):
        data = []
        labels = []
        for mode in modes:
            subset = numeric[
                (numeric["mode"] == mode) & (numeric["column"] == column)
            ]
            if subset.empty:
                continue
            data.append(subset["ks_mean"].to_numpy())
            labels.append(mode)
        axis.bar(range(len(data)), [float(v[0]) for v in data])
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=45, fontsize=7)
        axis.set_title(column, fontsize=9)
        axis.axhline(0.2, color="red", linestyle="--", linewidth=1)
    axes[0].set_ylabel("clean-window KS (mean)")
    fig.suptitle(
        "Clean-window KS by reference mode (red line = 0.2 target)"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main():
    config = load_reference_config()
    reference_df = load_reference_region(config=config)
    calibration_df = load_calibration_region(config=config)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in SEEDS:
        for mode in ["global", "per_device", "rolling"]:
            mode_rows = measure_mode(
                mode, reference_df, calibration_df, seed
            )
            # Rolling's first window has no clean history and falls back
            # to the global reference; drop that one warm-up window so the
            # comparison is like-for-like.
            if mode == "rolling":
                mode_rows = [r for r in mode_rows if r["window"] > 0]
            rows.extend(mode_rows)
            print(f"  measured {mode} seed {seed}: {len(mode_rows)} rows")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_ROOT / "clean_baseline_noise.csv", index=False)

    summary = summarize(df)
    summary.to_csv(
        OUTPUT_ROOT / "clean_baseline_noise_summary.csv", index=False
    )

    plot(summary, OUTPUT_ROOT / "clean_baseline_noise.png")

    print("\nCLEAN-WINDOW KS BY MODE (numeric columns, 3 seeds x 60 windows)")
    pivot = summary[summary["kind"] == "numeric"].pivot(
        index="column", columns="mode", values="ks_mean"
    )
    print(pivot.round(4).to_string())

    print("\nCLEAN-WINDOW KS p95 BY MODE")
    pivot_p95 = summary[summary["kind"] == "numeric"].pivot(
        index="column", columns="mode", values="ks_p95"
    )
    print(pivot_p95.round(4).to_string())

    print("\nCLEAN-WINDOW PSI mean BY MODE")
    pivot_psi = summary[
        (summary["kind"] == "numeric")
    ].pivot(index="column", columns="mode", values="psi_mean")
    print(pivot_psi.round(4).to_string())

    mode_means = (
        summary[summary["kind"] == "numeric"]
        .groupby("mode")["ks_mean"]
        .mean()
        .sort_values()
    )
    print("\nMEAN OF PER-COLUMN MEAN KS BY MODE")
    print(mode_means.round(6).to_string())

    with open(OUTPUT_ROOT / "clean_baseline_noise_meta.json", "w") as file:
        json.dump({
            "reference_region": config["reference_region"],
            "calibration_region": config["calibration_region"],
            "evaluation_regions": config["evaluation_regions"],
            "seeds": SEEDS,
            "n_windows_per_seed": N_WINDOWS,
            "window_size": WINDOW_SIZE,
            "mode_mean_ks": {
                mode: float(value)
                for mode, value in mode_means.items()
            },
        }, file, indent=2)

    return df, summary


if __name__ == "__main__":
    main()
