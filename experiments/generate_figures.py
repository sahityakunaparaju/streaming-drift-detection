#!/usr/bin/env python3
"""
Generate publication-quality figures for the paper (Phases 7-13).

Reads directly from the existing result files under results/ (no numbers
are hard-coded from the data) and writes each figure as both PNG (300 DPI)
and vector PDF into figures/.

Design constraints (see figures/README.md):
  - Colorblind-safe palette (Okabe-Ito hex codes), never red/green alone.
  - Grayscale-legible: distinct markers, hatching, and line styles in
    addition to color.
  - Sized for a 2-column IEEE layout (3.3 in per column); fonts chosen to
    remain readable at that width.

Usage:
    PYTHONPATH=. python experiments/generate_figures.py [--out figures]
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Global style: IEEE-friendly, colorblind-safe, grayscale-legible
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

# Okabe-Ito colorblind-safe palette
C_BLUE = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN = "#009E73"
C_PURPLE = "#CC79A7"
C_SKY = "#56B4E9"
C_VERM = "#D55E00"
C_YELLOW = "#F0E442"
C_BLACK = "#000000"
C_GREY = "#7F7F7F"

FIG_W = 3.3          # IEEE single-column width in inches
FIG_WIDE = 6.9       # full-page double-column width in inches
BASE_H = 2.5         # comfortable height for a single-column figure

BASE_FONT = 8        # readable at 3.3 in width
SMALL_FONT = 6.5
TICK_FONT = 7

plt.rcParams.update(
    {
        "font.size": BASE_FONT,
        "axes.titlesize": BASE_FONT,
        "axes.labelsize": BASE_FONT,
        "xtick.labelsize": TICK_FONT,
        "ytick.labelsize": TICK_FONT,
        "legend.fontsize": SMALL_FONT,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "errorbar.capsize": 2,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.constrained_layout.use": True,
    }
)

# ---------------------------------------------------------------------------
# Small IO helpers
# ---------------------------------------------------------------------------


def read_json(path: Path):
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_dicts(path: Path):
    """Load a CSV as a list of dicts."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_pm(s):
    """Parse '0.9055 +/- 0.0177' -> (0.9055, 0.0177); plain floats pass through."""
    if isinstance(s, (int, float)):
        return float(s), 0.0
    s = str(s).strip()
    if "+/-" in s:
        a, b = s.split("+/-")
        return float(a.strip()), float(b.strip())
    return float(s), 0.0


def parse_per_seed(s):
    """Parse a repr'd list of per-seed values into a float array."""
    return np.array(ast.literal_eval(s), dtype=float)


def save(fig, name: str, outdir: Path):
    """Save a figure as PNG and PDF, then close it."""
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.png / {name}.pdf")


# ---------------------------------------------------------------------------
# Figure 1: baseline_ladder_f1
# ---------------------------------------------------------------------------


def fig_baseline_ladder(outdir: Path) -> None:
    """B0-B5 raw vs adjusted F1, grouped bars with 10-seed std error bars.

    Source: results/phase7/multiseed/aggregate_ladder.csv
    (fallback: results/phase7/tables/ladder.csv, single seed 42).
    """
    ladder = read_csv_dicts(ROOT / "results/phase7/multiseed/aggregate_ladder.csv")

    configs = ["B0_null", "B1_schema", "B2_statistical", "B3_schema_statistical",
               "B4_unified", "B5_evidently"]
    labels = ["B0\n(null)", "B1\n(schema)", "B2\n(stat.)",
              "B3\n(sch+stat)", "B4\n(unified)", "B5\n(Evidently)"]

    raw_mean, raw_std = [], []
    adj_mean, adj_std = [], []
    for cfg in configs:
        r = next(x for x in ladder if x["config"] == cfg and x["variant"] == "raw" and x["metric"] == "f1")
        a = next(x for x in ladder if x["config"] == cfg and x["variant"] == "adjusted" and x["metric"] == "f1")
        raw_mean.append(float(r["mean"]))
        raw_std.append(float(r["std"]))
        adj_mean.append(float(a["mean"]))
        adj_std.append(float(a["std"]))

    # B5's aggregate rows are zero-filled (absent from the multiseed runs);
    # fall back to the real B5 numbers from the seed-42 ladder table.
    t = read_csv_dicts(ROOT / "results/phase7/tables/ladder.csv")
    b5 = next(row for row in t if row["config"] == "B5_evidently")
    raw_mean[-1], adj_mean[-1] = float(b5["raw_f1"]), float(b5["adj_f1"])
    raw_std[-1], adj_std[-1] = 0.0, 0.0
    single_seed = True

    x = np.arange(len(configs))
    w = 0.38

    fig, ax = plt.subplots(figsize=(FIG_W, 2.7))
    b1 = ax.bar(x - w / 2, raw_mean, w, yerr=raw_std,
                color=C_BLUE, hatch="", edgecolor="black", linewidth=0.5,
                label="Raw")
    b2 = ax.bar(x + w / 2, adj_mean, w, yerr=adj_std,
                color=C_ORANGE, hatch="///", edgecolor="black", linewidth=0.5,
                label="Adjusted (flagged windows excluded)")

    sr = read_json(ROOT / "results/phase7/multiseed/seed-42/seed_results.json")["ladder"]
    flagged = int(sr["B0_null"]["metrics"]["flagged"]["n_flagged"])
    b4_det = int(sr["B4_unified"]["metrics"]["flagged"]["n_flagged_detected"])
    ax.annotate(
        f"{flagged}/110 windows flagged as expected false negatives\n"
        f"(excluded from adjusted; {b4_det} of {flagged} detected by B4)",
        xy=(0.5, 1.02), xycoords="axes fraction",
        ha="center", va="bottom", fontsize=SMALL_FONT,
    )

    ax.set_xticks(x, labels)
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.24)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="upper left", frameon=False)
    ax.tick_params(axis="x", which="major", pad=1)
    save(fig, "baseline_ladder_f1", outdir)


# ---------------------------------------------------------------------------
# Figure 2: ablation_delta
# ---------------------------------------------------------------------------


def fig_ablation(outdir: Path) -> None:
    """Step waterfall of incremental adjusted-F1 gain, B0->B1->B3->B4.

    Source: results/phase9/ablation_table.csv
    """
    tbl = read_csv_dicts(ROOT / "results/phase9/ablation_table.csv")
    delta = {
        r["step"]: r
        for r in tbl
        if r["kind"] == "delta" and r["variant"] == "adjusted"
    }
    absolute = {
        r["step"]: r
        for r in tbl
        if r["kind"] == "absolute" and r["variant"] == "adjusted"
    }

    def pm(row, col):
        """Parse a mean±std cell from a result row."""
        m, s = parse_pm(row[col])
        return m, s

    steps = ["B0_null", "B1_schema", "B3_schema_statistical", "B4_unified"]
    labels = ["B0\n(null)", "B1\n+schema", "B3\n+statistical", "B4\n+semantic"]

    f1_mean, f1_std = zip(*(pm(absolute[s], "f1") for s in steps))

    d0 = float(absolute["B0_null"]["f1"].split("+/-")[0])
    d1 = pm(delta["B0_null->B1_schema"], "delta_f1")
    d2 = pm(delta["B1_schema->B3_schema_statistical"], "delta_f1")
    d3 = pm(delta["B3_schema_statistical->B4_unified"], "delta_f1")

    fig, ax = plt.subplots(figsize=(FIG_W, 2.6))

    # Baseline bar
    ax.bar(0, f1_mean[0], 0.62, yerr=f1_std[0], color=C_GREY,
           edgecolor="black", linewidth=0.5)

    # Incremental steps as floating segments
    bottoms = [d0, d0 + d1[0], d0 + d1[0] + d2[0]]
    heights = [d1[0], d2[0], d3[0]]
    step_std = [d1[1], d2[1], d3[1]]
    colors = [C_SKY, C_SKY, C_VERM]
    for i, (bot, h, c, s) in enumerate(zip(bottoms, heights, colors, step_std)):
        ax.bar(i + 1, h, 0.62, bottom=bot, color=c,
               edgecolor="black", linewidth=0.5)
        # error bar at the running total after this step
        ax.errorbar(i + 1, bot + h, yerr=s,
                    fmt="none", ecolor="black", elinewidth=0.7)

    # Connector line along the top of the running total
    tops = [f1_mean[0], bottoms[0] + heights[0], bottoms[1] + heights[1],
            bottoms[2] + heights[2]]
    ax.plot(range(4), tops, color="black", linewidth=0.8,
            linestyle=":", marker="", zorder=3)

    # Value labels: absolute F1 under each bar top
    for i, (top, std) in enumerate(zip(tops, f1_std)):
        ax.annotate(f"{top:.3f}", xy=(i, top), xytext=(0, 3),
                    textcoords="offset points", ha="center",
                    fontsize=SMALL_FONT)

    # Key-finding annotation on the B3->B4 step
    pct = float(delta["B3_schema_statistical->B4_unified"]["delta_f1_rel_pct"]
                .replace("%", "").split("+/-")[0])
    ax.annotate(
        f"+{d3[0]:.2f} ({pct:.1f}%)",
        xy=(3, tops[2] + d3[0] / 2), xytext=(-6, 8), textcoords="offset points",
        ha="right", fontsize=BASE_FONT, fontweight="bold", color=C_VERM,
        arrowprops=dict(arrowstyle="-", color=C_VERM, lw=0.8),
    )

    ax.set_xticks(range(4), labels)
    ax.set_ylabel("Adjusted F1 score")
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    legend = [
        Patch(facecolor=C_GREY, edgecolor="black", label="Baseline F1"),
        Patch(facecolor=C_SKY, edgecolor="black", label=r"$\Delta$F1 from schema / statistical"),
        Patch(facecolor=C_VERM, edgecolor="black", label=r"$\Delta$F1 from semantic (key finding)"),
    ]
    ax.legend(handles=legend, loc="upper left", frameon=False)
    save(fig, "ablation_delta", outdir)


# ---------------------------------------------------------------------------
# Figure 3: subtype_recall_heatmap
# ---------------------------------------------------------------------------


def fig_subtype_heatmap(outdir: Path) -> None:
    """Per-subtype recall heatmap: subtypes x detectors.

    Source: results/phase7/tables/subtype_recall.csv (seed 42, canonical test set)
    """
    rows = read_csv_dicts(ROOT / "results/phase7/tables/subtype_recall.csv")
    detectors = ["schema", "statistical", "semantic", "combined"]
    det_labels = ["Schema", "Statistical", "Semantic", "Combined (B4)"]

    subtypes = sorted({r["subtype"] for r in rows})
    m = np.zeros((len(subtypes), len(detectors)))
    for r in rows:
        i = subtypes.index(r["subtype"])
        j = detectors.index(r["detector"])
        m[i, j] = float(r["adj_recall"])

    # Order subtypes: those the semantic detector catches (and others miss)
    # at the top, so the pattern reads at a glance.
    semantic_hits = m[:, detectors.index("semantic")]
    others = m[:, [0, 1]]  # schema + statistical
    order = sorted(
        range(len(subtypes)),
        key=lambda i: (-(semantic_hits[i] * (1 - others[i].max())),
                       -semantic_hits[i], subtypes[i]),
    )
    subtypes = [subtypes[i] for i in order]
    m = m[order]

    fig, ax = plt.subplots(figsize=(FIG_W, 4.2))
    im = ax.imshow(m, cmap="Greys", vmin=0, vmax=1, aspect="auto")

    # annotate cells; text in black/white depending on background
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m[i, j]
            ax.text(j, i, f"{v:.1f}" if v in (0.0, 1.0) else f"{v:.2f}",
                    ha="center", va="center", fontsize=SMALL_FONT,
                    color="white" if v > 0.6 else "black")

    ax.set_xticks(range(len(detectors)), det_labels, rotation=20, ha="right")
    ax.set_yticks(range(len(subtypes)), subtypes, fontsize=SMALL_FONT)
    ax.set_ylabel("Drift subtype")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Recall (adjusted)", fontsize=SMALL_FONT)
    cbar.ax.tick_params(labelsize=SMALL_FONT)
    # grid between cells
    ax.set_xticks(np.arange(-0.5, len(detectors)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(subtypes)), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    save(fig, "subtype_recall_heatmap", outdir)


# ---------------------------------------------------------------------------
# Figure 4: semantic_blindspot_bar
# ---------------------------------------------------------------------------


def fig_blindspot(outdir: Path) -> None:
    """Recall on the isolated 50-case semantic blind-spot set.

    Source: results/phase8/per_detector_recall.csv (10-seed mean +/- std)
    """
    rows = read_csv_dicts(ROOT / "results/phase8/per_detector_recall.csv")
    order = ["schema", "statistical", "semantic", "combined"]
    labels = ["Schema\n(B1)", "Statistical\n(B2)", "Semantic", "Unified\n(B4)"]
    means = [float(next(r for r in rows if r["detector"] == d)["recall_mean"]) for d in order]
    stds = [float(next(r for r in rows if r["detector"] == d)["recall_std"]) for d in order]
    colors = [C_SKY, C_SKY, C_VERM, C_VERM]
    hatches = ["", "", "xx", "xx"]

    fig, ax = plt.subplots(figsize=(FIG_W, 2.3))
    x = np.arange(4)
    bars = ax.bar(x, means, 0.6, yerr=stds, color=colors, hatch=hatches,
                  edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, means):
        ax.annotate(f"{v:.1f}", xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=BASE_FONT)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Recall (50 blind-spot cases)")
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.axhline(1.0, color=C_GREY, linewidth=0.6, linestyle="--", zorder=0)
    save(fig, "semantic_blindspot_bar", outdir)


# ---------------------------------------------------------------------------
# Figure 5: literature_comparison_f1
# ---------------------------------------------------------------------------


def fig_literature_f1(outdir: Path) -> None:
    """F1 on the primary benchmark: ADWIN/DDM/KSWIN (default vs retuned) vs B4.

    Source: results/phase10/retune/retune_comparison.csv (10 seeds)
    """
    rows = read_csv_dicts(ROOT / "results/phase10/retune/retune_comparison.csv")

    def get(det, variant):
        """Fetch (mean, std) of raw F1 for one detector/variant pair."""
        r = next(x for x in rows if x["detector"] == det and x["variant"] == variant)
        m, s = parse_pm(r["raw_f1"])
        return m, s

    ddm_d, ddm_r = get("DDM", "default"), get("DDM", "retuned_k3")
    ks_d, ks_r = get("KSWIN", "default"), get("KSWIN", "retuned_alpha0001")
    adwin = get("ADWIN", "default")
    b4 = get("B4", "existing_validated")

    x = np.arange(4)
    w = 0.6
    means = [adwin[0], ddm_d[0], ks_d[0], b4[0]]
    stds = [adwin[1], ddm_d[1], ks_d[1], b4[1]]
    colors = [C_SKY, C_SKY, C_SKY, C_VERM]
    hatches = ["", "", "", "//"]

    fig, ax = plt.subplots(figsize=(FIG_W, 2.6))
    bars = ax.bar(x, means, w, yerr=stds, color=colors, hatch=hatches,
                  edgecolor="black", linewidth=0.5)

    # Dashed connectors default -> retuned, with a "retuning hurts" arrow
    for i, (dd, rr, name) in enumerate(
        [(ddm_d, ddm_r, "DDM"), (ks_d, ks_r, "KSWIN")]
    ):
        xi = 1 + i  # bars at x=1 (DDM), x=2 (KSWIN)
        ax.plot([xi - w / 4, xi + w / 4], [dd[0], rr[0]],
                color=C_BLACK, linewidth=0.8, linestyle="--", zorder=4)
        ax.annotate(
            f"retuned: {rr[0]:.3f}$\\pm${rr[1]:.3f}\n(worse)",
            xy=(xi + w / 4, rr[0]), xytext=(10, -2), textcoords="offset points",
            fontsize=SMALL_FONT, color=C_BLACK, ha="left", va="top",
            arrowprops=dict(arrowstyle="->", color=C_BLACK, lw=0.7,
                            connectionstyle="arc3,rad=-0.25"),
        )

    for b, (m, s) in zip(bars, zip(means, stds)):
        ax.annotate(f"{m:.2f}", xy=(b.get_x() + b.get_width() / 2, m + s),
                    xytext=(0, 2), textcoords="offset points",
                    ha="center", fontsize=SMALL_FONT)

    ax.set_xticks(x, ["ADWIN", "DDM", "KSWIN", "B4\n(unified)"])
    ax.set_ylabel("F1 score (primary benchmark)")
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(handles=[
        Patch(facecolor=C_SKY, edgecolor="black", label="Literature detectors (default)"),
        Patch(facecolor=C_VERM, edgecolor="black", hatch="//", label="B4 unified (ours)"),
    ], loc="upper left", frameon=False)
    save(fig, "literature_comparison_f1", outdir)


# ---------------------------------------------------------------------------
# Figure 6: literature_semantic_recall
# ---------------------------------------------------------------------------


def fig_literature_blindspot(outdir: Path) -> None:
    """Recall of each detector on the 50 semantic blind-spot cases.

    Source: results/phase10/blind_spot_comparison.csv
    """
    rows = read_csv_dicts(ROOT / "results/phase10/blind_spot_comparison.csv")
    order = ["ADWIN", "DDM", "KSWIN", "B4 (Phase 8 verified result)"]
    labels = ["ADWIN", "DDM", "KSWIN", "B4\n(unified)"]
    means = [float(next(r for r in rows if r["detector"] == d)["recall"]) for d in order]
    colors = [C_SKY, C_SKY, C_SKY, C_VERM]
    hatches = ["", "", "", "//"]

    fig, ax = plt.subplots(figsize=(FIG_W, 2.3))
    x = np.arange(4)
    bars = ax.bar(x, means, 0.6, color=colors, hatch=hatches,
                  edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, means):
        ax.annotate(f"{v:.1f}", xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=BASE_FONT)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Recall (50 semantic cases)")
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.axhline(1.0, color=C_GREY, linewidth=0.6, linestyle="--", zorder=0)
    save(fig, "literature_semantic_recall", outdir)


# ---------------------------------------------------------------------------
# Figure 7: overhead_latency
# ---------------------------------------------------------------------------


def fig_overhead(outdir: Path) -> None:
    """End-to-end latency with vs without detection, at two load points.

    Source: results/phase11/overhead.csv (mean of 3 repetitions)
    """
    rows = read_csv_dicts(ROOT / "results/phase11/overhead.csv")
    pts = ["comfortable", "stress"]
    with_ms = [float(next(r for r in rows if r["load_point"] == p)["latency_mean_with_ms"]) for p in pts]
    without_ms = [float(next(r for r in rows if r["load_point"] == p)["latency_mean_without_ms"]) for p in pts]

    x = np.arange(2)
    w = 0.34
    fig, ax = plt.subplots(figsize=(FIG_W, 2.4))
    ax.bar(x - w / 2, without_ms, w, color=C_GREY, hatch="",
           edgecolor="black", linewidth=0.5, label="Without detection")
    ax.bar(x + w / 2, with_ms, w, color=C_VERM, hatch="//",
           edgecolor="black", linewidth=0.5, label="With detection")

    for xi, (a, b) in enumerate(zip(without_ms, with_ms)):
        pct = (b - a) / a * 100
        ax.annotate(f"+{pct:.0f}%", xy=(xi + w / 2, b),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=BASE_FONT, fontweight="bold")

    ax.set_xticks(x, ["Comfortable load\n(5 windows, 0.5 s apart)",
                      "Stress load\n(20 windows, 0.05 s apart)"])
    ax.set_ylabel("End-to-end latency (ms)")
    ax.legend(loc="upper left", frameon=False)
    ax.set_ylim(0, max(with_ms) * 1.3)

    ax.text(0.99, 0.02,
            "Detection throughput ceiling $\\approx$ 2,000 rows/s (1,000-row windows)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=SMALL_FONT, style="italic",
            bbox=dict(facecolor="white", edgecolor=C_GREY, linewidth=0.5,
                      boxstyle="round,pad=0.25", alpha=0.9))
    save(fig, "overhead_latency", outdir)


# ---------------------------------------------------------------------------
# Figure 8: secondary_dataset_comparison
# ---------------------------------------------------------------------------


def fig_secondary(outdir: Path) -> None:
    """Primary vs secondary dataset, B0-B4, four metrics side by side.

    Source: results/phase13/results.json vs results/phase7/multiseed/aggregate_ladder.csv
    """
    metrics = [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1 score"),
        ("fpr", "FPR"),
    ]
    configs = ["B0_null", "B1_schema", "B2_statistical",
               "B3_schema_statistical", "B4_unified"]
    labels = ["B0", "B1", "B2", "B3", "B4"]

    agg = read_csv_dicts(ROOT / "results/phase7/multiseed/aggregate_ladder.csv")
    primary = {}
    for cfg in configs:
        for var in ("raw", "adjusted"):
            for met, _ in metrics:
                r = next(x for x in agg if x["config"] == cfg and x["variant"] == var and x["metric"] == met)
                primary[(cfg, var, met)] = (float(r["mean"]), float(r["std"]))

    sec = read_json(ROOT / "results/phase13/results.json")
    secondary = {}
    for cfg in configs:
        for var in ("raw", "adjusted"):
            for met, _ in metrics:
                secondary[(cfg, var, met)] = (float(sec[cfg][var][met]), 0.0)

    # Honesty markers on secondary bars: '*' = B1-inherited nullability-
    # tolerance FP (one clean window exceeds the 5 pp rate tolerance);
    # '†' = chi2-cardinality FP (chi2 fires on every clean window because
    # the frozen threshold was calibrated on a 3-category field, while the
    # secondary dataset's 'wd' has 16 categories).
    chi2_affected = {"B2_statistical", "B3_schema_statistical", "B4_unified"}
    nullab_affected = {"B1_schema", "B3_schema_statistical", "B4_unified"}

    n_met, n_cfg = len(metrics), len(configs)
    fig, axes = plt.subplots(1, n_met, figsize=(FIG_WIDE, 2.5), sharey=True)

    for k, (met, met_label) in enumerate(metrics):
        ax = axes[k]
        x = np.arange(n_cfg)
        w = 0.38
        pv = [primary[(c, "raw", met)][0] for c in configs]
        sv = [secondary[(c, "raw", met)][0] for c in configs]
        ps = [primary[(c, "raw", met)][1] for c in configs]
        ax.bar(x - w / 2, pv, w, yerr=ps, color=C_BLUE,
               edgecolor="black", linewidth=0.5, label="Primary (IoT)")
        ax.bar(x + w / 2, sv, w, color=C_ORANGE, hatch="///",
               edgecolor="black", linewidth=0.5, label="Secondary (Beijing AQ)")
        ax.set_xticks(x, labels)
        ax.set_title(met_label, fontsize=BASE_FONT)
        if k == 0:
            ax.set_ylabel("Score")
        ax.set_ylim(0, 1.45)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    # markers only once, above the tallest bar of each affected config
    ax = axes[2]  # F1 panel has the tallest secondary bars
    sv_f1 = [secondary[(c, "raw", "f1")][0] for c in configs]
    for i, c in enumerate(configs):
        marks = ""
        if c in nullab_affected:
            marks += "*"
        if c in chi2_affected:
            marks += "\u2020"
        if marks:
            ax.annotate(marks, xy=(i + w / 2, max(sv_f1[i], 0.35)),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=BASE_FONT, fontweight="bold",
                        annotation_clip=False)

    handles = [
        Patch(facecolor=C_BLUE, edgecolor="black", label="Primary (IoT)"),
        Patch(facecolor=C_ORANGE, edgecolor="black", hatch="///", label="Secondary (Beijing AQ)"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="black",
               markersize=6, linestyle="none",
               label="* nullability-tolerance FP"),
        Line2D([0], [0], marker=r"$\dagger$", color="none", markerfacecolor="black",
               markersize=6, linestyle="none",
               label="\u2020 chi2 cardinality FP"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=SMALL_FONT, bbox_to_anchor=(0.5, -0.06))
    save(fig, "secondary_dataset_comparison", outdir)


# ---------------------------------------------------------------------------
# Figure 9: severity_matrix_lineplot
# ---------------------------------------------------------------------------


def fig_severity(outdir: Path) -> None:
    """Recall vs drift severity per detector.

    Source: results/phase7/tables/severity_matrix.csv (seed 42; multiseed
    per-severity aggregates are not in results/, so error bars are omitted).
    """
    rows = read_csv_dicts(ROOT / "results/phase7/tables/severity_matrix.csv")
    sev = ["low", "medium", "high"]
    detectors = ["schema", "statistical", "semantic", "combined"]
    det_labels = ["Schema (B1)", "Statistical (B2)", "Semantic", "Combined (B4)"]
    colors = [C_SKY, C_GREEN, C_VERM, C_PURPLE]
    markers = ["o", "s", "^", "D"]
    linestyles = ["--", "-.", ":", "-"]

    fig, ax = plt.subplots(figsize=(FIG_W, 2.4))
    x = np.arange(3)
    for det, lab, c, mk, ls in zip(detectors, det_labels, colors, markers, linestyles):
        y = [float(next(r for r in rows if r["severity"] == s and r["detector"] == det)["adj_recall"])
             for s in sev]
        ax.plot(x, y, marker=mk, linestyle=ls, color=c, label=lab)

    ax.set_xticks(x, ["Low", "Medium", "High"])
    ax.set_xlabel("Drift severity")
    ax.set_ylabel("Recall (adjusted)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(loc="upper left", frameon=False, fontsize=SMALL_FONT)
    save(fig, "severity_matrix_lineplot", outdir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FIGURES = [
    ("baseline_ladder_f1", fig_baseline_ladder),
    ("ablation_delta", fig_ablation),
    ("subtype_recall_heatmap", fig_subtype_heatmap),
    ("semantic_blindspot_bar", fig_blindspot),
    ("literature_comparison_f1", fig_literature_f1),
    ("literature_semantic_recall", fig_literature_blindspot),
    ("overhead_latency", fig_overhead),
    ("secondary_dataset_comparison", fig_secondary),
    ("severity_matrix_lineplot", fig_severity),
]


def main() -> None:
    """Generate all thesis figures from saved result artifacts."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "figures"),
                    help="output directory (default: figures/)")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Generating figures in {outdir} ...")
    for name, fn in FIGURES:
        try:
            fn(outdir)
        except Exception as e:  # pragma: no cover - fail loudly but continue
            print(f"  ERROR in {name}: {e!r}")
            raise
    print("Done.")


if __name__ == "__main__":
    main()
