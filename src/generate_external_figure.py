#!/usr/bin/env python3
"""Plot capability-benchmark correlation changes reported in §4.6."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paths import FIGURES, RESULTS


LABELS = {
    "epoch_capabilities_index": "Epoch Capabilities Index",
    "gpqa_diamond": "GPQA Diamond",
    "frontiermath": "FrontierMath",
    "livebench": "LiveBench",
    "arc_agi_2": "ARC-AGI-2",
    "scicode": "SciCode",
    "aider_polyglot": "Aider Polyglot",
}
ORDER = list(LABELS)
COLORS = {
    "formatting_controlled": "#1f77b4",
    "joint_controlled": "#d95f02",
}


def interval_errors(point, interval):
    """Convert an interval into non-negative matplotlib error lengths."""
    return [[point - interval[0]], [interval[1] - point]]


def main():
    result_path = RESULTS / "external_leaderboard_results.json"
    with result_path.open(encoding="utf-8") as handle:
        results = json.load(handle)

    benchmarks = results["capability_benchmarks"]
    rows = [
        (key, benchmarks[key])
        for key in ORDER
        if benchmarks[key]["eligible"]
    ]

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    fig, ax = plt.subplots(figsize=(8.3, 5.1))
    y = np.arange(len(rows))[::-1]
    offsets = {
        "formatting_controlled": 0.12,
        "joint_controlled": -0.12,
    }
    markers = {
        "formatting_controlled": "o",
        "joint_controlled": "s",
    }
    legend_labels = {
        "formatting_controlled": "Formatting-controlled minus raw",
        "joint_controlled": "Full joint-controlled minus raw",
    }

    for ranking in ("formatting_controlled", "joint_controlled"):
        for index, (_, record) in enumerate(rows):
            comparison = record["delta_vs_raw"][ranking]
            point = comparison["point"]
            interval = comparison["bootstrap_95_ci"]
            ypos = y[index] + offsets[ranking]
            ax.errorbar(
                point,
                ypos,
                xerr=interval_errors(point, interval),
                fmt=markers[ranking],
                color=COLORS[ranking],
                markersize=5.5,
                capsize=2.5,
                linewidth=1.5,
                zorder=3,
            )

    ax.axvline(0, color="black", linewidth=0.9, linestyle="--", zorder=1)
    ax.grid(axis="x", color="#dddddd", linewidth=0.7, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([
        f"{LABELS[key]} (n={record['n_version_matches']})"
        for key, record in rows
    ])
    ax.set_xlabel("Change in Spearman correlation relative to raw Compar:IA")
    ax.set_title(
        "Independent capability-benchmark alignment\n"
        "paired 95% bootstrap intervals"
    )
    handles = [
        plt.Line2D(
            [], [], marker=markers[ranking], linestyle="", color=COLORS[ranking],
            label=legend_labels[ranking],
        )
        for ranking in ("formatting_controlled", "joint_controlled")
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True)
    fig.tight_layout()

    for extension in ("png", "pdf"):
        fig.savefig(FIGURES / f"fig12_external_alignment.{extension}")
    plt.close(fig)
    print(f"Wrote {FIGURES / 'fig12_external_alignment.png'}")


if __name__ == "__main__":
    main()
