#!/usr/bin/env python3
"""Analyze alternative lexical-diversity metrics for §4.10.

``mattr_alt.py`` extracts MTLD and two MATTR variants from vote-time final
answers. This script fits the same joint Bradley-Terry specification as the
main linguistic analysis, replacing MATTR with one alternative at a time.
Each metric uses its own available-case support, which is reported explicitly.

    python src/analyze_mattr_alt.py
        -> results/mattr_alt_results.json
"""

import json

import numpy as np
import pandas as pd

from modeling import FORMATTING, LINGUISTIC, MIN_BATTLES, fit_bt
from paths import BATTLES, DATA, RESULTS


ALTERNATIVES = ["mtld", "cwmattr", "nocapmattr"]
BASE_LINGUISTIC = [feature for feature in LINGUISTIC if feature != "mattr"]


def _fit_metric(battles, metric):
    features = FORMATTING + ["length"] + BASE_LINGUISTIC + [metric]
    required = [
        f"{feature}_{side}"
        for feature in features
        for side in ("a", "b")
    ]
    sample = battles.dropna(subset=required).copy()
    counts = pd.concat(
        [sample["model_a_name"], sample["model_b_name"]]
    ).value_counts()
    models = sorted(counts[counts >= MIN_BATTLES].index)
    sample = sample[
        sample["model_a_name"].isin(models)
        & sample["model_b_name"].isin(models)
    ]
    coefficient = fit_bt(sample, models, features)[1][metric]
    return {
        "n_battles": int(len(sample)),
        "n_models": len(models),
        "odds_pct": float((np.exp(coefficient) - 1) * 100),
    }


def main():
    battles = pd.read_parquet(BATTLES)
    alternatives = pd.read_parquet(DATA / "mattr_alt.parquet")
    merged = battles.merge(
        alternatives, on="conversation_pair_id", how="left", validate="one_to_one"
    )

    estimates = {"mattr": _fit_metric(battles, "mattr")}
    estimates.update({
        metric: _fit_metric(merged, metric)
        for metric in ALTERNATIVES
    })

    correlations = {}
    for metric in ALTERNATIVES:
        left = pd.concat(
            [merged["mattr_a"], merged["mattr_b"]], ignore_index=True
        )
        right = pd.concat(
            [merged[f"{metric}_a"], merged[f"{metric}_b"]], ignore_index=True
        )
        valid = left.notna() & right.notna()
        correlations[metric] = float(
            left[valid].corr(right[valid], method="spearman")
        )

    output = {
        "specification": (
            "joint BT model; MATTR replaced by one alternative at a time; "
            "metric-specific available-case support"
        ),
        "estimates": estimates,
        "spearman_with_mattr": correlations,
    }
    path = RESULTS / "mattr_alt_results.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
