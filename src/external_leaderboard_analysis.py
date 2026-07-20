#!/usr/bin/env python3
"""
Compare Compar:IA rankings with external LMArena leaderboard snapshots.

The external files are public, small, and pinned to a Hugging Face revision.
Only exact model identifiers are matched: no fuzzy aliases or model-family
substitutions.  We report Spearman rank correlations for Compar:IA's raw,
formatting-controlled, and full joint-controlled rankings against LMArena's
raw and style-controlled Text Arena rankings, both overall and for French.

    python src/external_leaderboard_analysis.py
        -> results/external_leaderboard_results.json
"""

from io import BytesIO
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from linguistic_analysis import fit_bt, FORMATTING, LINGUISTIC, MIN_BATTLES
from paths import BATTLES, DATA, RESULTS


LMARENA_REVISION = "afed939e10281b660a4369206ca505b2bf5e0208"
LMARENA_PUBLISH_DATE = "2026-07-16"
LMARENA_BASE = (
    "https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset/resolve/"
    f"{LMARENA_REVISION}"
)
SOURCES = {
    "raw": f"{LMARENA_BASE}/text/latest-00000-of-00001.parquet",
    "style_controlled": (
        f"{LMARENA_BASE}/text_style_control/latest-00000-of-00001.parquet"
    ),
}
CACHE = DATA / "external"
N_BOOTSTRAP = 10_000
SEED = 42


def load_external(name, url):
    """Read a pinned public snapshot, caching it for offline re-runs."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"lmarena_{name}_{LMARENA_REVISION[:8]}.parquet"
    if not path.exists():
        print(f"Downloading {url}")
        with urlopen(url) as response:
            path.write_bytes(response.read())
    return pd.read_parquet(BytesIO(path.read_bytes()))


def paired_bootstrap_delta(base, alternative, external, rng):
    """Bootstrap the change in Spearman rho over matched model identifiers."""
    deltas = []
    n = len(external)
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        rho_base = spearmanr(base[idx], external[idx]).statistic
        rho_alt = spearmanr(alternative[idx], external[idx]).statistic
        deltas.append(rho_alt - rho_base)
    return [float(x) for x in np.percentile(deltas, [2.5, 97.5])]


def main():
    battles = pd.read_parquet(BATTLES)
    joint_features = FORMATTING + ["length"] + LINGUISTIC
    required = [f"{feature}_{side}"
                for feature in joint_features for side in ("a", "b")]
    sub = battles.dropna(subset=required).copy()
    counts = pd.concat([sub["model_a_name"], sub["model_b_name"]]).value_counts()
    models = sorted(counts[counts >= MIN_BATTLES].index)
    sub = sub[sub["model_a_name"].isin(models)
              & sub["model_b_name"].isin(models)]

    ratings = {
        "raw": fit_bt(sub, models, [])[0],
        "formatting_controlled": fit_bt(sub, models, FORMATTING)[0],
        "joint_controlled": fit_bt(sub, models, joint_features)[0],
    }

    rng = np.random.default_rng(SEED)
    comparisons = {}
    for leaderboard, url in SOURCES.items():
        external = load_external(leaderboard, url)
        for category in ("overall", "french"):
            scores = (external[external["category"].eq(category)]
                      .dropna(subset=["rating"])
                      .set_index("model_name")["rating"])
            common = sorted(set(models) & set(scores.index))
            ext = np.array([scores[m] for m in common], dtype=float)
            comp = {
                name: np.array([model_ratings[m] for m in common], dtype=float)
                for name, model_ratings in ratings.items()
            }
            rho = {name: float(spearmanr(value, ext).statistic)
                   for name, value in comp.items()}
            delta = {}
            for name in ("formatting_controlled", "joint_controlled"):
                delta[name] = {
                    "point": rho[name] - rho["raw"],
                    "bootstrap_95_ci": paired_bootstrap_delta(
                        comp["raw"], comp[name], ext, rng
                    ),
                }
            key = f"lmarena_{leaderboard}_{category}"
            comparisons[key] = {
                "n_exact_matches": len(common),
                "spearman": rho,
                "delta_vs_raw": delta,
                "matched_models": common,
            }
            print(f"{key}: n={len(common)}  "
                  + "  ".join(f"{name}={value:.3f}" for name, value in rho.items()))

    output = {
        "comparia_common_support": {
            "battles": int(len(sub)),
            "models": len(models),
        },
        "matching": "exact model identifiers only",
        "lmarena_revision": LMARENA_REVISION,
        "lmarena_publish_date": LMARENA_PUBLISH_DATE,
        "n_bootstrap": N_BOOTSTRAP,
        "comparisons": comparisons,
    }
    out_path = RESULTS / "external_leaderboard_results.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
