#!/usr/bin/env python3
"""
Compare Compar:IA rankings with preference leaderboards and capability benchmarks.

The external files are public and fixed by a Hugging Face revision or content
hash. Only exact identifiers and audited same-build aliases are matched: no
fuzzy model-family substitutions. We report Spearman rank correlations for
Compar:IA's raw, formatting-controlled, and full joint-controlled rankings
against LMArena's raw and style-controlled Text Arena rankings, both overall
and for French. We also compare against the Epoch AI Benchmarking Hub when its
archive matches the audited hash. If that live archive changes, preference
results are still saved and capability results are marked unavailable rather
than silently switching snapshots.

    python src/external_leaderboard_analysis.py
        -> results/external_leaderboard_results.json
"""

from io import BytesIO
import hashlib
import json
import re
from urllib.request import urlopen
from zipfile import ZipFile

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

EPOCH_URL = "https://epoch.ai/data/benchmark_data.zip"
EPOCH_SNAPSHOT_DATE = "2026-07-19"
EPOCH_SHA256 = "9b19aff50418b2c61bffbb93656db4d5dda7dd9250a22f0c27816bd5a24ac1ae"
EPOCH_BENCHMARKS = {
    "epoch_capabilities_index": ("epoch_capabilities_index.csv", "ECI Score"),
    "gpqa_diamond": ("gpqa_diamond.csv", "mean_score"),
    "frontiermath": ("frontiermath.csv", "mean_score"),
    "swe_bench_verified": ("swe_bench_verified.csv", "mean_score"),
    "livebench": ("live_bench_external.csv", "Global average"),
    "arc_agi_2": ("arc_agi_2_external.csv", "Score"),
    "scicode": ("scicode_external.csv", "Score"),
    "aider_polyglot": ("aider_polyglot_external.csv", "Percent correct"),
}

# Only aliases where the Compar:IA identifier and Epoch identifier denote the
# same dated model build.  We deliberately do not collapse reasoning-effort
# variants or nearby releases merely because they share a family name.
EPOCH_MODEL_ALIASES = {
    "claude-3-5-sonnet-v2": "claude-3-5-sonnet-20241022",
    "claude-3-7-sonnet": "claude-3-7-sonnet-20250219",
    "gemini-2.0-flash": "gemini-2.0-flash-001",
    "gemma-3-12b": "gemma-3-12b-it",
    "gemma-3-27b": "gemma-3-27b-it",
    "gemma-3-4b": "gemma-3-4b-it",
    "llama-3.1-405b": "Llama-3.1-405B-Instruct",
    "llama-3.1-70b": "Llama-3.1-70B-Instruct",
    "llama-3.1-8b": "Llama-3.1-8B-Instruct",
    "llama-3.3-70b": "Llama-3.3-70B-Instruct",
    "mistral-small-24b-instruct-2501": "mistral-small-2501",
    "mistral-small-3.1-24b": "mistral-small-2503",
    "qwen2.5-coder-32b-instruct": "Qwen2.5-Coder-32B-Instruct",
}


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
        delta = rho_alt - rho_base
        if np.isfinite(delta):
            deltas.append(delta)
    return [float(x) for x in np.percentile(deltas, [2.5, 97.5])]


def normalized_model_id(value):
    """Normalize punctuation/case without collapsing model versions."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def load_epoch_archive():
    """Load the content-hashed Epoch AI snapshot, caching it for re-runs."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"epoch_benchmark_data_{EPOCH_SNAPSHOT_DATE}.zip"
    if path.exists():
        payload = path.read_bytes()
    else:
        print(f"Downloading {EPOCH_URL}")
        with urlopen(EPOCH_URL) as response:
            payload = response.read()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != EPOCH_SHA256:
            raise RuntimeError(
                "Epoch snapshot changed; audit the new data and update the "
                "snapshot date and SHA-256 before proceeding."
            )
        path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != EPOCH_SHA256:
        raise RuntimeError(f"Cached Epoch snapshot has unexpected SHA-256: {digest}")
    return ZipFile(BytesIO(payload))


def compare_scores(ratings, source_scores, common, rng):
    """Calculate correlations and paired deltas on one matched model set."""
    external = np.array([source_scores[m] for m in common], dtype=float)
    comp = {
        name: np.array([model_ratings[m] for m in common], dtype=float)
        for name, model_ratings in ratings.items()
    }
    rho = {
        name: float(spearmanr(value, external).statistic)
        for name, value in comp.items()
    }
    delta = {}
    for name in ("formatting_controlled", "joint_controlled"):
        delta[name] = {
            "point": rho[name] - rho["raw"],
            "bootstrap_95_ci": paired_bootstrap_delta(
                comp["raw"], comp[name], external, rng
            ),
        }
    return rho, delta


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
            score_dict = {m: float(scores[m]) for m in common}
            rho, delta = compare_scores(ratings, score_dict, common, rng)
            key = f"lmarena_{leaderboard}_{category}"
            comparisons[key] = {
                "source_type": "human_preference_leaderboard",
                "n_exact_matches": len(common),
                "spearman": rho,
                "delta_vs_raw": delta,
                "matched_models": common,
                "external_scores": score_dict,
            }
            print(f"{key}: n={len(common)}  "
                  + "  ".join(f"{name}={value:.3f}" for name, value in rho.items()))

    epoch_comparisons = {}
    epoch_error = None
    try:
        epoch = load_epoch_archive()
    except RuntimeError as error:
        epoch = None
        epoch_error = str(error)
        print(f"Epoch capability comparison unavailable: {epoch_error}")
    if epoch is not None:
        for key, (filename, score_column) in EPOCH_BENCHMARKS.items():
            frame = pd.read_csv(epoch.open(filename))
            frame = frame.dropna(subset=["Model version", score_column]).copy()
            frame["normalized_id"] = frame["Model version"].map(normalized_model_id)
            # Keep duplicate rows only when they report the same score. Differing
            # scores can encode an unlabelled reasoning budget, scaffold, or edit
            # format; averaging those would invent a model configuration.
            grouped = frame.groupby("normalized_id")[score_column]
            summary = grouped.agg(["count", "mean", "min", "max"])
            ambiguous_ids = set(summary[
                (summary["count"] > 1)
                & ((summary["max"] - summary["min"]).abs() > 1e-12)
            ].index)
            by_id = summary.loc[~summary.index.isin(ambiguous_ids), "mean"]
            matched = {}
            matched_source_ids = {}
            for model in models:
                source_id = EPOCH_MODEL_ALIASES.get(model, model)
                normalized = normalized_model_id(source_id)
                if normalized in by_id.index:
                    matched[model] = float(by_id[normalized])
                    matched_source_ids[model] = source_id
            common = sorted(matched)
            rho, delta = compare_scores(ratings, matched, common, rng)
            epoch_comparisons[key] = {
                "source_type": "capability_benchmark",
                "source_file": filename,
                "score_column": score_column,
                "n_version_matches": len(common),
                "spearman": rho,
                "delta_vs_raw": delta,
                "matched_models": common,
                "matched_source_ids": matched_source_ids,
                "external_scores": matched,
                "excluded_ambiguous_source_ids": sorted(ambiguous_ids),
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
        "epoch_snapshot": {
            "url": EPOCH_URL,
            "snapshot_date": EPOCH_SNAPSHOT_DATE,
            "sha256": EPOCH_SHA256,
            "matching": "exact normalized version IDs plus audited aliases",
            "status": "available" if epoch is not None else "unavailable",
            "error": epoch_error,
        },
        "capability_benchmarks": epoch_comparisons,
    }
    out_path = RESULTS / "external_leaderboard_results.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
