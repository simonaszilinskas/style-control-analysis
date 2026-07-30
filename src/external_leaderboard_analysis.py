#!/usr/bin/env python3
"""
Compare Compar:IA rankings with preference and capability benchmarks.

External inputs are pinned by immutable revision or SHA-256. Model matching is
exact after punctuation/case normalization, except for a short, audited alias
table whose entries denote the same model build. Nearby releases, reasoning
efforts, and model-family substitutions are deliberately not matched.

The primary external validation is against non-arena capability benchmarks.
LMArena comparisons are retained as secondary preference-leaderboard context.
For every eligible benchmark, raw, formatting-controlled, and joint-controlled
rankings are evaluated on exactly the same matched model set. Correlation
differences describe changes in alignment; they do not establish that a ranking
is better.

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
from scipy.stats import kendalltau, spearmanr

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
MIN_EXTERNAL_OVERLAP = 10

EPOCH_URL = "https://epoch.ai/data/benchmark_data.zip"
EPOCH_SNAPSHOT_DATE = "2026-07-27"
EPOCH_SHA256 = "08ed76781fe84ce0cf6c80500cdae7ed347aaf71b7ac74cd016d31198424f3e4"
EPOCH_BENCHMARKS = {
    "epoch_capabilities_index": {
        "file": "epoch_capabilities_index.csv",
        "score": "ECI Score",
        "role": "primary_broad_capability",
    },
    "gpqa_diamond": {
        "file": "gpqa_diamond.csv",
        "score": "mean_score",
        "role": "domain_specific_capability",
    },
    "frontiermath": {
        "file": "frontiermath.csv",
        "score": "mean_score",
        "role": "domain_specific_capability",
    },
    "swe_bench_verified": {
        "file": "swe_bench_verified.csv",
        "score": "mean_score",
        "role": "domain_specific_capability",
    },
    "livebench": {
        "file": "live_bench_external.csv",
        "score": "Global average",
        "role": "multi_domain_capability",
    },
    "arc_agi_2": {
        "file": "arc_agi_2_external.csv",
        "score": "Score",
        "role": "domain_specific_capability",
    },
    "scicode": {
        "file": "scicode_external.csv",
        "score": "Score",
        "role": "domain_specific_capability",
    },
    "aider_polyglot": {
        "file": "aider_polyglot_external.csv",
        "score": "Percent correct",
        "role": "domain_specific_capability",
    },
}

# Each alias was manually checked to denote the same dated model build. The
# analysis must not add aliases merely because two identifiers share a family.
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
}


def normalized_model_id(value):
    """Normalize punctuation and case without collapsing model versions."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def validate_aliases(aliases=EPOCH_MODEL_ALIASES):
    """Fail if two Compar:IA identifiers claim the same external build."""
    targets = [normalized_model_id(value) for value in aliases.values()]
    if len(targets) != len(set(targets)):
        raise ValueError("Epoch alias targets must be one-to-one")
    if any(normalized_model_id(source) == normalized_model_id(target)
           for source, target in aliases.items()):
        raise ValueError("Aliases must document a real identifier difference")


def load_external(name, url):
    """Read a pinned LMArena snapshot, caching it for offline reruns."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"lmarena_{name}_{LMARENA_REVISION[:8]}.parquet"
    if not path.exists():
        print(f"Downloading {url}")
        with urlopen(url) as response:
            path.write_bytes(response.read())
    return pd.read_parquet(BytesIO(path.read_bytes()))


def validate_epoch_payload(payload, expected_sha256=EPOCH_SHA256):
    """Verify the archived bytes before parsing any capability result."""
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise RuntimeError(
            "Epoch snapshot changed; expected SHA-256 "
            f"{expected_sha256}, received {digest}. Audit and pin the new "
            "payload before using capability comparisons."
        )
    return digest


def load_epoch_archive():
    """Load the content-hashed Epoch snapshot, caching verified bytes."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"epoch_benchmark_data_{EPOCH_SNAPSHOT_DATE}.zip"
    if path.exists():
        payload = path.read_bytes()
    else:
        print(f"Downloading {EPOCH_URL}")
        with urlopen(EPOCH_URL) as response:
            payload = response.read()
        validate_epoch_payload(payload)
        path.write_bytes(payload)
    validate_epoch_payload(payload)
    return ZipFile(BytesIO(payload))


def archive_manifest(archive):
    """Return a deterministic, machine-readable inventory of the zip."""
    return [
        {"file": item.filename, "bytes": item.file_size, "crc32": f"{item.CRC:08x}"}
        for item in sorted(archive.infolist(), key=lambda value: value.filename)
        if not item.is_dir()
    ]


def paired_bootstrap_delta(base, alternative, external, rng,
                           n_bootstrap=N_BOOTSTRAP):
    """Bootstrap paired changes in Spearman rho on one model support."""
    deltas = []
    n = len(external)
    if not (len(base) == len(alternative) == n):
        raise ValueError("All paired vectors must have identical support")
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        rho_base = spearmanr(base[idx], external[idx]).statistic
        rho_alt = spearmanr(alternative[idx], external[idx]).statistic
        delta = rho_alt - rho_base
        if np.isfinite(delta):
            deltas.append(delta)
    if not deltas:
        raise ValueError("Bootstrap produced no finite paired deltas")
    return [float(x) for x in np.percentile(deltas, [2.5, 97.5])]


def leave_one_group_out(ratings, source_scores, common, groups):
    """Sensitivity range after omitting each represented provider."""
    represented = sorted({groups[model] for model in common if groups.get(model)})
    estimates = {}
    for group in represented:
        kept = [model for model in common if groups.get(model) != group]
        if len(kept) < MIN_EXTERNAL_OVERLAP:
            continue
        external = np.array([source_scores[model] for model in kept], dtype=float)
        rho = {
            name: float(spearmanr(
                [model_ratings[model] for model in kept], external
            ).statistic)
            for name, model_ratings in ratings.items()
        }
        estimates[group] = {
            "n_models": len(kept),
            "spearman": rho,
            "delta_vs_raw": {
                name: rho[name] - rho["raw"]
                for name in ("formatting_controlled", "joint_controlled")
            },
        }
    if not estimates:
        return {"status": "unavailable", "reason": "insufficient grouped overlap"}
    ranges = {}
    for name in ("formatting_controlled", "joint_controlled"):
        values = [entry["delta_vs_raw"][name] for entry in estimates.values()]
        ranges[name] = [float(min(values)), float(max(values))]
    return {
        "status": "available",
        "grouping": "Epoch Organization field",
        "delta_vs_raw_range": ranges,
        "omissions": estimates,
    }


def compare_scores(ratings, source_scores, common, rng, groups=None,
                   n_bootstrap=N_BOOTSTRAP,
                   min_overlap=MIN_EXTERNAL_OVERLAP):
    """Calculate paired rank alignment on one explicitly shared model set."""
    common = list(common)
    missing = {
        name: sorted(set(common) - set(model_ratings))
        for name, model_ratings in ratings.items()
    }
    missing = {name: values for name, values in missing.items() if values}
    if missing:
        raise ValueError(f"Rating vectors do not share support: {missing}")
    if set(common) - set(source_scores):
        raise ValueError("External scores do not cover the declared common set")

    result = {
        "n_models": len(common),
        "minimum_overlap": min_overlap,
        "eligible": len(common) >= min_overlap,
    }
    if not result["eligible"]:
        result["reason"] = (
            f"fewer than {min_overlap} exact/audited same-build matches"
        )
        return result

    external = np.array([source_scores[model] for model in common], dtype=float)
    comp = {
        name: np.array([model_ratings[model] for model in common], dtype=float)
        for name, model_ratings in ratings.items()
    }
    spearman = {
        name: float(spearmanr(value, external).statistic)
        for name, value in comp.items()
    }
    kendall = {
        name: float(kendalltau(value, external).statistic)
        for name, value in comp.items()
    }
    delta = {}
    for name in ("formatting_controlled", "joint_controlled"):
        delta[name] = {
            "point": spearman[name] - spearman["raw"],
            "bootstrap_95_ci": paired_bootstrap_delta(
                comp["raw"], comp[name], external, rng, n_bootstrap
            ),
        }
    result.update({
        "spearman": spearman,
        "kendall_tau_b": kendall,
        "delta_vs_raw": delta,
    })
    if groups is not None:
        result["leave_one_provider_out"] = leave_one_group_out(
            ratings, source_scores, common, groups
        )
    return result


def prepare_epoch_benchmark(frame, models, score_column):
    """Validate schema and create scores plus complete match provenance."""
    required = {"Model version", score_column}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Epoch schema missing columns: {missing_columns}")

    work = frame.dropna(subset=["Model version", score_column]).copy()
    work[score_column] = pd.to_numeric(work[score_column], errors="raise")
    work["normalized_id"] = work["Model version"].map(normalized_model_id)
    grouped = work.groupby("normalized_id")[score_column]
    summary = grouped.agg(["count", "mean", "min", "max"])
    ambiguous_ids = set(summary[
        (summary["count"] > 1)
        & ((summary["max"] - summary["min"]).abs() > 1e-12)
    ].index)
    scores_by_id = summary.loc[
        ~summary.index.isin(ambiguous_ids), "mean"
    ].to_dict()

    source_rows = {}
    for normalized, rows in work.groupby("normalized_id", sort=True):
        source_rows[normalized] = {
            "source_model_ids": sorted(set(rows["Model version"].astype(str))),
            "organizations": (
                sorted(set(rows["Organization"].dropna().astype(str)))
                if "Organization" in rows else []
            ),
        }

    scores = {}
    source_ids = {}
    organizations = {}
    records = []
    claimed_source_ids = {}
    for model in sorted(models):
        alias_target = EPOCH_MODEL_ALIASES.get(model)
        requested_id = alias_target or model
        normalized = normalized_model_id(requested_id)
        record = {
            "comparia_model_id": model,
            "requested_epoch_id": requested_id,
            "match_rule": "audited_same_build_alias" if alias_target else "exact",
        }
        if normalized in ambiguous_ids:
            record["status"] = "excluded_ambiguous_source_scores"
            record.update(source_rows.get(normalized, {}))
        elif normalized in scores_by_id:
            if normalized in claimed_source_ids:
                raise ValueError(
                    "Epoch source model matched more than once: "
                    f"{requested_id!r} is claimed by "
                    f"{claimed_source_ids[normalized]!r} and {model!r}"
                )
            claimed_source_ids[normalized] = model
            metadata = source_rows[normalized]
            record["status"] = "matched"
            record.update(metadata)
            score = float(scores_by_id[normalized])
            scores[model] = score
            source_ids[model] = metadata["source_model_ids"]
            if len(metadata["organizations"]) == 1:
                organizations[model] = metadata["organizations"][0]
        else:
            record["status"] = "not_present"
        records.append(record)

    return {
        "scores": scores,
        "source_ids": source_ids,
        "organizations": organizations,
        "records": records,
        "ambiguous_normalized_ids": sorted(ambiguous_ids),
        "source_rows_after_missing_score_filter": int(len(work)),
        "source_columns": list(frame.columns),
    }


def main():
    validate_aliases()
    out_path = RESULTS / "external_leaderboard_results.json"
    previous_output = None
    if out_path.exists():
        try:
            previous_output = json.loads(out_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            previous_output = None

    battles = pd.read_parquet(BATTLES)
    joint_features = FORMATTING + ["length"] + LINGUISTIC
    required = [
        f"{feature}_{side}"
        for feature in joint_features for side in ("a", "b")
    ]
    sub = battles.dropna(subset=required).copy()
    counts = pd.concat([sub["model_a_name"], sub["model_b_name"]]).value_counts()
    models = sorted(counts[counts >= MIN_BATTLES].index)
    sub = sub[
        sub["model_a_name"].isin(models) & sub["model_b_name"].isin(models)
    ]

    ratings = {
        "raw": fit_bt(sub, models, [])[0],
        "formatting_controlled": fit_bt(sub, models, FORMATTING)[0],
        "joint_controlled": fit_bt(sub, models, joint_features)[0],
    }

    preference_comparisons = {}
    for offset, (leaderboard, url) in enumerate(SOURCES.items()):
        external = load_external(leaderboard, url)
        required_columns = {"category", "model_name", "rating"}
        if not required_columns.issubset(external.columns):
            raise ValueError(
                f"LMArena schema missing {sorted(required_columns-set(external.columns))}"
            )
        for category_index, category in enumerate(("overall", "french")):
            scores = (
                external[external["category"].eq(category)]
                .dropna(subset=["rating"])
                .set_index("model_name")["rating"]
            )
            common = sorted(set(models) & set(scores.index))
            score_dict = {model: float(scores[model]) for model in common}
            rng = np.random.default_rng(SEED + offset * 10 + category_index)
            statistics = compare_scores(ratings, score_dict, common, rng)
            key = f"lmarena_{leaderboard}_{category}"
            preference_comparisons[key] = {
                "source_type": "human_preference_leaderboard",
                "analysis_role": "secondary_context",
                "n_exact_matches": len(common),
                "matched_models": common,
                "external_scores": score_dict,
                **statistics,
            }
            if statistics["eligible"]:
                print(
                    f"{key}: n={len(common)}  "
                    + "  ".join(
                        f"{name}={value:.3f}"
                        for name, value in statistics["spearman"].items()
                    )
                )

    capability_benchmarks = {}
    epoch_error = None
    epoch_inventory = []
    reused_audited_result_cache = False
    try:
        epoch = load_epoch_archive()
    except RuntimeError as error:
        epoch = None
        epoch_error = str(error)
        print(f"Epoch capability comparison unavailable: {epoch_error}")

    if epoch is not None:
        epoch_inventory = archive_manifest(epoch)
        archived_files = {entry["file"] for entry in epoch_inventory}
        required_files = {spec["file"] for spec in EPOCH_BENCHMARKS.values()}
        missing_files = sorted(required_files - archived_files)
        if missing_files:
            raise ValueError(f"Epoch archive missing files: {missing_files}")

        for benchmark_index, (key, spec) in enumerate(EPOCH_BENCHMARKS.items()):
            frame = pd.read_csv(epoch.open(spec["file"]))
            prepared = prepare_epoch_benchmark(frame, models, spec["score"])
            common = sorted(prepared["scores"])
            rng = np.random.default_rng(SEED + 100 + benchmark_index)
            statistics = compare_scores(
                ratings,
                prepared["scores"],
                common,
                rng,
                groups=prepared["organizations"],
            )
            capability_benchmarks[key] = {
                "source_type": "capability_benchmark",
                "analysis_role": spec["role"],
                "source_file": spec["file"],
                "score_column": spec["score"],
                "score_direction": "higher_is_better",
                "n_version_matches": len(common),
                "matched_models": common,
                "matched_source_ids": prepared["source_ids"],
                "external_scores": prepared["scores"],
                "match_audit": prepared["records"],
                "excluded_ambiguous_normalized_ids": (
                    prepared["ambiguous_normalized_ids"]
                ),
                "source_rows_after_missing_score_filter": (
                    prepared["source_rows_after_missing_score_filter"]
                ),
                "source_columns": prepared["source_columns"],
                **statistics,
            }
            if statistics["eligible"]:
                print(
                    f"{key}: n={len(common)}  "
                    + "  ".join(
                        f"{name}={value:.3f}"
                        for name, value in statistics["spearman"].items()
                    )
                )
            else:
                print(f"{key}: n={len(common)}  ineligible ({statistics['reason']})")
    elif (
        previous_output
        and previous_output.get("epoch_snapshot", {}).get("status")
        in {"available", "reused_cached_scores"}
        and previous_output.get("epoch_snapshot", {}).get("sha256") == EPOCH_SHA256
        and previous_output.get("matching", {}).get("aliases") == EPOCH_MODEL_ALIASES
        and previous_output.get("capability_benchmarks")
    ):
        # The Epoch URL is mutable. A prior successful run records the
        # hash-verified snapshot's scores and complete match audit in the
        # generated result. Reuse that audited evidence when the live URL has
        # changed, but always recompute correlations against the current
        # Compar:IA ratings. If neither source is available, capability results
        # remain absent rather than silently accepting new bytes.
        reused_audited_result_cache = True
        epoch_inventory = previous_output["epoch_snapshot"].get(
            "archive_manifest", []
        )
        for benchmark_index, (key, cached) in enumerate(
            previous_output["capability_benchmarks"].items()
        ):
            score_dict = {
                model: float(score)
                for model, score in cached["external_scores"].items()
                if model in models
            }
            common = sorted(score_dict)
            groups = {}
            for item in cached.get("match_audit", []):
                organizations = item.get("organizations", [])
                model = item.get("comparia_model_id")
                if (
                    item.get("status") == "matched"
                    and model in common
                    and len(organizations) == 1
                ):
                    groups[model] = organizations[0]
            rng = np.random.default_rng(SEED + 100 + benchmark_index)
            statistics = compare_scores(
                ratings, score_dict, common, rng, groups=groups or None
            )
            record = {
                field: value
                for field, value in cached.items()
                if field not in {
                    "eligible", "reason", "spearman", "kendall_tau_b",
                    "delta_vs_raw", "leave_one_group_out",
                    "leave_one_provider_out",
                }
            }
            record["matched_models"] = common
            record["n_version_matches"] = len(common)
            record.update(statistics)
            capability_benchmarks[key] = record
        print("Reused audited Epoch scores from the hash-verified result cache")

    output = {
        "interpretation": (
            "Correlation differences measure changes in external alignment; "
            "they do not establish that any ranking is better."
        ),
        "analysis_priority": {
            "primary": "non-arena capability benchmarks",
            "secondary": "LMArena preference leaderboards",
        },
        "comparia_common_support": {
            "battles": int(len(sub)),
            "models": len(models),
            "model_ids": models,
        },
        "matching": {
            "rule": "exact normalized version IDs plus audited same-build aliases",
            "fuzzy_or_family_substitution": False,
            "aliases": EPOCH_MODEL_ALIASES,
        },
        "minimum_external_overlap": MIN_EXTERNAL_OVERLAP,
        "n_bootstrap": N_BOOTSTRAP,
        "bootstrap_seed": SEED,
        "lmarena_snapshot": {
            "revision": LMARENA_REVISION,
            "publish_date": LMARENA_PUBLISH_DATE,
        },
        "preference_comparisons": preference_comparisons,
        # Compatibility key for existing downstream readers.
        "comparisons": preference_comparisons,
        "epoch_snapshot": {
            "url": EPOCH_URL,
            "snapshot_date": EPOCH_SNAPSHOT_DATE,
            "sha256": EPOCH_SHA256,
            "status": (
                "available"
                if epoch is not None
                else "reused_cached_scores"
                if reused_audited_result_cache
                else "unavailable"
            ),
            "error": (
                None
                if epoch is not None or reused_audited_result_cache
                else epoch_error
            ),
            "live_download_error": (
                epoch_error if reused_audited_result_cache else None
            ),
            "reused_audited_result_cache": reused_audited_result_cache,
            "archive_manifest": epoch_inventory,
        },
        "capability_benchmarks": capability_benchmarks,
    }
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
