# What Wins a Vote? Formatting, Length, and Lexical Diversity in the French Compar:IA LLM Arena

Style-control analysis of how presentation (markdown formatting, length, readability, lexical diversity, sentence structure) is associated with human preference votes in **Compar:IA**, a French government-backed LLM arena. The retained dataset contains 137,293 decisive French battles; the primary analysis uses 137,113 battles across the 116 models meeting the minimum-battle threshold.

## Research Question

Which presentation features retain an association with votes after joint adjustment, and how sensitive are model rankings to that adjustment?

## Key Findings

- **Presentation is mostly one collinear "verbosity" dimension.** Length, bold, and lists rise together and trade coefficient weight; their individual attributions are unstable across specifications.
- **Two signals stand apart and survive full joint control:** **bold formatting** (+11.0% win odds per standard deviation in the joint model) and **comparatively length-robust lexical diversity** (MATTR, +16.8%). Both are conditional correlates, not causal effects.
- **Much apparent linguistic association is between models:** several coefficients shrink substantially when per-model fixed effects are added. This does not identify presentation as either bias or a quality signal.
- **Exploratory heterogeneity varies with depth visible at the vote.** In the pooled interaction model, the formatting-only bold association is smaller in genuine multi-turn conversations (+30.1% to +7.4% odds/SD), while MATTR is essentially unchanged. Length's interaction is not significant. Vote-time depth remains endogenous and is not a direct measure of attention.
- **The bold association is not solely a subject-matter artefact.** It is positive within every large topic and remains after topic × formatting controls; this is still observational rather than causal.
- **Production shifts are striking, but external validation is inconclusive.** In the live 27 July 2026 leaderboard, style control moves GPT-5.3 from rank 47 to 1 and Mistral Medium 2508 from 2 to 28, which supplies useful face-validity evidence. The independent capability comparison has lower formatting-controlled point correlations on all seven eligible benchmarks, with every difference including zero. It also omits GPT-5.3, Mistral Medium 2508, and Gemini 3.1 Flash Lite, so it cannot validate the largest live shifts. Four LMArena preference comparisons move slightly in the other direction and are similarly uncertain.
- **Leaderboard impact:** controlling for the full presentation bundle moves 36 of 116 models by ≥10 ranks (r = 0.931); more adjustment is not automatically better.
- **Vote timing is now corrected and audited.** Source conversations continue after the retained vote in 11.2% of battles, but no later turn enters the measured text, cumulative token length, or depth stratum. The audit validates every retained vote and final-turn index against the raw release.
- **Hidden reasoning is excluded without discarding the final answer.** The parser removes paired `<think>...</think>` spans, ignores `reasoning_content`, and retains only unambiguous voter-visible text. The pinned release contains no retained vote prefix with missing final content but non-empty `reasoning_content`; untagged traces in `content` cannot be detected from these fields alone.

## Data

A single dataset: **`ministere-culture/comparia-fr-arena`** (Ministère de la Culture), revision `8cd6488` of the consolidated Compar:IA release. It is turn-level (641K rows); we retain the last decisive French reaction per conversation and truncate both response text and cumulative per-turn token totals at that vote. No CamemBERT perplexity is computed (GPU-only).

## Quickstart

```bash
# Recreate the tested Python environment.
uv sync --locked --all-groups

# Verify code and publication consistency.
uv run ruff check src tests run.py
uv run pytest -q

# Recompute the main analyses and Figures 1–3 from committed derived data.
uv run python run.py --profile core

# Recompute all analyses supported by the committed derived/auxiliary data.
uv run python run.py --profile extended
```

Every script also runs on its own (e.g. `python src/topic_analysis.py`) and
resolves `data/`, `results/`, `figures/` relative to the repo root, so it works
from any directory.

### Full rebuild from the gated source

The full profile streams the immutable Hugging Face revision, rebuilds all
derived tables, runs the vote-time and hidden-reasoning audits, and then runs
the extended analysis:

```bash
hf auth whoami
uv run python run.py --profile full
```

Row-group checkpoints carry manifests containing the upstream revision,
selected columns, and preprocessing-function hash. A missing or mismatched
manifest fails closed. Use `--reset-checkpoints` only when intentionally
discarding and rebuilding generated checkpoints.

For a local copy of the source Parquet, both the explicit path and its SHA-256
are required:

```bash
export COMPARIA_FR_ARENA_PARQUET=/absolute/path/comparia-fr-arena.parquet
export COMPARIA_FR_ARENA_SHA256=<sha256-of-that-file>
uv run python run.py --profile full
```

## What produces what

| Script (`src/`) | Section | Output |
|---|---|---|
| `build_fr_arena.py` | §2 | `data/fr_battles.parquet` (one row per battle) |
| `analyze_core.py` | §4.1–4.4 | `results/core_results.json` |
| `linguistic_analysis.py` | §4.5 | `results/linguistic_results.json` |
| `leaderboard_shift.py` | §4.5 | `results/leaderboard_shift_results.json` |
| `turn_depth_analysis.py` | §4.6 | `results/turn_depth_results.json` |
| `topic_analysis.py` | §4.7 | `results/topic_results.json` |
| `extract_prompts.py` → `task_classify.py` → `task_analysis.py` | §4.8 | `results/task_results.json` (the extended profile uses the committed text-free task table; rebuilding that table requires the gated raw dataset) |
| `robustness_random.py`; `time_block_bootstrap.py` | §4.9 | `results/robustness_random_results.json`, `time_block_results.json` (block bootstrap needs `data/timestamps.parquet`) |
| `mattr_stress.py`; `mattr_alt.py`; `analyze_mattr_alt.py` | §4.10 | `results/mattr_stress_results.json`, `data/mattr_alt.parquet`, `results/mattr_alt_results.json` (`mattr_alt.py` needs the raw dataset) |
| `external_leaderboard_analysis.py` | §4.11 | `results/external_leaderboard_results.json` (audited cached capability scores, pinned LMArena snapshots, model-match audits, correlations, and paired intervals) |
| `generate_external_figure.py` | §4.11 | `figures/fig12_external_alignment.*` |
| Live leaderboard audit | §4.11 | `results/production_ranking_examples.json` (dated production examples, kept distinct from the research release) |
| `audit_vote_timing.py` | §2.4, §4.6 | `results/vote_timing_audit_results.json` |
| `audit_reasoning_content.py` | §2.3 | `results/reasoning_content_audit_results.json` |
| `endogeneity_analysis.py` | §5.3 | `results/endogeneity_results.json` |
| `qualitative_analysis.py` | §5.4 | `results/qualitative_results.json` |
| `generate_linguistic_figure.py` | §4.5 | `figures/fig9_linguistic.*` (Figure 1) |
| `generate_polish_figures.py` | §4.6–4.7 | `figures/fig10_*`, `fig11_*` (Figures 2–3) |

## Reproducible environment and builds

Python 3.11–3.14 and [`uv`](https://docs.astral.sh/uv/) are supported.
`pyproject.toml` pins direct dependencies and `uv.lock` pins the complete
environment. `Makefile` exposes the standard checks:

```bash
make setup
make verify        # lint, tests, two-pass XeLaTeX build, artifact manifest
```

The CI workflow runs the Python checks and a clean two-pass XeLaTeX build.
`results/artifact_manifest.json` records SHA-256 hashes for every distributed
data, result, figure, source-manuscript, and PDF artifact.

## Repository Structure

```
├── manuscript/         # publication LaTeX source and compiled PDF
├── paper_draft.md      # readable manuscript source
├── run.py              # explicit core / extended / full pipeline profiles
├── pyproject.toml      # direct dependency pins
├── uv.lock             # complete tested environment lock
├── CITATION.cff        # citation metadata
├── README.md  LICENSE  feedback.md
├── src/                # analysis scripts (+ paths.py, shared path helper)
├── data/               # derived, text-free analysis tables + data README
├── results/            # *_results.json (computed outputs)
├── figures/            # publication figures (PNG + vector PDF)
└── tests/              # parser, provenance, and publication consistency tests
```

The source release is gated and may contain user-generated text. Raw prompts and
conversation text are never committed. The distributed battle table contains
derived features, model identifiers, outcomes, and opaque comparison IDs only;
see [`data/README.md`](data/README.md) for its schema and privacy boundary.

## License

Unless otherwise noted, the repository's original code, manuscript, and derived
artifacts are released under the
[Open Licence 2.0 / Licence Ouverte 2.0](LICENSE) (Etalab). The upstream
Compar:IA data remain subject to their source terms (Etalab 2.0 / CC-BY-4.0).
External leaderboard snapshots are not redistributed; only derived statistics,
source revisions, match audits, and hashes are included.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). For a public
preprint release, archive the exact repository state and use its DOI or immutable
release URL in both `CITATION.cff` and the manuscript's Code Availability
statement.
