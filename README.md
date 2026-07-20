# What Wins a Vote? Formatting, Length, and Lexical Diversity in the French Compar:IA LLM Arena

Style-control analysis of how presentation (markdown formatting, length, readability, lexical diversity, sentence structure) is associated with human preference votes in **Compar:IA**, a French government-backed LLM arena. Built on the consolidated `comparia-fr-arena` release (about 138,000 decisive French battles, 116 models).

## Research Question

Which presentation features retain an association with votes after joint adjustment, and how sensitive are model rankings to that adjustment?

## Key Findings

- **Presentation is mostly one collinear "verbosity" dimension.** Length, bold, and lists rise together and trade coefficient weight; their individual attributions are unstable across specifications.
- **Two signals stand apart and survive full control:** **bold formatting** (~+10% win odds/SD in the joint model) and **length-independent lexical diversity** (MATTR, ~+18%). Both are conditional correlates, not causal effects.
- **Much apparent linguistic association is between models:** several coefficients shrink substantially when per-model fixed effects are added. This does not identify presentation as either bias or a quality signal.
- **Exploratory heterogeneity varies with completed conversation length.** Formatting and length associations are smaller in the eventual multi-turn stratum (bold +35%→+8% odds/SD), while MATTR is unchanged. Because the current fields include post-vote turns, this cannot be interpreted as pre-vote depth or attention.
- **The bold association is not solely a subject-matter artefact.** It is positive within every large topic and remains after topic × formatting controls; this is still observational rather than causal.
- **Leaderboard impact:** controlling for the full presentation bundle moves 37 of 116 models by ≥10 ranks (r = 0.94). In exact-name external comparisons, formatting-only control has the highest correlation with all four LMArena rankings tested, but its gains over raw Compar:IA are not statistically decisive; full joint control usually correlates less strongly.
- **Important timing limitation:** the current builder uses completed `full_conversation_*` fields. Post-vote turns enter the measured text in 11.2% of battles, and 40.5% of the nominal multi-turn stratum was single-turn at the retained vote. The paper now flags the depth analysis as uninterpretable and the primary estimates as requiring a vote-truncated rerun.

## Data

A single dataset: **`ministere-culture/comparia-fr-arena`** (Ministère de la Culture), revision `8cd6488` of the consolidated Compar:IA release. It is turn-level (641K rows); we retain the last decisive French reaction per conversation. The current pipeline computes features on the completed conversation, including later turns in 11.2% of cases; this must be changed to vote-truncated prefixes before publication. No CamemBERT perplexity is computed (GPU-only).

## Quickstart

```bash
# One-time: build the single battle table from comparia-fr-arena (slow; reads a
# local copy of the gated parquet if present, else streams it with a HF token).
python src/build_fr_arena.py           # -> data/fr_battles.parquet

# Then run the whole analysis (results -> results/, figures -> figures/):
python run.py
```

Every script also runs on its own (e.g. `python src/topic_analysis.py`) and
resolves `data/`, `results/`, `figures/` relative to the repo root, so it works
from any directory.

## What produces what

| Script (`src/`) | Section | Output |
|---|---|---|
| `build_fr_arena.py` | §2 | `data/fr_battles.parquet` (one row per battle) |
| `analyze_core.py` | §4.1–4.4 | `results/core_results.json` |
| `linguistic_analysis.py` | §4.5 | `results/linguistic_results.json` |
| `leaderboard_shift.py` | §4.5 | `results/leaderboard_shift_results.json` |
| `turn_depth_analysis.py` | §4.6 | `results/turn_depth_results.json` |
| `topic_analysis.py` | §4.7 | `results/topic_results.json` |
| `extract_prompts.py` → `task_classify.py` → `task_analysis.py` | §4.8 | `results/task_results.json` (task proxy; needs the raw dataset, run separately from `run.py`) |
| `robustness_random.py`; `time_block_bootstrap.py` | §4.9 | `results/robustness_random_results.json`, `time_block_results.json` (block bootstrap needs `data/timestamps.parquet`) |
| `mattr_stress.py`; `mattr_alt.py` | §4.10 | `results/mattr_stress_results.json`, `mattr_alt_results.json` (`mattr_alt.py` needs the raw dataset) |
| `external_leaderboard_analysis.py` | §4.11 | `results/external_leaderboard_results.json` (downloads pinned public LMArena snapshots) |
| `audit_vote_timing.py` | §2.4, §4.6 | `results/vote_timing_audit_results.json` |
| `endogeneity_analysis.py` | §5.3 | `results/endogeneity_results.json` |
| `qualitative_analysis.py` | §5.4 | `results/qualitative_results.json` |
| `generate_linguistic_figure.py` | §4.5 | `figures/fig9_linguistic.*` (Figure 1) |
| `generate_polish_figures.py` | §4.6–4.7 | `figures/fig10_*`, `fig11_*` (Figures 2–3) |

## Requirements

Python 3.10+, with: `pandas`, `numpy`, `scipy`, `scikit-learn`, `pyarrow`, `textstat`, `matplotlib`, `huggingface_hub`.

## Repository Structure

```
├── paper_draft.md      # the paper
├── run.py              # runs the full analysis pipeline in order
├── README.md  LICENSE  feedback.md
├── src/                # analysis scripts (+ paths.py, shared path helper)
├── data/               # fr_battles.parquet (the single battle table)
├── results/            # *_results.json (computed outputs)
└── figures/            # publication figures (PNG + PDF)
```

## License

[Open Licence 2.0 / Licence Ouverte 2.0](LICENSE) (Etalab). Data sourced from the [Compar:IA](https://comparia.beta.gouv.fr/) platform by the French Ministry of Culture, published under Etalab 2.0 / CC-BY-4.0.
