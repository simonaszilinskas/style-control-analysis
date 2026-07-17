# What Wins a Vote? Formatting, Length, and Lexical Diversity in the French Compar:IA LLM Arena

Style-control analysis of how presentation (markdown formatting, length, readability, lexical diversity, sentence structure) biases human preference votes in **Compar:IA**, a French government-backed LLM arena. Built on the consolidated `comparia-fr-arena` release (about 138,000 decisive French battles, 116 models).

## Research Question

Which presentation features independently change which models rank highest, and which apparent effects are really model skill?

## Key Findings

- **Presentation is mostly one collinear "verbosity" dimension.** Length, bold, and lists rise together and trade coefficient weight; their individual attributions are unstable across specifications.
- **Two signals stand apart and survive full control:** **bold formatting** (~+10% win odds/SD in the joint model) and **length-independent lexical diversity** (MATTR, ~+18%, richer vocabulary wins even at equal length).
- **Much apparent linguistic effect is model skill:** coefficients roughly halve when per-model strengths are added (confounder vs mediator).
- **Presentation acts through reading depth.** Proxying attention by conversation length, the pull of formatting *and* length fades sharply once readers engage over several turns (bold +38%→+6% odds/SD), while MATTR is unchanged. Only vocabulary richness survives an attentive read.
- **The premium is not a subject-matter artefact.** Bold is positive within every topic, and the reading-depth effect survives topic × formatting controls.
- **Leaderboard impact:** controlling for the full presentation bundle moves 33 of 116 models by ≥10 ranks (r = 0.95); heavy formatters fall ~30 places, concise strong models rise.

## Data

A single dataset: **`ministere-culture/comparia-fr-arena`** (Ministère de la Culture), the consolidated Compar:IA release. It is turn-level (641K turns); we keep decisive French votes, take the full conversation the voter saw, and compute all features on it. Topic (`categories`) and reading depth (turn count) are native to the export. Votes only, no reactions; no CamemBERT perplexity (GPU-only).

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
| `robustness_random.py` | §4.9 | `results/robustness_random_results.json` |
| `mattr_stress.py`; `mattr_alt.py` | §4.10 | `results/mattr_stress_results.json`, `mattr_alt_results.json` (`mattr_alt.py` needs the raw dataset) |
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
