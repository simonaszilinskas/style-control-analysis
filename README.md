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

## Analysis Pipeline

```bash
# 0. Build the single battle table from comparia-fr-arena (reads a local copy of the
#    gated parquet if present, else streams it with a HF token).
python build_fr_arena.py               # -> fr_battles.parquet

# 1. Core formatting: BT rankings, rank changes, position bias, ablation (§4.1-4.4)
python analyze_core.py

# 2. Length + linguistic joint model (§4.5)
python linguistic_analysis.py
python leaderboard_shift.py            # standard vs formatting vs joint ranking shift

# 3. Reading depth: formatting/length x multi-turn interactions (§4.6)
python turn_depth_analysis.py

# 4. Topic controls: within-topic fits + topic x style interactions (§4.7)
python topic_analysis.py

# 5. Endogeneity (confounder vs mediator) and qualitative winner-flips (§5.3-5.4)
python endogeneity_analysis.py
python qualitative_analysis.py

# 6. Figures
python generate_linguistic_figure.py   # Fig 1 (joint coefficients + shrinkage)
python generate_polish_figures.py      # Fig 2 (reading depth), Fig 3 (topic controls)
```

## Requirements

Python 3.10+, with: `pandas`, `numpy`, `scipy`, `scikit-learn`, `pyarrow`, `textstat`, `matplotlib`, `huggingface_hub`.

## Repository Structure

```
├── build_fr_arena.py             # comparia-fr-arena -> fr_battles.parquet (one battle table)
├── analyze_core.py               # formatting BT, rank changes, position bias, ablation (§4.1-4.4)
├── linguistic_analysis.py        # joint formatting+length+linguistic model (§4.5)
├── leaderboard_shift.py          # standard vs formatting vs joint ranking shift (§4.5)
├── turn_depth_analysis.py        # reading-depth interactions (§4.6)
├── topic_analysis.py             # topic controls (§4.7)
├── endogeneity_analysis.py       # confounder vs mediator (§5.3)
├── qualitative_analysis.py       # winner-flip prevalence/asymmetry (§5.4)
├── generate_linguistic_figure.py # Figure 1
├── generate_polish_figures.py    # Figures 2-3
├── paper_draft.md                # Paper draft
├── *_results.json                # computed outputs
├── fr_battles.parquet            # the single battle table
├── figures/                      # publication figures
└── feedback.md                   # peer review tracker
```

## License

[Open Licence 2.0 / Licence Ouverte 2.0](LICENSE) (Etalab). Data sourced from the [Compar:IA](https://comparia.beta.gouv.fr/) platform by the French Ministry of Culture, published under Etalab 2.0 / CC-BY-4.0.
