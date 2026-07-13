# What Wins a Vote? Presentation Bias in the French Compar:IA LLM Arena

Analysis of how presentation, formatting, length, and linguistic properties of the text, biases human preference judgments in the **Compar:IA** dataset, a French government-backed LLM arena.

## Research Question

Which presentation features (markdown formatting, length, readability, lexical diversity, sentence structure, perplexity) independently change which models rank highest, and which apparent effects are really model skill?

## Key Findings

- **Presentation is mostly one collinear "verbosity" dimension.** Length, bold, and lists rise together (Δlength–Δbold ρ = 0.65) and trade coefficient weight; their individual attributions are not stable across datasets.
- **Two signals survive full control and replicate on a second dataset:** **bold formatting** (~+13% win odds/SD) and **length-independent lexical diversity** (MATTR, +15–19%, richer vocabulary wins even at equal length).
- **Presentation acts on the vote through reading depth.** Proxying attention by conversation length, the formatting premium is concentrated in quick single-turn votes and fades by ~three-quarters (bold: +42%→+9% odds/SD) once readers engage over several turns, whereas length turns from null to positive and MATTR is unchanged. This maps the three feature families onto three levels of reading (shape / argument / words).
- **The premium is not a subject-matter artefact.** Bold, lists, and headers are positive within every topic (`categories` taxonomy), and the reading-depth result survives topic × formatting interactions. Topic is controlled; task type (coding vs summarising vs translating) is not, and is left to future work.
- **Readability and perplexity add essentially nothing** once length and formatting are controlled.
- **Much apparent linguistic effect is model skill:** coefficients roughly halve when per-model strengths are added (confounder vs mediator).
- **Formatting-only view (for comparison with English style control):** bold +19.0%, lists +18.0%, headers +15.6%; 76/89 models shift significantly; heavy formatters drop sharply. Controlling for the *full* presentation bundle moves the leaderboard more (r = 0.92 vs 0.95; 21/83 models move ≥10 ranks).
- **Robustness:** results replicate across two independently built French datasets (~145K votes+reactions and ~126K comparia-fr-arena battles).

## Data

The primary analysis uses three datasets from HuggingFace (`ministere-culture/comparia-*`), auto-downloaded on first run:

| Dataset | Rows | Content |
|---------|------|---------|
| conversations | 396K | All conversations with model responses |
| reactions | 82K | Message-level likes/dislikes + quality attributes |
| votes | 133K | Conversation-level winner selections |

The linguistic extension (§4.7) is replicated on a second, independent export, **`ministere-culture/comparia-fr-arena`** (the newer consolidated arena dataset, ~138K decisive French battles), as a robustness check (§4.8). See `build_fr_arena.py` / `robustness_fr_arena.py`.

## Analysis Pipeline

```bash
# 1. Main analysis: logistic regression, Bradley-Terry rankings, bootstrap CIs
python clean_and_analyze.py

# 2. Endogeneity analysis: confounder vs mediator tests
python endogeneity_analysis.py

# 3. Qualitative analysis: winner-flipping battles
python qualitative_analysis.py

# 4. Generate publication figures
python generate_figures.py

# 5. Linguistic extension: length + readability/diversity/structure/perplexity
python linguistic_analysis.py

# 6. Robustness of the linguistic extension on comparia-fr-arena (independent dataset)
HF_TOKEN=hf_... python build_fr_arena.py   # streams the gated dataset (~10-15 min)
python robustness_fr_arena.py

# 7. Reading depth: does presentation matter less when the answer is read more carefully? (§4.9)
python turn_depth_analysis.py   # formatting x multi-turn interactions, needs the votes parquet for turn counts

# 8. Topic controls: is the formatting premium a subject-matter proxy? (§4.10)
python topic_analysis.py        # within-topic fits + topic x formatting interactions, needs the conversations parquet
```

### Style Features

Five dimensions extracted from response text via regex:
- **Markdown headers** (h1–h6)
- **Markdown lists** (ordered/unordered)
- **Markdown bold**
- **Code blocks**
- **Emoji**

Token length is excluded from style control (confounded with completeness).

## Requirements

Python 3.10+, with: `pandas`, `numpy`, `scipy`, `scikit-learn`, `pyarrow`, `tqdm`

## Repository Structure

```
├── clean_and_analyze.py          # Main analysis pipeline
├── endogeneity_analysis.py       # Tier-stratified interaction tests
├── qualitative_analysis.py       # Winner-flipping battle analysis
├── linguistic_analysis.py        # Length + linguistic-feature joint model (§4.7)
├── build_fr_arena.py             # Stream comparia-fr-arena -> fr_arena_battles.parquet
├── robustness_fr_arena.py        # Replicate §4.7 on comparia-fr-arena (§4.8)
├── turn_depth_analysis.py        # Reading depth: formatting x multi-turn interactions (§4.9)
├── topic_analysis.py             # Topic controls: within-topic fits + topic x style interactions (§4.10)
├── generate_figures.py           # Publication-quality figures
├── generate_linguistic_figure.py # Figure 9 (linguistic extension)
├── paper_draft.md                # Paper draft
├── clean_analysis_results.json   # Main analysis outputs
├── endogeneity_results.json      # Endogeneity test outputs
├── qualitative_results.json      # Qualitative analysis outputs
├── linguistic_results.json       # Linguistic extension outputs
├── fr_arena_results.json         # comparia-fr-arena robustness outputs
├── battles_bt_styled.parquet     # Intermediate styled battle data
├── linguistic_features.parquet   # Per-conversation linguistic features (merge key: conversation_pair_id)
├── fr_arena_battles.parquet      # comparia-fr-arena battles + features
├── figures/                      # publication figures (PDF + PNG)
└── feedback.md                   # Peer review tracker
```

## License

[Open Licence 2.0 / Licence Ouverte 2.0](LICENSE) (Etalab). Data sourced from the [Compar:IA](https://comparia.beta.gouv.fr/) platform by the French Ministry of Culture, which is itself published under Etalab 2.0 / CC-BY-4.0. The vendored `style_control.py` originates from compar:IA (Apache-2.0); its upstream notice is retained in the file.
