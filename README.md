# Style Control Bias in French LLM Evaluation

Analysis of style control bias in the **Compar:IA** dataset — a French government-backed LLM arena for evaluating language models through human preference judgments.

## Research Question

Do formatting features (bold, lists, headers, emoji, code blocks) bias human preference judgments, inflating rankings for "stylish" models?

## Key Findings

- **Bold** (+19.0%), **lists** (+18.0%), and **headers** (+15.6%) significantly increase win probability
- Code blocks and emoji have no significant effect
- Standard vs style-controlled Bradley-Terry rankings correlate at **r = 0.976**, but **76/89 models** show significant rank changes
- Reasoning models gain **+2.3 ranks** on average after style control
- Style bias is stronger among lower-quality models (bottom-tier: bold +24.6%) than top-tier (+16.0%)
- **Linguistic extension** (§4.7): adding length and linguistic features shows formatting survives these controls (bold +13.0%, lists +10.8%, headers +9.3%), that length carries a substantial share of the raw formatting effect, and that the one additional robust signal is length-independent lexical diversity (MATTR +19.3%); perplexity adds essentially nothing
- **Replication** (§4.8): on the independent `comparia-fr-arena` dataset (126K French battles), bold (+13.7%) and length-independent diversity (MATTR +15.2%) reproduce almost exactly; the divergent features (length↔lists, readability) confirm the collinearity/instability caveats

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

Research use. Data sourced from the [Compar:IA](https://comparia.beta.gouv.fr/) platform by the French Ministry of Culture.
