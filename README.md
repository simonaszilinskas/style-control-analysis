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

## Data

The analysis uses three datasets from HuggingFace (`ministere-culture/comparia-*`), auto-downloaded on first run:

| Dataset | Rows | Content |
|---------|------|---------|
| conversations | 396K | All conversations with model responses |
| reactions | 82K | Message-level likes/dislikes + quality attributes |
| votes | 133K | Conversation-level winner selections |

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
├── generate_figures.py           # Publication-quality figures
├── paper_draft.md                # Paper draft
├── clean_analysis_results.json   # Main analysis outputs
├── endogeneity_results.json      # Endogeneity test outputs
├── qualitative_results.json      # Qualitative analysis outputs
├── qualitative_examples.json     # Illustrative battle examples
├── battles_bt_styled.parquet     # Intermediate styled battle data
├── figures/                      # 8 publication figures (PDF + PNG)
└── feedback.md                   # Peer review tracker
```

## License

Research use. Data sourced from the [Compar:IA](https://comparia.beta.gouv.fr/) platform by the French Ministry of Culture.
