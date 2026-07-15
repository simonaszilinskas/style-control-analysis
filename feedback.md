# Review status: Style Control Analysis for Compar:IA

Tracker for the paper (`paper_draft.md`), rebuilt on the consolidated
`comparia-fr-arena` dataset (single dataset, ~137K decisive French battles, 116
models, votes only).

## Addressed

- **Proper tokenization.** Length is the output-token count from metadata, not a
  whitespace split, which settles the French-tokenization concern.
- **Bootstrap 1000x + Benjamini-Hochberg** on all style coefficients and rank
  changes (§3.3-3.4).
- **Joint model.** Formatting + length + linguistic features in one
  style-controlled BT (§4.5); bold and MATTR survive, most of the rest is one
  collinear verbosity bundle.
- **Endogeneity: confounder vs mediator** with three tests (§5.3).
- **Qualitative winner-flips** (§5.4).
- **Reading depth (C. Benavent).** Formatting and length fade with reading depth,
  only MATTR persists; framed by the three levels of reading (§4.6, §1, §5.1).
- **Topic controls.** Bold positive within every subject; reading-depth survives
  topic x formatting interactions (§4.7).
- **References.** Arena / evaluation-bias literature and method citations added;
  inline and matched to entries.

## Open

- **Reading-science references are placeholders.** The comprehension citations
  (Kintsch-van Dijk, Rayner, etc.) are our reading of Christophe's proposal;
  he should confirm or replace them. A dual-process framing (heuristic vs
  systematic reading) is floated in discussion but not committed.
- **Task-type control.** §4.7 controls for topic (subject matter), not task type
  (summarise / translate / code). Mining a task signal from prompt text and
  repeating §4.7 is the main remaining validity step.
- **Perplexity deliberately omitted.** CamemBERT pseudo-perplexity (Farzana's
  metric) is GPU-only and was null on the earlier export; readability is also
  null here, so it was not recomputed. Noted as a limitation.
- **No independent replication.** Single corpus from one platform.
- **Authors.** Author list and order to be settled by the authors; Maayeesha
  Farzana is credited in-text as the linguistic feature-set designer.

## Reproduce

Everything reruns from `fr_battles.parquet` via the scripts in the README.
