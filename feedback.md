# Review status: Style Control Analysis for Compar:IA

Tracker for the paper (`paper_draft.md`), rebuilt on the consolidated
`comparia-fr-arena` dataset (137,214 retained decisive French battles, 116
models, votes only).

## Addressed

- **Proper tokenization.** Length is the output-token count from metadata, not a
  whitespace split, which settles the French-tokenization concern.
- **Bootstrap 1000x + Benjamini-Hochberg** on all style coefficients and rank
  changes (§3.3–3.4).
- **Joint model.** Formatting + length + linguistic features in one
  style-controlled Bradley-Terry model (§4.5); bold and MATTR survive, while
  most of the remaining features form one collinear verbosity bundle.
- **Endogeneity: confounder vs mediator** with three tests (§5.3).
- **Qualitative winner-flips** (§5.4).
- **Vote-time conversation depth.** The formatting-only bold association is
  smaller in genuine multi-turn conversations, while the joint-model length
  interaction is not significant. Depth is observational and is not treated as
  a direct measure of attention (§4.6, §1, §5.1).
- **Topic controls.** Bold is positive within every large subject class; the
  vote-time depth pattern survives topic × formatting controls (§4.7).
- **Task-type robustness check.** A coarse rule-based task taxonomy and
  within-task fits have been added. Its limited held-out agreement makes this
  exploratory rather than a definitive task control (§4.8).
- **Vote-time reconstruction.** Response text, cumulative token totals, and
  depth are truncated at the retained vote and independently audited against
  the source release (§2.4, §4.6).
- **External preference comparison.** The raw, formatting-controlled, and full
  joint-controlled Compar:IA rankings are compared with four pinned LMArena
  preference rankings (§4.11).
- **Independent capability comparison.** A content-hashed 27 July 2026 Epoch
  archive is audited at file, schema, model-match, and exclusion level. Seven
  benchmarks meet the pre-specified ten-model overlap rule. The live production
  shifts are now reported separately as face-validity evidence. The aggregate
  capability differences remain within bootstrap uncertainty and omit three
  models behind the most prominent live shifts (§4.11).
- **References.** Arena / evaluation-bias literature and method citations added;
  inline and matched to entries.

## Open

- **Validated task-type control.** The current task taxonomy is a rough proxy.
  A validated classifier or embedding-based approach remains future work.
- **Small external-benchmark overlaps.** The broad capability index matches 38
  model versions, but several domain benchmarks match only 10–13. Exact and
  audited same-build matching avoids invented comparisons, at the cost of wide
  intervals and limited provider-omission checks. GPT-5.3, Mistral Medium 2508,
  and Gemini 3.1 Flash Lite are not matched in the broad index.
- **Perplexity deliberately omitted.** CamemBERT pseudo-perplexity (Farzana's
  metric) is GPU-only and was null on the earlier export; readability is also
  null here, so it was not recomputed. Noted as a limitation.
- **No independent replication.** Single corpus from one platform.
- **Authors.** Author list and order to be settled by the authors; Maayeesha
  Farzana is credited in-text as the linguistic feature-set designer.

## Reproduce

Everything reruns from `fr_battles.parquet` via the scripts in the README.
