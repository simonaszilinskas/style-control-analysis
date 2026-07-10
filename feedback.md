# Consolidated Review: Style Control Analysis for Compar:IA

## Overall Assessment

**Paper-worthy?** Yes, suitable for a workshop or short/findings paper (ACL Findings, EMNLP Findings, LREC-COLING, Eval4NLP). A full conference paper requires significant additional work.

**Core contribution:** First style control analysis on a non-English (French) government-backed LLM evaluation dataset, with three-dataset triangulation (conversations, reactions, votes).

---

## P0, Blocking Issues (fix before anything else)

### Summary contradicts actual output

The final summary table states effect sizes of ~0.02–0.03 (odds ratio ~1.03), but the code outputs coefficients of 0.07–0.17 (odds ratios 1.08–1.18, i.e. **8–18% changes in odds**). The summary also claims "most rank changes are NOT significant," while cell D5 shows **10/10 biggest rank changes are significant**. This appears to be a stale copy-paste from the LMSYS blog post template. A reviewer would reject on this alone.

**Fix:** Rewrite the summary to match computed results. Estimated effort: 1 hour.

---

## P1, Methodological Fixes (required for any submission)

### Tokenization is fundamentally flawed

`len(text.split())` is invalid for French. Contractions (`l'ordinateur`, `qu'il`), punctuation rules, and clitics make whitespace splitting unreliable. This contaminates the primary "length" confounder that drives the largest style coefficient.

**Fix:** Replace with proper tokenization (tiktoken, HuggingFace tokenizers, or model-specific token counts from the dataset). Re-run all length-dependent analyses. Estimated effort: 1 day.

### The correlation drop is the most interesting finding, and it's unexplored

The correlation between like rate and BT rating drops from **0.829 → 0.645** after style control (~22% decrease). This is the paper's most provocative result: is style control removing bias or removing genuine quality signal?

**Fix:** Dedicate a full section to investigating this drop. Stratify by model tier, task type, and style intensity. Test whether the drop is driven by specific models or is uniform. Estimated effort: 2–3 days.

---

## P2, Statistical Rigor (expected for peer review)

### Bootstrap sample sizes are too small

100 iterations for coefficients and 50 for ratings are below publication standards. CIs may not be stable.

**Fix:** Increase to 1,000+ samples for both. Effort is primarily compute time.

### No multiple comparison correction

96 models and numerous pairwise tests with no Bonferroni or FDR correction. False positive risk is elevated.

**Fix:** Apply appropriate correction (Benjamini-Hochberg recommended). Estimated effort: 1 day.

### ~~Endogeneity is not addressed~~ ✅ DONE

~~Style is treated as a confounder to remove, but it may be a **mediator**, better models may produce better-structured output because they are more capable, not because users are fooled by formatting. No causal framework is presented.~~

**Resolved:** Added dedicated section 5.3 "Endogeneity: Confounder or Mediator?" with three empirical tests: (1) quality-formatting correlation (r=0.60), (2) tier-stratified style effects showing ~2x larger formatting bias for bottom-tier battles vs top-tier, (3) interaction model with bootstrap CIs. Conclusion: formatting is both partial mediator and partial confounder. Analysis in `endogeneity_analysis.py`, results in `endogeneity_results.json`.

### No content or task controls

No stratification by topic, task type, or difficulty. Style features may proxy for task complexity (e.g., coding tasks naturally produce more structured output).

**Fix:** If task metadata is available, add stratified analysis. If not, acknowledge as a limitation. Estimated effort: 1–2 days if data permits.

---

## P3, Enhancements (strengthen the contribution)

### Ablation study on style features

Which features actually drive rank changes? Length alone vs. markdown features vs. combined. Currently all four are bundled together.

**Fix:** Run the BT model with subsets of style features. Show incremental contribution. Estimated effort: 2 days.

### ~~Qualitative analysis~~, DONE

~~Sample 30–50 conversations where style control flips the winner. Manual inspection: did the style-adjusted winner actually produce better content?~~

**Done:** Added Section 5.4 with analysis of 5,847 winner-flipping battles (5.3% of non-tie battles). Identified 3 patterns: similar content with different formatting, appropriate brevity penalized, and users sometimes seeing through formatting. Script: `qualitative_analysis.py` → `qualitative_results.json`.

### Cross-lingual comparison

Compare findings with English Chatbot Arena results (same methodology, same time period if possible). Key question: are French users less influenced by formatting than English users?

**Fix:** Replicate on a subset of English data. Estimated effort: 1 week.

### ~~Visualization~~, DONE

~~The notebook is entirely tables. Add forest plots for style coefficients with CIs, scatter plots of standard vs. style-controlled ELO, and distribution plots of markdown features by model.~~

**Done:** Created 4 publication-quality figures in `figures/`: (1) forest plot of style coefficients with bootstrap CIs, (2) scatter of standard vs. style-controlled BT ratings with reasoning models highlighted, (3) rank change waterfall for top 20 movers, (4) tier-stratified style effects grouped bar chart. Script: `generate_figures.py`. Output in PDF and PNG.

---

## Strengths to Preserve

- **Novel dataset:** Compar:IA is rare, a French government-backed arena with 133K votes across 100+ models.
- **Three-dataset triangulation:** Combining message-level reactions with conversation-level votes is a strong validation design.
- **Statistical self-awareness (Part D):** Bootstrap CIs, effect size interpretation, and CI overlap analysis demonstrate rigor.
- **Honest findings:** Modest but real style effects, and the observation that most adjacent models are statistically indistinguishable, are valuable contributions.

---

## Suggested Paper Framing

**Title:** *"Style Control in Non-English LLM Evaluation: Evidence from the French Compar:IA Dataset"*

**Angle:** Lead with the cross-cultural and multilingual novelty. Use the three-dataset design as a methodological contribution. Position the modest-but-real style effects as an honest empirical finding. Make the correlation drop (0.829 → 0.645) a central discussion point.

**Target venues:** ACL Rolling Review (Evaluation track), EMNLP Findings, LREC-COLING, Eval4NLP workshop.

---

## Summary Table

| Priority | Issue | Effort | Impact |
|----------|-------|--------|--------|
| P0 | Fix summary to match outputs | 1 hour | Credibility |
| P1 | Proper tokenization | 1 day | Validity |
| P1 | Analyze correlation drop | 2–3 days | Core finding |
| P2 | Bootstrap 1,000+ samples | Compute | Rigor |
| P2 | Multiple comparison correction | 1 day | Rigor |
| ~~P2~~ | ~~Discuss endogeneity~~ ✅ | ~~1 day~~ | ~~Framing~~ |
| P2 | Task/content controls | 1–2 days | Validity |
| P3 | Ablation study | 2 days | Depth |
| P3 | Qualitative validation | 2–3 days | Credibility |
| P3 | Cross-lingual comparison | 1 week | Novelty |
| ~~P3~~ | ~~Publication-quality figures~~ ✅ | ~~2 days~~ | ~~Presentation~~ |