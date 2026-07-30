# Preprint readiness review

## Current status

The statistical claims, analysis scripts, derived artifacts, LaTeX source, and
rendered PDF have been re-audited against one another. Automated checks cover
the vote-time parser, hidden-reasoning handling, checkpoint provenance,
external-source matching, headline manuscript values, conversation-depth
derivations, and deterministic exploratory bootstraps.

The paper is technically ready for author review. It is not ready for public
deposit until the author-supplied declarations below are confirmed.

All reviewed changes remain local and uncommitted. No branch, commit, push, or
pull request has been created.

## Resolved in this review

- Corrected Table 4 to report the single- and multi-turn slopes implied by the
  pooled interaction model rather than separate descriptive stratum fits.
- Added the exact complete-case flow from 137,293 retained battles to the
  127,092-battle joint sample and changed the nested bold comparison to the
  same-support estimate.
- Documented which analyses winsorize contrasts, every bootstrap count,
  multiple-testing families, and the battle-level independence assumption.
- Explicitly recentered Bradley-Terry model coefficients to a sum-zero,
  mean-1000 rating convention.
- Added an aggregate-only, reproducible audit for hidden reasoning and final
  content; raw text is not persisted.
- Made task bootstraps deterministic and relabelled unadjusted topic/task
  intervals as exploratory.
- Added checkpoint manifests, local-source SHA-256 verification, a locked
  Python environment, CI, build targets, artifact hashes, a data codebook, and
  citation metadata.
- Clarified that the mutable Epoch source is currently reproducible from an
  audited derived cache, not from a redistributed raw archive.
- Added missing MTLD and SWE-bench references, data/code availability, and
  ethics/privacy statements.
- Removed forced figure placement, duplicated figure numbering, clipped topic
  labels, long-identifier overflow, appendix-table overlap, and orphaned table
  headings.
- Recomputed the complete extended analysis from the current derived tables,
  including 1,000-resample primary/joint/depth models, 400-resample
  topic/task strata, 500 weekly blocks, and 10,000 paired external
  correlation resamples.

## Final verification evidence

- `uv run ruff check src tests run.py`: passed.
- `uv run pytest -q`: 35 tests and 11 subtests passed.
- `uv run python -m compileall -q src tests run.py`: passed.
- `git diff --check`: passed.
- Two-pass XeLaTeX build: 19 pages, with no overfull/underfull boxes, LaTeX
  warnings, or package warnings.
- All 19 pages were rendered at 144 dpi and inspected; the extracted text
  bounding boxes contain no words outside the PDF page.
- The full raw-data reasoning audit found 86 newly retained visible-final
  battles, 7 ambiguous legacy cases removed, a net increase of 79, and zero
  retained vote prefixes with missing final content plus nonempty
  `reasoning_content`.

## Author confirmation required before deposit

1. Confirm that all three authors consent to publication and to this order:
   Simonas Zilinskas, Maayeesha Farzana, Christophe Benavent.
2. Supply and approve each author's affiliation and, if desired, ORCID.
3. Confirm authorization to use `contact@comparia.beta.gouv.fr` as the shared
   correspondence address.
4. Approve a CRediT author-contribution statement.
5. Supply or approve the funding and competing-interest declarations.
6. After creating an immutable repository release or archive, replace the
   pending Code Availability wording with its DOI or release URL.

The Epoch capability sensitivity analysis is verifiable from the retained
hash-audited scores, archive inventory, and model-match records, but the
original source payload is not redistributed and its live URL is mutable. If
licensing permits, an immutable archive of that exact payload would improve
end-to-end reproducibility.

## Verification commands

```bash
uv sync --locked --all-groups
make verify
```
