# Contributing

Changes to analysis code and publication claims should remain reproducible and
auditable.

1. Create the locked environment with `uv sync --locked --all-groups`.
2. Run `uv run ruff check src tests run.py` and `uv run pytest -q`.
3. Recompute the smallest pipeline profile affected by the change.
4. If a headline value changes, update both `paper_draft.md` and
   `manuscript/paper.tex`, then add or update a consistency test.
5. Build the manuscript twice with `make manuscript` and visually inspect every
   rendered page.
6. Regenerate `results/artifact_manifest.json` last.

Never commit raw prompts, raw conversations, gated-source exports, access
tokens, or unmanifested row-group checkpoints. Do not reuse checkpoints after a
manifest mismatch; rebuild them explicitly.
