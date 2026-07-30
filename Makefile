PYTHON := uv run python

.PHONY: setup test lint core extended full manuscript manifest verify

setup:
	uv sync --locked --all-groups

test:
	uv run pytest -q

lint:
	uv run ruff check src tests run.py

core:
	$(PYTHON) run.py --profile core

extended:
	$(PYTHON) run.py --profile extended

full:
	$(PYTHON) run.py --profile full

manuscript:
	cd manuscript && xelatex -interaction=nonstopmode -halt-on-error paper.tex
	cd manuscript && xelatex -interaction=nonstopmode -halt-on-error paper.tex

manifest:
	$(PYTHON) src/write_artifact_manifest.py

verify: lint test manuscript manifest
