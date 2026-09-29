PYTHON := uv run python

.PHONY: setup test lint core extended full manuscript arxiv manifest verify

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

# arXiv compiles with pdfLaTeX and rejects paths outside the upload, so the
# bundle rewrites ../figures/ to figures/ and is test-built before packing.
ARXIV := build/arxiv

arxiv:
	rm -rf $(ARXIV) && mkdir -p $(ARXIV)/figures
	sed 's|\.\./figures/|figures/|g' manuscript/paper.tex > $(ARXIV)/paper.tex
	cp figures/fig9_linguistic.pdf figures/fig10_reading_depth.pdf \
	   figures/fig11_topic_controls.pdf figures/fig12_external_alignment.pdf $(ARXIV)/figures/
	cd $(ARXIV) && pdflatex -interaction=nonstopmode -halt-on-error paper.tex >/dev/null
	cd $(ARXIV) && pdflatex -interaction=nonstopmode -halt-on-error paper.tex >/dev/null
	cd $(ARXIV) && COPYFILE_DISABLE=1 tar -czf ../arxiv.tar.gz paper.tex figures
	@echo "wrote build/arxiv.tar.gz (test build: $(ARXIV)/paper.pdf)"

manifest:
	$(PYTHON) src/write_artifact_manifest.py

verify: lint test manuscript manifest
