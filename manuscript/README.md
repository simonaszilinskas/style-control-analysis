# LaTeX manuscript

`paper.tex` is the compiled LaTeX version of the repository's research
manuscript. It uses the vector figures in `../figures/` directly.

To rebuild the PDF from this directory:

```bash
xelatex -interaction=nonstopmode -halt-on-error paper.tex
xelatex -interaction=nonstopmode -halt-on-error paper.tex
```

The title page lists Simonas Zilinskas, Maayeesha Farzana, and Christophe
Benavent. The shared correspondence address is
`contact@comparia.beta.gouv.fr`.

For arXiv, which builds with pdfLaTeX, run `make arxiv` from the repository
root. It writes an upload-ready `build/arxiv.tar.gz` after a test build.

