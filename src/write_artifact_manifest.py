#!/usr/bin/env python3
"""Write cryptographic provenance for the publication's distributable artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

from paths import ROOT


OUTPUT = ROOT / "results" / "artifact_manifest.json"
GLOBS = (
    "data/*.parquet",
    "results/*_results.json",
    "results/production_ranking_examples.json",
    "figures/*.pdf",
    "figures/*.png",
    "src/*.py",
    "tests/*.py",
    "manuscript/paper.tex",
    "manuscript/paper.pdf",
    "run.py",
    "pyproject.toml",
    "uv.lock",
    "Makefile",
    "CITATION.cff",
    "README.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "data/README.md",
    "manuscript/README.md",
    "manuscript/REVIEW.md",
    ".github/workflows/*.yml",
)
PACKAGES = (
    "huggingface-hub",
    "matplotlib",
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "scipy",
    "textstat",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    paths = {
        path
        for pattern in GLOBS
        for path in ROOT.glob(pattern)
        if path.is_file() and path != OUTPUT
    }
    output = {
        "format_version": 1,
        "hash_algorithm": "SHA-256",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            package: importlib.metadata.version(package) for package in PACKAGES
        },
        "artifacts": {
            str(path.relative_to(ROOT)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(paths)
        },
    }
    OUTPUT.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote provenance for {len(paths)} artifacts -> {OUTPUT}")


if __name__ == "__main__":
    main()
