"""Safe, resumable checkpoint helpers for raw-data streaming jobs.

Checkpoint parts are ignored by Git and may survive code changes.  A manifest
therefore records the immutable upstream revision, selected columns, and a hash
of the functions that determine each part's contents.  Existing parts are never
silently adopted: a missing or mismatched manifest fails closed.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import shutil
from pathlib import Path
from typing import Callable, Iterable


MANIFEST_NAME = "checkpoint_manifest.json"
_VERIFIED_LOCAL_FILES: set[tuple[str, str]] = set()


def processor_sha256(functions: Iterable[Callable]) -> str:
    """Hash the source of the functions that determine checkpoint contents."""
    source = "\n\n".join(inspect.getsource(function) for function in functions)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def prepare_checkpoint_dir(
    directory: Path,
    expected_manifest: dict,
    *,
    reset: bool = False,
) -> None:
    """Create or validate a checkpoint directory.

    ``reset=True`` is an explicit request to discard generated parts.  Without
    it, any pre-existing Parquet parts must have an exactly matching manifest.
    """
    directory = Path(directory)
    if reset and directory.exists():
        shutil.rmtree(directory)

    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / MANIFEST_NAME
    parts = sorted(directory.glob("*.parquet"))

    if manifest_path.exists():
        observed = json.loads(manifest_path.read_text(encoding="utf-8"))
        if observed != expected_manifest:
            raise RuntimeError(
                f"checkpoint manifest mismatch in {directory}; rerun with "
                "--reset-checkpoints to rebuild generated parts"
            )
        return

    if parts:
        raise RuntimeError(
            f"{directory} contains checkpoint parts without a provenance "
            "manifest; rerun with --reset-checkpoints"
        )

    manifest_path.write_text(
        json.dumps(expected_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def verify_local_sha256(path: str | Path, expected_sha256: str | None) -> str:
    """Verify a caller-supplied immutable hash before accepting a local input."""
    path = Path(path)
    if not expected_sha256:
        raise RuntimeError(
            "COMPARIA_FR_ARENA_SHA256 is required when "
            "COMPARIA_FR_ARENA_PARQUET is set"
        )
    expected_sha256 = expected_sha256.lower()
    cache_key = (str(path.resolve()), expected_sha256)
    if cache_key in _VERIFIED_LOCAL_FILES:
        return expected_sha256

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    observed = digest.hexdigest()
    if observed != expected_sha256:
        raise RuntimeError(
            f"local dataset SHA-256 mismatch for {path}: expected "
            f"{expected_sha256}, observed {observed}"
        )
    _VERIFIED_LOCAL_FILES.add(cache_key)
    return observed
