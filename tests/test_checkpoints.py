import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from checkpoints import prepare_checkpoint_dir, verify_local_sha256  # noqa: E402


class CheckpointSafetyTests(unittest.TestCase):
    def test_matching_manifest_allows_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "parts"
            manifest = {"format_version": 1, "processor_sha256": "abc"}
            prepare_checkpoint_dir(directory, manifest)
            (directory / "part_000.parquet").write_bytes(b"checkpoint")

            prepare_checkpoint_dir(directory, manifest)
            observed = json.loads(
                (directory / "checkpoint_manifest.json").read_text()
            )
            self.assertEqual(observed, manifest)

    def test_unmanifested_parts_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "parts"
            directory.mkdir()
            (directory / "part_000.parquet").write_bytes(b"stale")

            with self.assertRaisesRegex(RuntimeError, "without a provenance"):
                prepare_checkpoint_dir(directory, {"format_version": 1})

    def test_mismatched_manifest_requires_explicit_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "parts"
            prepare_checkpoint_dir(directory, {"processor_sha256": "old"})
            (directory / "part_000.parquet").write_bytes(b"old")

            with self.assertRaisesRegex(RuntimeError, "manifest mismatch"):
                prepare_checkpoint_dir(directory, {"processor_sha256": "new"})

            prepare_checkpoint_dir(
                directory,
                {"processor_sha256": "new"},
                reset=True,
            )
            self.assertFalse((directory / "part_000.parquet").exists())

    def test_local_file_hash_is_required_and_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.parquet"
            path.write_bytes(b"immutable source")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()

            with self.assertRaisesRegex(RuntimeError, "SHA256 is required"):
                verify_local_sha256(path, None)
            self.assertEqual(verify_local_sha256(path, digest), digest)
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                verify_local_sha256(path, "0" * 64)


if __name__ == "__main__":
    unittest.main()
