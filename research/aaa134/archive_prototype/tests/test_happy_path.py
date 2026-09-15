"""The one path that is allowed to succeed -- and what it is *not* evidence of."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from research.aaa134.archive_prototype import PROTOTYPE_LABEL
from research.aaa134.archive_prototype.enumeration import enumerate_members, read_identity
from research.aaa134.archive_prototype.publish import publish_archive, retrieve_archive, verify_retrieval

from .helpers import IDENTITY, backend, build_attempt, set_credential


class PublishRetrieveVerifyTest(unittest.TestCase):
    def setUp(self) -> None:
        set_credential(self)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.attempt = build_attempt(self.base / "attempt")
        self.store = backend(self.base / "store")

    def test_full_cycle_into_a_fresh_empty_directory(self) -> None:
        result = publish_archive(self.attempt, self.store, archive_id="arch-0001")
        self.assertEqual(result.producer_checksum_cross_check["status"], "PASS")
        self.assertEqual(result.locator.object_count, len(enumerate_members(self.attempt)))

        fresh = self.base / "retrieved"
        written = retrieve_archive(self.store, result.locator, fresh)
        self.assertEqual(written, result.locator.object_count)

        verdict = verify_retrieval(
            self.store, result.locator, fresh, expected_identity=read_identity(self.attempt)
        )
        self.assertEqual(verdict.verdict, "PASS", verdict.problems)
        self.assertEqual(verdict.objects_verified, result.locator.object_count)

    def test_retrieved_bytes_are_identical_without_consulting_the_original(self) -> None:
        result = publish_archive(self.attempt, self.store, archive_id="arch-0002")
        fresh = self.base / "retrieved2"
        retrieve_archive(self.store, result.locator, fresh)
        for member in enumerate_members(self.attempt):
            self.assertEqual(
                (self.attempt / member.relative_path).read_bytes(),
                (fresh / member.relative_path).read_bytes(),
                member.relative_path,
            )

    def test_the_locator_says_it_is_not_durable(self) -> None:
        result = publish_archive(self.attempt, self.store, archive_id="arch-0003")
        self.assertFalse(result.locator.durable)
        self.assertEqual(result.locator.label, PROTOTYPE_LABEL)
        manifest = json.loads(self.store.read_manifest(result.locator))
        self.assertFalse(manifest["durable"])
        self.assertEqual(manifest["label"], PROTOTYPE_LABEL)

    def test_identity_is_carried_from_the_run_not_invented(self) -> None:
        result = publish_archive(self.attempt, self.store, archive_id="arch-0004")
        manifest = json.loads(self.store.read_manifest(result.locator))
        self.assertEqual(manifest["identity"], IDENTITY.as_dict())

    def test_a_missing_metadata_field_becomes_unknown_rather_than_a_guess(self) -> None:
        root = build_attempt(self.base / "attempt-thin")
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        del metadata["source_commit"]
        (root / "metadata.json").write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
        self.assertEqual(read_identity(root).source_commit, "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
