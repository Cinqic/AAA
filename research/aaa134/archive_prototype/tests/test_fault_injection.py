"""Every way this prototype is supposed to fail.

A durable-evidence tool tested only on the happy path is a data-loss utility
with good manners. Each test below breaks one specific thing and asserts that
the break is *detected*, not merely that something raised.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from research.aaa134.archive_prototype.contract import (
    CredentialError,
    DuplicateAttemptError,
    FinalizationError,
    ImmutabilityError,
    IntegrityError,
    LocatorError,
    PreflightError,
    UnsafePathError,
)
from research.aaa134.archive_prototype.enumeration import (
    MANIFEST_NAME,
    canonical_manifest,
    enumerate_members,
)
from research.aaa134.archive_prototype.local_backend import CREDENTIAL_ENV, LocalImmutableBackend
from research.aaa134.archive_prototype.publish import (
    publish_archive,
    refuse_finalized_overwrite,
    retrieve_archive,
    verify_retrieval,
)

from .helpers import IDENTITY, backend, build_attempt, set_credential


class FaultInjectionTest(unittest.TestCase):
    def setUp(self) -> None:
        set_credential(self)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.attempt = build_attempt(self.base / "attempt")
        self.store = backend(self.base / "store")

    # -- helpers -------------------------------------------------------------

    def _published(self, archive_id: str = "arch"):
        result = publish_archive(self.attempt, self.store, archive_id=archive_id)
        fresh = self.base / f"retrieved-{archive_id}"
        retrieve_archive(self.store, result.locator, fresh)
        return result, fresh

    def _manifest_path(self, archive_id: str) -> Path:
        return self.base / "store" / archive_id / MANIFEST_NAME

    # -- retrieval-side corruption ------------------------------------------

    def test_missing_shard_is_detected(self) -> None:
        result, fresh = self._published("missing")
        (fresh / "records" / "shard-0001.jsonl.gz").unlink()
        verdict = verify_retrieval(self.store, result.locator, fresh)
        self.assertEqual(verdict.verdict, "FAIL")
        self.assertTrue(any("missing after retrieval" in problem for problem in verdict.problems))

    def test_corrupted_shard_is_detected(self) -> None:
        result, fresh = self._published("corrupt")
        target = fresh / "records" / "shard-0000.jsonl.gz"
        data = bytearray(target.read_bytes())
        data[-1] ^= 0xFF  # same length, different bytes: only a checksum catches this
        target.write_bytes(bytes(data))
        verdict = verify_retrieval(self.store, result.locator, fresh)
        self.assertEqual(verdict.verdict, "FAIL")
        self.assertTrue(any("checksum mismatch" in problem for problem in verdict.problems))

    def test_truncated_shard_is_detected(self) -> None:
        result, fresh = self._published("truncated")
        target = fresh / "records" / "shard-0002.jsonl.gz"
        target.write_bytes(target.read_bytes()[:-3])
        verdict = verify_retrieval(self.store, result.locator, fresh)
        self.assertEqual(verdict.verdict, "FAIL")
        self.assertTrue(any("truncated or extended" in problem for problem in verdict.problems))

    def test_incomplete_retrieval_is_detected(self) -> None:
        result, fresh = self._published("incomplete")
        for path in (fresh / "records").glob("shard-*.jsonl.gz"):
            path.unlink()
        verdict = verify_retrieval(self.store, result.locator, fresh)
        self.assertEqual(verdict.verdict, "FAIL")
        self.assertGreaterEqual(len([p for p in verdict.problems if "missing" in p]), 3)

    def test_unexpected_extra_file_is_detected(self) -> None:
        result, fresh = self._published("extra")
        (fresh / "records" / "not-in-the-manifest.jsonl.gz").write_bytes(b"surprise")
        verdict = verify_retrieval(self.store, result.locator, fresh)
        self.assertEqual(verdict.verdict, "FAIL")
        self.assertTrue(any("unexpected file" in problem for problem in verdict.problems))

    # -- manifest and index tampering ---------------------------------------

    def test_manifest_checksum_tampering_is_detected(self) -> None:
        result, fresh = self._published("tamper")
        path = self._manifest_path("tamper")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["objects"][0]["sha256"] = "0" * 64
        path.write_bytes(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n")
        with self.assertRaises(IntegrityError):
            retrieve_archive(self.store, result.locator, self.base / "tamper-fresh")
        verdict = verify_retrieval(self.store, result.locator, fresh)
        self.assertEqual(verdict.verdict, "FAIL")

    def test_manifest_edited_to_be_self_consistent_is_still_detected(self) -> None:
        """The interesting case: rows *and* the internal digest changed together.

        Only the locator's independently held manifest digest catches this, which
        is exactly why the locator carries one instead of trusting the manifest.
        """

        result, _ = self._published("selfconsistent")
        path = self._manifest_path("selfconsistent")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        rows = [dict(row) for row in manifest["objects"]]
        rows[0]["size_bytes"] = 1
        from research.aaa134.archive_prototype.contract import ArchiveIdentity

        path.write_bytes(
            canonical_manifest(
                manifest["archive_id"],
                ArchiveIdentity(**manifest["identity"]),
                rows,
                durable=False,
                label=manifest["label"],
            )
        )
        with self.assertRaises(IntegrityError):
            retrieve_archive(self.store, result.locator, self.base / "selfconsistent-fresh")

    def test_index_tampering_is_detected(self) -> None:
        result, fresh = self._published("index")
        target = fresh / "records" / "index.json"
        payload = json.loads(target.read_text(encoding="utf-8"))
        payload["shards"] = payload["shards"][:1]
        target.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        verdict = verify_retrieval(self.store, result.locator, fresh)
        self.assertEqual(verdict.verdict, "FAIL")
        self.assertTrue(any("records/index.json" in problem for problem in verdict.problems))

    def test_producer_checksum_manifest_is_cross_checked_not_adopted(self) -> None:
        root = build_attempt(self.base / "attempt-badsums")
        rows = json.loads((root / "checksums.json").read_text(encoding="utf-8"))
        rows["records/shard-0000.jsonl.gz"] = "1" * 64
        (root / "checksums.json").write_text(json.dumps(rows, sort_keys=True), encoding="utf-8")
        with self.assertRaises(IntegrityError):
            publish_archive(root, backend(self.base / "store-badsums"), archive_id="badsums")

    def test_absent_producer_checksums_is_not_verified_not_pass(self) -> None:
        root = build_attempt(self.base / "attempt-nosums", with_checksums=False)
        result = publish_archive(root, backend(self.base / "store-nosums"), archive_id="nosums")
        self.assertEqual(result.producer_checksum_cross_check["status"], "NOT_VERIFIED")

    # -- locator and version identity ---------------------------------------

    def test_invalid_locator_is_refused(self) -> None:
        result, _ = self._published("locator")
        broken = type(result.locator)(**{**result.locator.as_dict(), "archive_id": "does-not-exist"})
        with self.assertRaises(LocatorError):
            retrieve_archive(self.store, broken, self.base / "locator-fresh")

    def test_locator_object_count_disagreement_is_refused(self) -> None:
        result, _ = self._published("count")
        broken = type(result.locator)(**{**result.locator.as_dict(), "object_count": 1})
        with self.assertRaises(IntegrityError):
            retrieve_archive(self.store, broken, self.base / "count-fresh")

    def test_wrong_remote_version_is_refused(self) -> None:
        result, _ = self._published("version")
        with self.assertRaises(LocatorError):
            list(self.store.open_object(result.locator, "records/shard-0000.jsonl.gz", "0" * 32))

    def test_wrong_archive_identity_is_reported(self) -> None:
        result, fresh = self._published("identity")
        from research.aaa134.archive_prototype.contract import ArchiveIdentity

        wrong = ArchiveIdentity(**{**IDENTITY.as_dict(), "candidate_id": "causal-innovation-clip-050-v1"})
        verdict = verify_retrieval(self.store, result.locator, fresh, expected_identity=wrong)
        self.assertEqual(verdict.verdict, "FAIL")
        self.assertTrue(any("identity mismatch for candidate_id" in p for p in verdict.problems))

    def test_mismatched_protocol_and_source_metadata_are_reported(self) -> None:
        result, fresh = self._published("meta")
        from research.aaa134.archive_prototype.contract import ArchiveIdentity

        wrong = ArchiveIdentity(
            **{**IDENTITY.as_dict(), "protocol_hash": "f" * 64, "source_commit": "deadbeef"}
        )
        verdict = verify_retrieval(self.store, result.locator, fresh, expected_identity=wrong)
        self.assertEqual(verdict.verdict, "FAIL")
        self.assertTrue(any("protocol_hash" in p for p in verdict.problems))
        self.assertTrue(any("source_commit" in p for p in verdict.problems))

    # -- write-once and duplication -----------------------------------------

    def test_duplicate_object_is_refused(self) -> None:
        members = enumerate_members(self.attempt)
        from research.aaa134.archive_prototype.enumeration import stream_file

        self.store.put_object("dup", members[0], stream_file(self.attempt / members[0].relative_path))
        with self.assertRaises(ImmutabilityError):
            self.store.put_object("dup", members[0], stream_file(self.attempt / members[0].relative_path))

    def test_duplicate_attempt_upload_is_refused(self) -> None:
        publish_archive(self.attempt, self.store, archive_id="dupattempt-1")
        with self.assertRaises(DuplicateAttemptError):
            publish_archive(self.attempt, self.store, archive_id="dupattempt-2")

    def test_duplicate_finalization_is_refused(self) -> None:
        result = publish_archive(self.attempt, self.store, archive_id="finaltwice")
        with self.assertRaises(FinalizationError):
            self.store.finalize("finaltwice", IDENTITY, result.stored)

    def test_finalize_refuses_an_empty_archive(self) -> None:
        refuse_finalized_overwrite(self.store, "emptyarchive", IDENTITY)

    def test_write_after_finalization_is_refused(self) -> None:
        publish_archive(self.attempt, self.store, archive_id="afterfinal")
        members = enumerate_members(self.attempt)
        from research.aaa134.archive_prototype.enumeration import stream_file

        with self.assertRaises(ImmutabilityError):
            self.store.put_object(
                "afterfinal", members[0], stream_file(self.attempt / members[0].relative_path)
            )

    def test_finalize_refuses_an_incomplete_stored_set(self) -> None:
        members = enumerate_members(self.attempt)
        from research.aaa134.archive_prototype.enumeration import stream_file

        stored = [
            self.store.put_object("partial", member, stream_file(self.attempt / member.relative_path))
            for member in members[:2]
        ]
        with self.assertRaises(FinalizationError):
            self.store.finalize("partial", IDENTITY, stored[:1])

    # -- interruption and resume --------------------------------------------

    def test_interrupted_upload_leaves_no_finalized_archive_and_resumes(self) -> None:
        store = backend(self.base / "store-interrupt")
        store.interrupt_after_objects = 3
        with self.assertRaises(ConnectionError):
            publish_archive(self.attempt, store, archive_id="interrupted")
        self.assertFalse((self.base / "store-interrupt" / "interrupted" / MANIFEST_NAME).exists())
        self.assertFalse(any((self.base / "store-interrupt").glob("*.locator.json")))
        # No partial object is left readable.
        self.assertFalse(list((self.base / "store-interrupt" / "interrupted").rglob("*.partial")))

        store.interrupt_after_objects = None
        resume_store = backend(self.base / "store-interrupt")
        result = publish_archive(self.attempt, resume_store, archive_id="interrupted-resumed")
        fresh = self.base / "interrupt-fresh"
        retrieve_archive(resume_store, result.locator, fresh)
        self.assertEqual(verify_retrieval(resume_store, result.locator, fresh).verdict, "PASS")

    def test_resume_reuploads_a_member_whose_bytes_changed(self) -> None:
        members = enumerate_members(self.attempt)
        from research.aaa134.archive_prototype.contract import StoredObject

        stale = StoredObject(
            relative_path=members[0].relative_path,
            object_key="stale",
            version_id="stale",
            size_bytes=members[0].size_bytes,
            backend_sha256="9" * 64,
        )
        result = publish_archive(self.attempt, self.store, archive_id="resume", already_stored=[stale])
        self.assertEqual(result.resumed_objects, 0)

    # -- preflight, capacity, credentials -----------------------------------

    def test_capacity_shortfall_is_refused_before_uploading(self) -> None:
        small = backend(self.base / "store-small", capacity_bytes=16)
        with self.assertRaises(PreflightError):
            publish_archive(self.attempt, small, archive_id="toobig")
        self.assertFalse(list((self.base / "store-small").glob("*/objects")))

    def test_missing_credentials_are_refused(self) -> None:
        previous = os.environ.pop(CREDENTIAL_ENV, None)
        self.addCleanup(lambda: os.environ.__setitem__(CREDENTIAL_ENV, previous or ""))
        with self.assertRaises(CredentialError):
            publish_archive(self.attempt, backend(self.base / "store-nocred"), archive_id="nocred")

    def test_no_credential_value_appears_in_the_locator_or_manifest(self) -> None:
        result, _ = self._published("secrets")
        blob = json.dumps(result.as_dict()) + self.store.read_manifest(result.locator).decode("utf-8")
        self.assertNotIn("prototype-token-not-a-secret", blob)
        self.assertNotIn(CREDENTIAL_ENV, blob)

    # -- unsafe paths --------------------------------------------------------

    def test_symlinked_member_is_refused(self) -> None:
        root = build_attempt(self.base / "attempt-symlink")
        (root / "records" / "link.jsonl.gz").symlink_to(root / "records" / "shard-0000.jsonl.gz")
        with self.assertRaises(UnsafePathError):
            enumerate_members(root)

    def test_symlinked_directory_is_refused(self) -> None:
        root = build_attempt(self.base / "attempt-symdir")
        (root / "mirror").symlink_to(root / "records", target_is_directory=True)
        with self.assertRaises(UnsafePathError):
            enumerate_members(root)

    def test_manifest_traversal_path_is_refused_on_retrieval(self) -> None:
        """A *self-consistent* manifest carrying a traversal path.

        Re-encoded canonically and with the locator digest updated to match, so
        every integrity check passes and the only thing standing between the
        archive and a write outside the destination is the path guard.
        """

        import hashlib

        from research.aaa134.archive_prototype.contract import ArchiveIdentity

        result, _ = self._published("traversal")
        path = self._manifest_path("traversal")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        rows = [dict(row) for row in manifest["objects"]]
        rows[0]["relative_path"] = "../escaped.bin"
        raw = canonical_manifest(
            manifest["archive_id"],
            ArchiveIdentity(**manifest["identity"]),
            rows,
            durable=False,
            label=manifest["label"],
        )
        path.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        relocated = type(result.locator)(
            **{**result.locator.as_dict(), "manifest_sha256": digest, "manifest_version_id": digest}
        )
        destination = self.base / "traversal-fresh"
        with self.assertRaises(UnsafePathError):
            retrieve_archive(self.store, relocated, destination)
        self.assertFalse((self.base / "escaped.bin").exists())

    def test_absolute_member_path_is_refused(self) -> None:
        from research.aaa134.archive_prototype.enumeration import safe_relative_path

        with self.assertRaises(UnsafePathError):
            safe_relative_path(self.attempt, Path("/etc/passwd"))

    def test_relative_traversal_is_refused(self) -> None:
        from research.aaa134.archive_prototype.enumeration import safe_relative_path

        with self.assertRaises(UnsafePathError):
            safe_relative_path(self.attempt, Path("records/../../escape.bin"))

    def test_unsafe_archive_id_is_refused(self) -> None:
        with self.assertRaises(LocatorError):
            self.store._archive_dir("../escape")

    def test_retrieval_refuses_a_non_empty_destination(self) -> None:
        result, _ = self._published("nonempty")
        occupied = self.base / "occupied"
        occupied.mkdir()
        (occupied / "leftover.bin").write_bytes(b"from the original attempt directory")
        with self.assertRaises(UnsafePathError):
            retrieve_archive(self.store, result.locator, occupied)

    def test_empty_archive_root_is_refused(self) -> None:
        empty = self.base / "empty-attempt"
        empty.mkdir()
        with self.assertRaises(UnsafePathError):
            enumerate_members(empty)


class StreamingBoundsTest(unittest.TestCase):
    """The prototype must never need the whole archive resident or duplicated."""

    def setUp(self) -> None:
        set_credential(self)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)

    def test_upload_reads_in_bounded_chunks(self) -> None:
        from research.aaa134.archive_prototype.contract import CHUNK_BYTES
        from research.aaa134.archive_prototype.enumeration import stream_file

        root = build_attempt(self.base / "attempt")
        big = root / "records" / "shard-0000.jsonl.gz"
        big.write_bytes(b"x" * (CHUNK_BYTES * 3 + 17))
        sizes = [len(chunk) for chunk in stream_file(big)]
        self.assertEqual(sizes, [CHUNK_BYTES, CHUNK_BYTES, CHUNK_BYTES, 17])
        self.assertLessEqual(max(sizes), CHUNK_BYTES)

    def test_no_second_local_copy_is_made_under_the_attempt_root(self) -> None:
        root = build_attempt(self.base / "attempt2")
        before = set(root.rglob("*"))
        store = LocalImmutableBackend(self.base / "store2")
        publish_archive(root, store, archive_id="nocopy")
        self.assertEqual(before, set(root.rglob("*")))


if __name__ == "__main__":
    unittest.main()
