"""Publish, retrieve and independently verify an archive through the contract.

The verification half is deliberately written as if it distrusts the publish
half. It re-reads the manifest from the destination, recomputes every digest
from retrieved bytes, and checks identity fields against what the caller
expected. Nothing is accepted because the producer said so.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contract import (
    CHUNK_BYTES,
    ArchiveBackend,
    ArchiveIdentity,
    DurableLocator,
    FinalizationError,
    IntegrityError,
    LocatorError,
    StoredObject,
    UnsafePathError,
)
from .enumeration import (
    MANIFEST_SCHEMA,
    canonical_manifest,
    cross_check_against_aaa_checksums,
    enumerate_members,
    read_identity,
    safe_relative_path,
    stream_file,
)
from .local_backend import duplicate_guard, preflight_or_raise


@dataclass
class PublishResult:
    locator: DurableLocator
    stored: list[StoredObject]
    preflight: dict[str, Any]
    producer_checksum_cross_check: dict[str, Any]
    resumed_objects: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "locator": self.locator.as_dict(),
            "object_count": len(self.stored),
            "total_bytes": sum(row.size_bytes for row in self.stored),
            "preflight": self.preflight,
            "producer_checksum_cross_check": self.producer_checksum_cross_check,
            "resumed_objects": self.resumed_objects,
        }


@dataclass
class VerificationResult:
    verdict: str
    objects_verified: int
    bytes_verified: int
    problems: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "objects_verified": self.objects_verified,
            "bytes_verified": self.bytes_verified,
            "problems": self.problems,
        }


def publish_archive(
    attempt_root: str | Path,
    backend: ArchiveBackend,
    *,
    archive_id: str,
    identity: ArchiveIdentity | None = None,
    already_stored: list[StoredObject] | None = None,
) -> PublishResult:
    """Enumerate, preflight, stream-upload and finalize one attempt directory.

    ``already_stored`` resumes an interrupted upload: those members are skipped
    and their recorded identities are carried into finalization. Resume is by
    path *and* digest -- a member whose bytes changed since the interrupted
    attempt is re-uploaded rather than trusted.
    """

    root = Path(attempt_root).resolve()
    members = enumerate_members(root)
    resolved_identity = identity or read_identity(root)
    cross_check = cross_check_against_aaa_checksums(root, members)
    if hasattr(backend, "root"):
        duplicate_guard(backend, resolved_identity)  # type: ignore[arg-type]
    report = preflight_or_raise(backend.preflight(members, resolved_identity))
    carried = {row.relative_path: row for row in (already_stored or [])}
    stored: list[StoredObject] = []
    resumed = 0
    for member in members:
        previous = carried.get(member.relative_path)
        if previous is not None and previous.backend_sha256 == member.sha256:
            stored.append(previous)
            resumed += 1
            continue
        stored.append(backend.put_object(archive_id, member, stream_file(root / member.relative_path)))
    locator = backend.finalize(archive_id, resolved_identity, stored)
    return PublishResult(
        locator=locator,
        stored=stored,
        preflight={
            "object_count": report.object_count,
            "total_bytes": report.total_bytes,
            "largest_object_bytes": report.largest_object_bytes,
            "capacity_known": report.capacity_known,
            "supports_versioning": report.supports_versioning,
            "supports_object_checksums": report.supports_object_checksums,
        },
        producer_checksum_cross_check=cross_check,
        resumed_objects=resumed,
    )


def _load_manifest(backend: ArchiveBackend, locator: DurableLocator) -> dict[str, Any]:
    raw = backend.read_manifest(locator)
    if hashlib.sha256(raw).hexdigest() != locator.manifest_sha256:
        raise IntegrityError("manifest bytes do not match the digest recorded in the locator")
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise IntegrityError("manifest is malformed or has an unknown schema")
    if manifest.get("archive_id") != locator.archive_id:
        raise LocatorError("manifest archive id disagrees with the locator")
    objects = manifest.get("objects")
    if not isinstance(objects, list) or not objects:
        raise IntegrityError("manifest contains no objects")
    if len(objects) != locator.object_count:
        raise IntegrityError(
            f"manifest lists {len(objects)} objects; the locator claims {locator.object_count}"
        )
    # Re-derive the manifest bytes from their own content: a manifest whose
    # recorded digest was edited to match tampered rows still fails here,
    # because the canonical encoding is a pure function of the rows.
    recomputed = canonical_manifest(
        manifest["archive_id"],
        ArchiveIdentity(**manifest["identity"]),
        [dict(row) for row in objects],
        durable=bool(manifest["durable"]),
        label=str(manifest["label"]),
    )
    if hashlib.sha256(recomputed).hexdigest() != locator.manifest_sha256:
        raise IntegrityError("manifest does not re-encode to its recorded digest")
    return manifest


def retrieve_archive(
    backend: ArchiveBackend,
    locator: DurableLocator,
    destination: str | Path,
) -> int:
    """Retrieve a complete archive into a fresh, empty directory.

    Refuses a destination that already holds anything, so a retrieval can never
    silently inherit bytes from the original local attempt directory. Every
    member path is re-validated on the way *in*, because the manifest is
    untrusted input at this point.
    """

    target = Path(destination).resolve()
    if target.exists() and any(target.iterdir()):
        raise UnsafePathError(f"retrieval destination is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(backend, locator)
    written = 0
    for row in manifest["objects"]:
        relative = str(row["relative_path"])
        candidate = Path(relative)
        if candidate.is_absolute() or any(part in {"..", "."} for part in candidate.parts):
            raise UnsafePathError(f"manifest member path is unsafe: {relative}")
        path = (target / candidate).resolve()
        if target not in path.parents:
            raise UnsafePathError(f"manifest member path escapes the destination: {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            for chunk in backend.open_object(locator, relative, str(row["version_id"])):
                handle.write(chunk)
        written += 1
    return written


def verify_retrieval(
    backend: ArchiveBackend,
    locator: DurableLocator,
    retrieved_root: str | Path,
    *,
    expected_identity: ArchiveIdentity | None = None,
) -> VerificationResult:
    """Recompute every digest from the retrieved bytes and report problems.

    Returns a result rather than raising, so a caller gets the *complete* list
    of what is wrong with an archive instead of only the first thing.
    """

    root = Path(retrieved_root).resolve()
    problems: list[str] = []
    try:
        manifest = _load_manifest(backend, locator)
    except (IntegrityError, LocatorError) as exc:
        return VerificationResult(verdict="FAIL", objects_verified=0, bytes_verified=0, problems=[str(exc)])

    if expected_identity is not None:
        for key, expected in expected_identity.as_dict().items():
            actual = manifest["identity"].get(key)
            if actual != expected:
                problems.append(
                    f"identity mismatch for {key}: expected {expected!r}, manifest has {actual!r}"
                )

    if manifest.get("durable") is not False:
        problems.append("manifest claims durability; this prototype has no durable backend")

    verified = 0
    total = 0
    declared_paths: set[str] = set()
    for row in manifest["objects"]:
        relative = str(row["relative_path"])
        declared_paths.add(relative)
        path = root / relative
        if not path.is_file():
            problems.append(f"missing after retrieval: {relative}")
            continue
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
        if size != int(row["size_bytes"]):
            problems.append(f"truncated or extended: {relative} ({size} bytes, expected {row['size_bytes']})")
            continue
        if digest.hexdigest() != str(row["sha256"]):
            problems.append(f"checksum mismatch: {relative}")
            continue
        verified += 1
        total += size

    actual_paths: set[str] = set()
    for path in sorted(root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        try:
            actual_paths.add(safe_relative_path(root, path))
        except UnsafePathError as exc:
            problems.append(str(exc))
    for extra in sorted(actual_paths - declared_paths):
        problems.append(f"unexpected file not covered by the manifest: {extra}")

    verdict = "PASS" if not problems and verified == len(manifest["objects"]) else "FAIL"
    return VerificationResult(
        verdict=verdict, objects_verified=verified, bytes_verified=total, problems=problems
    )


def refuse_finalized_overwrite(backend: ArchiveBackend, archive_id: str, identity: ArchiveIdentity) -> None:
    """Assert that a finalized archive refuses a second finalization."""

    try:
        backend.finalize(archive_id, identity, [])
    except FinalizationError:
        return
    raise AssertionError("finalized archive accepted a second finalization")
