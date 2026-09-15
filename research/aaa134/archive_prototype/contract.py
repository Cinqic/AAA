"""The backend contract a real durable archive destination must satisfy.

Nothing here talks to a network. The contract is deliberately written against
capabilities that a mainstream versioned object store actually exposes --
multipart upload, per-object strong checksums, immutable object versions --
so that an adapter is a thin translation rather than a redesign.

Two rules shape every signature below.

1. **Nothing may require the whole archive in memory, or a second local copy.**
   Objects are handed to the backend as a path plus a streaming reader, and the
   backend is expected to consume them chunk by chunk.
2. **No producer-side summary is ever trusted as evidence.** The backend
   reports the digest *it* computed from the bytes it stored; verification
   recomputes from the bytes it retrieved. A manifest is a claim to be checked,
   never a result.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

CHUNK_BYTES = 1 << 20


class ArchiveError(RuntimeError):
    """Base class for every prototype archive failure."""


class UnsafePathError(ArchiveError):
    """An archive member path is not a safe relative file path."""


class PreflightError(ArchiveError):
    """The archive or the destination failed a pre-upload check."""


class CredentialError(ArchiveError):
    """The backend has no usable credentials."""


class DuplicateAttemptError(ArchiveError):
    """An attempt identity has already been uploaded to this destination."""


class FinalizationError(ArchiveError):
    """Finalization was refused, repeated, or attempted out of order."""


class ImmutabilityError(ArchiveError):
    """A write was attempted against an already-finalized or existing object."""


class LocatorError(ArchiveError):
    """A durable locator is malformed, unknown, or points at the wrong thing."""


class IntegrityError(ArchiveError):
    """Retrieved bytes do not match the recorded identity."""


@dataclass(frozen=True)
class ArchiveIdentity:
    """Who produced this archive, and under which frozen scientific inputs.

    Every field is copied from evidence the run itself wrote. A mismatch on
    retrieval is an integrity failure, not a cosmetic difference: it means the
    bytes in durable storage were not produced by the thing the locator claims.
    """

    attempt_id: str
    batch_id: str
    role: str
    protocol_version: str
    protocol_hash: str
    scientific_fingerprint_sha256: str
    candidate_id: str
    source_commit: str
    invocation: str
    outcome: str

    def as_dict(self) -> dict[str, str]:
        return {
            "attempt_id": self.attempt_id,
            "batch_id": self.batch_id,
            "role": self.role,
            "protocol_version": self.protocol_version,
            "protocol_hash": self.protocol_hash,
            "scientific_fingerprint_sha256": self.scientific_fingerprint_sha256,
            "candidate_id": self.candidate_id,
            "source_commit": self.source_commit,
            "invocation": self.invocation,
            "outcome": self.outcome,
        }


@dataclass(frozen=True)
class ArchiveMember:
    """One file to be stored, addressed by its path relative to the attempt root."""

    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class StoredObject:
    """What the backend reports after it has durably stored one member.

    ``version_id`` is the backend's own immutable version identity. A backend
    that cannot supply one cannot satisfy AAA-134, because without it a locator
    names a mutable key rather than fixed bytes.
    """

    relative_path: str
    object_key: str
    version_id: str
    size_bytes: int
    backend_sha256: str


@dataclass(frozen=True)
class DurableLocator:
    """The finalized, immutable address of a complete archive.

    ``durable`` is False for every backend in this prototype. A consumer that
    treats a non-durable locator as evidence is doing the exact thing AAA-134
    exists to prevent, so the flag is part of the locator rather than a
    docstring.
    """

    scheme: str
    root: str
    archive_id: str
    manifest_object_key: str
    manifest_version_id: str
    manifest_sha256: str
    object_count: int
    total_bytes: int
    durable: bool
    label: str

    def as_dict(self) -> dict[str, object]:
        return {
            "scheme": self.scheme,
            "root": self.root,
            "archive_id": self.archive_id,
            "manifest_object_key": self.manifest_object_key,
            "manifest_version_id": self.manifest_version_id,
            "manifest_sha256": self.manifest_sha256,
            "object_count": self.object_count,
            "total_bytes": self.total_bytes,
            "durable": self.durable,
            "label": self.label,
        }


@dataclass
class PreflightReport:
    """What a destination says about an archive before a byte is uploaded."""

    object_count: int
    total_bytes: int
    largest_object_bytes: int
    capacity_bytes: int | None
    capacity_known: bool
    supports_multipart: bool
    supports_versioning: bool
    supports_object_checksums: bool
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


@runtime_checkable
class ArchiveBackend(Protocol):
    """The contract every durable destination adapter must implement.

    An S3-compatible adapter maps this almost one to one: ``put_object`` onto a
    multipart upload with ``ChecksumSHA256``, ``version_id`` onto the object
    version, ``finalize`` onto writing the manifest object last and reading its
    version back.
    """

    scheme: str
    durable: bool

    def preflight(self, members: list[ArchiveMember], identity: ArchiveIdentity) -> PreflightReport:
        """Validate the archive and the destination before uploading anything."""

    def put_object(
        self,
        archive_id: str,
        member: ArchiveMember,
        chunks: Iterator[bytes],
    ) -> StoredObject:
        """Stream one member into the destination and report what was stored.

        Must refuse to overwrite an existing object for the same key, and must
        report the digest computed from the bytes it actually received.
        """

    def finalize(
        self,
        archive_id: str,
        identity: ArchiveIdentity,
        stored: list[StoredObject],
    ) -> DurableLocator:
        """Write the canonical manifest last and return the immutable locator.

        Must be refused if called twice, or if the stored set is incomplete.
        """

    def open_object(self, locator: DurableLocator, relative_path: str, version_id: str) -> Iterator[bytes]:
        """Stream one stored object back by path *and* explicit version."""

    def read_manifest(self, locator: DurableLocator) -> bytes:
        """Return the canonical manifest bytes named by the locator."""

    def list_objects(self, locator: DurableLocator) -> list[StoredObject]:
        """Enumerate what the destination actually holds for this archive."""
