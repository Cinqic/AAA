"""A local write-once backend, for contract testing only.

**PROTOTYPE_ONLY_NOT_DURABLE.** A directory on the same disk as the run is not
durable storage and never will be. This backend exists to answer one narrow
question: does the contract in ``contract.py`` actually catch the failures it
claims to catch? It emulates the semantics a real object store provides --
immutable versions, refusal to overwrite, atomic finalization, credentials
supplied out of band -- so the fault-injection tests exercise the same code
paths a real adapter would.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from collections.abc import Iterator
from pathlib import Path

from .contract import (
    CHUNK_BYTES,
    ArchiveBackend,
    ArchiveIdentity,
    ArchiveMember,
    CredentialError,
    DuplicateAttemptError,
    DurableLocator,
    FinalizationError,
    ImmutabilityError,
    IntegrityError,
    LocatorError,
    PreflightError,
    PreflightReport,
    StoredObject,
)
from .enumeration import MANIFEST_NAME, canonical_manifest

CREDENTIAL_ENV = "AAA134_PROTOTYPE_ARCHIVE_TOKEN"
LABEL = "PROTOTYPE_ONLY_NOT_DURABLE"


def _encode_key(relative_path: str) -> str:
    """Map a member path onto a flat, collision-free object key.

    Hashing the path rather than reusing it means a hostile member name cannot
    influence where the backend writes, which keeps the store's own layout
    independent of anything inside the archive.
    """

    return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()


class LocalImmutableBackend(ArchiveBackend):
    """Write-once object store backed by a directory.

    Every stored object gets a random ``version_id``; reads require the caller
    to name the version, so a locator that drifts from the bytes it was issued
    against is detectable rather than silently satisfied.
    """

    scheme = "aaa134-prototype-local"
    durable = False

    def __init__(
        self, root: str | Path, *, capacity_bytes: int | None = None, require_credential: bool = True
    ) -> None:
        self.root = Path(root).resolve()
        self.capacity_bytes = capacity_bytes
        self.require_credential = require_credential
        self.root.mkdir(parents=True, exist_ok=True)
        self.interrupt_after_objects: int | None = None
        self._objects_written = 0

    # -- credentials ---------------------------------------------------------

    def _check_credential(self) -> None:
        """Credentials live in the environment, never in the repository.

        The prototype deliberately reads a token it does not use for anything:
        the point is that a missing credential must fail loudly *before* any
        byte moves, and that the value must never be logged or serialized.
        """

        if not self.require_credential:
            return
        token = os.environ.get(CREDENTIAL_ENV)
        if not token:
            raise CredentialError(
                f"no archive credential: set {CREDENTIAL_ENV} in the environment, never in the repository"
            )

    # -- contract ------------------------------------------------------------

    def preflight(self, members: list[ArchiveMember], identity: ArchiveIdentity) -> PreflightReport:
        self._check_credential()
        total = sum(member.size_bytes for member in members)
        largest = max((member.size_bytes for member in members), default=0)
        problems: list[str] = []
        if self.capacity_bytes is not None and total > self.capacity_bytes:
            problems.append(f"archive needs {total} bytes; destination reports {self.capacity_bytes}")
        if identity.attempt_id == "UNKNOWN":
            problems.append("archive identity has no attempt_id")
        if identity.scientific_fingerprint_sha256 == "UNKNOWN":
            problems.append("archive identity has no scientific fingerprint")
        if (self.root / f"{identity.attempt_id}.locator.json").exists():
            problems.append(f"attempt {identity.attempt_id} has already been archived at this destination")
        return PreflightReport(
            object_count=len(members),
            total_bytes=total,
            largest_object_bytes=largest,
            capacity_bytes=self.capacity_bytes,
            capacity_known=self.capacity_bytes is not None,
            supports_multipart=False,
            supports_versioning=True,
            supports_object_checksums=True,
            problems=problems,
        )

    def _archive_dir(self, archive_id: str) -> Path:
        if "/" in archive_id or "\\" in archive_id or archive_id in {"", ".", ".."}:
            raise LocatorError(f"unsafe archive id: {archive_id!r}")
        return self.root / archive_id

    def put_object(self, archive_id: str, member: ArchiveMember, chunks: Iterator[bytes]) -> StoredObject:
        self._check_credential()
        directory = self._archive_dir(archive_id)
        if (directory / MANIFEST_NAME).exists():
            raise ImmutabilityError(f"archive {archive_id} is finalized and cannot accept more objects")
        key = _encode_key(member.relative_path)
        version_id = secrets.token_hex(16)
        target = directory / "objects" / key / version_id
        if (directory / "objects" / key).exists():
            raise ImmutabilityError(f"object {member.relative_path} already exists and is write-once")
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        # Write to a temporary name and rename, so an interrupted upload never
        # leaves a readable partial object behind.
        staging = target.with_suffix(".partial")
        with staging.open("wb") as handle:
            for chunk in chunks:
                handle.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if size != member.size_bytes or digest.hexdigest() != member.sha256:
            staging.unlink(missing_ok=True)
            raise IntegrityError(
                f"upload of {member.relative_path} did not match the declared identity "
                f"(size {size} vs {member.size_bytes})"
            )
        staging.rename(target)
        self._objects_written += 1
        if self.interrupt_after_objects is not None and self._objects_written >= self.interrupt_after_objects:
            raise ConnectionError("simulated upload interruption")
        return StoredObject(
            relative_path=member.relative_path,
            object_key=key,
            version_id=version_id,
            size_bytes=size,
            backend_sha256=digest.hexdigest(),
        )

    def finalize(
        self, archive_id: str, identity: ArchiveIdentity, stored: list[StoredObject]
    ) -> DurableLocator:
        from .contract import DurableLocator

        self._check_credential()
        directory = self._archive_dir(archive_id)
        manifest_path = directory / MANIFEST_NAME
        if manifest_path.exists():
            raise FinalizationError(f"archive {archive_id} is already finalized")
        if not stored:
            raise FinalizationError("refusing to finalize an archive with no objects")
        present = {path.parent.name for path in (directory / "objects").glob("*/*") if path.is_file()}
        declared = {row.object_key for row in stored}
        if present != declared:
            raise FinalizationError(
                f"stored set is incomplete: destination holds {len(present)} objects, caller declared {len(declared)}"
            )
        payload = canonical_manifest(
            archive_id,
            identity,
            [
                {
                    "relative_path": row.relative_path,
                    "object_key": row.object_key,
                    "version_id": row.version_id,
                    "size_bytes": row.size_bytes,
                    "sha256": row.backend_sha256,
                }
                for row in stored
            ],
            durable=self.durable,
            label=LABEL,
        )
        staging = manifest_path.with_suffix(".partial")
        staging.write_bytes(payload)
        staging.rename(manifest_path)
        locator = DurableLocator(
            scheme=self.scheme,
            root=str(self.root),
            archive_id=archive_id,
            manifest_object_key=MANIFEST_NAME,
            manifest_version_id=hashlib.sha256(payload).hexdigest(),
            manifest_sha256=hashlib.sha256(payload).hexdigest(),
            object_count=len(stored),
            total_bytes=sum(row.size_bytes for row in stored),
            durable=self.durable,
            label=LABEL,
        )
        (self.root / f"{identity.attempt_id}.locator.json").write_text(
            json.dumps(locator.as_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        return locator

    def open_object(self, locator: DurableLocator, relative_path: str, version_id: str) -> Iterator[bytes]:

        directory = self._archive_dir(locator.archive_id)
        path = directory / "objects" / _encode_key(relative_path) / version_id
        if not path.is_file():
            raise LocatorError(f"no object {relative_path!r} at version {version_id!r}")
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(CHUNK_BYTES)
                if not chunk:
                    return
                yield chunk

    def read_manifest(self, locator: DurableLocator) -> bytes:
        path = self._archive_dir(locator.archive_id) / MANIFEST_NAME
        if not path.is_file():
            raise LocatorError(f"archive {locator.archive_id} has no finalized manifest")
        return path.read_bytes()

    def list_objects(self, locator: DurableLocator) -> list[StoredObject]:
        manifest = json.loads(self.read_manifest(locator))
        return [
            StoredObject(
                relative_path=row["relative_path"],
                object_key=row["object_key"],
                version_id=row["version_id"],
                size_bytes=row["size_bytes"],
                backend_sha256=row["sha256"],
            )
            for row in manifest["objects"]
        ]


def duplicate_guard(backend: LocalImmutableBackend, identity: ArchiveIdentity) -> None:
    """Refuse a second upload of the same attempt identity."""

    if (backend.root / f"{identity.attempt_id}.locator.json").exists():
        raise DuplicateAttemptError(f"attempt {identity.attempt_id} is already archived at this destination")


def preflight_or_raise(report: PreflightReport) -> PreflightReport:
    if not report.ok:
        raise PreflightError("; ".join(report.problems))
    return report
