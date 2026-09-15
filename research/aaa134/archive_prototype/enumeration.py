"""Safe, streaming enumeration of an AAA attempt directory.

Everything here reads in ``CHUNK_BYTES`` blocks. No function loads a whole
shard, let alone a whole archive, and no function makes a second local copy.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .contract import (
    CHUNK_BYTES,
    ArchiveIdentity,
    ArchiveMember,
    IntegrityError,
    UnsafePathError,
)

MANIFEST_NAME = "aaa134_archive_manifest.json"
MANIFEST_SCHEMA = "aaa.research.aaa134.archive_manifest.v1"
AAA_CHECKSUM_NAME = "checksums.json"


def safe_relative_path(root: Path, path: Path) -> str:
    """Return a POSIX relative path, or refuse.

    Refuses absolute paths, ``..`` traversal, anything resolving outside the
    root, symlinks at any component, and anything that is not a regular file.
    Checked component by component so a symlinked *directory* is caught too,
    not just a symlinked leaf.
    """

    if path.is_absolute():
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise UnsafePathError(f"absolute path outside the archive root: {path}") from exc
    else:
        relative = path
    parts = relative.parts
    if not parts:
        raise UnsafePathError("empty archive member path")
    if any(part in {"..", "."} for part in parts):
        raise UnsafePathError(f"archive member path contains traversal: {relative}")
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise UnsafePathError(f"archive member path crosses a symlink: {relative}")
    resolved = (root / relative).resolve()
    if root.resolve() not in resolved.parents:
        raise UnsafePathError(f"archive member path escapes the archive root: {relative}")
    if not resolved.is_file():
        raise UnsafePathError(f"archive member is not a regular file: {relative}")
    return relative.as_posix()


def stream_file(path: Path) -> Iterator[bytes]:
    """Yield a file in fixed-size chunks. Never materializes the whole file."""

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_BYTES)
            if not chunk:
                return
            yield chunk


def digest_file(path: Path) -> tuple[str, int]:
    """Return ``(sha256_hex, size_bytes)`` computed by streaming."""

    digest = hashlib.sha256()
    size = 0
    for chunk in stream_file(path):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def enumerate_members(root: Path) -> list[ArchiveMember]:
    """Enumerate every regular file under ``root``, safely and deterministically.

    The prototype's own manifest is excluded so that re-running enumeration on
    a directory that already holds one is not self-referential.
    """

    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise UnsafePathError(f"archive root is not a real directory: {root}")
    members: list[ArchiveMember] = []
    for path in sorted(root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        relative = safe_relative_path(root, path)
        if relative == MANIFEST_NAME:
            continue
        sha256, size = digest_file(root / relative)
        members.append(ArchiveMember(relative_path=relative, size_bytes=size, sha256=sha256))
    if not members:
        raise UnsafePathError(f"archive root contains no files: {root}")
    return members


def cross_check_against_aaa_checksums(root: Path, members: list[ArchiveMember]) -> dict[str, Any]:
    """Compare our independent digests with the run's own ``checksums.json``.

    This is the point of the exercise. AAA's ``checksums.json`` is a
    *producer-side* summary. Agreeing with it is evidence; adopting it is not.
    If the file is absent the result says so rather than quietly passing.
    """

    checksum_path = root / AAA_CHECKSUM_NAME
    if not checksum_path.is_file():
        return {"status": "NOT_VERIFIED", "reason": f"{AAA_CHECKSUM_NAME} is absent"}
    try:
        rows = json.loads(checksum_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IntegrityError(f"{AAA_CHECKSUM_NAME} is not valid JSON: {exc}") from exc
    if not isinstance(rows, dict) or not rows:
        raise IntegrityError(f"{AAA_CHECKSUM_NAME} must be a nonempty object")
    ours = {
        member.relative_path: member.sha256 for member in members if member.relative_path != AAA_CHECKSUM_NAME
    }
    theirs = {str(key): str(value) for key, value in rows.items()}
    missing = sorted(set(ours) - set(theirs))
    unexpected = sorted(set(theirs) - set(ours))
    mismatched = sorted(key for key in set(ours) & set(theirs) if ours[key] != theirs[key])
    if missing or unexpected or mismatched:
        raise IntegrityError(
            "producer checksum manifest disagrees with independently computed digests: "
            f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}"
        )
    return {"status": "PASS", "entries": len(theirs)}


def read_identity(root: Path) -> ArchiveIdentity:
    """Build the archive identity from the run's own retained metadata.

    Absent fields are recorded as ``"UNKNOWN"`` rather than invented. A caller
    that needs a real identity should check for them; a caller that fabricates
    one is defeating the purpose.
    """

    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise IntegrityError("metadata.json is required to identify an archive")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise IntegrityError("metadata.json must be an object")

    def field(*names: str) -> str:
        for name in names:
            value = metadata.get(name)
            if isinstance(value, str) and value:
                return value
        return "UNKNOWN"

    return ArchiveIdentity(
        attempt_id=field("attempt_id"),
        batch_id=field("batch_id"),
        role=field("role"),
        protocol_version=field("protocol_version"),
        protocol_hash=field("protocol_hash"),
        scientific_fingerprint_sha256=field("scientific_fingerprint_sha256", "source_fingerprint_sha256"),
        candidate_id=field("selected_candidate", "candidate_id"),
        source_commit=field("source_commit"),
        invocation=field("invocation", "command"),
        outcome=field("outcome", "status"),
    )


def canonical_manifest(
    archive_id: str,
    identity: ArchiveIdentity,
    stored: list[dict[str, Any]],
    *,
    durable: bool,
    label: str,
) -> bytes:
    """Serialize the canonical record index deterministically.

    Sorted keys, no NaN, compact separators, one trailing newline: the bytes
    are a function of the content alone, so the manifest digest is stable
    across machines and reruns.
    """

    payload = {
        "schema_version": MANIFEST_SCHEMA,
        "archive_id": archive_id,
        "identity": identity.as_dict(),
        "durable": durable,
        "label": label,
        "objects": sorted(stored, key=lambda row: str(row["relative_path"])),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
