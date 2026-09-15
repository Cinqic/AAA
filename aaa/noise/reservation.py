"""Repository-wide, append-only confirmation batch reservations."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any


class ReservationError(RuntimeError):
    """Raised when a formal batch cannot be reserved without reuse."""


def _git(root: Path, *args: str, input_text: str | None = None, check: bool = True) -> str:
    process = subprocess.run(
        ["git", "-C", str(root), *args],
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and process.returncode != 0:
        raise ReservationError(process.stderr.strip() or f"git {' '.join(args)} failed")
    return process.stdout.strip()


def _claim_ref(batch_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", batch_id):
        raise ReservationError("batch ID is unsafe for a reservation ref")
    return f"refs/heads/aaa-confirmation-claims/{batch_id}"


def _read_remote_claim(root: Path, remote: str, ref: str) -> dict[str, Any] | None:
    listing = _git(root, "ls-remote", "--heads", remote, ref, check=False)
    if not listing:
        return None
    oid = listing.split()[0]
    _git(root, "fetch", "--no-tags", remote, f"{ref}:refs/aaa-confirmation-claims/{oid}")
    message = _git(root, "show", "-s", "--format=%B", f"refs/aaa-confirmation-claims/{oid}")
    try:
        payload = json.loads(message)
    except json.JSONDecodeError as exc:
        raise ReservationError(f"remote reservation {ref} has malformed ownership metadata") from exc
    if not isinstance(payload, dict):
        raise ReservationError(f"remote reservation {ref} ownership metadata is not an object")
    return payload


def reserve_confirmation_batch(
    root: str | Path,
    *,
    batch_id: str,
    role: str,
    attempt_id: str,
    scientific_fingerprint_sha256: str,
    resume: bool,
    remote: str = "origin",
) -> dict[str, Any]:
    """Atomically create an immutable remote ref, or resume its exact owner.

    A normal Git ref creation is atomic at the shared remote. No force update or
    deletion is used, so separate processes, output roots, worktrees, and clones
    cannot all create ownership for the same batch.
    """

    repository = Path(root).resolve()
    ref = _claim_ref(batch_id)
    head = _git(repository, "rev-parse", "HEAD")
    payload = {
        "schema_version": "aaa.observation_noise_batch_reservation.v1",
        "batch_id": batch_id,
        "role": role,
        "attempt_id": attempt_id,
        "source_commit": head,
        "scientific_fingerprint_sha256": scientific_fingerprint_sha256,
        "claim_ref": ref,
    }
    existing = _read_remote_claim(repository, remote, ref)
    if existing is not None:
        if resume and existing == payload:
            return existing
        owner = existing.get("attempt_id", "unknown")
        raise ReservationError(f"batch {batch_id!r} is already reserved by attempt {owner!r}")
    tree = _git(repository, "rev-parse", "HEAD^{tree}")
    commit = _git(
        repository,
        "commit-tree",
        tree,
        "-p",
        head,
        input_text=json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
    )
    pushed = subprocess.run(
        ["git", "-C", str(repository), "push", remote, f"{commit}:{ref}"],
        text=True,
        capture_output=True,
        check=False,
    )
    if pushed.returncode != 0:
        winner = _read_remote_claim(repository, remote, ref)
        owner = None if winner is None else winner.get("attempt_id")
        raise ReservationError(
            f"batch {batch_id!r} reservation lost an atomic race"
            + ("" if owner is None else f" to attempt {owner!r}")
        )
    return payload
