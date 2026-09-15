"""Non-self-referential scientific identity for observation-noise confirmation.

The observation-noise confirmation freeze is generated provenance, not a
scientific input. It is therefore excluded from this fingerprint so committing
the freeze cannot change the identity it records. The design/source-freeze
manifest is also an explicitly generated provenance artifact. Results and run
archives are excluded as generated evidence. Every other tracked scientific
file is covered, and every nonignored untracked file fails closed.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

NOISE_CONFIRMATION_FREEZE = "benchmarks/observation_noise_freeze.json"
NOISE_SOURCE_FREEZE = "benchmarks/observation_noise_source_freeze.json"
BATCH_REGISTRY = "benchmarks/observation_noise_registry.json"
GENERATED_PREFIXES = frozenset({"results/", "runs/", "build/", "dist/"})
GENERATED_FILES = frozenset(
    {
        NOISE_CONFIRMATION_FREEZE,
        NOISE_SOURCE_FREEZE,
        "docs/final_audit.md",
        "docs/handoff_sol.md",
        "docs/sol_review.md",
        "docs/evidence/observation_noise_development_selection.json",
    }
)
MUTABLE_BATCH_FIELDS = frozenset({"status", "consumed_by", "outcome", "claimed_by", "claim_started_at"})
REQUIRED_SCIENTIFIC_PATHS = frozenset(
    {
        ".github/workflows/ci.yml",
        "pyproject.toml",
        "requirements-lock.txt",
        "aaa/benchmark/source_identity.py",
        "aaa/noise/data/observation_noise_v1.json",
        "aaa/noise/freeze.py",
        "aaa/noise/runner.py",
        "aaa/noise/statistics.py",
        "aaa/noise/verifier.py",
        "benchmarks/observation_noise_candidate_ledger.json",
        "benchmarks/observation_noise_registry.json",
        "docs/observation_noise_protocol.md",
        "docs/reproduction.md",
        "tests/test_observation_noise.py",
    }
)


class ScientificIdentityError(ValueError):
    """Raised when the scientific source cannot be identified safely."""


def _git_paths(root: Path, *args: str) -> list[str]:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "-z", *args], stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as exc:
        raise ScientificIdentityError("scientific source requires a readable Git repository") from exc
    return sorted(path for path in output.decode("utf-8").split("\0") if path)


def _git_toplevel(root: Path) -> Path:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"], stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as exc:
        raise ScientificIdentityError("scientific source requires a readable Git repository") from exc
    return Path(output.decode("utf-8").strip()).resolve()


def _is_generated(name: str) -> bool:
    return name in GENERATED_FILES or any(name.startswith(prefix) for prefix in GENERATED_PREFIXES)


def _normalized_content(name: str, path: Path) -> bytes:
    content = path.read_bytes()
    if name != BATCH_REGISTRY:
        return content
    try:
        registry = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ScientificIdentityError(f"batch registry is not valid JSON: {exc}") from exc
    if not isinstance(registry, dict) or not isinstance(registry.get("batches"), list):
        raise ScientificIdentityError("batch registry must contain a batches list")
    normalized = dict(registry)
    normalized_batches: list[dict[str, Any]] = []
    for index, batch in enumerate(registry["batches"]):
        if not isinstance(batch, dict):
            raise ScientificIdentityError(f"batch registry entry {index} is not an object")
        normalized_batches.append(
            {key: value for key, value in batch.items() if key not in MUTABLE_BATCH_FIELDS}
        )
    normalized["batches"] = normalized_batches
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def scientific_fingerprint(project_root: str | Path, *, require_project_shape: bool = True) -> dict[str, Any]:
    """Hash scientific files and fail closed on unsafe or untracked inputs.

    The returned file map includes normalized batch-registry content and
    executable-bit metadata, making the result independently inspectable.
    ``require_project_shape=False`` is intended only for isolated unit-test
    repositories.
    """

    root = Path(project_root).resolve()
    if _git_toplevel(root) != root:
        raise ScientificIdentityError("scientific source must be a Git repository root")
    tracked = set(_git_paths(root, "--cached"))
    untracked = set(_git_paths(root, "--others", "--exclude-standard"))
    names = sorted(tracked | untracked)
    if require_project_shape:
        missing = sorted(REQUIRED_SCIENTIFIC_PATHS - tracked)
        if missing:
            raise ScientificIdentityError(f"scientific source is missing required files: {missing}")
    files: dict[str, dict[str, Any]] = {}
    for name in names:
        path = root / name
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ScientificIdentityError(f"scientific path escapes repository root: {name}") from exc
        if path.is_symlink():
            raise ScientificIdentityError(f"scientific source contains a symlink: {name}")
        if name in untracked and not _is_generated(name):
            raise ScientificIdentityError(f"untracked scientific file is not permitted: {name}")
        if _is_generated(name):
            continue
        if not path.is_file():
            raise ScientificIdentityError(f"scientific source is missing or not a regular file: {name}")
        content = _normalized_content(name, path)
        files[name] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "executable": bool(path.stat().st_mode & 0o111),
        }
    if not files:
        raise ScientificIdentityError("scientific source file set is empty")
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {
        "schema": "aaa.observation_noise_scientific_source.v1",
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "files": files,
        "excluded_generated_prefixes": sorted(GENERATED_PREFIXES),
        "excluded_generated_files": sorted(GENERATED_FILES),
        "normalized_registry_fields": sorted(MUTABLE_BATCH_FIELDS),
    }


def fingerprints_equal(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    """Compare the complete scientific file map, not only a cached digest."""

    return (
        expected.get("schema") == actual.get("schema") == "aaa.observation_noise_scientific_source.v1"
        and expected.get("sha256") == actual.get("sha256")
        and expected.get("files") == actual.get("files")
    )
