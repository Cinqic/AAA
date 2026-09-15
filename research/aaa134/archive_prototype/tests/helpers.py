"""Fixtures for the archive-prototype tests."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path

from research.aaa134.archive_prototype.contract import ArchiveIdentity
from research.aaa134.archive_prototype.local_backend import CREDENTIAL_ENV, LocalImmutableBackend

IDENTITY = ArchiveIdentity(
    attempt_id="prototype-attempt-0001",
    batch_id="prototype-batch-0001",
    role="development",
    protocol_version="aaa.observation_noise.v1.1",
    protocol_hash="546e2434cc15850779107c1a329af1e2f8581cd6a1c3b31807eecf5f6e7b427c",
    scientific_fingerprint_sha256="28a5e55e6866f0825e86cd49b69d88ab08dd28ce7562d23d3015acb8cbb5ac19",
    candidate_id="incumbent-no-refinement-v1",
    source_commit="58e440462ce2e8fb551c5a5975bef85d4a54fef3",
    invocation="research/aaa134 prototype fixture",
    outcome="PROTOTYPE_FIXTURE",
)


def set_credential(case) -> None:
    """Install a throwaway token for the test and remove it afterwards.

    The value is fixed and meaningless. A real adapter would read a real
    credential from the environment; neither ever belongs in the repository.
    """

    previous = os.environ.get(CREDENTIAL_ENV)
    os.environ[CREDENTIAL_ENV] = "prototype-token-not-a-secret"

    def restore() -> None:
        if previous is None:
            os.environ.pop(CREDENTIAL_ENV, None)
        else:
            os.environ[CREDENTIAL_ENV] = previous

    case.addCleanup(restore)


def build_attempt(root: Path, *, shards: int = 3, with_checksums: bool = True) -> Path:
    """Write a miniature attempt directory shaped like a real noise archive."""

    root = Path(root)
    (root / "records").mkdir(parents=True, exist_ok=True)
    (root / "schedules").mkdir(parents=True, exist_ok=True)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)

    index_rows = []
    for shard in range(shards):
        relative = f"records/shard-{shard:04d}.jsonl.gz"
        lines = [
            json.dumps({"trial_id": f"t-{shard:04d}", "step": step, "value": step * 0.5}, sort_keys=True)
            for step in range(4)
        ]
        payload = ("\n".join(lines) + "\n").encode("utf-8")
        (root / relative).write_bytes(gzip.compress(payload, mtime=0))
        index_rows.append({"path": relative, "records": len(lines)})

    (root / "records" / "index.json").write_text(
        json.dumps(
            {
                "schema_version": "aaa.observation_noise_record_shards.v1",
                "ordering": "trial_id ascending, step ascending",
                "shards": index_rows,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "schedules" / "index.json").write_text(
        json.dumps([{"trial_id": "t-0000", "digest": "0" * 64}], sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "checkpoints" / "replica-00.json").write_text(
        json.dumps({"weights": [0.0, 0.004, -0.001]}, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "attempt_id": IDENTITY.attempt_id,
                "batch_id": IDENTITY.batch_id,
                "role": IDENTITY.role,
                "protocol_version": IDENTITY.protocol_version,
                "protocol_hash": IDENTITY.protocol_hash,
                "scientific_fingerprint_sha256": IDENTITY.scientific_fingerprint_sha256,
                "selected_candidate": IDENTITY.candidate_id,
                "source_commit": IDENTITY.source_commit,
                "invocation": IDENTITY.invocation,
                "outcome": IDENTITY.outcome,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "summary.json").write_text(
        json.dumps({"verdict": "PROTOTYPE"}, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "report.md").write_text("# prototype fixture\n", encoding="utf-8")

    if with_checksums:
        rows = {}
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name == "checksums.json":
                continue
            rows[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        (root / "checksums.json").write_text(
            json.dumps(rows, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
    return root


def backend(root: Path, **kwargs) -> LocalImmutableBackend:
    return LocalImmutableBackend(root, **kwargs)
