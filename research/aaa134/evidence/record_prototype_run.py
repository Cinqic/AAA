"""Run the archive prototype end to end and record what happened.

Writes nothing outside the path given on the command line and a temporary
directory. Consumes no confirmation identity and contacts no network.

    python research/aaa134/evidence/record_prototype_run.py \
        research/aaa134/evidence/prototype_run.json
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from research.aaa134.archive_prototype import PROTOTYPE_LABEL
from research.aaa134.archive_prototype.enumeration import read_identity
from research.aaa134.archive_prototype.local_backend import CREDENTIAL_ENV, LocalImmutableBackend
from research.aaa134.archive_prototype.publish import publish_archive, retrieve_archive, verify_retrieval
from research.aaa134.archive_prototype.tests.helpers import build_attempt


def main(output: Path) -> None:
    os.environ.setdefault(CREDENTIAL_ENV, "prototype-token-not-a-secret")
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        attempt = build_attempt(base / "attempt", shards=8)
        store = LocalImmutableBackend(base / "store")
        published = publish_archive(attempt, store, archive_id="prototype-archive-0001")
        fresh = base / "fresh"
        retrieved = retrieve_archive(store, published.locator, fresh)
        verdict = verify_retrieval(store, published.locator, fresh, expected_identity=read_identity(attempt))
        payload = {
            "schema_version": "aaa.research.aaa134.prototype_run.v1",
            "label": PROTOTYPE_LABEL,
            "durable": False,
            "closes_aaa_134": False,
            "closes_aaa_144": False,
            "backend": store.scheme,
            "fixture": "synthetic miniature attempt directory, not an AAA run",
            "published": published.as_dict(),
            "objects_retrieved_into_fresh_empty_directory": retrieved,
            "verification": verdict.as_dict(),
            "disclaimer": (
                "The locator above addresses a directory on the machine that produced it. "
                "It is not a durable locator, it is not evidence, and it must never be recorded "
                "in an AAA attempt as one."
            ),
        }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
