#!/usr/bin/env python3
"""Generate the current observation-noise review handoff from retained evidence."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "handoff_sol.md"


def load(relative: str) -> dict[str, Any]:
    value = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"{relative} must contain a JSON object")
    return value


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def main() -> None:
    protocol = load("aaa/noise/data/observation_noise_v1.json")
    ledger = load("benchmarks/observation_noise_candidate_ledger.json")
    selection = load("docs/evidence/observation_noise_development_selection.json")
    pilot = load("docs/evidence/observation_noise_scale_pilot.json")
    registry = load("benchmarks/observation_noise_registry.json")
    source_freeze = load("benchmarks/observation_noise_source_freeze.json")
    confirmation_path = ROOT / "benchmarks" / "observation_noise_freeze.json"
    confirmation = load("benchmarks/observation_noise_freeze.json") if confirmation_path.is_file() else None
    protocol_hash = str(source_freeze["protocol_hash"])
    planned_batches = set(source_freeze["planned_batches"])
    batches = [row for row in registry["batches"] if row.get("batch_id") in planned_batches]
    selected = str(ledger["selected_candidate"])
    selected_entry = next(row for row in ledger["entries"] if row["candidate_id"] == selected)
    eligible = selection["ranking"]["eligible_refinements"]
    measured = pilot["measured"]
    fresh = pilot["streaming_sharded_repair_pilot"]
    full = pilot["full_development_selection_verification"]
    projection = pilot["repaired_projection_per_batch"]
    freeze_status = (
        f"prepared at `{confirmation['scientific_fingerprint_sha256']}`; not authorization to run"
        if confirmation is not None
        else "not yet generated"
    )
    lines = [
        "# Observation-noise v1.1 Sol review handoff — PR #11",
        "",
        "This is the current AI-review handoff for the separately versioned",
        "`aaa.observation_noise.v1.1` phase. It supersedes the old v1 handoff as an",
        "active instruction, while the prior v2.1 review remains preserved in Git",
        "history and its dedicated evidence. This document is not human review,",
        "scientific confirmation, merge approval, or release approval.",
        "",
        "## Frozen identities",
        "",
        "| Item | Value |",
        "|---|---|",
        "| PR | [#11](https://github.com/Cinqic/AAA/pull/11) |",
        f"| current local head | `{git('rev-parse', 'HEAD')}` |",
        f"| scientific source commit | `{source_freeze['source']['commit']}` |",
        f"| source dirty when frozen | `{source_freeze['source']['dirty']}` |",
        f"| protocol | `{protocol['protocol_version']}` |",
        f"| protocol hash | `{protocol_hash}` |",
        f"| scientific fingerprint | `{source_freeze['scientific_fingerprint_sha256']}` |",
        f"| v2.1 reference commit | `{source_freeze['reference']['commit']}` |",
        f"| v2.1 raw / resolved hash | `{source_freeze['reference']['raw_sha256']}` / `{source_freeze['reference']['resolved_sha256']}` |",
        f"| dependency lock | `{source_freeze['dependency_lock']['sha256']}` |",
        f"| selected candidate | `{selected}` |",
        f"| candidate configuration | `{selected_entry['configuration_hash']}` |",
        f"| confirmation freeze | {freeze_status} |",
        "",
        "## Independent findings and repairs",
        "",
        "Sol reproduced and retained the failures recorded as `AAA-135` through",
        "`AAA-149` in `docs/issue_ledger.md`. The repairs include mandatory complete",
        "checksums, finite comparisons, ratio-of-means adaptation, complete selection",
        "constraints, fixed formal draw/batch identity, atomic cross-clone reservation,",
        "bounded-memory sharded evidence, isolated pinned-reference replay, causal",
        "prediction intervals, vectorized hierarchical bootstrap, and archive-path",
        "containment. Negative and interrupted probes remain retained.",
        "",
        "## Development selection",
        "",
        f"The corrected full 2 x 2 x 1 search completed with `quick: {str(selection['quick']).lower()}`.",
        f"Eligible refinements: `{eligible}`. Selected: `{selected}`. No-refinement",
        "remained possible and won because none of the three clipping mechanisms met",
        "every preregistered utility, baseline, retention, adaptation, stability,",
        "practical-gain, and adjusted-evidence condition. This is a development",
        "decision only; it establishes no formal endpoint.",
        "",
        f"Each of four candidate archives contained {full['scored_records_per_candidate']:,} scored records,",
        f"{full['trials_per_candidate']:,} trials, and {full['training_records_per_candidate']:,} training",
        "records and independently verified `PASS` before selection.",
        "",
        "## Local validation",
        "",
        "The final maintained-checkout suite passed 442 tests in 77.860 seconds.",
        "The instrumented run passed the same 442 tests with 71% total coverage",
        "against the configured 70% floor. Ruff lint/format, mypy, the exact 24-pin",
        "dependency lock, confirmation exit-code contract, strict JSON parsing and",
        "diff whitespace checks passed. `aaa-0.2.0-py3-none-any.whl` built with the",
        "locked backend and contained both canonical protocol files and the license.",
        "A fresh Python 3.12 environment outside the checkout installed the lock and",
        "wheel; both protocol hashes passed, its 58,944-record noise smoke recomputed",
        "`PASS`, and the checkout-only pinned-reference field was truthfully",
        "`NOT_VERIFIED`. Final-head GitHub CI is external evidence and is not claimed",
        "by this pre-push generated file.",
        "",
        "## Scale and archive boundary",
        "",
        f"The old quick layout peaked at {measured['maximum_resident_set_bytes']:,} bytes RSS.",
        f"The repaired sharded pilot retained {fresh['scored_records']:,} records, peaked at",
        f"{fresh['maximum_resident_set_bytes']:,} bytes RSS, occupied {fresh['attempt_bytes']:,} bytes,",
        "and independently verified. Timing-neutral primitive digests and computed",
        "metrics matched the prior streaming implementation exactly.",
        "",
        f"One formal batch is exactly {pilot['formal_plan']['scored_records']:,} scored records",
        f"and is projected at about {projection['compressed_record_bytes_approximate'] / 2**30:.1f} GiB compressed,",
        "plus schedules and metadata. There is no approved immutable object-store, Git",
        "LFS, or other durable locator and no demonstrated independent retrieval.",
        "Consequently neither fresh batch has been observed:",
        "",
        "| Batch | Role | Status |",
        "|---|---|---|",
        *[f"| `{row['batch_id']}` | `{row['role']}` | `{row['status']}` |" for row in batches],
        "",
        "## Review and scientific verdict",
        "",
        "**Engineering review verdict: BLOCKED for completion and normal merge.**",
        "",
        "**Scientific outcome: NOT ESTABLISHED.** No A/B observation, durable upload,",
        "fresh-location retrieval, joint recomputation, or primary endpoint decision",
        "exists. Missing confirmation is not an unfavorable or inconclusive result.",
        "The exact blocker is `AAA-144`: an approved immutable destination with at",
        "least roughly 80 GiB for the two projected compressed record streams, plus",
        "schedule/metadata overhead and practical retrieval headroom.",
        "",
        "The code and development evidence may be pushed to PR #11 for review and CI,",
        "but the brief forbids merge until the full confirmation and durable retrieval",
        "requirements are actually satisfied. No tag or release is authorized.",
        "",
        "## Reproduction entry points",
        "",
        "```bash",
        "python -m aaa.cli observation-noise-protocol-hash",
        "python -m aaa.cli observation-noise-fingerprint",
        "python -m aaa.cli observation-noise-development-select --reuse-root <selection-root> \\",
        "  --output docs/evidence/observation_noise_development_selection.json",
        "python -m aaa.cli observation-noise-recompute <attempt>",
        "python -m aaa.cli observation-noise-confirmation-evaluate <A-archive> <B-archive> \\",
        "  --output docs/evidence/observation_noise_joint_evaluation.json",
        "```",
        "",
        "The formal runner additionally requires the committed confirmation freeze, an",
        "explicit attempt label, the declared batch, and successful atomic remote",
        "reservation. Running it is intentionally deferred until durable storage and",
        "retrieval are real rather than aspirational.",
        "",
    ]
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
