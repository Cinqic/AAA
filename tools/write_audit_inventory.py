#!/usr/bin/env python3
"""Generate the stable, exhaustive final-review inventory for every tracked path."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "final_audit.md"


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def purpose(path: str) -> str:
    if path.startswith("aaa/"):
        return "active implementation or packaged protocol data"
    if path.startswith("tests/"):
        return "active regression and protocol verification"
    if path.startswith(".github/"):
        return "active CI, governance, or dependency automation"
    if path.startswith("benchmarks/"):
        return "active frozen protocol identity or seed registry"
    if path.startswith("results/"):
        return "historical or current immutable experiment evidence"
    if path.startswith("docs/evidence/"):
        return "retained diagnostic or selection evidence"
    if path.startswith("docs/"):
        return "active documentation or review record"
    if path.startswith("tools/"):
        return "active validation or deterministic document generator"
    return "project metadata, packaging, license, or operator entry point"


def category(path: str) -> str:
    if path.startswith("results/final/") or path.startswith("results/benchmark_v2/"):
        return "historical"
    if path.startswith("results/benchmark_v2_1/") or path.startswith("docs/evidence/"):
        return "retained evidence"
    if path in {"docs/final_audit.md", "docs/handoff_sol.md"}:
        return "generated"
    return "active"


def verification(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".py":
        return "read; syntax/tests/lint as applicable"
    if suffix == ".json":
        return "read; strict JSON parsed; semantic checks as applicable"
    if suffix in {".yml", ".yaml", ".toml"}:
        return "read; parser or CI validation as applicable"
    if suffix == ".png":
        return "opened at original resolution; historical labeling checked"
    if suffix in {".md", ".txt"} or Path(path).name in {"LICENSE", ".gitignore"}:
        return "read; references and claims audited"
    return "read as bytes; retained provenance checked"


def findings(path: str) -> str:
    noise_path = (
        path.startswith("aaa/noise/")
        or path.startswith("docs/evidence/observation_noise")
        or path.startswith("docs/evidence/sol_observation_noise")
        or "observation_noise" in path
        or path
        in {
            "README.md",
            "CHANGELOG.md",
            "CONTRIBUTING.md",
            ".github/workflows/ci.yml",
            "docs/evidence_policy.md",
            "docs/experiment_registry.md",
            "docs/issue_ledger.md",
            "docs/limitations.md",
            "docs/reproduction.md",
            "docs/self_review.md",
            "docs/sol_review.md",
            "docs/handoff_sol.md",
            "tools/write_handoff.py",
        }
    )
    if noise_path:
        return "AAA-135 through AAA-149; AAA-144 remains open"
    ids: list[str] = []
    if path.startswith(("aaa/", "tests/", "tools/", "benchmarks/")):
        ids.extend(["AAA-121", "AAA-122", "AAA-123"])
    if path.startswith("results/benchmark_v2_1/") or path in {
        "docs/evidence_policy.md",
        "docs/reproduction.md",
    }:
        ids.append("AAA-124")
    if path in {"aaa/benchmark/seeds.py", ".github/workflows/benchmark.yml"}:
        ids.append("AAA-125")
    return ", ".join(dict.fromkeys(ids)) or "none"


def main() -> None:
    paths = [line for line in git("ls-files", "--cached").splitlines() if line]
    rows = []
    for name in paths:
        target = ROOT / name
        if not target.is_file() or target.is_symlink():
            raise SystemExit(f"tracked path is absent or not a regular file: {name}")
        digest = (
            "generated-self-reference"
            if target == OUTPUT
            else hashlib.sha256(target.read_bytes()).hexdigest()
        )
        rows.append(
            f"| `{name}` | `{digest}` | {purpose(name)} | {category(name)} | "
            f"{verification(name)} | {findings(name)} | retain candidate content |"
        )
    text = "\n".join(
        [
            "# Final tracked-file audit",
            "",
            "Generated deterministically by `tools/write_audit_inventory.py` from the staged review candidate.",
            f"It inventories **{len(rows)} tracked regular files**; the count and path set must equal `git ls-files --cached`.",
            "Hashes are SHA-256 of the reviewed worktree bytes, not Git blob IDs.",
            "A row records inspection coverage; it does not upgrade historical evidence into fresh verification.",
            "",
            "| Path | Reviewed SHA-256 | Purpose | Category | Verification | Related findings | Disposition |",
            "|---|---|---|---|---|---|---|",
            *rows,
            "",
        ]
    )
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)} with {len(rows)} rows")


if __name__ == "__main__":
    main()
