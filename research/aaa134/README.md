# `research/aaa134` — isolated prototype tree

**Status: exploratory. Not approved. Not merged. Not confirmation evidence.**

This directory is an additive research tree exploring a solution path for
`AAA-134` (and the later restatement of the same storage blocker, `AAA-144`),
plus a design-only proposal for the experiment that should follow the
observation-noise phase.

## Branch and base identity

| | |
|---|---|
| branch | `opus/aaa134-durable-archive-prototype` |
| base branch | `codex/aaa-observation-noise-v1` (the PR #11 head branch) |
| base commit | `58e440462ce2e8fb551c5a5975bef85d4a54fef3` |
| base PR | #11, *Repair and freeze observation-noise v1.1 phase* |
| base of the base | `main` @ `25b6c32c9040d0f934314a2139993d12763afc99` |

The intended pull request is **stacked on PR #11**, not on `main`. Basing it on
`main` would drag the entire unmerged observation-noise implementation into an
unrelated diff.

## Hard boundaries

This branch:

- **modified no existing file.** The diff is additive only. Verified with
  `git diff --diff-filter=MDR --name-only <base>..HEAD` returning empty.
- **did not touch PR #11's branch, PR #12's branch, or `main`.**
- **did not execute observation-noise confirmation A or B.**
- **reserved and consumed no confirmation batch**, created no remote
  reservation ref, generated no confirmation identity, and changed no registry
  lifecycle field.
- **fabricated no durable locator.** Every artifact this tree can produce is
  labelled `PROTOTYPE_ONLY_NOT_DURABLE`.
- **changed no protocol, threshold, gate, seed, schedule, freeze manifest,
  candidate definition, ledger disposition or scientific conclusion.**
- **added no scientific-fingerprint exclusion** and moved nothing under
  `results/` or `runs/` to dodge one.

## This branch is not eligible for formal confirmation

`aaa/noise/scientific_identity.py::scientific_fingerprint` covers every tracked
non-generated file. Adding tracked files under `research/` therefore **changes
this branch's scientific fingerprint**, and
`benchmarks/observation_noise_source_freeze.json` will not match it.

That is correct behaviour, not a bug, and it was not worked around:

- PR #11's scientific identity, `28a5e55e6866f0825e86cd49b69d88ab08dd28ce7562d23d3015acb8cbb5ac19`
  over 111 files, remains untouched **on PR #11's own branch**;
- this prototype branch has a different identity because it contains additional
  tracked files;
- therefore **this branch must never be used to run a confirmation attempt.**

The measured effect is recorded in [`evidence/fingerprint_effect.json`](evidence/fingerprint_effect.json).

## AAA-134 status

> **AAA-134 PROTOTYPE VALIDATED; BLOCKER NOT RESOLVED.**

A local prototype can show that the proposed architecture is technically
plausible. It cannot close `AAA-134`, which requires a repository-approved
durable destination and a demonstrated production upload → immutable locator →
fresh-location retrieval → full checksum verification → independent
recomputation cycle. None of that exists yet. `AAA-144` likewise remains open.

## Contents

| Path | What it is |
|---|---|
| [`analysis/existing_aaa_behavior.md`](analysis/existing_aaa_behavior.md) | read-only behavioural analysis of current AAA |
| [`analysis/existing_aaa_behavior.json`](analysis/existing_aaa_behavior.json) | the same findings, machine-readable, with per-claim provenance |
| [`analysis/eiv_diagnostic.py`](analysis/eiv_diagnostic.py) | the fresh non-confirmatory diagnostic a reviewer can rerun |
| [`analysis/eiv_diagnostic.json`](analysis/eiv_diagnostic.json) | its recorded output |
| [`archive_prototype/`](archive_prototype/) | vendor-neutral archive contract, a local write-once test backend, and the publish/retrieve/verify prototype |
| [`archive_solution_design.md`](archive_solution_design.md) | the real `AAA-134` requirements, production architecture, durable-storage comparison, threat model, integration plan, and what stays unresolved |
| [`post_noise_experiment/design.md`](post_noise_experiment/design.md) | the next experiment, design only |
| [`post_noise_experiment/protocol_draft.json`](post_noise_experiment/protocol_draft.json) | machine-readable preregistration proposal, `DESIGN_ONLY_UNFROZEN_UNEXECUTED` |
| [`evidence/`](evidence/) | recorded validation and prototype-run evidence for this branch |
| [`self_review.md`](self_review.md) | an adversarial self-audit of this tree |

## Running the prototype

From the repository root, with the locked environment:

```bash
python -m unittest discover -s research/aaa134/archive_prototype/tests -t .
```

```bash
python research/aaa134/analysis/eiv_diagnostic.py
```

Neither command touches `benchmarks/`, `results/`, `docs/evidence/`, the batch
registry, or any remote.
