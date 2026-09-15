# Observation-noise v1.1 Sol review handoff — PR #11

This is the current AI-review handoff for the separately versioned
`aaa.observation_noise.v1.1` phase. It supersedes the old v1 handoff as an
active instruction, while the prior v2.1 review remains preserved in Git
history and its dedicated evidence. This document is not human review,
scientific confirmation, merge approval, or release approval.

## Frozen identities

| Item | Value |
|---|---|
| PR | [#11](https://github.com/Cinqic/AAA/pull/11) |
| current local head | `a03e9c1bfa5fea4d472dc7124f61ed9e2d877082` |
| scientific source commit | `404280e050b05ec7cd46f8cacc3e3bb56d74003c` |
| source dirty when frozen | `False` |
| protocol | `aaa.observation_noise.v1.1` |
| protocol hash | `546e2434cc15850779107c1a329af1e2f8581cd6a1c3b31807eecf5f6e7b427c` |
| scientific fingerprint | `28a5e55e6866f0825e86cd49b69d88ab08dd28ce7562d23d3015acb8cbb5ac19` |
| v2.1 reference commit | `25b6c32c9040d0f934314a2139993d12763afc99` |
| v2.1 raw / resolved hash | `4993c5e6e173f9dd5ef002bc84ff4c484da853b066ea45662d41ad826d10d48e` / `f8e1090bf5b1aeb02cb8a129fec0ec9c83ab1b50cb2c500496b862fe6a1a5e37` |
| dependency lock | `2aa52a7deefd05403b9fb6441bef42e13e47a4384baaf780f3b4b879bfcf3bc7` |
| selected candidate | `incumbent-no-refinement-v1` |
| candidate configuration | `dcc82cb0396985976419733a817d02027c45683db4f76d3ea37ac48290f6c649` |
| confirmation freeze | prepared at `28a5e55e6866f0825e86cd49b69d88ab08dd28ce7562d23d3015acb8cbb5ac19`; not authorization to run |

## Independent findings and repairs

Sol reproduced and retained the failures recorded as `AAA-135` through
`AAA-149` in `docs/issue_ledger.md`. The repairs include mandatory complete
checksums, finite comparisons, ratio-of-means adaptation, complete selection
constraints, fixed formal draw/batch identity, atomic cross-clone reservation,
bounded-memory sharded evidence, isolated pinned-reference replay, causal
prediction intervals, vectorized hierarchical bootstrap, and archive-path
containment. Negative and interrupted probes remain retained.

## Development selection

The corrected full 2 x 2 x 1 search completed with `quick: false`.
Eligible refinements: `[]`. Selected: `incumbent-no-refinement-v1`. No-refinement
remained possible and won because none of the three clipping mechanisms met
every preregistered utility, baseline, retention, adaptation, stability,
practical-gain, and adjusted-evidence condition. This is a development
decision only; it establishes no formal endpoint.

Each of four candidate archives contained 235,776 scored records,
1,024 trials, and 3,552 training
records and independently verified `PASS` before selection.

## Local validation

The final maintained-checkout suite passed 442 tests in 77.860 seconds.
The instrumented run passed the same 442 tests with 71% total coverage
against the configured 70% floor. Ruff lint/format, mypy, the exact 24-pin
dependency lock, confirmation exit-code contract, strict JSON parsing and
diff whitespace checks passed. `aaa-0.2.0-py3-none-any.whl` built with the
locked backend and contained both canonical protocol files and the license.
A fresh Python 3.12 environment outside the checkout installed the lock and
wheel; both protocol hashes passed, its 58,944-record noise smoke recomputed
`PASS`, and the checkout-only pinned-reference field was truthfully
`NOT_VERIFIED`. Final-head GitHub CI is external evidence and is not claimed
by this pre-push generated file.

## Scale and archive boundary

The old quick layout peaked at 2,731,806,720 bytes RSS.
The repaired sharded pilot retained 58,944 records, peaked at
155,054,080 bytes RSS, occupied 47,281,979 bytes,
and independently verified. Timing-neutral primitive digests and computed
metrics matched the prior streaming implementation exactly.

One formal batch is exactly 59,043,840 scored records
and is projected at about 35.9 GiB compressed,
plus schedules and metadata. There is no approved immutable object-store, Git
LFS, or other durable locator and no demonstrated independent retrieval.
Consequently neither fresh batch has been observed:

| Batch | Role | Status |
|---|---|---|
| `observation-noise-a-0002` | `confirmation_a` | `planned` |
| `observation-noise-b-0002` | `confirmation_b` | `planned` |

## Review and scientific verdict

**Engineering review verdict: BLOCKED for completion and normal merge.**

**Scientific outcome: NOT ESTABLISHED.** No A/B observation, durable upload,
fresh-location retrieval, joint recomputation, or primary endpoint decision
exists. Missing confirmation is not an unfavorable or inconclusive result.
The exact blocker is `AAA-144`: an approved immutable destination with at
least roughly 80 GiB for the two projected compressed record streams, plus
schedule/metadata overhead and practical retrieval headroom.

The code and development evidence may be pushed to PR #11 for review and CI,
but the brief forbids merge until the full confirmation and durable retrieval
requirements are actually satisfied. No tag or release is authorized.

## Reproduction entry points

```bash
python -m aaa.cli observation-noise-protocol-hash
python -m aaa.cli observation-noise-fingerprint
python -m aaa.cli observation-noise-development-select --reuse-root <selection-root> \
  --output docs/evidence/observation_noise_development_selection.json
python -m aaa.cli observation-noise-recompute <attempt>
python -m aaa.cli observation-noise-confirmation-evaluate <A-archive> <B-archive> \
  --output docs/evidence/observation_noise_joint_evaluation.json
```

The formal runner additionally requires the committed confirmation freeze, an
explicit attempt label, the declared batch, and successful atomic remote
reservation. Running it is intentionally deferred until durable storage and
retrieval are real rather than aspirational.
