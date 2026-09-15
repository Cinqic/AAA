# Research history

## Observation-noise v1.1 independent repair boundary (2026-09-14)

Sol independently reproduced fifteen observation-noise review findings
(`AAA-135` through `AAA-149`) before repairing the implementation. The phase is
now explicitly versioned `aaa.observation_noise.v1.1`; v1 is retained as an
unobserved superseded protocol because its adaptation prose and machine
operator disagreed at the exact 10% boundary.

The corrected full 2 x 2 x 1 development selection retained and independently
verified four 235,776-record candidate archives. None of the three innovation
clipping refinements met every preregistered constraint, so the unchanged
`incumbent-no-refinement-v1` control remains selected. This is a development
decision, not confirmation evidence.

Record production and verification are bounded-memory and use deterministic
per-trial gzip shards. A measured quick pilot reduced peak RSS from about 2.67
GiB to 148 MiB and retained one primitive copy with exact timing-neutral
semantic equivalence. Formal admission now requires a committed freeze, fixed
4,000 draws, frozen batch identities, and an atomic append-only remote claim.
The verifier requires complete checksums and rejects non-finite values,
incomplete cells, unsafe paths, symlinks, ambiguous record layouts, and
tampered schedules or statistics.

Formal A/B remains deliberately unobserved. One batch is projected at 38.5 GB
of compressed records plus schedules and metadata, and no approved immutable
archive/retrieval destination exists. The phase is therefore `BLOCKED`, not
negative or inconclusive, and PR #11 is not approved for merge at this boundary.

## Observation-noise v1 completion boundary (2026-09-14, superseded)

The observation-noise phase now has a non-self-referential scientific source
fingerprint. It hashes the tracked scientific source map, rejects untracked
scientific files and symlinks, normalizes only mutable batch lifecycle fields,
and excludes generated result/freeze/review artifacts. Committing the exact
freeze therefore cannot invalidate its own recorded identity; Git commit and
tree values remain provenance only.

Development refinement is bounded by the declared 12-configuration/3-mechanism
budget. Four immutable candidate identities were evaluated on the committed
development plan. All three innovation-clipping refinements were rejected on
the preregistered constraints, so `incumbent-no-refinement-v1` is the selected
candidate. No A/B conclusion follows from this development result.

The joint confirmation evaluator is present as
`observation-noise-confirmation-evaluate <A> <B>`. It independently verifies
both archives, requires shared candidate/fingerprint/training compatibility,
reconstructs absolute utility, baseline competitiveness, first-50 adaptation,
and unchanged-law retention claims, and applies one Holm family across A+B.
No confirmation freeze or A/B archive was executed in this completion boundary.

Dates are the dates of the work, not of any release. Nothing here is a product
announcement.

## Observation-noise v1 design freeze (2026-09-13)

Added the separately versioned `aaa.observation_noise.v1` protocol and a
controlled sensor channel for next-position prediction. The implementation
keeps latent truth and noisy observations separate, caches one schedule sample
per timestamp, updates only from the revealed noisy observation, records full
primitive evidence, and independently recomputes its arithmetic. Development
smoke output is explicitly inconclusive; no refinement or capability promotion
is claimed.

## v2.1 — benchmark protocol repair (2026-09-10 / 2026-09-11)

A full engineering and scientific-methodology repair following independent
reviews by GPT-6 Astra, Claude Opus 5 and GPT-5.6 Sol. Every finding was
reproduced against the pre-repair tree before anything was changed; the probe
output is committed at `docs/evidence/pre_repair_probes.json` and every item is
tracked in `docs/issue_ledger.md`.

**The benchmark can now say no.** Formal confirmation returns a non-zero exit
status when any required gate is not `PASS`, including `NOT_VERIFIED` and
`INSUFFICIENT_EVIDENCE`. Absence of evidence is never converted into success.

**Gates measure what they are named after.** `straight_learning` — which a
parameterless analytic rule satisfied — is replaced by three separate claims.
The bounce criterion no longer compares a candidate that may reflect against a
baseline that may not; development measurement showed essentially the entire
reported margin was the reflection transform.

**The specification is executable.** 26 declared values that the runner never
read are now consumed, under a typed strict schema, with a test that fails if
any declared leaf stops being used.

**The learner is numerically sound.** Covariance-form RLS with forgetting lost
positive semidefiniteness at update 323 and overflowed at 6,642 on weakly
exciting input. It is replaced by a square-root formulation with trace-bounded
forgetting, verified against an independent batch least-squares reference.
Invalid state is refused rather than silently repaired.

**Statistics match their estimands.** A genuinely hierarchical paired bootstrap
that recomputes the gate's own statistic on every draw, with declared event
weighting, no-event episodes excluded from event resampling, intervals on
non-regression claims, Holm-Bonferroni multiplicity that counts repeated
confirmation attempts, and missing intervals reported rather than assumed.

**Confirmation discipline is mechanical.** Predeclared immutable batch
identities that *are* the seed input, a batch registry with a lifecycle, a
freeze manifest checked before a confirmation runs, a clean-source-tree
requirement, and invariants enforced by the runner rather than only the CLI.

**Evidence is auditable.** Versioned artifact schemas, an experiment registry
with safe resume, and a `recompute` command that rebuilds every metric and gate
from retained raw evidence without retraining or re-simulating.

Also: recovery redefined from peak shock with predictor-specific references and
a six-way status partition; one normalized-unit contract; multi-wall bounce
accounting; a headless, testable animation with checkpoint-type dispatch; plots
through the object-oriented Matplotlib API with bounds-derived axes; a real
learning curve on a fixed frozen probe bank; an always-online deployment track;
in-domain training with extrapolation measured separately; formatter, linter,
type checker and coverage with no blanket ignores; and CI that installs the
lock file and proves it.

**The repaired benchmark said no, and that is the headline.** The first v2.1
confirmation round failed: confirmation A and confirmation B, on independent
fresh streams, both failed the required `always_online_stability` gate and both
exited non-zero. Thirteen of fourteen gates passed in each. The failure was
traced on development data to a precise mechanism — a continuously updating
instance is damaged by the one window after each wall contact whose
displacement feature is a folded difference — and repaired in the candidate.
Both batches were retired permanently, no threshold was touched, and the failed
attempts are committed. A first attempt at the repair did not work and is
recorded too. See `AAA-120` in `docs/issue_ledger.md`.

Round 2, with the repaired candidate and newly declared batches against a new
specification hash, is recorded alongside it. Both rounds are in
`benchmarks/confirmation_batches.json` and `results/benchmark_v2_1/`.

Historical v1 and v2 evidence is preserved. The v2 confirmations are
reclassified as historical/provisional under a superseded methodology; see
`docs/errata.md`.

## v2 — benchmark introduction (2026-09-09)

Introduced a versioned benchmark separate from the v1 snapshot, with four
families, compressed step-level evidence, checkpoint lineage and gate
decisions. Superseded; see `docs/errata.md` for which of its claims were
affected.

## v1 — initial prototype (2026-09-09)

The original moving-dot evaluation with an online linear SGD learner over
absolute-position features. The result was unfavourable and is preserved
unchanged in `results/final/`: the learner did not convincingly improve with
experience, generalized poorly when frozen, and lost to constant motion.
Continued updates did improve it relative to its matched frozen copy after a
change, but it remained inferior to the analytic baseline. The learner is
retained as the `legacy_linear_sgd` diagnostic arm and reproduces that
behaviour under the repaired protocol.
