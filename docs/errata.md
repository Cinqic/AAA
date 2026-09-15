# Errata: which earlier AAA claims were affected, and why

This document exists so that nothing inconvenient is quietly deleted. The
historical artifacts stay where they are; this is the correction notice that
travels with them.

## Status of previously recorded results

| Evidence | Previous description | Correct description |
|---|---|---|
| [`results/final/`](../results/final/) | historical v1 snapshot | unchanged. Still historical v1 evidence, still scientifically useful |
| `confirmation-a-20260909-clean`, `confirmation-b-20260909-clean` | "development provenance" | historical/provisional evidence under a superseded methodology |
| `confirmation-a-20260909-strata-clean`, `confirmation-b-20260909-strata-clean` | "**the acceptance evidence**" | **superseded.** Historical/provisional evidence under a superseded methodology |

All four v2 attempts are preserved. None of them is acceptance evidence for any
current claim. They were produced by a benchmark whose defects changed what
several of its gates meant, which is documented below and reproduced in
[`evidence/pre_repair_probes.json`](evidence/pre_repair_probes.json).

## The v1 result stands, and it was unfavourable

The original v1 finding is preserved exactly because it was useful:

- the online linear learner did not convincingly improve from experience;
- its frozen generalization was poor;
- constant motion was substantially better;
- continued updates did improve it relative to its matched frozen copy after a
  change, but it remained inferior to the analytic constant-motion baseline.

Under the repaired v2.1 protocol the same learner is retained as the
`legacy_linear_sgd` diagnostic arm, and it reproduces that behaviour: in the
bouncing family it is worse than persistence. That is a real result about the
original design, not something to be tidied away.

## What was wrong with v2, and what it changed

### Claims that were overstated

**"All required gates PASS."** True of the numbers v2 computed, but several of
those gates did not measure what their names said.

**`straight_learning`.** The criterion was a relative reduction against a
zero-initialized control on constant-velocity motion. A parameterless analytic
rule satisfies it. It was not evidence of learning, and it is gone. See
[`benchmark_protocol.md`](benchmark_protocol.md) for the three separate claims
that replace it.

**The 20% bounce-event improvement.** The candidate was allowed to reflect its
prediction into the public bounds; the baseline it was compared against was
not. Measured on development streams, the entire margin is the reflection
transform:

| Predictor (bouncing family, normalized bounce-event MAE) | Value |
|---|---|
| `constant_motion` (raw) | ~3.1e-03 |
| `constant_motion_reflected` (like-for-like) | ~3.7e-17 |
| candidate with reflection | ~4.2e-17 |
| candidate with reflection removed | ~3.1e-03 — indistinguishable from raw constant motion |

Against the fair baseline the candidate is not 20% better; it is marginally
*worse*, at a magnitude that is floating-point noise. The gate was replaced by
an absolute accuracy requirement plus non-regression against the reflected
baseline. This was decided on development evidence and frozen before any
confirmation batch existed.

**"Randomized event timing."** The active v2 benchmark fixed the changed-law
intervention at transition 600 while describing the timing as randomized. v2.1
draws timing and post-change coefficients from predeclared ranges, and says so.

**"Independent models."** Confirmation A and B trained from identical
`development` streams, so their checkpoints were identical. The `role`
parameter threaded into training was accepted and never used. This is a
defensible design — it measures generalization of one frozen selected model set
— but it was not what "independent" implied. v2.1 declares the relationship
explicitly and records the checkpoint hashes.

**"Fresh confirmation seeds."** `attempt_id` never entered the seed derivation,
so changing only the label re-ran identical streams. Freshness now comes from a
predeclared batch identity that *is* the seed input.

**"Reproducibility PASS."** The v2 gate compared a deterministic pure function
with itself and checked a checkpoint format string. No rerun happened. v2.1
executes a deterministic rerun, a save/resume equivalence check and a golden
seed-mapping fixture, and reports `NOT_VERIFIED` when they have not run.

**"Compact evidence."** Tens of megabytes of generated JSON, with the sole copy
of the acceptance evidence in a git-ignored directory or a 30-day CI artifact.
See [`evidence_policy.md`](evidence_policy.md).

**"CI validation."** The old confirmation workflow could not fail on a gate
result, and CI ignored the published lock file while testing one Python version
out of a declared `>=3.10` range.

**`475 / 475` recovery.** Eligibility was decided by a 50-transition average,
so a large single-step shock followed by fast recovery was diluted below the
threshold and dropped out of the denominator entirely. A 10x shock over a
0.02 baseline became "no meaningful post-change error increase". Recovery rates
computed that way are not comparable to rates computed from peak shock.
Additionally, every predictor's recovery used the online learner's pre-event
error sequence as its reference.

### Corrections to earlier corrective text

The v2 results README said the first clean A/B pair "used only a low-speed
training range and therefore did not exercise all declared speed strata". The
training range was one contributing factor, but the defect was in the
confirmation helper and the resulting stratum coverage, and the coverage gate
could not detect it: with empty or missing strata the gate passed vacuously. In
v2.1, coverage is planned as an explicit Cartesian product and a missing
stratum is `INSUFFICIENT_EVIDENCE`.

Separately, the v2 training range (0.02-0.06) did not match the v2 confirmation
range (0.08-0.20). That is a speed extrapolation, and describing the result as
ordinary held-out generalization was wrong. v2.1 trains in-domain on the
confirmation range and measures extrapolation separately in a family that is
labelled out-of-distribution.

## Numerical stability

The v2 selected learner used covariance-form RLS with forgetting 0.90. On
weakly exciting input it exhibits classical covariance windup. Reproduced
against the pre-repair tree:

| Probe | Outcome |
|---|---|
| repeated identical input | covariance overflowed to non-finite at update 6,642 |
| stationary observation stream | non-finite covariance at update 6,642 |
| slow constant velocity | lost positive semidefiniteness at update 323 (min eigenvalue -6.7e-02) |

Checkpoint loading also accepted negative-definite, indefinite, asymmetric,
singular, zero and badly conditioned covariance matrices, and silently
symmetrized asymmetric ones.

These did not fire on the benchmark's own trajectories, which is precisely why
"it passed" was not evidence that it was sound.

## The first v2.1 confirmation round failed

This belongs here rather than in a footnote. The first formal confirmation
round under the repaired protocol **failed**: confirmation A and confirmation B,
on independent fresh streams, both failed the required `always_online_stability`
gate and both exited non-zero. Thirteen of fourteen gates passed in each.

The failure was diagnosed to a specific mechanism on development data — a
continuously updating instance is damaged by the one window after each wall
contact whose displacement feature is a folded difference — and repaired in the
candidate. Both batches are retired permanently, no threshold was altered, and
the failed attempts are committed at
[`../results/benchmark_v2_1/`](../results/benchmark_v2_1/).

See `AAA-120` in [`issue_ledger.md`](issue_ledger.md) for the full diagnosis,
including an alternative repair that was tried and did not work.

## What survived

Not everything was wrong, and the repair did not assume it was.

- The strict predict/advance/score/update temporal boundary was correct and is
  preserved, with leakage tests.
- The matched changed-law branch design — common prefix, cloned state, frozen
  and updating copies in the same changed world, plus an unchanged control — is
  the soundest part of v2 and is kept and regression-tested.
- The covariance-form RLS recursion was algebraically right. With no forgetting
  it matches a ridge batch least-squares reference to 1.7e-18. The failure was
  forgetting under weak excitation, not the algebra.
- The ~55% changed-law improvement of the updating copy over its frozen twin is
  a real signal and not an artifact of the old statistics. Whether it survives
  the repaired protocol on fresh confirmation streams is reported in the
  confirmation evidence, not assumed here, and 55% was never used as a target.

## Active v2.1 timing and episodic semantics (2026-09-13)

Some historical review prose described change timing as randomized. The active
v2.1 executable specification uses the fixed transition-300 change point; the
current v2.1 runner does not sample a new change time per episode. Its
always-online learner carries predictor state across the sequence of episodes,
while the simulated world and observation history are reset for each episode.
This correction describes the current executable behavior; historical review
wording and archived attempt provenance remain unchanged. The separately
versioned observation-noise phase follows the same fixed 300-transition law
change for its matched branch and does not reinterpret a sensor shift as a
physical-law detection.
