# Research limitations

Stated plainly, because a benchmark that cannot say what it does not show is
not measuring much.

## The environment

One dimension. One moving dot. Deterministic, noiseless, fully observed
positions. Two dynamical laws: constant velocity with reflection, and a damped
harmonic oscillator. Both are analytically simple, and one of them —
constant-velocity motion with reflection — is *exactly* solved by a
parameterless baseline given the public boundary map. That is why the reflected
constant-motion baseline exists and why the bounce criterion is an absolute
accuracy requirement rather than a relative one.

## The learner

Three parameters. A linear map from `[1, scaled displacement, centered
position]` to the next normalized displacement. It has no explicit bounce
detector, no change detector that the evaluator informs, and no memory beyond
four positions and its own covariance state. It cannot represent a law outside
that family.

## What the results can and cannot support

These claims are kept separate, and none implies another:

1. the implementation runs correctly;
2. the model improves with experience;
3. what it learned generalizes to unfamiliar episodes;
4. continued updates help after a change;
5. it beats a baseline.

In particular:

- **Boundary handling is programmed, not learned.** Reflection is public
  knowledge of the observation format, supplied by the programmer.
- **The oscillator's pre-change dynamics are not held out.** The candidate is
  allowed to learn them from its own observations during the prefix. The
  post-change coefficients and the intervention are held out.
- **"Autonomous" means unattended.** The observe / predict / score / update loop
  runs without intervention after launch. It does not mean general
  intelligence, physical understanding, or independent goal formation, and
  nothing here is evidence for any of those.
- **Five training replicas is a routine engineering minimum**, labelled as such.
  Five checkpoints evaluated on 500 episodes is 500 episodes from five
  learners.

## What the benchmark has already refused

The first confirmation round under this protocol failed. Both fresh streams
failed `always_online_stability`: a continuously updating instance regressed
against the reflected constant-motion baseline, because it was fitting the one
window after each wall contact whose displacement feature is a folded
difference. That was repaired in the candidate, on development evidence, with
the failed attempts kept and no threshold touched.

Worth stating plainly: the thirteen gates that passed in that round include
every *frozen* track. The gate that failed is the only one that asks what
happens when the system is left running. That asymmetry is the most interesting
thing this version has measured.

## Known open items

- Raw per-step evidence is regenerable rather than durably archived
  (`AAA-077`).
- The always-online family rotates through a fixed sequence of the existing
  regimes; it is not an open-ended stream (`AAA-008`).

## Historical next-experiment rationale

Before the separate noise phase existed, controlled observation noise was the
next experiment proposed for benchmark v2.1. That rationale remains historical:
perfect observations make an analytic extrapolator unusually strong. The
experiment is now implemented as `aaa.observation_noise.v1.1`, but its formal
scientific result is not established.

## Observation-noise phase status

`aaa.observation_noise.v1.1` has a corrected full 2 x 2 x 1 development search
and a smaller smoke path. Together they demonstrate schedule generation,
causal noisy updates, matched branch construction, direct realized stratum
identities, primitive recomputation, bounded-memory sharding, uncertainty and
raw-data plots. They do not establish a supported operating envelope.
Confirmation A/B, adjusted uncertainty over the fixed formal replication plan,
durable full-archive publication and retrieval are not represented.

The earlier quick refinement smoke was not a valid completion of the frozen
2 x 2 x 1 development selection plan. The repaired full selection has now run:
all four candidate archives independently verified, no refinement had the
preregistered practical gain or adjusted positive evidence, and the unchanged
incumbent was retained as the explicit no-refinement control. This does not
establish that the incumbent meets any confirmation endpoint. The source
fingerprint is non-self-referential and fail-closed, but a durable archive
locator is still absent. The exact confirmation freeze may be prepared, but
full ten-lineage A/B execution, joint Holm analysis and durable retrieval are
therefore `NOT_VERIFIED` or `BLOCKED`, not silently treated as passing. Sol's AI
review is recorded separately and does not claim independent human review.

The noise model corrupts observations only. It does not study process noise,
dropout, bias, irregular sampling, hidden state, actions, goals, language,
vision, or general intelligence. A favorable detector response to a sensor
shift is not evidence that a physical-law change was detected.
