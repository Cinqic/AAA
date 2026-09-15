# Observation-noise phase: `aaa.observation_noise.v1.1`

This is a separately versioned study of one bounded question: how well does
AAA predict a moving dot's next latent position when only the observation
channel is noisy? It is not a revision of benchmark v2.1 and it does not add
language, vision, goals, actions, process noise, missingness, or variable
sampling intervals.

The executable protocol is
[`aaa/noise/data/observation_noise_v1.json`](../aaa/noise/data/observation_noise_v1.json).
It is design-frozen before comparative candidate work. The protocol hash is
printed by `python -m aaa.cli observation-noise-protocol-hash` and is recorded
in every attempt.

## Reference boundary

The v2.1 specification remains byte-for-byte unchanged and is pinned to commit
`25b6c32c9040d0f934314a2139993d12763afc99`. The phase records both
its raw SHA-256 (`4993c5e6e173f9dd5ef002bc84ff4c484da853b066ea45662d41ad826d10d48e`)
and its resolved specification hash
(`f8e1090bf5b1aeb02cb8a129fec0ec9c83ab1b50cb2c500496b862fe6a1a5e37`). The
hashes address different representations. A zero-noise equivalence fixture
compares latent trajectories, observations, forecasts, targets, updates,
reflection handling, counters, and metrics against an isolated v2.1 reference
checkout before any confirmation claim. The isolated reference is exported
from the pinned commit rather than imported from the current working tree.

The old v2.1 source fingerprint is not weakened to accommodate this phase.
Adding a new namespace legitimately changes the current checkout. Historical
reference checks use the pinned commit and the repository lock in an isolated
checkout.

## Causal observation model

For latent position `x_t`, interval width `L`, and sensor innovation `eta_t`,
the only predictor-visible value is

```text
y_t = x_t + L * eta_t
```

The runner creates and caches exactly one sensor sample for every timestamp,
including warmup. Repeated observation calls return the same cached value. A
primitive record keeps `latent_position`, `raw_observation`, and
`observed_history` as separate fields. Noise is never clipped or reflected at
the physical boundary.

The order is:

```text
predict from observed history
record the forecast
advance the unchanged latent simulator
reveal latent truth and the new noisy observation
score latent and noisy-observation tasks
update from the new noisy observation only
append the noisy observation to history
```

Predictors never receive latent truth, latent velocity, simulator parameters,
intervention labels, noise innovations, event times, or future samples. The
evaluator may use latent truth privately for scoring.

## Why differencing needs an explicit diagnostic

The incumbent uses a displacement feature from observations. With

```text
y_t = x_t + L eta_t
```

the observed displacement is

```text
y_t - y_(t-1) = (x_t - x_(t-1)) + L(eta_t - eta_(t-1))
```

and the next noisy displacement target is

```text
y_(t+1) - y_t = (x_(t+1) - x_t) + L(eta_(t+1) - eta_t)
```

Thus the current noise term `-L eta_t` occurs in both the regressor and the
target. For independent innovations their covariance contribution is
`Cov(L(eta_t-eta_(t-1)), L(eta_(t+1)-eta_t)) = -L^2 s^2`, before any physical
signal covariance is considered. The predictor is therefore exposed to a
shared-noise errors-in-variables problem. For the AR(1) channel, the same
calculation must retain the declared lag covariance instead of treating the
terms as independent. Numerical square-root RLS stability does not establish
unbiased estimation or noise robustness.

This derivation motivates the generator tests, clean/noise-trained comparison,
causal smoothing baseline, and the separate noise-shift controls. It is not a
claim that a refinement succeeds.

## Frozen channels and cells

The dimensionless RMS scale set is `0`, `0.0005`, `0.002`, and `0.01`. Gaussian
and uniform at the two lower nonzero levels are primary cells. The highest
level, correlated AR(1), and impulsive channels are mandatory stress cells.
The exact definitions and stream derivation are in the machine-readable
protocol. Actual schedule arrays are saved and hashed before a cell runs.

The stationary study covers constant velocity, bouncing, speed change, and the
changed-law oscillator. A separate factorial records unchanged/changed
dynamics crossed with unchanged/changed noise. Law changes occur at transition
300 with a 100-transition branch. Sensor shifts are `0.0005 -> 0.002` and
`0.002 -> 0.0005`, aligned at 300 or staggered at 340. A noise shift is never
counted as successful adaptation to a physical-law change.

## Models and matched comparisons

The mandatory comparator set is persistence, raw constant motion, reflected
constant motion, causal smoothing-plus-motion, a fixed-parameter alpha-beta
filter, the unchanged square-root RLS incumbent, and a no-learning control.
The reflected constant-motion model receives the same public boundary map as
the incumbent; boundary handling is not counted as learned capability.

There are two fixed training conditions: clean-trained and noise-trained. The
noise mixture is fixed to Gaussian and uniform at the two primary lower levels.
The complete learner/filter state is cloned at interventions. A frozen arm
freezes learned parameters and adaptive predictor state, while ordinary
observation history and fixed-parameter filter state continue to evolve in both
arms. Initial forecasts must agree. Point-prediction adaptation and interval
calibration adaptation are separate analyses.

Candidate work is bounded at twelve configurations and three substantive
mechanism changes. The ledger includes unsuccessful configurations. A
refinement is not promoted merely because it is more complex or because a
confirmation outcome was unfavorable.

The committed selection plan is
[`../benchmarks/observation_noise_development_selection.json`](../benchmarks/observation_noise_development_selection.json).
It fixes the development namespace, sample, estimands, conjunctive ranking,
practical margin, tie-break, baseline separation, and rejection policy before
variants are compared. The four current identities are the unchanged
`incumbent-no-refinement-v1` control and three causal innovation-clipping
variants. Development selected the unchanged incumbent; the complete ledger
and compact evidence retain the three rejected variants and their reasons.

The confirmation candidate is resolved solely from the committed freeze and
ledger. Its configuration hash is checked before any batch is consumed. The
candidate receives only predictor-visible noisy history and the newly revealed
noisy target; the clipping mechanism uses the candidate's own raw forecast and
does not receive a law-change, bounce, truth, or evaluator-event label.

## Replication and uncertainty

The fixed confirmation plan is ten independently trained learner lineages,
32 latent episodes per family and stationary cell per lineage, and three
independent sensor realizations per latent episode. The existing 32 strata are
checked directly. Confirmation A and B share the frozen training lineages but
have disjoint evaluation streams; they are replication across schedules, not
independent retraining studies.

Aggregate uncertainty resamples training lineage, latent episode within
lineage, and sensor realization within episode, preserving paired models and
branches. The protocol declares 4,000 draws, 90% and 95% intervals, fixed
balanced cell weighting, and one Holm family across primary endpoint/cell/batch
claims. Undefined ratios remain undefined.

The machine-readable acceptance formulas are in
`acceptance.machine_criteria`. The joint evaluator enumerates primary
condition/family/channel/scale cells in A and B, computes absolute utility,
candidate-versus-reflected-baseline margin, first-50 changed-law online versus
complete-state frozen reduction (including the first surprise), and unchanged
law retention. It applies one one-sided 95% Holm-adjusted bound family over all
of those claims. An added mechanism has an additional preregistered positive
gain margin over the unchanged incumbent; the no-refinement control makes that
promotion gate not applicable rather than inventing a gain claim.

## Evidence and outcomes

Every attempt has an immutable ID and atomic lifecycle records. Confirmation
also atomically reserves an append-only remote Git ref, so separate clones
cannot claim the same batch. Full primitive
records, schedule arrays, checkpoint state, source identity, lock identity,
exact invocation, diagnostics, summaries, plots, exclusions, and checksums are
retained. Failed and cancelled attempts remain visible. Independent
recomputation reads primitive artifacts without importing the production metric
collector or gate evaluator. Reproduction reruns generation from declared
inputs; recomputation verifies retained evidence. They are distinct verdicts.

The current primitive record schema is `aaa.observation_noise_step.v3`. In
addition to latent and noisy point-prediction fields, it stores the causal
prediction intervals available before each reveal. Intervals are calibrated
from past noisy residuals only; warmup rows carry an explicit unavailable
value. The statistical artifact reports coverage, width, and interval score
separately from point-prediction adaptation. Completed trials are written to
deterministic atomic per-trial gzip shards, which makes an interrupted attempt
safely resumable without replacing retained evidence. Finalization renames the
shard directory to `records/` and writes a canonical `records/index.json`
ordered by trial ID then step. The manifest binds the concatenated uncompressed
semantic stream, the index, and every compressed shard; no duplicate combined
JSONL copy is retained.

Protocol v1 was superseded before either confirmation batch was executed or
observed. Its exact hash, source commit, retired batch IDs, and audit reason are
retained in `benchmarks/observation_noise_registry.json`. v1.1 resolves the
10%-boundary prose/operator contradiction, pins the v2.1 reference commit, and
declares the bounded sharded archive. The fresh confirmation IDs are
`observation-noise-a-0002` and `observation-noise-b-0002`.

Training updates are retained in `training_records.jsonl` and are bound to the
saved training schedules. Each evaluated lineage also receives a fixed four-cell
Gaussian/Uniform calibration prefix from the declared training mixture; its
schedule files, sample counts, and cold-start policy are recorded separately.
Primitive diagnostics include update norm, detector/counter activity, measured
predict/update latency, and a static parameter/state-cost inventory. These are
resource diagnostics, not evidence that a predictor is scientifically better.

The phase uses five separate outcomes: engineering complete, scientifically
supported, negative, inconclusive, and blocked by missing evidence. Only an
explicit `PASS` satisfies a required engineering gate. A valid negative result
may be merged as evidence but cannot promote an unsupported refinement or
authorize the next capability stage.

The formal command is:

```bash
python -m aaa.cli observation-noise-confirmation-evaluate <A> <B> \
  --output docs/evidence/observation_noise_joint_evaluation.json
```

It is intentionally a joint operation. It independently verifies both full
archives, rejects mismatched protocol/fingerprint/candidate/training inputs,
reconstructs the primitive claims, records the claim order and family size,
and recomputes the final outcome without trusting either stored conclusion.
