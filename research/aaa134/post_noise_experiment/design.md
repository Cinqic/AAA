# The experiment after observation noise — design only

> **`DESIGN_ONLY_UNFROZEN_UNEXECUTED`.**
>
> Nothing here was implemented or run. Execution is **gated on the
> observation-noise phase completing and being interpreted**: PR #11 must obtain
> valid confirmation evidence first. Until then the observation-noise result is
> unresolved and this design is a proposal, not a plan of record.

---

## 1. The question observation noise cannot answer

Observation-noise v1.1 asks what happens when the sensor is **noisy**. Every
channel it registers — gaussian, uniform, correlated AR(1), impulsive — is
**zero-mean**. Even the sensor-shift controls only change the *scale*
(`0.0005 <-> 0.002`), which is still a symmetric, zero-mean perturbation.

So the phase can answer: *does zero-mean sensor noise degrade prediction, and
does continued updating still help after a law change under that noise?*

It cannot answer the question sitting immediately behind it:

> **What happens when the sensor is wrong in a direction?**

That matters here specifically, and not as a generic robustness wish, because of
what the existing evidence already shows.

**First**, the learner regresses displacement on displacement
(`analysis/existing_aaa_behavior.md` §3.2). A *constant* sensor offset cancels
in a first difference and is therefore invisible to the displacement feature —
but it does *not* cancel in the centered-position feature. A *linear drift* adds
a constant to every displacement, which is **arithmetically indistinguishable
from a real change in velocity**. AAA's learner has no mechanism that could tell
those apart, and it is not obvious from the outside whether that is a problem in
practice or a rounding error.

**Second**, AAA's detector is self-triggered from its own scalar residual
(`aaa/benchmark/data/benchmark_v2_1.json`, `self_triggered_forgetting`). A
scalar residual cannot in principle separate "the world changed" from "my sensor
drifted". Protocol hypothesis `h2` already forbids treating detector activity as
evidence of a successful law-change response — but v1.1 never puts that
prohibition under load, because it never presents a sensor change that *looks
like* a law change.

**Third**, and this is the sharp point: an always-online learner facing a sensor
drift will adapt *toward the drift*. That **reduces observation-space error while
increasing latent error.** AAA measures latent error as its primary endpoint, so
it can actually see this happen — most systems cannot. The experiment is
therefore a direct test of the failure mode that always-online adaptation is
most exposed to, on a benchmark instrumented to detect it.

`AAA-120`'s history makes this concrete rather than theoretical: the first
confirmation round failed `always_online_stability`, the gate that asks whether
continuous updating hurts. Continuous updating has already hurt AAA once.

## 2. Candidates considered, and why this one

| Direction | Verdict |
|---|---|
| **Persistent sensor bias and drift** | **Selected.** One added deterministic term in the observation equation, `b_t`. Everything else — families, seed derivation, schedule hashing, matched branch construction, statistics, archive format, gates, verifier — is reused unchanged. Directly attacks the `h2` discrimination question and the claim-4/claim-5 divergence. Falsifiable in both directions. |
| Missing observations / partial observability | Deferred. Changes the *timing contract*, not just the observation value: "predict the next position" is ambiguous when the next observation is absent. That risks confounding an information change with an interface change, and AAA's carefully verified temporal ordering is the last thing to destabilise. A strong experiment — but it is the one *after* this one. |
| Process noise | Deferred. Makes the latent world stochastic, so the optimal predictor is no longer deterministic and **every absolute threshold in the protocol has to be re-derived** (the 0.02 line-length bound, the 1e-5 competitiveness margin, the near-machine-precision identification gate). That is a protocol rewrite wearing an environment change's clothes. |
| Irregular sampling intervals | Deferred. `dt` would become per-step, which changes the candidate's own feature normalization `dx/(dt * speed)`. That is a **candidate identity change**, not an environment change, and it would need its own selection phase. |

Rejected outright as premature by an order of magnitude: language, vision, audio,
autonomous goals, personality, general-intelligence claims, arbitrary-environment
adaptation, large neural models, multimodal agents. AAA currently predicts one
scalar in a bounded interval and loses to reflected constant motion on easy
families under noise (`analysis/existing_aaa_behavior.md` §3.4). Nothing in that
record licenses a jump.

The selection rule the assignment asked for — *the smallest experiment that
meaningfully increases uncertainty or partial observability while preserving
causal interpretability* — picks sensor bias/drift cleanly. It adds one scalar
term to one equation.

## 3. Research question and hypotheses

**Research question.** When the sensor acquires a persistent bias or drift and
the physical law does **not** change, does continued online updating help or harm
latent-state prediction; and can any predeclared control distinguish a sensor
change from a law change?

**H1 (primary, and the one the experiment exists for).** Under an unchanged law
with a linear sensor drift beginning at the declared onset, the online arm's
latent normalized MAE over the first 50 post-onset transitions is **not lower**
than its matched frozen clone's. Formally, the multiplicity-adjusted lower bound
on the relative reduction `(frozen - online) / frozen` does **not** exceed
`+0.10`.

This is deliberately stated so that the *expected* result is a non-effect or a
harm, and a benefit would falsify it. An experiment whose hypothesis can only be
confirmed is not an experiment.

**H2 (positive control, and it can invalidate the whole run).** Under a changed
law with **no** sensor drift, the online arm *does* beat its matched frozen clone
on the same endpoint, with an adjusted lower bound above `+0.10` — reproducing
the established `changed_law_adaptation` result under this protocol. If H2 fails,
the run is `NOT_VERIFIED` in its entirety: the machinery failed to reproduce a
known result, so nothing it says about H1 can be trusted.

**H3 (discrimination).** The candidate's self-triggered forgetting rate under
(law changed, no drift) and under (law unchanged, drift) is not distinguishable
at the declared level. That is, **the detector cannot tell the two apart.**

**H4 (observation-versus-latent divergence).** Under drift with an unchanged law,
the online arm's *noisy-observation* error decreases relative to frozen while its
*latent* error does not. This is the mechanism claim: the learner adapts
successfully to the wrong target.

### Explicit non-hypotheses

None of the following is claimed, tested, or inferable from any outcome:

1. that AAA models, represents, or understands a sensor;
2. that AAA can estimate or correct a bias;
3. that detector activity indicates a law change, a sensor change, or correct
   adaptation of any kind;
4. that any result generalizes to families, channels or bias forms outside the
   registered grid;
5. that a `FAIL` on H1 is a defect in AAA — absorbing a sensor drift is the
   *correct* behaviour for a learner that is only shown observations. The
   experiment measures a consequence of the information available, not a bug;
6. anything about baseline competitiveness beyond the cells measured;
7. any claim about intelligence, understanding, autonomy, or generality.

## 4. Environment, latent state, observation channel

**Latent state.** Unchanged. The existing scalar bounded simulator: `L = 1.0`,
`dt = 0.02`, speeds in `[0.12, 0.32]`. Two registered families only:

- `constant_velocity_400` — straight motion over 400 steps. This is where the
  EIV bias is worst and where a drift is maximally confusable with velocity.
- `changed_law` — the existing damped oscillator with a coefficient intervention
  at transition 300 and a 100-step branch. Unchanged from v1.1.

**Observation channel.** One added term:

```
y_t = x_t + L * ( b_t + eta_t )

eta_t ~ gaussian, scale 0.002          (the upper primary v1.1 level, unchanged)

b_t:  "none"   0
      "step"   0           t <  t_onset
               0.005       t >= t_onset
      "drift"  0                                   t <  t_onset
               0.0025 * (t - t_onset) * dt         t >= t_onset
```

`t_onset = 300`, matched exactly to the existing law-change timing so the 2x2 is
clean. Both magnitudes are chosen to be *small*: the step bias is 2.5x the noise
RMS, and the drift reaches the same 0.005 offset over the 100-step branch, which
is a displacement perturbation of about 1-2% of true velocity. Large enough to be
a real effect, small enough that "it obviously breaks" is not the answer.

`b_t` is **deterministic** given the condition, and is hashed into the schedule
digest before execution exactly as the noise path already is. It is generated and
saved before the run, never sampled during it.

**Intervention grid — a 2x3 factorial:**

| | sensor `none` | sensor `step` | sensor `drift` |
|---|---|---|---|
| **law unchanged** | double control | secondary | **H1 primary** |
| **law changed** | **H2 primary / positive control** | secondary | secondary |

Primary family: the two bolded cells plus the double control and the
(law changed, drift) cell — 4 cells. The `step` column is secondary diagnostic
throughout, because a constant offset differences away and is the less
interesting of the two forms.

**Predictor-visible information.** `observed_history` — a tuple of noisy, biased
scalar observations. Nothing else, ever.

**Evaluator-only information.** Latent `x_t`; the realized `eta_t`; the realized
`b_t`; the sensor condition label; the law-change flag and schedule; family and
branch identity; bounce labels.

**Temporal ordering.** Identical to `aaa/noise/runner.py`: observe -> predict ->
record the prediction -> advance the world -> reveal the target -> score against
latent truth -> update on the *noisy, biased* observation -> append. The learner
is never told that an onset occurred.

## 5. Arms

**Baselines** — the unchanged v1.1 comparator set, in the same fixed order:
`persistence`, `constant_motion`, `constant_motion_reflected`,
`causal_smoothing_motion`, `alpha_beta_filter`, `no_learning_control`.

`constant_motion_reflected` remains the primary deployable baseline, and it is
the *right* comparator here for a specific reason: it differences the observation
too, so a constant bias is invisible to it and a drift is absorbed by it in
exactly the same arithmetic way. Any divergence between it and the candidate is
therefore attributable to the learner, not to the bias.

**Candidate.** The v1.1 selected identity, `incumbent-no-refinement-v1`,
**unchanged**. Mechanism budget: **zero**. No refinement is proposed, searched,
or permitted.

That is a deliberate restraint. AAA's selection machinery exists to promote
mechanisms, and `analysis/existing_aaa_behavior.md` §3.5 shows what happens when
a mechanism family is proposed before the failure it addresses is understood: all
three clip candidates turned out numerically inert and the search tested nothing.
This experiment characterizes a failure mode. Designing a fix for a failure not
yet measured is how that happens again. If this phase finds something, the fix is
a **separate**, separately preregistered candidate-search phase.

**Matched arms.** At `t_onset` the candidate's complete state is cloned into
`selected_candidate_frozen` (updates off) and `selected_candidate_online`
(updates on). Both consume the identical continued trajectory. The clone state is
hashed so a reviewer can verify the match without trusting the runtime.

## 6. Splits, seeds, and freeze

| | Development | Confirmation A / B |
|---|---|---|
| Seed namespace | `post_noise_bias_development` | `post_noise_bias_confirmation_a` / `_b` |
| Lineages | 2 | 5 (shared across A and B) |
| Episodes per condition per lineage | 2 | 12 |
| Sensor realizations per episode | 1 | 2 |
| Evaluation streams | — | disjoint latent and sensor namespaces |

**Development tuning budget: zero candidate configurations.** Development may
only perform three pass/fail engineering checks, all declared before it runs:

1. the measured pilot resource use is within the projection in §9;
2. the `b_t` generator produces the declared deterministic path, verified against
   a closed form on a sample outside the benchmark streams;
3. the H2 positive control is non-degenerate — the frozen and online arms are not
   numerically identical, the failure that made v1.1's refinement search vacuous.

No threshold, seed, schedule, magnitude, onset or candidate parameter may be
changed after any development result is read. If a check fails, the design is
revised and **re-frozen from scratch** with fresh confirmation batch identities,
exactly as v1.1 did when its protocol hash changed.

**Seed derivation.** `aaa/noise/observation.py::derive_seed`, unchanged:
`SeedSequence([root, namespace, condition, family, lineage, episode, realization])`
with stable UTF-8 label words. No process-randomized hashing.

**Frozen schedule generation.** Every noise path and every `b_t` path is
generated, saved and hashed **before** execution; per-trial digests go in
`schedules/index.json`; the manifest records the count. Identical to v1.1.

**Freeze sequence.** Design freeze (this protocol, generator, split definitions,
budget) -> development -> confirmation freeze naming the candidate -> declare two
fresh, never-used batch identities -> execute A -> execute B -> joint analysis.
Nothing is tuned between A and B. A failed attempt retires its batch permanently.

## 7. Endpoints and statistics

**Primary endpoint.** Latent next-position MAE normalized by `L`, over the first
50 transitions after `t_onset`, including the first surprise — online versus the
matched frozen clone, as a relative reduction `(frozen - online) / frozen`.

**Secondary diagnostics**, reported and never promoted:

- the same endpoint computed on **noisy-observation** error rather than latent
  error — this is the H4 divergence measurement;
- signed latent bias (not just magnitude) — a drift should produce a *signed*
  error, which is the direct fingerprint;
- detector firing rate and update-norm trajectory per condition (H3);
- weight trajectories, specifically the displacement weight, which is where a
  drift should be absorbed;
- calibration interval coverage and width, reported separately from point
  prediction;
- the `step` bias column throughout;
- latency, peak memory, resets, skips, non-finite counts.

**Normalization.** All errors divided by `L`. The relative reduction is
dimensionless. A zero or missing denominator makes the ratio **undefined and
fails the claim**; it is never replaced with zero. (v1.1's
`undefined_ratio_rule`, carried over.)

**Uncertainty.** Paired hierarchical bootstrap over
`(training_lineage, episode_within_lineage, sensor_realization_within_episode)`,
4,000 draws, common random numbers pairing arms within a cell. 95% intervals,
90% also reported.

**Multiplicity.** **One** Holm correction across the joint A+B primary family:
4 primary cells x 2 endpoints x 2 batches = **16 claims**. Secondary diagnostics
are outside the family and can never support a claim.

**Practical margins.** Relative-reduction margin `0.10` (matched to v1.1's
`moderate_noise_adaptation`). Absolute latent MAE bound `0.02` line lengths.

**Resampling units** are the units of independence, and the design is sized for
them: 5 lineages x 12 episodes x 2 realizations = **120 independent units per
cell**.

## 8. Decision rules

| Outcome | Condition |
|---|---|
| **H1 `PASS`** (hypothesis supported) | adjusted lower bound on relative reduction under (law unchanged, drift) is `<= 0.10` in both A and B |
| **H1 `FAIL`** (hypothesis falsified) | that bound exceeds `0.10` in both A and B — online updating genuinely helps under a pure sensor drift, and the premise of this design is wrong |
| **H2 `FAIL`** | positive control does not reproduce -> **the entire run is `NOT_VERIFIED`** |
| `NOT_VERIFIED` | a required check did not execute |
| `INSUFFICIENT_EVIDENCE` | intervals too wide to separate the margin, or A and B disagree in direction |
| `BLOCKED` | no approved durable archive locator exists at execution time — see §10 |

**Falsification conditions, stated plainly.** H1 is falsified by a clear online
benefit under drift-only. H3 is falsified by a detector-rate difference that
clears the adjusted bound — which would be a *positive* and genuinely surprising
result, and would immediately motivate a follow-up. H4 is falsified if
observation-space and latent-space effects move together.

**Stopping rules.** No interim analysis of confirmation data. No peeking at A
before B is planned; both are executed under the same freeze. A run that crashes
is retained, its batch retired, and a fresh batch declared.

**Promotion rules.** **None.** No mechanism may be promoted from this phase under
any outcome. A finding motivates a separate preregistered search.

**Exit criteria.** The phase is complete when: both batches executed under one
freeze; both archives independently verified byte for byte; both retrieved from a
durable locator into a fresh location; joint Holm analysis produced; every
primary claim assigned exactly one of the five statuses; and the result written up
with H1, H2, H3, H4 reported separately and no cross-implication.

## 9. Resource plan

Derived by scaling v1.1's **measured** sharded pilot (58,944 records, 47.28 MB
attempt, 155 MB peak RSS, 124 s), not by guessing.

| Quantity | Value | Derivation |
|---|---:|---|
| conditions | 6 | 2 laws x 3 sensor conditions |
| lineages | 5 | design |
| episodes per condition per lineage | 12 | design |
| sensor realizations per episode | 2 | design |
| **trials per batch** | **720** | `6 x 5 x 12 x 2` |
| scored transitions per trial | ~500 | 300 prefix + 100 frozen + 100 online |
| **scored records per batch** | **~360,000** | `720 x 500` |
| scale vs the v1.1 pilot | **6.1x** | `360,000 / 58,944` |
| **compressed archive per batch** | **~290 MB** | `47.28 MB x 6.1` |
| **A + B, with schedules, metadata and headroom** | **~0.8 GB** | x2 plus overhead and one retained failed attempt |
| record shards per batch | ~1,500 | one per trial-branch |
| CPU | single core, ~475 records/s measured | pilot |
| **runtime per batch** | **~13 min** | `360,000 / 475` |
| peak RSS | **~160 MB** | streaming shard architecture, unchanged |
| bootstrap operations | ~130,000 resamples | `4,000 draws x (16 primary + ~16 secondary) cells x 2 batches` |
| bootstrap runtime | seconds | v1.1 vectorized: 1,888 cells x 4,000 draws in 18.7 s |

### The scale is a design decision, not an accident

v1.1's formal plan is **38.5 GB per batch** and is blocked on `AAA-134`/`AAA-144`
for exactly that reason. This design is **~290 MB per batch — roughly 130x
smaller** — and it gets there by cutting the grid to what the hypotheses actually
need: 2 families instead of 4, 1 channel instead of 4, 1 scale instead of 4,
5 lineages instead of 10, 12 episodes instead of 32, 2 realizations instead of 3.

Nothing was cut from the *statistics*: 120 independent resampling units per cell
with 4,000 draws, one Holm family, matched arms, a positive control.

Disk is not cheap when it makes an experiment unrunnable. v1.1 has been blocked
for its entire life on a storage prerequisite. Designing the successor at 0.8 GB
means that whatever `AAA-134` resolution is eventually approved — object store,
Zenodo DOI, or something else — this experiment fits inside it comfortably, and
a reviewer can retrieve and recompute the whole thing over an ordinary internet
connection in minutes.

**A development pilot must prove the resource behaviour before any confirmation
freeze.** The numbers above are projections from a measured pilot at a different
scale; they are not measurements of this experiment.

## 10. Durable-evidence and reproduction requirements

Unchanged in kind from v1.1, and **this design does not relax them**:

- complete primitive archive, canonical record index, schedules, checkpoints,
  metadata, reports, plots, `checksums.json` covering every retained byte;
- failed and superseded attempts retained;
- an immutable or version-addressed durable locator recorded in the evidence
  before any reproduction claim;
- retrieval into a fresh location with no reference to the original attempt
  directory;
- independent byte verification of every object after retrieval;
- independent recomputation of every metric and gate from the retrieved primitive
  records, by the reference verifier that owns its own arithmetic.

At ~0.8 GB the range of acceptable destinations is much wider than v1.1's — a
Zenodo record with a DOI becomes viable on its own, as well as the object-store
path in `../archive_solution_design.md`.

**But the requirement does not go away.** If no approved durable locator exists
when this experiment would run, its status is `BLOCKED`, exactly as v1.1's is.
Smaller is easier; smaller is not exempt.

## 11. Claim classes stay separate

| Class | What this experiment can say |
|---|---|
| 1. implementation correctness | the usual verifier/recompute gates, nothing new |
| 2. learning with experience | **nothing** — not tested here |
| 3. generalization | **nothing** — no held-out family |
| 4. benefit from continued updates after change | **the whole point**, and specifically that the benefit is *conditional on the change being in the world* |
| 5. baseline competitiveness | reported per cell; never inferred from class 4 |

A H1 `PASS` says online updating does not help under a sensor drift. It says
nothing about whether AAA learns, generalizes, or beats a baseline. A H2 `PASS`
replicates a known class-4 result under a new protocol and implies nothing about
H1. These are kept apart in the report by construction, not by discipline.
