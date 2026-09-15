# Read-only behavioural analysis of current AAA

**Scope.** What AAA actually does today, verified against code and retained
evidence rather than against summaries, PR descriptions or ledger prose.
Nothing in AAA was changed to produce this document.

**Trees inspected.**

| Label | Ref | Commit |
|---|---|---|
| `main` | `origin/main` | `25b6c32c9040d0f934314a2139993d12763afc99` |
| PR #11 head (the base of this branch) | `origin/codex/aaa-observation-noise-v1` | `58e440462ce2e8fb551c5a5975bef85d4a54fef3` |
| PR #12 head (context only, **not** canonical) | `origin/opus/aaa-playground-tiny-nn` | `697d44916ca77aaad6accfabd68e7dff7f18e2c5` |

Every quantitative statement below is tagged with its evidence class:

- **H** — historical committed evidence (frozen v2.1 confirmation results);
- **D** — development evidence committed on the PR #11 branch;
- **F** — a fresh non-confirmatory diagnostic run for this analysis, from
  [`eiv_diagnostic.py`](eiv_diagnostic.py), recorded in
  [`eiv_diagnostic.json`](eiv_diagnostic.json);
- **C** — a statement about code, cited by file and line.

No confirmation batch was reserved or consumed and no unobserved confirmation
outcome was inspected to produce any number here.

---

## 1. Core temporal behaviour

### 1.1 The causal order is what the documentation claims

**(C)** `aaa/noise/runner.py:325-424` implements exactly one ordering per step,
and it is the documented one:

1. `observed_history = tuple(history[-history_length:])` — the predictor-visible
   window is materialised first (`:325`);
2. `raw = float(predictor.predict(observed_history))` for every predictor,
   and every forecast plus its calibrated interval is captured (`:332-337`);
3. `transition = environment.advance()` — the world moves only *after* all
   forecasts exist (`:339`);
4. `target_observation = environment.observe()` — the target is revealed (`:340`);
5. scoring against `transition.position` (latent) and `target_observation`
   (noisy) (`:343-354`);
6. the complete record dict is constructed and appended (`:355-404`);
7. `predictor.update(observed_history, target_observation)` (`:411`);
8. `history.append(target_observation)` (`:424`).

The record is built **before** any predictor is updated, so a record can never
be retro-fitted with post-update state. `run_noisy_segment` (`:456-523`) and the
matched-branch paths (`:701-728`, `:801-815`) repeat the identical ordering.

A structural guard at `:359-403` raises `ObservationNoiseError("record
construction drifted")` if the record's key set changes at all — a cheap but
effective defence against silently widening what gets retained.

### 1.2 What a predictor receives, and what it is denied

**(C)** The predictor interface is `predict(history) -> float` and
`update(history, target_position) -> None`. `history` is a tuple of floats and
nothing else. In a noise run those floats are **noisy observations**
(`aaa/noise/observation.py:NoisyObservationEnvironment.observe`), never latent
positions.

Evaluator-only information, present in the record but never passed to a
predictor:

| Field | Why it is evaluator-only |
|---|---|
| `latent_position` | ground truth `x_t`; the primary metric is scored against it |
| `bounced`, `changed` | event labels; a predictor that saw them would be told when to adapt |
| `noise_innovation`, `noise_channel`, `noise_scale` | the sensor's own realisation and parameters |
| `trial.family`, `trial.branch` | scenario identity |

`NoisyObservationEnvironment.position` carries an explicit in-code comment
calling itself *"Latent evaluator truth; the prediction runner never passes this
value"* (`aaa/noise/observation.py`). The predictor sees `observed_history`;
the runner passes `transition.position` only into the scoring dictionary.

**Caveat, stated plainly.** This is a *structural* argument from reading the
call sites, not a proof. The repository does carry tests asserting the
predictor signatures and the absence of evaluator fields
(`tests/test_observation_noise.py`, `tests/test_predictors.py`), and I did not
find a path that leaks. I did not attempt an exhaustive taint analysis.

---

## 2. Where the predictive performance actually comes from

The baseline suite is designed so each member isolates one source. Ranked from
"no knowledge" upward:

| Arm | What it encodes | Learned? |
|---|---|---|
| `persistence` | `y_t` | no |
| `constant_motion` | `y_t + (y_t - y_{t-1})` | no |
| `constant_motion_reflected` | the same, plus the **public** boundary reflection map | no |
| `causal_smoothing_motion` | two-window causal mean, then constant motion, then reflection | no |
| `alpha_beta_filter` | fixed `alpha=0.72`, `beta=0.18` tracker; **state** updates, **parameters** do not | no parameter learning |
| `no_learning_control` | the incumbent architecture, zero-initialised, updates disabled | no |
| `incumbent_square_root_rls` | three RLS weights over `[1, dx/(dt*0.2), (x-mid)/L]` | yes |

### 2.1 Programmed boundary knowledge is not learning, and the evidence says so

**(H)** In confirmation `aaa-v2_1-confirmation-b-0004`
(`results/benchmark_v2_1/aaa-v2_1-confirmation-b-0004/report.md`):

- `bounce.decomposition_vs_persistence` = `1 [1, 1]`, p = 2.5e-4
- `bounce.decomposition_vs_raw_constant_motion` = `1 [1, 1]`, p = 2.5e-4
- `bounce.decomposition_vs_reflected_constant_motion` = **`-1.237e+06`
  [-1.761e+06, -8.128e+05], p = 1**

Read correctly: against arms that lack the boundary map, the candidate's bounce
advantage is total. Against the arm that has *the same* boundary map, the
comparison is decisively negative with p = 1. **Essentially all of the
candidate's bounce-transition accuracy is the programmed reflection, not
learned capability.** The repository already says this in prose; the retained
interval is the number that makes it checkable.

### 2.2 Where learning does contribute

**(H)** `aaa-v2_1-confirmation-b-0004`, required gates:

- `changed_law_adaptation` = **0.7542** — the updating copy beats its
  bitwise-identical frozen copy on 50-transition cumulative post-change error.
  This is a comparison against *itself*, which is why it is a clean claim about
  online updating and nothing else.
- `constant_velocity_identification` = 3.895e-11 normalized MAE — the RLS
  weights genuinely identify the linear law to near machine precision in a
  noiseless world.
- `frozen_prediction_accuracy` = 5.467e-06 — frozen generalisation to the
  unfamiliar bouncing regime.
- `unchanged_control` = -5.341e-08 — continuing to update in an unchanged world
  does not degrade against the frozen twin.
- `always_online_stability` = 6.841e-06, margin
  `-1.002e-05 [-1.016e-05, -9.872e-06]`.

### 2.3 The `always_online_stability` history

**(H)** `README.md:54-60` and `AAA-120`: the *first* confirmation round under
the repaired protocol **failed** this gate on both independent fresh streams
and both exited non-zero; 13 of 14 gates passed in each. The failure was traced
on development data to a specific mechanism, the candidate was repaired, no
threshold was touched, and the failed attempts stay committed. The repaired
margin now sits at roughly `-1.0e-05`, i.e. it clears the declared bound by
about the bound's own width — a pass, but not a comfortable one.

This is the single most load-bearing fact about AAA's credibility: the protocol
has demonstrably said no, and the response was to fix the learner rather than
the threshold.

---

## 3. Observation-noise behaviour (PR #11)

### 3.1 The sensor model

**(C)** `y_t = x_t + L * eta_t` with `L = 1.0`, one sample per timestamp, no
clipping or reflection of the sensor, and repeated observation idempotent
(`aaa/noise/data/observation_noise_v1.json`, `aaa/noise/observation.py`).
Channels: `gaussian` and `uniform` primary; `correlated` (AR(1), rho=0.8) and
`impulsive` (1% Bernoulli, 10-sigma spikes) as stress-only, explicitly *not*
promotable. Scales `{0, 0.0005, 0.002, 0.01}`, primary `{0.0005, 0.002}`.

### 3.2 The errors-in-variables structure is real, and it is worse than attenuation

This is the mechanically important part, and the code makes it precise.

**(C)** `aaa/predictors.py:561-568` — the feature vector is

```
phi_t = [ 1,  (y_t - y_{t-1}) / D,  (y_t - midpoint) / L ]      D = dt * 0.2 = 0.004
```

**(C)** `aaa/predictors.py:617` — the regression target is **a displacement,
not a position**:

```
z_t = (y_{t+1} - y_t) / L
```

So the same observation `y_t` enters the regressor with coefficient `+L/D` and
the target with coefficient `-1`. With white `eta`:

- regressor noise variance = `2 * (L/D)^2 * sigma^2` = **125,000 · sigma²**
- regressor/target noise covariance = `-(L/D) * sigma^2` = **−250 · sigma²**

That negative covariance is *not* classical attenuation. Attenuation shrinks
the slope toward zero; this term pushes it *past* zero.

**(F)** Batch OLS of `z_t` on `phi_1` over a pure constant-velocity latent path,
5 seeds × 3 speeds, 4,000 steps:

| noise scale | mean estimated displacement weight |
|---:|---:|
| 0.0005 | **−0.0019812** |
| 0.002 | **−0.0019812** |
| 0.01 | **−0.0019812** |

The closed-form noise-only asymptote is `−D/(2L)` = **−0.002**; the true value
for this law is `+D/L` = **+0.004**. The estimate lands on the asymptote to
three figures, is **sign-flipped relative to truth**, and is *scale-invariant*
— exactly as predicted, because under constant velocity the true feature has no
variance at all, so the regression identifies the noise and nothing else.

> *Honest caveat.* The `scale = 0.0` rows in `eiv_diagnostic.json`
> (0.00106 / 0.00200 / 0.00288) are **not** the true weight. With no noise the
> constant-velocity design matrix is rank-deficient and `lstsq` returns a
> minimum-norm solution that splits arbitrarily between intercept and slope.
> They are reported rather than suppressed; they carry no meaning. The noisy
> rows are full-rank and are the ones that matter.

**(F)** Driving the *real* incumbent (`make_candidate`, updates on) through a
noisy sinusoidal latent path chosen to never touch a wall — so no reflection,
unfolding or straddle skip is active:

| noise scale | displacement weight | centered-position weight | intercept |
|---:|---:|---:|---:|
| 0.0 | 0.0040000 | −0.000987 | −1.9e-14 |
| 0.0005 | 0.0039376 | −0.000995 | −1.9e-06 |
| 0.002 | 0.0033570 | **+0.17558** | 0.00688 |
| 0.01 | 0.0036319 | **+1.70504** | 0.06208 |

Two things are visible. The displacement weight attenuates (0.0040 → 0.0034 at
the upper primary scale). More importantly the contamination **leaks into the
other features**: the centered-position weight moves by three orders of
magnitude and changes sign. The learner does not fail gracefully by shrinking;
it re-apportions the fit across whatever features are available.

### 3.3 What that predicts, and what the development evidence shows

If a learner is trained under noise and then *frozen*, it carries a
systematically wrong velocity coefficient into deployment. It should be
catastrophic precisely where the law is a pure constant velocity.

**(D)** `docs/evidence/observation_noise_development_selection.json`, the
`noise_trained` constant-velocity primary cells:

| channel | scale | online MAE | **frozen MAE** | reflected baseline MAE |
|---|---:|---:|---:|---:|
| gaussian | 0.0005 | 0.002546 | **0.061255** | 0.000802 |
| uniform | 0.0005 | 0.002659 | **0.061260** | 0.000904 |
| gaussian | 0.002 | 0.004960 | **0.061539** | 0.003609 |
| uniform | 0.002 | 0.005031 | **0.061158** | 0.003795 |

The frozen noise-trained state is ~24× worse than online and ~68× worse than
the analytic reflected baseline, and — the tell — its error is **almost
independent of the noise scale** (0.0612 at both 0.0005 and 0.002). That is the
signature of a fixed wrong coefficient, not of sensor noise. The EIV prediction
in §3.2 and the retained development numbers agree.

Mean retention delta (online − frozen) over the 16 unchanged-law primary cells:
**−0.024743** (median −0.020181, max +1.95e-05).

So: **online updating is what rescues the noise-trained learner, and it rescues
it from a bias that the noisy training itself created.** That is a real and
interesting result, but it is not the same claim as "online updating beats a
strong baseline".

### 3.4 Baseline competitiveness fails on development evidence

**(D)** Preregistered constraint (`aaa/noise/data/observation_noise_v1.json`,
`acceptance.baseline_competitiveness`): the adjusted upper bound on
`MAE_online − 1.10 · MAE_reflected` must be `<= 0.00001`.

Development point estimates over the incumbent's 32 primary cells:

| statistic | value |
|---|---:|
| mean margin | **+0.000412** |
| min | −0.001911 |
| max | +0.015378 |
| fraction of cells with a positive (failing-direction) margin | 0.281 |

By family:

| family | mean margin |
|---|---:|
| `bouncing` | −0.000178 |
| `changed_law` | −0.000892 |
| `constant_velocity` | **+0.000431** |
| `speed_change` | **+0.002287** |

The learner beats the analytic baseline where the world is hard (`bouncing`,
`changed_law`) and **loses to it where the world is easy** (`constant_velocity`,
`speed_change`). In a noisy but otherwise simple world, causal smoothing plus
reflected constant motion remains the thing to beat. On development evidence
the incumbent does not beat it on average.

This is a development point estimate, not a confirmation verdict, and it does
not predetermine the outcome of A/B. It does mean the baseline-competitiveness
claim is the one most at risk in confirmation.

### 3.5 The development selection outcome, verified independently

The repository states the selection retained the unchanged incumbent with no
eligible refinements. I verified that directly and found the reason is stronger
than the prose suggests.

**(D)** For all four catalogued candidates, in
`docs/evidence/observation_noise_development_selection.json`:

```
max |incumbent_mae − candidate_mae| over all 32 cells = 3.25e-19
```

and `incumbent_gain_mean = 0.0` for every one of the three refinements.
3.25e-19 is floating-point dust. **The three `causal-innovation-clip-*`
candidates are numerically indistinguishable from the incumbent: the innovation
clip never bound at any tested noise scale.**

The three eligibility constraints they fail (`aaa/noise/selection.py:315-335`):

1. `incumbent_gain_mean` (0.0) must exceed the practical margin 1e-05 — fails,
   because the mechanism is inert;
2. `baseline_margin_mean` (+4.12e-04) must be `<= 1e-05` — fails, per §3.4,
   and this failure is the **incumbent's**, shared by all four;
3. `promotion_adjusted_positive_evidence` is `False`.

> **The correct wording, and it matters.** No proposed refinement satisfied the
> preregistered constraints, so the unchanged incumbent was retained as an
> explicit no-refinement control. That is a legitimate selection outcome. It is
> **not** evidence that the incumbent is robust to observation noise. The
> search space consisted of one mechanism family at three clip levels, and that
> family turned out to be *inactive* on the development grid — it tested
> nothing. Reading "no refinement promoted" as "the incumbent is good enough"
> would be a straightforward misreading of a null search.

### 3.6 Noise-shift versus law-change

**(C)** The protocol separates these deliberately. `schedules.sensor_shifts`
declares `0.0005 -> 0.002` and `0.002 -> 0.0005` at two predeclared timings
(`sensor_aligned = 300`, `sensor_staggered = 340`), crossed with unchanged and
changed dynamics. Hypothesis `h2` states that a noise-level shift is
distinguishable from a changed physical law *only through predeclared
controls*, and explicitly forbids treating detector activity as evidence of a
successful law-change response. Hypothesis `h0` anticipates that observation
noise alone can trigger the incumbent's self-triggered forgetting without any
law change.

Given §3.2, `h0` is very likely to hold: an 8× EWMA surprise trigger fed by a
noise-inflated innovation stream will fire on sensor noise. The detector cannot
in principle separate the two from a scalar residual alone. **This is the
uncertainty the observation-noise phase is least equipped to resolve, and it is
where the next experiment should go (see `../post_noise_experiment/design.md`).**

### 3.7 Calibration and uncertainty

**(C)** `aaa/noise/calibration.py` computes causal residual-quantile intervals
at 90% and 95% from past noisy residuals only, and the protocol requires
coverage, width and interval score to be reported by channel, level and regime,
with calibration adaptation reported **separately** from point-predictor
adaptation. No calibration outcome has been confirmed.

---

## 4. Claim classes, kept separate

AAA's five claim classes and where each currently stands:

| # | Claim | Status | Basis |
|---|---|---|---|
| 1 | implementation correctness | supported at v2.1 | **H** `correctness`, `reproducibility` gates PASS in `b-0004` |
| 2 | learning with experience | supported at v2.1 | **H** `learning_progress`, `constant_velocity_identification` |
| 3 | generalization | narrowly supported at v2.1 | **H** `frozen_prediction_accuracy` = 5.467e-06 on the unfamiliar bouncing regime |
| 4 | benefit from continued updates after change | supported at v2.1; **unresolved under noise** | **H** `changed_law_adaptation` = 0.7542; **D** only for noise |
| 5 | baseline competitiveness | supported at v2.1; **at risk under noise** | **H** `always_online_stability` margin ≈ −1.0e-05; **D** §3.4 |

None of these implies another. In particular (4) holding under noise on
development data says nothing about (5) under noise, and §3.4 is a direct
warning about exactly that pair.

---

## 5. PR #12 is context, not canon

PR #12's 17-parameter `TinyMLP` is exploratory. Its own PR description records
that it lost to *every* baseline including `persistence`, and by ~18× to its
like-for-like `constant_motion_reflected` comparator. It is useful as an
independent re-implementation of AAA's causal contract, and its
source-fingerprint collision is the same mechanism described in this branch's
README. It is **not** used as evidence anywhere in this analysis.

---

## 6. Limitations of this analysis

- The causal-boundary argument in §1.2 is structural, not an exhaustive taint
  analysis.
- The record-shard count of 245,760 per batch is taken from
  `docs/evidence/observation_noise_scale_pilot.json`. I confirmed the archive
  *byte* projection independently (§ `../archive_solution_design.md`) but did
  **not** re-derive the shard count from the plan's first principles; the
  linear scaling of the pilot's 256 shards gives 256,434, and I could not
  reconcile the two exactly from the committed plan alone. Treat 245,760 as
  reported, not as verified.
- The **F** diagnostics use synthetic latent paths chosen to isolate the EIV
  term, not AAA's registered families. They explain a mechanism; they do not
  measure AAA's benchmark.
- No confirmation outcome was inspected, so nothing here predicts A/B.
