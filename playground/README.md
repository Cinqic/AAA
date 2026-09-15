# AAA Playground — exploratory tooling

A live visualizer for one AAA prediction loop, and a tiny neural experiment
that runs inside the same loop.

## What this is

- A way to *watch* prediction, error, learning, adaptation and model state
  evolve one step at a time.
- A 17-parameter neural learner, written by hand in NumPy, dropped into AAA's
  existing observe / predict / reveal / score / update contract to see what
  happens.

## What this is not

- **Not** benchmark v2.1, and not a proposed v2.2.
- **Not** a confirmation run, an acceptance gate, or evidence for any AAA
  claim.
- **Not** a replacement for `OnlineRLSPredictor`, and not "the new AAA
  architecture".
- **Not** connected to the separate controlled observation-noise research
  phase.

Nothing in `playground/` changes a benchmark specification, a confirmation
batch, a registry, a freeze manifest, a threshold, a gate or a historical
result. It reuses the public `aaa` package read-only.

## Launch

From a repository checkout:

```bash
python -m playground
```

```bash
python -m playground --scenario bouncing --seed 42 --frozen
```

It is deliberately not wired into `aaa.cli` and deliberately not packaged, so
the scientific package's command surface acquires nothing while it is under
independent review.

### Controls

| Input | Effect |
|---|---|
| `Space` | play / pause |
| `Right` | single step |
| `r` | reset (deterministic) |
| `f` | freeze / unfreeze the tiny network |
| Buttons | Pause, Step, Reset, Freeze TinyMLP, Next seed |
| Radio | scenario: `straight`, `bouncing`, `changed`, `dynamics_change` |
| Slider | steps per animation frame |

Matplotlib's own figure key bindings are released when the dashboard is built.
It binds `f` to fullscreen, `r` to home and `Right` to forward by default —
the same three keys documented above — so without that release, pressing `f`
froze the tiny network *and* threw the window into fullscreen at the same
time. Save, zoom and pan remain on the toolbar, and the window manager still
closes the window.

Pausing freezes the simulation completely: no environment transition, no
prediction, no update. If a hyperparameter is changed mid-run the session is
latched as `EXPLORATORY / MODIFIED DURING RUN`, because a run whose protocol
moved while it was running is not fixed-protocol evidence.

### Panels

- **world** — the bounded line, the actual dot, and every predictor's
  next-position prediction *as it was issued before that position was
  revealed*. Nothing is recomputed after the reveal. The TinyMLP's raw
  prediction is drawn separately whenever public reflection actually moved it.
- **network** — the real 2 → 4 → 1 model: its actual weights (blue positive,
  red negative, thickness by magnitude), its actual hidden activations, its
  actual biases. There are no decorative neurons.
- **error** — rolling normalized MAE per predictor, one linear axis shared by
  every series, with revealed law changes marked. Nothing is rescaled to
  flatter the network. Note that `f` freezes only the tiny network: the AAA RLS
  learner keeps updating, because it is shown as the project's current *online*
  learner. A frozen TinyMLP is therefore not a like-for-like comparison against
  it, and the panel says `FROZEN` in that state for exactly that reason.
- **state** — step, scenario, environment seed, model seed, online/frozen,
  learning rate, latest loss, update count, cumulative gradient norm, and the
  cumulative MAE of every arm.

Bounce and change labels are **evaluator display metadata**. They are attached
only after the corresponding observation has been revealed, and they never
reach a predictor.

## TinyMLP

```text
2 inputs  ->  4 tanh hidden units  ->  1 linear output

W1: 4 x 2 = 8      b1: 4
W2: 1 x 4 = 4      b2: 1
---------------------------
total             = 17 trainable parameters
```

### Why 17

The world is one dot on a line. A learner small enough to print in full is a
learner whose every weight, activation and gradient a reviewer can check by
hand, which is the same reason the scientific candidate has three parameters.
Backpropagation is written out explicitly rather than delegated to a framework,
and `tests/test_playground_neural.py` verifies it against central finite
differences for `W1`, `b1`, `W2` and `b2`. The parameter count itself has a
regression test.

### Inputs

```text
recent_displacement = (x[t] - x[t-1]) / (dt * speed_max)
centered_position   = (x[t] - midpoint) / interval_width
```

Causally available positions and public constants only. No velocity, no event
indicator, no scenario name, no change schedule, no oscillator coefficient, no
future observation, and no statistic computed from the complete episode.

### Ordering

```text
observe available history
-> predict the next observation
-> permanently record that prediction
-> advance the environment
-> reveal the next observation
-> score the pre-reveal prediction
-> update the online learner
-> append the revealed observation
```

`PlaygroundSession` in `session.py` owns this and has no Matplotlib import;
`rendering.py` only reads it. The renderer never causes learning.

### Public boundary reflection

The prediction is folded into the public bounds with
`aaa.predictors.reflect_prediction`, and the update target is unfolded with
`aaa.predictors.unfold_observation` so the regression target lives in the same
coordinate as the raw prediction. **Both are programmed public knowledge of the
observation format, not learned intelligence.** That is exactly why
`ReflectedConstantMotionPredictor` is the like-for-like baseline in every
comparison here: without it, a boundary transform is trivially mistaken for
learned bounce anticipation.

### Learning

Plain SGD on `loss = 0.5 * (prediction - target)^2`, where `target` is the next
displacement normalized by the interval width. Non-finite parameters,
gradients, activations, loss or predictions raise; a broken model is never
silently reset.

## The tiny neural AAA experiment

```bash
python -m playground.experiment --quick          # smoke form, truncated horizon
python -m playground.experiment                  # evaluation seeds
python -m playground.experiment --seeds development
python -m playground.experiment --select-learning-rate
```

`--quick` is a check that the protocol runs, at a truncated horizon. It is not
the evaluation result whichever seed group it is given; the recorded evaluation
is the full-horizon run in
[`evidence/evaluation_summary.json`](evidence/evaluation_summary.json).

For each seed: a fresh TinyMLP learns causally through a common pre-change
prefix under the damped oscillator's pre-change law; at the intervention its
complete state is cloned into `tiny_nn_frozen` and `tiny_nn_online`; both
branches then consume the *same* continued trajectory under changed
coefficients. Neither is ever told a change occurred — the evaluator knows the
change point only in order to branch and to report. The AAA RLS learner is
carried through the identical construction as a matched comparison. The clone
is hashed so a reviewer can confirm the two branches started from the same
numbers without trusting the runtime.

### Seed discipline

```text
development environment seeds:  701, 702, 703
evaluation  environment seeds:  711, 712, 713, 714, 715
model seed = 900000 + environment seed
```

Model seeds are offset far from environment seeds so the two can never collide,
and the model uses its own `np.random.default_rng`, never the environment's.

The learning rate was chosen on development seeds only. The declared grid is
`0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0`; implementation began at a conservative
`0.01`. Every attempt is recorded in
[`evidence/learning_rate_selection.json`](evidence/learning_rate_selection.json),
including `3.0`, which diverged loudly. The criterion, fixed before any
evaluation seed was run, is: among rates with no numerical failure and at least
one grid step below the smallest diverging rate, the lowest mean first-window
post-change MAE of `tiny_nn_online` across development seeds. That selected
**0.3**. The one-step stability margin exists because `1.0` scored better but
sits immediately below the divergence point, and a hyperparameter chosen there
is a gamble on whether an unseen seed blows up, not a choice.

### Interpretation limits

These are five different statements and none implies another:

1. the implementation works;
2. the model learns on some stream;
3. the model improves over its own frozen copy;
4. the model generalizes;
5. the model beats a baseline.

This experiment addresses (3) on one fixed toy protocol at five seeds, reports
(5) descriptively with no hypothesis test, and does not address (4) at all. The
correct form of words for a favourable result is:

> Under this fixed toy protocol, continued online updates reduced next-position
> error relative to an identical frozen neural state.

Not: that AAA understands motion, learned physics, is intelligent, or that the
neural network proves general adaptation. It does not.

One thing worth saying plainly: in a deterministic, noiseless world, reflected
constant-motion extrapolation is an extremely strong baseline, and the tiny
network loses to it by a wide margin. That is reported rather than hidden, and
the error panel is on a shared linear axis so it stays visible on screen too.

## Relationship to the rest of the repository

- **Benchmark v2.1** is untouched. No specification, batch, registry, freeze
  manifest, threshold, gate or historical result is modified by this package.
- **The observation-noise research phase** is untouched. This work is based on
  `main` and does not depend on, modify or reinterpret it.
- `aaa/` is imported read-only. Environment dynamics, bounds, configuration,
  reflection, the baselines, the RLS learner and the causal runner are reused,
  not reimplemented.

## Generated output

Runs write to `runs/playground/<label>/` (`runs/` is already git-ignored):

```text
runs/playground/<label>/summary.json
runs/playground/<label>/steps/seed-<n>-prefix.jsonl
runs/playground/<label>/steps/seed-<n>-changed-law.jsonl
```

`summary.json` carries the experiment identity, git commit, dirty-tree status,
the full configuration, both seed groups, the architecture and parameter count,
the learning rate, the public-boundary policy, per-seed metrics, aggregates
with per-seed variability retained, failures, and the claim boundaries above.

Only the committed files under [`evidence/`](evidence/) are kept in the
repository, and they are exploratory evidence, not acceptance evidence.

## Tests

```bash
MPLBACKEND=Agg python -m unittest tests.test_playground_neural \
    tests.test_playground_session tests.test_playground_experiment
```

The rendering smoke test runs under Agg and needs no display.
