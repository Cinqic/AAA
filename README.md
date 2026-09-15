# AAA — Accurate Autonomous Adaptation

An experimental research project asking one deliberately small question:

> Can an autonomous system observe an environment, learn to predict what
> happens next, detect when its existing knowledge is no longer adequate, adapt
> from subsequent evidence, and improve its future predictions?

The environment is one moving dot on a line. The learner has three parameters.
Neither is an accident: the point of this version is not a capable system, it
is a **measuring instrument you can check**.

AAA makes no claim to general intelligence, physical understanding, or
independent goal formation. "Autonomous" here means the observe / predict /
score / update loop runs unattended after launch, and nothing more.

## Why the benchmark looks the way it does

A previous version of this benchmark reported that all required gates passed.
Three independent reviews then found that several of those gates could not have
failed. Among other things:

- a **parameterless analytic formula** satisfied the gate named
  `straight_learning`;
- a run with **zero** required strata passed the coverage gate, because
  `all()` of an empty collection is `True`;
- the candidate was allowed to reflect its prediction into the public bounds
  while the baseline it was compared against was not — and essentially the
  entire reported bounce advantage was that transform, not learned parameters;
- the reproducibility gate compared a deterministic function with itself;
- a confirmation run that recorded failed gates still exited 0;
- `475 / 475` recovery counted only events whose 50-step average stayed
  elevated, so large, fast shocks were diluted out of the denominator;
- 26 values in the "frozen specification" were never read by the code.

Every one of those was reproduced against the old tree before anything was
changed. The probes are committed at
[`docs/evidence/pre_repair_probes.json`](docs/evidence/pre_repair_probes.json)
and each defect is tracked in
[`docs/issue_ledger.md`](docs/issue_ledger.md).

The current protocol is built so that it can say **no**:

| Status | Meaning |
|---|---|
| `PASS` | the required property was measured and holds |
| `FAIL` | it was measured and does not hold |
| `NOT_VERIFIED` | the check did not run |
| `INSUFFICIENT_EVIDENCE` | there was not enough evidence to decide |

Only `PASS` satisfies a required gate, and formal confirmation exits non-zero
on anything else. Absence of evidence is never turned into success.

**It has already said no.** The first confirmation round under the repaired
protocol failed: both independent fresh streams failed the required
`always_online_stability` gate and both exited non-zero. Thirteen of fourteen
gates passed in each. The failure was traced on development data to a specific
mechanism, repaired in the candidate, and the failed attempts are committed
alongside everything else. No threshold was touched. See `AAA-120` in
[`docs/issue_ledger.md`](docs/issue_ledger.md).

## What is implemented

- A seeded, bounded CPU simulator: constant velocity, reflection at the
  boundaries, unannounced speed changes, and a damped harmonic oscillator whose
  coefficients change mid-episode without notice.
- A strict temporal boundary — predict, record, advance, reveal, score, then
  update — with tests that fail if a scenario name, event flag, velocity,
  change schedule, hidden coefficient or future observation reaches a
  predictor.
- A baseline suite where each member isolates one source of predictive power,
  including a **reflected constant-motion** baseline that uses exactly the same
  public boundary map the candidate may use.
- A three-parameter square-root recursive-least-squares candidate with
  trace-bounded, self-triggered forgetting, verified against an independent
  batch least-squares reference.
- The historical v1 SGD learner, retained and reported as a named diagnostic
  arm rather than quietly replaced.
- Benchmark v2.1: a typed, hash-identified, fully executable specification;
  planned stratified coverage; a paired hierarchical bootstrap that recomputes
  each gate's own statistic; predeclared confirmation batches with a registry
  and a freeze manifest; an experiment registry with safe resume; and
  independent recomputation of every metric and gate from retained raw
  evidence.
- Observation-noise v1.1: a separately versioned sensor model with cached
  schedules, latent/observed field separation, causal noisy-target updates,
  matched intervention branches, deterministic sharded evidence, and an
  independent reference verifier. The corrected full 2 x 2 x 1 development
  search evaluated four candidate identities and retained the unchanged
  incumbent as an explicit no-refinement control. Formal A/B remains blocked
  on an approved immutable archive/retrieval destination; development evidence
  is not a confirmation result. See
  [`benchmarks/observation_noise_candidate_ledger.json`](benchmarks/observation_noise_candidate_ledger.json).

## The learner

Feature vector, from the last four observed positions only:

```text
phi = [ 1,  (x[t] - x[t-1]) / (dt * speed_max),  (x[t] - midpoint) / L ]
```

It predicts the next displacement normalized by the interval width `L` and adds
it to `x[t]`. Parameters start at zero, so before any learning it predicts
persistence.

Boundary reflection is **programmed public knowledge of the observation
format**, not a learned capability. That is precisely why the reflected
constant-motion baseline exists: so the transform is never counted as
intelligence. The report decomposes any apparent bounce advantage into analytic
extrapolation, public boundary handling, and learned parameters, separately.

The covariance is propagated as a square-root factor, so symmetry and positive
semidefiniteness hold by construction, and forgetting is suspended when the
covariance trace would exceed its declared bound — with suspensions counted and
reported. The previous covariance-form learner lost positive semidefiniteness
after 323 updates on a slow constant-velocity stream and overflowed at 6,642 on
a stationary one.

## Install

```bash
git clone https://github.com/Cinqic/AAA.git
cd AAA
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python tools/check_lock.py
```

CPU only. NumPy and Matplotlib are the runtime dependencies. No GPU, no
external API, no pretrained model, no paid service.

## Commands

```bash
python -m unittest discover -s tests -t .        # the test suite
python -m aaa.cli spec-hash                      # canonical specification identity
python -m aaa.cli observation-noise-protocol-hash # separate noise protocol identity
python -m aaa.cli observation-noise-fingerprint    # scientific source identity
python -m aaa.cli observation-noise-development-select --quick \
    --output runs/development-selection/observation_noise_development_smoke.json
python -m aaa.cli observation-noise --role development --quick \
    --attempt-label noise-smoke-001 --output-root runs
python -m aaa.cli observation-noise-recompute runs/observation-noise-v1_1/noise-smoke-001
python -m aaa.cli observation-noise-confirmation-evaluate <A> <B> \
    --output docs/evidence/observation_noise_joint_evaluation.json
python -m aaa.cli benchmark --role development --attempt-label dev-001 \
    --replicas 2 --episodes 3 --output-root runs
python -m aaa.cli recompute runs/benchmark-v2_1/dev-001
python -m aaa.cli diagnose --output docs/evidence/diagnosis
python -m aaa.cli select-candidate
python -m aaa.cli animate --checkpoint runs/<attempt>/checkpoints/replica-00.json
```

Formal confirmation requires a predeclared batch, a committed freeze manifest,
the selected candidate ledger entry, the exact lock, and a matching
non-self-referential scientific fingerprint; see
[`docs/reproduction.md`](docs/reproduction.md). The confirmation evaluator
reconstructs the complete primary A+B claim family and applies one Holm
adjustment without trusting stored conclusions.

## Interpreting a result

These are five different claims, and none of them implies another:

1. the implementation runs correctly;
2. the model improves with experience;
3. what it learned generalizes to unfamiliar episodes;
4. continued updates help after a change;
5. it beats a baseline.

Constant-motion extrapolation is an extremely strong baseline in a
deterministic, noiseless world — with the public boundary map it is *exact* at
bounce transitions. A learned model losing to it is a valid and informative
result, and this repository is built to report that rather than to avoid it.

## Documentation

| Document | What it covers |
|---|---|
| [Benchmark protocol](docs/benchmark_protocol.md) | the active v2.1 protocol, gates, statistics and confirmation discipline |
| [Observation-noise protocol](docs/observation_noise_protocol.md) | the separately versioned sensor study, causal boundary, schedules, replication and limits |
| [Issue ledger](docs/issue_ledger.md) | every defect: reproduction, root cause, repair, regression test, status |
| [Errata](docs/errata.md) | which earlier claims were affected, and why |
| [Candidate selection](docs/candidate_selection.md) | why this candidate, with the full table including what lost |
| [Diagnosis](docs/diagnosis.md) | controlled ablations on the legacy learner |
| [Reproduction](docs/reproduction.md) | exact commands from a clean checkout |
| [Evidence policy](docs/evidence_policy.md) | what is committed, what is regenerable, and the known limitation |
| [Experiment registry](docs/experiment_registry.md) | trial state, resume semantics, verification |
| [Limitations](docs/limitations.md) | what this cannot show and the current confirmation blocker |
| [Research history](CHANGELOG.md) | v1, v2, v2.1 and observation-noise v1/v1.1 |
| [Self-review](docs/self_review.md) | what was checked after the repair, and what stayed weak |
| [Review handoff](docs/handoff_sol.md) | identity, confirmation outcomes, reproduction commands |

## Historical evidence

[`results/final/`](results/final/) is the v1 snapshot, preserved unchanged. Its
result was unfavourable — the original learner did not convincingly improve,
generalized poorly, and lost to constant motion — and that is exactly why it is
kept.

[`results/benchmark_v2/`](results/benchmark_v2/) holds the v2 confirmation
attempts. They are **superseded**: historical and provisional evidence produced
under a methodology since found defective. They are not acceptance evidence for
anything.

## License

Apache License 2.0. See [LICENSE](LICENSE).
