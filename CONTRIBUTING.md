# Contributing to AAA

AAA is a small research prototype whose whole point is that its measurements can
be trusted. Contributions are welcome; the conventions below exist to keep that
property.

## Before you change anything

```bash
python3 -m venv .venv && . .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m unittest discover -s tests -t .
```

## The rules that matter most

**Reproduce before you repair.** If you believe something is wrong, write a
probe that demonstrates it against the current code first, and keep that probe
as a regression test. Every entry in [`docs/issue_ledger.md`](docs/issue_ledger.md)
follows this shape.

**A gate must measure what its name says.** If a criterion can be satisfied by
something that does not have the property being claimed — a parameterless rule
passing a "learning" gate, an empty collection passing a coverage gate — the
gate is broken even when it is green.

**Absence of evidence is not `PASS`.** A missing interval, a missing stratum, an
unrun check: `NOT_VERIFIED` or `INSUFFICIENT_EVIDENCE`. Never silently
satisfied.

**Never tune a threshold to a result you have seen.** Thresholds are frozen
before confirmation batches are generated. If a criterion turns out to measure
the wrong thing, replace the *concept* and document the scientific reason —
and do it on development data.

When a confirmation fails, the failure is the deliverable. Keep it, retire its
batch, diagnose on development data, change the *system* if the diagnosis
warrants it, declare a fresh batch, and let the multiplicity accounting charge
you for the extra attempt. This has happened once already and the whole chain is
in `AAA-120`; follow that shape.

**Keep confirmation streams clean.** Any confirmation stream that has been
looked at is contaminated for model selection forever. Do selection on
development data.

**Every specification value must be read.** `tests/test_spec.py` fails if a
declared leaf stops being consumed. If you add a value to the specification,
use it.

**Keep the causal boundary.** Predict, record, advance, reveal, score, then
update. No scenario name, event flag, velocity, change schedule, hidden
coefficient or future observation may reach a predictor.

**Failures stay visible.** A numerical problem raises. A failed confirmation is
retained and its batch retired. Nothing is quietly reset, clipped, symmetrized
or dropped.

## Checks that must pass

```bash
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m unittest discover -s tests -t .
python tools/check_lock.py
python tools/check_exit_codes.py
```

`unittest` is the project's test framework. Please do not introduce a second
one. No blanket `# type: ignore`: if the checker objects, the annotation is
usually the thing that is wrong.

## Adding a gate

1. Declare it in `aaa/benchmark/data/benchmark_v2_1.json` with its evaluator,
   `required` flag, description and thresholds.
2. Implement the evaluator in `aaa/benchmark/gates.py`.
3. Add deterministic synthetic cases in `tests/test_gates.py` proving `PASS`,
   `FAIL`, and `NOT_VERIFIED` or `INSUFFICIENT_EVIDENCE` where reachable.
4. Changing the specification changes its hash, which retires every batch
   declared against the old one. That is intentional.

## Scope

Observation noise is implemented only in the separately versioned
`aaa.observation_noise.v1.1` phase. Do not fold it into benchmark v2.1 or change
either protocol's frozen thresholds, samples, endpoints, or historical evidence.
Larger models, extra dimensions, process noise, missing observations and other
research directions remain out of scope. See
[`docs/limitations.md`](docs/limitations.md) for the current evidence boundary.
