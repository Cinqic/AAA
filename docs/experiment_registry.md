# Experiment registry and resume

Output-directory existence is not a notion of experiment state. Each attempt
carries `experiment_registry.json`, a durable per-trial record.

## States

| State | Meaning |
|---|---|
| `PLANNED` | identity and seed allocated; nothing run |
| `RUNNING` | started; not yet complete |
| `COMPLETE` | finished, outputs written, checksums recorded |
| `FAILED` | failed, with the stage and error preserved |
| `INTERRUPTED` | was `RUNNING` when the registry was reloaded |
| `SUPERSEDED` | replaced by a later trial with the same identity |

## Per trial

Immutable identity (trial id, family, branch, replica, episode, environment
seed), state, output paths, per-file checksums, checkpoint hash, start and end
times, failure stage and error message.

## Resume

```bash
python -m aaa.cli benchmark --role development --attempt-label dev-001 \
  --output-root runs --resume
```

A resume:

- does not re-run completed trials;
- **cannot change a completed trial's seed** — replanning an existing trial
  with a different family, branch, replica, episode or environment seed raises,
  because that is a different experiment wearing the same name;
- does not overwrite existing evidence;
- preserves failure records;
- marks trials that were `RUNNING` at the previous exit as `INTERRUPTED` rather
  than silently assuming they finished.

## Verification

`verification/` in each attempt holds the executed reproducibility evidence: the
deterministic rerun comparison, the save/resume equivalence probe, and the
golden seed-mapping check. The correctness report records whether every expected
trial exists, whether trial ids are unique, whether record counts match, whether
frozen branch state was measured unchanged, whether the online branch actually
updated, whether checkpoints load, whether checksums verify and whether the
stored summary recomputes from raw evidence.

A malformed, partial or interrupted run cannot earn a correctness `PASS`.

## Observation-noise registry

The observation-noise phase does not reuse the v2.1 registry. Its protocol
declares `benchmarks/observation_noise_registry.json` for batch declarations and
each attempt retains an atomic `lifecycle.jsonl` with `created`, `started`,
`consumed`, `completed`, `cancelled`, and `failed` transitions. Schedule and
trial identities are allocated before the first observation. Existing attempt
directories are never overwritten; an interrupted run is a preserved failure
until a verified resume implementation reuses its original schedule and
checkpoint state. Confirmation additionally creates an append-only remote Git
reservation ref before attempt creation; normal atomic ref creation permits one
claimant across separate clones and exact-owner resume only.

The candidate ledger is a separate immutable-identity registry. Its entries
contain candidate ID, parent, mechanism, exact parameters and configuration
hash, trainable/fixed state, training condition, mechanism count, all
development attempt IDs, and the outcome/reason. The selected candidate for a
confirmation run is resolved from the committed confirmation freeze and ledger
entry; a command-line candidate override cannot substitute for it.

The development plan is committed before comparative work and limits the
search to 12 configurations and three substantive mechanism changes. The first
quick search was development smoke and could not validly complete the frozen
selection plan. A fresh full 2 x 2 x 1 selection under v1.1 retained and
independently verified all four archives; no refinement was eligible, so the
unchanged incumbent is the explicit no-refinement control. The old quick
attempts remain historical evidence. A future pair of full
confirmation archives is evaluated jointly so A/B do not receive separate
multiplicity corrections.
