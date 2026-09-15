# Reproduction

Every command below is run from a clean checkout. No private filesystem path is
required.

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

`tools/check_lock.py` prints the dependency-lock hash and fails if the installed
set drifts from the pins. CI runs the same check, so "the locked environment"
means the environment that was actually exercised.

## Verification suite

```bash
python -m unittest discover -s tests -t . -v
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m coverage run -m unittest discover -s tests -t . && python -m coverage report
python tools/check_exit_codes.py
```

Set `MPLBACKEND=Agg` in a headless environment.

## Identity

```bash
python -m aaa.cli spec-hash          # canonical specification path, version, hash
python -m aaa.cli batches            # declared confirmation batches and their status
python -m aaa.cli observation-noise-protocol-hash  # separate v1.1 noise protocol identity
```

The v2.1 command and its specification are unchanged. Observation-noise runs
use a separate namespace and output root:

```bash
python -m aaa.cli observation-noise --role development --quick \
  --attempt-label noise-smoke-001 --output-root runs
python -m aaa.cli observation-noise-recompute \
  runs/observation-noise-v1_1/noise-smoke-001
```

An observation-noise attempt writes each completed trial to an immutable,
deterministic gzip shard under `trial_records/` before moving to the next trial.
Finalization atomically renames that directory to `records/`, writes the
canonical shard index, and retains no duplicate combined primitive file. If the process is
interrupted, resume the same attempt with its exact label:

```bash
python -m aaa.cli observation-noise --role development --quick \
  --resume --attempt-label noise-smoke-001 --output-root runs
```

Resume refuses a finalized attempt, requires the existing label, regenerates
only deterministic missing inputs, and compares any replayed trial byte for
byte with the retained shard. A divergent replay is an error; it is never
silently merged into the archive. Failed attempts and their lifecycle records
remain in place for inspection.

The quick command exercises every registered stationary channel and scale,
both training conditions, direct realized stratum identities, matched
changed-law branches, unchanged/changed-dynamics sensor-shift controls at both
predeclared timings and directions, schedule hashes, primitive records, plots,
and the independent reference verifier. The plot bundle includes uncertainty,
matched first-error trajectories, noise-only controls, worst realized strata,
detector/update diagnostics, and `plot_provenance.json` binding each figure to
the retained primitive record hash. Its scientific endpoints remain
`INSUFFICIENT_EVIDENCE`; it is an engineering smoke fixture, not confirmation.
Formal A/B execution requires the separate source-freeze manifest, declared
batch identities, the exact lock, the fixed ten-lineage plan, durable full
archive retention/retrieval, and independent recomputation.

## Installed-wheel boundary

The wheel contains both protocol files and supports development mechanics from
outside a checkout. The isolated zero-noise comparison against the pinned v2.1
Git commit necessarily requires the maintained repository history. An installed
development smoke records that check and its reference gate as `NOT_VERIFIED`;
it never substitutes an in-process comparison or reports a false `PASS`.
Formal confirmation is maintained-checkout-only and refuses an installed
package without the committed freeze and Git provenance.

The non-self-referential scientific identity can be inspected before a freeze:

```bash
python -m aaa.cli observation-noise-fingerprint
```

It covers the scientific source map and fails closed on nonignored untracked
scientific files or symlinks. Only mutable batch lifecycle fields are
normalized. Generated result, freeze and review files are excluded so writing
the exact freeze cannot alter the fingerprint it records.

Before any candidate comparison, run the committed bounded development plan:

```bash
python -m aaa.cli observation-noise-development-select --quick \
  --output runs/development-selection/observation_noise_development_smoke.json \
  --runs-root runs/development-selection
```

This evaluates all four catalogued IDs, retains every attempt, and updates the
candidate ledger. The current selected identity is the unchanged incumbent
control, which is a valid no-refinement result. A confirmation invocation does
not accept a candidate override; it resolves the ID from the committed
confirmation freeze and checks the ledger's selected entry and configuration
hash.

The design/source freeze is written only after the implementation is committed:

```bash
python -m aaa.cli observation-noise-freeze \
  --batch observation-noise-a-0002 --batch observation-noise-b-0002 \
  --notes "design freeze for observation-noise v1.1"
```

After bounded development, a confirmation freeze is a separate file and must
name the selected candidate explicitly. The command refuses to create that
file without a candidate identity. A no-refinement incumbent selection is a
legitimate candidate identity; it does not waive the freeze, full archive, or
joint-analysis requirements.

## Development work

Development runs may use small overrides and may record failed gates. They exit
0 regardless, because exploration is allowed to fail.

```bash
python -m aaa.cli benchmark --role development --attempt-label dev-001 \
  --replicas 2 --episodes 3 --output-root runs
```

Development-only evidence:

```bash
python -m aaa.cli diagnose --output docs/evidence/diagnosis
python -m aaa.cli select-candidate --output docs/evidence/candidate_selection.json
```

## Formal confirmation

Confirmation refuses a custom specification, an undeclared or already-consumed
batch, weakened minimums, a dirty source tree, or drift from the committed
freeze manifest. It exits non-zero if any required gate is not `PASS`.

```bash
python -m aaa.cli declare-batch <unused-a-id> --role confirmation_a
python -m aaa.cli declare-batch <unused-b-id> --role confirmation_b
python -m aaa.cli freeze \
  --batch <unused-a-id> \
  --batch <unused-b-id>
git add benchmarks/freeze_manifest.json benchmarks/confirmation_batches.json
git commit -m "Freeze the confirmation plan"

python -m aaa.cli benchmark --role confirmation_a \
  --batch-id <unused-a-id> --output-root runs
git add benchmarks/confirmation_batches.json results/benchmark_v2_1/<unused-a-id>
git commit -m "Record confirmation A evidence"

python -m aaa.cli benchmark --role confirmation_b \
  --batch-id <unused-b-id> --output-root runs
```

Nothing may be tuned between A and B. A failed attempt is kept, its batch is
retired permanently, and a new predeclared batch is required for a fresh
attempt.

This has already happened. The round-1 pair failed and was retired; the
candidate was repaired on development evidence, which changed the specification
hash, which in turn meant round 2 needed newly declared batches. A batch
declared against one specification hash is refused under another. Both rounds
are in `benchmarks/confirmation_batches.json` and `results/benchmark_v2_1/`.

Observation-noise A and B are evaluated together only after both full archives
exist:

```bash
python -m aaa.cli observation-noise-confirmation-evaluate \
  runs/observation-noise-v1_1/<confirmation-a-attempt> \
  runs/observation-noise-v1_1/<confirmation-b-attempt> \
  --output docs/evidence/observation_noise_joint_evaluation.json
```

The command enumerates the complete primary family, applies one Holm
adjustment across A+B, and returns non-zero on any failed or missing required
claim. It is not a substitute for durable archive publication or independent
review.

A higher-replication track with 20 independent training lineages:

```bash
python -m aaa.cli benchmark --role high_replication --output-root runs
```

## Recomputing evidence without rerunning anything

```bash
python -m aaa.cli recompute runs/benchmark-v2_1/<attempt>
```

This verifies checksums and the specification hash, rebuilds every per-step
metric, episode and replica summary, interval and gate from the retained raw
records, and exits non-zero if a stored gate status fails to reproduce. It never
retrains and never re-simulates.

## Regenerating raw evidence for a recorded attempt

```bash
git checkout <recorded commit>
python -m aaa.cli benchmark --role confirmation_a \
  --batch-id <recorded batch id> --reproduce --output-root runs
python -m aaa.cli recompute runs/benchmark-v2_1/<recorded batch id>
```

The original manifest verifies the original bytes only while all named files
are available. A regenerated attempt must instead match the recorded scientific
identity, trial design, metrics and gates. Runtime timestamps and provenance can
differ, so regenerated whole-directory checksums are not claimed to be equal.

## Historical v1 track

```bash
python -m aaa.cli smoke --output-root runs
python -m aaa.cli full --output-root runs
```

Preserved for regression and provenance. Not acceptance evidence. Nothing is
written outside the requested output root.

## Optional animation

```bash
python -m aaa.cli animate --checkpoint runs/<attempt>/checkpoints/replica-00.json \
  --scenario changed --seed 201
```

Space pauses and resumes, `r` restarts. A display is required for this command
and for nothing else; its temporal semantics are tested headlessly.

## Hardware

No GPU is required and none is used. The model has three parameters. Every
attempt records the CPU model, core count, memory, OS, Python, NumPy, BLAS
build metadata and available disk space, so a latency number can be read in
context.
