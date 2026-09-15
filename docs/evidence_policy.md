# Evidence and artifact policy

## What lives in git

| Committed | Why |
|---|---|
| compact per-attempt `summary.json` and `report.md` | the result, readable without regenerating anything |
| `metadata.json`, `benchmark_spec.json` | provenance and the exact resolved specification |
| `checksums.json` | integrity of every byte retained in the compact archive |
| `full_attempt_checksums.json`, when present | identity of the complete local attempt, including omitted bytes; not clean-clone verification |
| `experiment_registry.json` | per-trial state, seeds and completion |
| selected checkpoints (a few kB of JSON) | the model the claim is about |
| `benchmarks/freeze_manifest.json`, `benchmarks/confirmation_batches.json` | what was frozen, and which batches were spent |
| `benchmarks/golden_seeds.json` | a fixture that detects accidental RNG-mapping changes |
| development evidence under `docs/evidence/` | diagnosis and candidate-selection tables, including unsuccessful variants |

## What does not

Compressed per-step record files (`raw/**/*.jsonl.gz`) and the plot images.
A full confirmation attempt writes roughly 100 MB of per-step records. The
previous repository committed tens of megabytes of generated JSON while calling
it compact evidence; this policy is the correction.

## How raw evidence stays recoverable

Raw step records are **semantically regenerable**, not archived, from four committed things:

1. public source at a recorded commit and tree hash;
2. the immutable specification, identified by hash;
3. the immutable seed mapping, keyed by batch identity;
4. the checkpoint lineage, identified by checkpoint hash.

```bash
git checkout <recorded commit>
python -m aaa.cli benchmark --role confirmation_a \
  --batch-id <recorded batch id> --reproduce --output-root runs
python -m aaa.cli recompute runs/benchmark-v2_1/<batch id>
```

`recompute` verifies byte checksums only when the files named by the manifest
are present. A regenerated run is then compared semantically (schemas, trial
identities, metrics and gates); byte equality is not promised for timestamps or
other runtime provenance. Byte integrity and semantic reproducibility are
deliberately reported as different claims.

CI additionally retains the full attempt directory for 90 days.

`tools/archive_attempt.py` creates the compact Git archive. It omits `raw/` and
`plots/`, retains the original complete-attempt manifest under the explicit
`full_attempt_checksums.json` name, and writes a new `checksums.json` that covers
every retained file. Thus a clean clone can verify the compact bytes without
pretending the omitted bytes are present.

Formal confirmation is run only in the canonical maintained checkout. The
manual Actions workflow accepts development and high-replication roles, but
refuses confirmation batch IDs: independent ephemeral checkouts cannot safely
persist the repository's single durable batch claim. This prevents two remote
jobs from spending the same planned stream while both believe they own it.

## Known limitation

Regenerability is weaker than durable archival. A sufficiently unlucky
combination — the recorded commit lost and the CI artifact expired — would
leave the summaries and checksums without the bytes they describe.

Git LFS, or an external archive with content addresses recorded here, would be
stronger. Neither is in place. This is stated as a recommendation rather than
described as done, and it is tracked as `AAA-077` in the issue ledger.

The four historical v2.1 attempt directories committed before the Sol review
have checksum manifests that name omitted raw and generated files. In a clean
clone, those manifests therefore fail integrity verification because the named
bytes are absent. Their summaries remain historical evidence, not currently
byte-verifiable archives. This reproduced limitation is tracked as `AAA-124`.

## Failed attempts

A confirmation that records a required-gate failure is evidence and is kept on
exactly the same terms as one that passes: the same compact summary, report,
provenance, checksums and checkpoints. Its batch is retired permanently in
`benchmarks/confirmation_batches.json`.

This has happened. The round-1 v2.1 pair failed `always_online_stability` on
both fresh streams and is committed at
[`../results/benchmark_v2_1/`](../results/benchmark_v2_1/) alongside the
round-2 pair.

## Historical evidence

`results/final/` and `results/benchmark_v2/` are preserved unchanged. They are
historical provenance. See [`errata.md`](errata.md) for which of their claims
were affected and why.

## Observation-noise v1.1

The observation-noise phase is additive and separately addressed. Its protocol,
source-freeze manifest, candidate ledger, registry, attempt metadata, schedule
files, primitive records, plots, reports, and checksums live under the names
declared in [`observation_noise_protocol.md`](observation_noise_protocol.md).
An attempt retains deterministic per-trial `records/*.jsonl.gz` shards and a
canonical index rather than duplicate combined compressed and uncompressed
copies. The manifest hashes the concatenated uncompressed record bytes in
canonical trial/step order, while every shard also has a compressed-byte hash.
A compact summary without the full schedule and primitive archive cannot
support an independent-reproduction claim.

Noise records contain both latent truth and predictor-visible observations, but
the predictor-generation boundary receives only the latter. The presence of a
truth field in evaluator evidence is not permission for a predictor or
calibration callback to read it. A report must state whether a result is
engineering complete, scientifically supported, negative, inconclusive, or
blocked by missing evidence.

### Observation-noise completion boundary

The scientific identity used for a future confirmation freeze is the output of
`observation-noise-fingerprint`, not a commit/tree equality check. It covers
the scientific source map, including the protocol, candidate ledger and plan,
runner, statistics, verifier, tests, lock and relevant documentation. It
normalizes only mutable registry lifecycle fields and fails closed on
nonignored untracked scientific files, unsafe paths and symlinks. The exact
confirmation freeze, design/source freeze, result archives and review handoff
are generated provenance and are excluded so their own commit does not create a
self-reference.

The committed development selection evidence is
[`evidence/observation_noise_development_selection.json`](evidence/observation_noise_development_selection.json),
and the full candidate ledger retains every attempted identity and outcome.
The selected result is the unchanged incumbent control. Four full 2 x 2 x 1
development archives were independently verified before that decision. Their
locator is still explicitly transient local evidence; it is not a durable
confirmation archive.

The joint command
`python -m aaa.cli observation-noise-confirmation-evaluate <A> <B>` is the only
supported route to a final A+B conclusion. It independently verifies both
primitive archives, checks shared training/checkpoint and candidate identity,
recomputes the endpoint family, records Holm order and adjusted bounds, and
does not trust a stored summary conclusion. No durable archive mechanism is
approved or recorded for this phase. Formal A/B therefore remains unobserved
and the confirmation boundary is `BLOCKED` until an immutable destination is
supplied, development upload/retrieval is proved, and the fresh v1.1 batches
are legitimately spent.
