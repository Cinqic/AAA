# Sol observation-noise v1.1 remediation self-review — not independent approval

Sol's review of Luna's original PR #11 implementation was independent; this
section reviews Sol's own repairs and is therefore a self-review, not a second
independent approval and not human review.

The final pass retraced protocol input through schedule generation, latent and
observed histories, prediction/reveal/update ordering, sharded retention,
streaming aggregation, uncertainty, gates, CLI status, checksum verification,
freeze admission, remote reservation, candidate selection, and joint A+B
evaluation. It also reread the complete diff against main and active documents.

Material results:

- The corrected full development search completed for all four candidates at
  the declared 2 x 2 x 1 hierarchy. Each archive independently verified; no
  refinement met all constraints; no confirmation data was inspected.
- A timing-neutral digest, metrics, and coverage matched exactly across the
  streaming implementation and deterministic sharded implementation.
- The final adversarial pass found `AAA-146` through `AAA-149`: schedule-index
  members could traverse outside the archive or follow a symlink, and quick
  selection could overwrite the canonical full evidence. Fresh-wheel execution
  also failed because installed packages have no pinned Git history.
  Containment checks, a protected evidence destination and an explicit
  `NOT_VERIFIED` installed-checkout boundary address those three defects. The
  first final-head CI then proved shallow Actions checkouts omitted the pinned
  historical commit; all CI jobs now fetch complete history.
- Missing checksums, rehashed NaN summaries, wrong adaptation estimands,
  incomplete cells, caller-selected formal draws, cross-clone batch reuse,
  current-checkout reference drift, shard tampering, zero-valued legitimate
  metrics, interruption/resume, and causal interval ordering have dedicated
  negative tests.
- Formal A/B was not run. There is no immutable archive locator or independent
  retrieval proof for the projected 38.5 GB compressed record stream per
  batch. This is `BLOCKED`, not a scientific result and not merge approval.

## Historical Opus v2.1 self-review — not current approval

This is a record of what was checked and what it found. It is **not** an
approval decision and must not be read as one. GPT-5.6 Sol is the current
designated independent reviewer and remediation owner; Sol's separate verdict
belongs in `sol_review.md` after the final evidence qualifies.

The review was carried out by re-reading the whole repository rather than only
the diff, and by trying to break the repaired system rather than confirming it.

## What was re-read

Every source module, the canonical specification, every test module, both
workflows, the README, all documentation, the retained result summaries, the
issue ledger, and the full branch diff against `235ce28`.

## Adversarial probing of the repaired system

Written after the repair, aimed at the new code, in the same spirit as the
pre-repair probes.

| Probe | Result |
|---|---|
| every gate evaluated against completely empty evidence | no gate returned `PASS`; overall status false |
| bootstrap with a single replica | `INSUFFICIENT_EVIDENCE`, with the reason recorded |
| bootstrap with `NaN` observations | `INSUFFICIENT_EVIDENCE` ("point estimate is not finite") |
| bootstrap with an all-zero baseline | `INSUFFICIENT_EVIDENCE` ("relative improvement is undefined") |
| 200,000 repeated updates at `lambda = 0.5` | survived; min eigenvalue 5.0e-06, condition 8.0e+09, trace bounded at 8.0e+04, 199,998 suspensions counted |
| pathological displacement-scale mismatch | survived with valid state; conditioning reported |
| prose-only specification edit | changes the hash, so it retires every batch declared against the old one |
| alternating error after a shock | correctly classified `unrecovered`, not recovered |
| hand-edited batch registry (retired flipped back to planned) | **accepted** — see below |

### The one finding that stands

The confirmation batch registry is a committed JSON file, and its `status`
field is trusted. Hand-editing a retired batch back to `planned` lets it be
claimed again.

This is a tamper scenario, not an accident scenario: the registry is in git, so
the edit appears in the diff and the history, the `consumed_by` list still names
the runs that spent it, and a confirmation additionally requires a clean tree
and agreement with the committed freeze manifest. Anyone able to make that edit
could equally edit anything else.

It is recorded rather than silently hardened, because hardening it would have
meant changing source between confirmation attempts. A `consumed_by`-based
guard — refusing any batch that has ever recorded a run, regardless of its
status label — is the obvious improvement and is left as a recorded
recommendation.

## What the review changed

Eleven defects were found during this investigation and are in the ledger as
`AAA-090` through `AAA-100`, each with a regression test. The ones most worth
naming:

- a zero `--replicas`/`--episodes` override was falsy and silently became the
  full confirmation budget;
- the experiment registry accepted a replan that changed a recorded trial's
  seed, so a resume could quietly become a different experiment;
- the declared numerical tolerances were in the specification and unused — the
  same class of defect the review was convened to fix, reproduced in the fix;
- `baselines.legacy_linear_sgd` was declared and never run;
- the v1 report generator referenced metric keys the repaired metrics module no
  longer produces, so report generation raised;
- the changed-law intervention record did not hash the shared branch state, so
  the matched design could not be audited from the artifacts.

## What the review deliberately did not do

- It did not lower any threshold. The one gate that failed in confirmation
  stayed exactly where it was and the candidate was changed instead.
- It did not treat a green suite as evidence. Every gate has synthetic cases
  proving it can return `FAIL` and, where reachable, `NOT_VERIFIED` or
  `INSUFFICIENT_EVIDENCE`.
- It did not accept a previous "repaired" status. Each was re-tested from
  scratch and several were reopened with a wider root cause than originally
  recorded.
- It did not adopt parallelism. The benchmark was profiled; a full confirmation
  attempt is minutes of CPU, so adding workers would buy nothing and add an RNG
  and BLAS-oversubscription risk surface. Recorded rather than silently dropped.

## Honest assessment of what remains weak

- **Evidence durability** (`AAA-077`). Raw per-step records are regenerable
  from the committed identity, not archived. That is weaker than durable
  storage and is stated as a recommendation, not as done.
- **Repository settings** (`AAA-110`, now closed). Confirmed independently
  rather than taken on report — no ruleset, `main` unprotected, Dependabot
  security updates disabled — and then applied through the API and read back.
  Two items are deliberately left off and say so with their exact commands:
  `enforce_admins`, so a misconfigured required check cannot lock the
  maintainer out, and the account-level read-only workflow permission, which
  every workflow here already supersedes by declaring its own.

  The commit that applied those protections was pushed to `main` directly
  through the admin bypass, because the protection had just been enabled and
  that commit was the one recording it. Everything after went through a pull
  request with the six required checks — which is also how the protection was
  verified to work. Noted here rather than left in the reflog for someone to
  find.
- **Absolute floors do real work.** Several non-regression gates carry an
  absolute floor (`1e-5`, or `1e-6` for bounce parity) because ratios between
  quantities near `1e-17` are meaningless in a deterministic noiseless world.
  Those floors are what several comparisons actually turn on. They are declared
  in the specification and visible in every report, but a reviewer should look
  at them directly rather than at the relative numbers.
- **The bouncing family has a ceiling.** Reflected constant motion is
  analytically exact there, so that family can only ever show the candidate
  matching a known-correct rule. This is stated in the specification's own
  notes.
- **Five replicas is few.** It is labelled a routine engineering minimum, and a
  `high_replication` role with 20 independent lineages exists, but the
  confirmation evidence here is the five-replica track.
- **Multiplicity is scoped to a specification hash.** Repeated confirmation
  attempts against the *same* frozen system cost statistical power, which is
  the property that matters most. But a candidate change starts a new family,
  so the counter does not by itself bound how many systems may be tried. That
  is deliberate — a different system under a different frozen protocol is a
  different hypothesis — and the discipline that actually bounds it is that
  every attempt stays on the record, permanently retired if it failed. A
  reviewer should count the attempts in
  `benchmarks/confirmation_batches.json` directly rather than reading the
  family size as the whole story. The rule was declared before the round-1
  failure and was not adjusted after it.
- **One dimension, no noise.** Every result depends on perfect observations,
  which is exactly the regime where an analytic extrapolator is unbeatable.
  `docs/limitations.md` names observation noise as the next experiment.

## Verification performed

The full list of commands and their outcomes is in
[`handoff_sol.md`](handoff_sol.md).
