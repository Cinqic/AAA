# Self-review: trying to reject this branch

Written as if the goal were to find a reason to refuse it. Everything below that
is a real defect was fixed inside the prototype; everything that is a real
limitation is recorded rather than hidden.

**I am not approving this work.** A self-review is not independent review.

---

## Defects found and fixed during the work

| # | Defect | Fix |
|---|---|---|
| 1 | The manifest-traversal test asserted `assertRaises((UnsafePathError, IntegrityError))`. It was passing on the *integrity* branch — the tampered manifest no longer matched its digest — so **the path guard it claimed to test was never reached.** A test passing for the wrong reason. | Rewritten to build a canonically re-encoded manifest with an updated locator digest, so every integrity check passes and only the path guard can stop it. It now also asserts the escaped file was not written. |
| 2 | `_online_weights` in the diagnostic contained three dead reassignments of `latent`, leftovers from an abandoned construction. Arithmetically harmless, but unreadable and an invitation to misread what was measured. | Rewritten as a single sinusoidal path chosen so no wall policy activates, which is also what makes the measurement clean. |
| 3 | `DurableLocator` was imported lazily inside four methods, leaving `F821` undefined-name errors under Ruff and making the type contract invisible. No circular import justified it. | Imported at module scope. |
| 4 | `mypy` could not check the prototype at all — the same file resolved under two module names. | Added `research/__init__.py` and `research/aaa134/__init__.py`. Confirmed `find_packages(include=["aaa*"])` still yields only the three `aaa` packages, so packaging is unaffected. |

## Findings against the assignment's own rejection checklist

**Fake durability.** The locator carries `durable: False` as a *field*, the
manifest carries it too, `verify_retrieval` **fails** an archive whose manifest
claims durability, and a test asserts it. The word `PROTOTYPE_ONLY_NOT_DURABLE`
appears in the package docstring, the backend, the locator, the manifest and the
recorded run. No production-shaped locator exists anywhere in the branch.

**Accidental reliance on local paths.** `retrieve_archive` refuses a non-empty
destination, so a retrieval cannot inherit bytes from the original attempt
directory. `verify_retrieval` reads only the retrieved tree and the backend.
Tested by `test_retrieval_refuses_a_non_empty_destination`.
*Residual:* the local backend's locator `root` field is an absolute path on the
producing machine. That is honest for a local backend — it is exactly what makes
it non-durable — but a real adapter must emit a bucket/key/version triple, not a
filesystem path. Not implemented.

**Hidden duplicate storage.** `test_no_second_local_copy_is_made_under_the_attempt_root`
asserts the attempt tree is byte-for-byte unchanged in file set after publishing.
*Residual:* it checks the attempt root only. A backend rooted elsewhere on the
same disk still doubles total disk use. For a real object store that is moot; for
the local backend it is inherent.

**Whole-archive memory loading.** Every read path is a chunked generator at
1 MiB. `test_upload_reads_in_bounded_chunks` asserts the exact chunk sequence.
*Residual, and it is real:* `read_manifest` returns the whole manifest as bytes.
At ~246,000 objects a JSON manifest would be on the order of 50–100 MB, which is
survivable but not elegant. The container-packing design in
`archive_solution_design.md` §4 would cut it to ~150 rows; **that design is not
implemented.**

**Unsafe filesystem behaviour.** Symlinks are rejected per path *component*, so a
symlinked directory is caught, not just a symlinked leaf. Absolute paths, `.`,
`..` and post-resolution escapes are rejected on the way in and again on the way
out, because the manifest is untrusted input at retrieval time. Seven tests.

**Unchecked partial uploads.** Objects stream to a `.partial` name and are
`fsync`ed and renamed only after size *and* digest match. The manifest is written
last, so an interrupted upload leaves no manifest and no locator — the archive
simply does not exist. Tested, including that no readable `.partial` survives.

**Checksum schemes that trust the producer's summary.** This was the specific
thing to get right. The prototype computes its own digests and *requires* them to
agree with AAA's `checksums.json`; disagreement raises. Absence reports
`NOT_VERIFIED`, never `PASS`. Mutation-testing confirms the cross-check is
load-bearing.

**Locator mutability and ambiguous version identity.** Reads take an explicit
`version_id`. A wrong version raises. The locator holds the manifest digest
independently of the manifest, so a manifest edited *and* re-encoded consistently
still fails — `test_manifest_edited_to_be_self_consistent_is_still_detected`.

**Accidental secrets.** The credential is read from `AAA134_PROTOTYPE_ARCHIVE_TOKEN`
and used for nothing but a presence check. A test asserts neither the value nor
the variable name appears in the locator or manifest. A pattern scan over the
complete diff finds nothing. The only token-shaped string in the branch is the
literal `prototype-token-not-a-secret` in a test fixture.

**Tests that pass without testing the intended failure.** Defect 1 above was
exactly this. Four guards were then mutation-tested and each probe made its test
fail — recorded in `evidence/mutation_probe.json`, **including the one partial
result**: removing the symlink guard breaks only one of the two symlink tests,
because a symlinked directory is independently caught by the regular-file check.
Four of thirty-four guards were probed. That is a spot check, not exhaustive
mutation coverage, and it is stated as such.

## Findings against the scientific checklist

**Scientific overclaiming.** The analysis reports things that are *unfavourable*
to AAA and reports them prominently: baseline competitiveness fails on
development point estimates in 28% of primary cells with a mean margin 41x the
preregistered threshold; the three proposed refinements were numerically inert
so the null search tested nothing; essentially all bounce accuracy is programmed
reflection at p=1 against the like-for-like baseline. Every claim carries an
evidence class (**H**/**D**/**F**/**C**) and a source.

Where the diagnostic produced a meaningless number — the rank-deficient
`scale = 0.0` least-squares rows — it is **reported and labelled meaningless**
rather than deleted. Deleting it would have made a cleaner table and a worse
document.

**Post-noise design tuned to existing evidence.** This is the sharpest risk, and
I do not think it is fully escaped. The design *is* motivated by the EIV analysis
in §3.2, which is derived from AAA's committed code and development evidence. The
mitigations: the mechanism budget is **zero** (nothing is tuned, because nothing
new is proposed); the development budget permits only three declared pass/fail
engineering checks; H1 is stated so that a *benefit* falsifies it; and H2 is a
positive control that invalidates the entire run if it fails.

**The residual risk that remains:** the drift magnitude `0.0025` and the step
magnitude `0.005` were chosen by me, from reasoning about the noise scale, without
measurement. They are the one genuinely tunable quantity in the design. They must
be fixed at the design freeze and never revisited after any confirmation data is
read — and a reviewer should scrutinise them specifically, because they are the
place where a result could be engineered.

**Confirmation leakage.** No confirmation batch reserved or consumed, no remote
reservation ref created, no registry lifecycle field touched, no unobserved
confirmation outcome inspected, no confirmation-role command run. The **F**-class
diagnostics use synthetic latent paths in a throwaway process and touch no
namespace the protocol owns.

**Implicit evaluator information.** The §1.2 argument is structural — reading
call sites — not an exhaustive taint analysis, and it says so. The prototype
itself never touches a predictor.

**Changed canonical behaviour / source-freeze weakening.** Zero existing files
modified, verified by `git diff --diff-filter=MDR` returning empty. No exclusion
added; nothing relocated under `results/` or `runs/`. The fingerprint changed from
`28a5e55e…` (111 files) to `caf29255…` (133 files) and that is recorded in
`evidence/fingerprint_effect.json` as a fact, not a problem to route around.

**Stale documentation.** The README, design and analysis were written against the
final state of the tree and the recorded commands were rerun after the last code
change.

## Limitations I could not resolve

1. **The prototype proves nothing about durability.** A local directory is not
   durable storage. `AAA-134` is untouched by anything in this branch.
2. **No real backend adapter exists.** No object-store credential was available
   and none was requested, so no real upload was attempted and no provider's
   documented behaviour was verified in practice.
3. **AWS S3 pricing could not be verified** from the official pricing page, whose
   tables render dynamically. Those cells say "not verified" rather than carrying
   a remembered number.
4. **Cloudflare R2's compliance-mode retention and versioning semantics could not
   be verified** from the official docs reachable at analysis time. That is why
   R2 is the runner-up and not the recommendation, despite free egress.
5. **Container packing is designed, not implemented**, and would need its own
   fault tests before use.
6. **The 245,760 per-batch shard count was not independently re-derived.** Linear
   scaling of the pilot's 256 shards gives 256,434 and I could not reconcile the
   two from the committed plan. The byte projections, which the design actually
   depends on, *were* recomputed independently.
7. **The post-noise resource numbers are projections**, scaled from a measured
   pilot at a different scale. A development pilot must measure them before any
   freeze, and the protocol draft says so in the artifact itself.
8. **Four of thirty-four guards were mutation-probed.**
9. **The bias and drift magnitudes are unmeasured design choices** — see above.
10. **This is a self-review.** It is not independent review, and it does not
    substitute for one.

## Hosted CI, reported rather than fixed

Two jobs fail on this PR — `Locked environment` and `Fresh install` — and **they
fail identically on PR #11's own head**, at the same step, for the same reason.

Every substantive step passes: locked install, lock verification, lint, format,
mypy, tests with coverage, both identity hashes, the observation-noise
development smoke, independent recomputation, and the confirmation exit-code
contract. The only failing step is `Retain failure evidence`
(`actions/upload-artifact`, `if: always()`), which rejects the colons in schedule
filenames such as
`development:clean_trained:bouncing:correlated:000000:l000:e0000:r00.json`.

This branch adds no schedule file and modifies neither `.github/workflows/ci.yml`
nor `aaa/noise/runner.py`, so it cannot be the cause. **I did not fix it**, because
fixing it means modifying an existing file, which this assignment forbids.

Worth flagging beyond the red tick: it means the Actions-artifact path cannot
currently retain a noise run's evidence *at all*, which independently weakens any
fallback story in which CI artifacts stand in for durable storage.
