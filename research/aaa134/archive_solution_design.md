# AAA-134: durable archive path — requirements, architecture, and what stays unresolved

> **AAA-134 PROTOTYPE VALIDATED; BLOCKER NOT RESOLVED.**
> **AAA-144 REMAINS OPEN.**
>
> Nothing in this document or in [`archive_prototype/`](archive_prototype/) closes
> either ledger entry. Neither entry was renamed, merged, closed or rewritten.

---

## 1. What AAA-134 actually is

`AAA-134` (`docs/issue_ledger.md:1127`) records that *no approved durable archive
locator exists for the observation-noise A/B evidence*. Its disposition is that no
locator is fabricated and the confirmation boundary stays blocked.

`AAA-144` (`docs/issue_ledger.md:1252`) is the **later independent Sol audit of the
same underlying blocker, restated after the archive implementation was repaired and
its real scale was measured**. The relationship, precisely:

| | AAA-134 | AAA-144 |
|---|---|---|
| Raised by | fresh observation-noise completion audit | independent Sol archive audit |
| When | before the sharded-archive repair | after it, with measured scale |
| Formulation | "there is no approved durable mechanism at all" | "the repaired single-copy format projects ~38.5 GB per batch, the repository has no LFS config, release asset or object-store credentials, and only ~69.8 GB of local disk was available" |
| Severity / status | blocking / open | blocking / open |

They are **not duplicates and must not be merged.** `AAA-134` is the policy-level
statement that durable storage is a protocol prerequisite. `AAA-144` is the
engineering-level statement of the same prerequisite *at a now-known scale*, and it
is the one that carries actionable numbers. Closing one does not close the other:
`AAA-134` closes when an approved mechanism exists and has been demonstrated;
`AAA-144` closes when that mechanism demonstrably fits the measured payload.

This document targets `AAA-134` as requested and cross-references `AAA-144`
wherever the repaired implementation's measured scale is what actually binds.

## 2. Independently confirmed scale

Recomputed from `docs/evidence/observation_noise_scale_pilot.json` rather than
copied from it:

```
quick pilot scored records                58,944
formal plan scored records            59,043,840
scale factor                           1,001.694
quick pilot compressed record bytes   38,475,272
```

| Quantity | Value | How |
|---|---:|---|
| compressed records, one batch | **38.54 GB** | `38,475,272 × 1001.694` |
| full sharded attempt, one batch | **47.36 GB** | `47,281,979 × 1001.694` |
| compressed records, A + B | **77.08 GB** | ×2 |
| planning figure with overhead and headroom | **~100 GB** | A+B plus schedules, metadata, checkpoints, reports, plots, verification, and room for a retained failed attempt |
| record shards, one batch | 245,760 | *as reported by the pilot*; see the caveat below |
| local disk free today | **68.76 GB** | `df -B1 /home/cinqic` at analysis time |

**68.76 GB < 77.08 GB.** The combined A+B compressed record payload does not fit on
the machine at all, before anything is copied anywhere. `AAA-144`'s blocking claim
reproduces.

> *Caveat.* I did not independently re-derive the 245,760 shard count from the plan.
> Linearly scaling the pilot's 256 shards gives 256,434, and I could not reconcile
> the two from the committed plan alone. The **byte** projections above are
> independently recomputed and are the ones the design depends on; the object count
> is treated as "roughly a quarter of a million per batch", which is all the
> architecture needs.

### Why the existing mechanisms do not satisfy the protocol

| Mechanism | Why it is not the answer |
|---|---|
| **Local output root** | `docs/evidence_policy.md:131` already calls the development locator "explicitly transient local evidence". A path on the producing machine is not retrievable from a fresh location by a third party, and here it does not even fit. |
| **The Git reservation ref** (`aaa/noise/reservation.py`) | This is *batch ownership coordination* and nothing else: an append-only `refs/heads/aaa-confirmation-claims/<batch>` commit whose message is a small JSON ownership payload, created atomically so two clones cannot both own a batch. It stores no evidence. Confusing it with an archive would be a category error. |
| **Git LFS** | GitHub's per-file limit is 2 GB on Free/Pro, 4 GB on Team, 5 GB on Enterprise Cloud. Individual shards fit; 77 GB of metered LFS storage and bandwidth does not, and LFS pointers are not immutable versions. |
| **GitHub release assets** | 2 GB per asset, and assets can be replaced or deleted — a mutable locator. |
| **GitHub Actions artifacts** | Retention is capped (90 days maximum). The protocol requires evidence that outlives a retention window by construction. The PR #11 description already says these are not the final solution. |

## 3. Requirements, derived from the repository

Taken from `aaa/noise/data/observation_noise_v1.json` (`artifacts.*`,
`compute_storage.retention`), `aaa/noise/verifier.py`, `docs/evidence_policy.md`
and `docs/reproduction.md` — not invented here.

**What must be preserved** (the complete attempt directory, not a summary):

- the complete primitive archive: `records/` deterministic per-trial gzip shards;
- `records/index.json`, the canonical record index (schema
  `aaa.observation_noise_record_shards.v1`, ordering `trial_id ascending, step
  ascending`, with per-shard compressed checksums and a total record count);
- `schedules/` and `schedules/index.json` with per-trial digests;
- `checkpoints/`, including per-budget checkpoints;
- `metadata.json`, `run_manifest.json`, `summary.json`, `report.md`, plots and
  `plot_provenance.json`;
- `checksums.json`, which `verify_attempt` requires and which must cover **every**
  retained file exactly once;
- **failed, cancelled and superseded attempts too** — `compute_storage.retention`
  says so explicitly, and `benchmarks/observation_noise_registry.json` records
  `failed_attempts_are_retained: true`.

**What identity must travel with the bytes:** attempt ID, batch ID, role, protocol
version and hash, scientific fingerprint, candidate ID, source commit, invocation,
outcome. A retrieval that cannot prove *which frozen inputs produced these bytes* is
not evidence.

**What the destination must provide:** an immutable or version-addressed durable
locator; retrieval into a fresh location with no reference to the original attempt
directory; independent byte verification of every retained object after retrieval.

**Two hard engineering constraints**, both from `AAA-145`/`AAA-146` history: never
load the archive into memory, and never require a duplicate local copy. The repair
that made a batch feasible at all reduced peak RSS from 2.73 GB to 155 MB and attempt
size from 788 MB to 47 MB at pilot scale precisely by streaming shard by shard. An
archive layer that reintroduces either failure undoes that repair.

## 4. Architecture

```
                attempt directory (streamed, never copied)
                               |
                    enumerate_members()            safe paths, streaming sha256
                               |
             cross_check_against_aaa_checksums()   our digests vs the run's own
                               |
                       backend.preflight()         size, capacity, duplicate, identity
                               |
        for each member:  backend.put_object(path, chunk iterator)
                               |                   -> StoredObject{key, version_id, sha256}
                       backend.finalize()          canonical manifest written LAST
                               |
                         DurableLocator            {scheme, archive_id, manifest
                               |                     version + digest, counts, durable}
        ------------------------------------------------
        |                                              |
  retrieve_archive(fresh empty dir)            verify_retrieval()
        re-validates every manifest path        recomputes every digest from
        refuses a non-empty destination         retrieved bytes; checks identity
```

Four design decisions carry the weight.

**1. The manifest is written last, and finalization is the only thing that creates a
locator.** An interrupted upload therefore leaves no locator and no finalized
manifest — the archive simply does not exist yet. There is no state in which a
partially uploaded archive looks complete.

**2. The locator holds the manifest digest independently.** A manifest can be
tampered with; if it is, the digest in the locator no longer matches. And because the
manifest's canonical encoding is a pure function of its rows, an attacker who edits
rows *and* re-encodes cleanly still fails, because the re-derived digest will not
match the one the locator was issued with. `test_manifest_edited_to_be_self_consistent_is_still_detected`
covers exactly this.

**3. Reads name the version, not just the key.** `open_object(locator, path,
version_id)`. A backend that only offers a mutable key cannot satisfy `AAA-134`,
because the locator would name a *name* rather than *bytes*.

**4. The producer's `checksums.json` is cross-checked, never adopted.** AAA already
computes its own checksum manifest. The archive layer computes digests independently
and requires the two to agree. Agreement is evidence; adoption would be the producer
grading its own homework.

### Object-count design note (designed, **not implemented**)

A quarter of a million objects per batch is 245,760 PUTs — roughly $1.10 of Class A
operations on R2, which is fine, but 245,760 round trips is operationally slow and
fragile. The intended production shape packs shards into sequential container objects
of ~256 MB (about 150 per batch) with byte offsets recorded in the canonical index, so
an independent reviewer can still fetch one trial's shard with a single HTTP Range
request. Per-shard checksums stay in the index, so a corrupt container is still
localized to the shards it damages.

**The prototype does not implement this.** It uploads one object per member. Container
packing is a design proposal that would need its own fault tests before use.

## 5. Durable backend evaluation

Capability claims below come from official provider documentation, fetched at
analysis time. Where I could not verify something from an official page, the cell
says so rather than guessing.

| | **Backblaze B2** | **Cloudflare R2** | **AWS S3** | **Zenodo** | Git LFS / Releases |
|---|---|---|---|---|---|
| Immutability | Object Lock: compliance + governance + legal hold | "Bucket locks" documented; compliance-mode semantics **not verified** from official docs I could reach | Object Lock compliance mode: not deletable by anyone, including the account root, until the retain-until date | Published records; files editable within 30 days, then only in "duly justified cases"; new versions otherwise | none — both are mutable/replaceable |
| Versioned objects | yes (S3-compatible) | versioning **not verified** | yes; Object Lock *requires* versioning | record-level versioning, not object-level | no |
| Capacity > 100 GB | yes | yes | yes | **no** — 50 GB default, 200 GB on request; 100 files per record | no |
| Max object size | S3-compatible | not verified | 48.8 TiB | n/a | 2 GB Free/Pro, 4 GB Team, 5 GB Enterprise |
| Multipart / resume | yes (S3 API) | not verified | yes: 10,000 parts, 5 MiB–5 GiB each | no | no |
| Strong object checksums | S3 API `ChecksumSHA256` | S3-compatible API | `ChecksumSHA256` | file-level checksums in metadata | no |
| Stable locator | bucket/key + version ID | bucket/key (+version, unverified) | bucket/key + version ID | **DOI** — the strongest citable locator of the five | commit/tag, mutable |
| Storage cost at ~100 GB | **$6.95/TB/mo → ≈ $0.70/mo** | $0.015/GB-mo → **≈ $1.50/mo** | rate **not verified** (the official pricing page renders its tables dynamically) | free | metered |
| Retrieval cost, one full 100 GB fetch | **free** — free egress up to 3× stored covers it | **free** — R2 charges no internet egress | charged | free | metered bandwidth |
| Operation cost, ~246k PUTs/batch | included | ≈ $1.10 Class A per batch, ≈ $0.09 Class B per full read | not verified | n/a | n/a |
| API/CLI automation | S3-compatible | S3-compatible | native | REST API | git |
| Vendor lock-in | low (S3 API) | low (S3 API) | moderate | low (DOI is portable) | high |
| Operational complexity | low | low | moderate (IAM, Object Lock config) | low but manual | low |

### Recommendation

**Primary: an S3-compatible bucket with Object Lock in compliance mode, targeting
Backblaze B2, with AWS S3 as the reference semantics.**

The reasoning is deliberately boring, which is the point:

- B2 and S3 speak the same API, so **one adapter serves both** and the choice is
  reversible. Picking R2 today would not be wrong, but I could not verify its
  compliance-mode and versioning semantics from official documentation, and
  "convenient and probably fine" is not a basis for choosing where scientific
  evidence lives.
- Compliance mode is the only retention setting that means what `AAA-134` needs:
  not deletable or overwritable by *anyone*, including the account root, for the
  retention period. Governance mode is bypassable by a permission, which makes the
  locator's immutability a policy claim rather than a storage property.
- Object Lock requires versioning, which gets version-addressed reads for free.
- Free egress up to 3× stored covers the independent full retrieval the protocol
  requires, so verification does not carry a cost disincentive. That matters: a
  verification step people avoid because it is expensive is a verification step that
  does not happen.
- At ~$0.70/month for 100 GB the cost is not a decision variable.

**Secondary, and genuinely useful: publish the compact evidence bundle to Zenodo for
a DOI.** Zenodo cannot hold the primitive archive — 50 GB default quota and a
100-file-per-record limit against 77 GB and a quarter-million shards — but a DOI over
`summary.json`, `report.md`, `checksums.json`, the canonical index and the object-store
locator is the most citable immutable pointer available, and it survives a storage
vendor disappearing. This is complementary, not an alternative.

**Rejected:** Git LFS and release assets (mutable, size-capped, metered), Actions
artifacts (bounded retention by design), and any local path.

### What was deliberately not done

No account was created, no storage purchased, no billing changed, no credentials
requested. No pre-existing object-store credential was found in the environment, so
**no real adapter was prototyped and no real upload was attempted.** Per the
assignment, the work stops at the backend contract and this documented integration
design. Absence of credentials is not permission to fake success, and no locator that
could be mistaken for a production one exists anywhere in this branch.

## 6. Threat and failure model

| # | Failure | Detected by | Tested |
|---|---|---|---|
| 1 | shard missing after retrieval | manifest coverage check | `test_missing_shard_is_detected` |
| 2 | shard silently corrupted, same length | per-object sha256 recomputed from retrieved bytes | `test_corrupted_shard_is_detected` |
| 3 | shard truncated | size **and** digest | `test_truncated_shard_is_detected` |
| 4 | manifest rows edited | locator-held manifest digest | `test_manifest_checksum_tampering_is_detected` |
| 5 | manifest edited *and* re-encoded consistently | independently held locator digest | `test_manifest_edited_to_be_self_consistent_is_still_detected` |
| 6 | canonical index tampered | index is itself a checksummed member | `test_index_tampering_is_detected` |
| 7 | producer `checksums.json` wrong or forged | independent digests must agree with it | `test_producer_checksum_manifest_is_cross_checked_not_adopted` |
| 8 | producer `checksums.json` absent | reported `NOT_VERIFIED`, never `PASS` | `test_absent_producer_checksums_is_not_verified_not_pass` |
| 9 | duplicate object write | write-once key | `test_duplicate_object_is_refused` |
| 10 | same attempt uploaded twice | duplicate-attempt guard | `test_duplicate_attempt_upload_is_refused` |
| 11 | double finalization | finalization guard | `test_duplicate_finalization_is_refused` |
| 12 | write after finalization | immutability guard | `test_write_after_finalization_is_refused` |
| 13 | finalize with an incomplete stored set | destination/caller reconciliation | `test_finalize_refuses_an_incomplete_stored_set` |
| 14 | invalid locator | manifest lookup fails | `test_invalid_locator_is_refused` |
| 15 | locator disagrees with manifest object count | cross-check | `test_locator_object_count_disagreement_is_refused` |
| 16 | wrong remote version requested | version-addressed read | `test_wrong_remote_version_is_refused` |
| 17 | interrupted upload | no manifest, no locator, no readable partial | `test_interrupted_upload_leaves_no_finalized_archive_and_resumes` |
| 18 | resume against changed bytes | resume matches on digest, not path | `test_resume_reuploads_a_member_whose_bytes_changed` |
| 19 | missing credentials | preflight refuses before any byte moves | `test_missing_credentials_are_refused` |
| 20 | credential leaked into an artifact | asserted absent from locator and manifest | `test_no_credential_value_appears_in_the_locator_or_manifest` |
| 21 | capacity shortfall | preflight | `test_capacity_shortfall_is_refused_before_uploading` |
| 22 | symlinked member file | per-component symlink check | `test_symlinked_member_is_refused` |
| 23 | symlinked directory | same | `test_symlinked_directory_is_refused` |
| 24 | `../` traversal in a manifest path | destination containment check | `test_manifest_traversal_path_is_refused_on_retrieval` |
| 25 | absolute member path | path guard | `test_absolute_member_path_is_refused` |
| 26 | relative traversal | path guard | `test_relative_traversal_is_refused` |
| 27 | unsafe archive id | archive-id guard | `test_unsafe_archive_id_is_refused` |
| 28 | unexpected extra file after retrieval | manifest coverage, both directions | `test_unexpected_extra_file_is_detected` |
| 29 | incomplete retrieval | coverage | `test_incomplete_retrieval_is_detected` |
| 30 | wrong archive identity | identity comparison | `test_wrong_archive_identity_is_reported` |
| 31 | mismatched protocol/source metadata | identity comparison | `test_mismatched_protocol_and_source_metadata_are_reported` |
| 32 | retrieval into a dirty destination | non-empty destination refused | `test_retrieval_refuses_a_non_empty_destination` |
| 33 | whole-archive memory load | chunk sizes asserted bounded | `test_upload_reads_in_bounded_chunks` |
| 34 | hidden duplicate local copy | attempt tree asserted unchanged | `test_no_second_local_copy_is_made_under_the_attempt_root` |

The guards were **mutation-tested**: removing the symlink check, the producer
cross-check, the retrieval traversal guard, or the write-once guard each makes the
corresponding test fail. They are load-bearing, not decorative. Recorded in
[`evidence/mutation_probe.json`](evidence/mutation_probe.json).

### Residual threats the prototype does not address

- **A malicious producer.** If the run itself writes wrong numbers, every checksum in
  the world agrees with them. This is what independent recomputation
  (`observation-noise-recompute`, `verify_attempt`) is for, and it is a separate
  control from archival.
- **Silent bit rot in the destination** between upload and retrieval. Detected on
  retrieval, not prevented; mitigation is periodic re-verification, which is an
  operational policy this design does not set.
- **Retention expiry.** Compliance-mode locks expire. Somebody has to decide the
  retention period and renew it. That is a maintainer decision, not a code one.
- **Credential compromise.** Out of scope here beyond "credentials live in the
  environment and never in the repository, and never appear in a log or artifact".

## 7. Integration plan (for a future, separate decision)

None of this should happen before PR #11 resolves.

1. **Maintainer approves a destination** and records it in `docs/evidence_policy.md`
   as the repository-approved durable mechanism. Until this exists, `AAA-134` cannot
   close no matter how good the code is.
2. **Credentials are provisioned out of band** into the CI/operator environment. A
   write-only, Object-Lock-enabled, versioned bucket; no credential in the repository.
3. **A real S3-compatible adapter** implementing `ArchiveBackend`, mapping
   `put_object` to multipart upload with `ChecksumSHA256`, `version_id` to the object
   version, and `finalize` to writing the manifest last and reading its version back.
4. **A development-scale end-to-end proof first**: upload a *development* attempt,
   retrieve into a fresh location on a different machine, verify every byte, and
   record the result. This is the step that would let `AAA-144` close.
5. **Only then**: extend the confirmation flow to require a finalized durable locator
   before a batch may be marked consumed, and record the locator in the attempt's
   retained evidence.
6. **Then** the freeze/confirmation sequence proceeds unchanged, on PR #11's branch,
   with PR #11's untouched scientific identity.

Steps 1–4 are prerequisites for confirmation, not deliverables of it.

## 8. What the prototype proved, and what it did not

**Proved** (local, `PROTOTYPE_ONLY_NOT_DURABLE`):

1. an attempt archive can be safely enumerated with streaming digests and no second
   local copy;
2. its objects upload through the proposed interface in bounded chunks;
3. a finalized locator and canonical manifest can be produced atomically;
4. the archive retrieves into a fresh empty directory with no reference to the
   original;
5. independent verification detects all 34 failure modes above, and the guards are
   mutation-tested rather than assumed.

**Did not prove, and cannot:**

- that any durable destination exists or has been approved;
- that a real backend behaves as its documentation says;
- that 100 GB moves through a real network within an acceptable window;
- that an independent third party can retrieve and recompute;
- anything at all about the scientific outcome of observation-noise A/B.

A local directory is not durable storage. That is the entire content of `AAA-134`,
and this prototype does not change it.
