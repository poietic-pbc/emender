# Final E97 S3 seed integration validation

Date: 2026-07-22 UTC. Task: `integrate-final-e97-s3-seed`.

## Authority cross-check

The Frontier login environment loaded successfully with
`scripts/frontier/activate_emender_frontier.sh`. The site environment does not
currently expose an `aws` executable or `boto3`; the canonical public S3 HTTPS
endpoint was therefore used by both validation and the runtime materializer.
No local/shared checkpoint was used as an authority or staging intermediary.

Both JSON objects returned HTTP 200 and independently contained this exact
identity:

- immutable checkpoint: `s3://spinozans/emender/e97-diloco/emender_E97_1.3B_20260709_084606/step_2300930/checkpoint_step_2300930_loss_2.4365.pt`
- step manifest: `s3://spinozans/emender/e97-diloco/emender_E97_1.3B_20260709_084606/step_2300930/manifest.json`
- discovery pointer: `s3://spinozans/emender/e97-diloco/latest_emender_E97_1.3B.json`
- size: `7719680116`
- SHA-256: `0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2`
- step: `2300930`
- loss: `2.4365`
- tokens: `150793748480`

An HTTP HEAD of the immutable object returned HTTP 200, `Content-Length:
7719680116`, `Last-Modified: Wed, 22 Jul 2026 08:03:56 GMT`, and multipart ETag
`f3f88f4a11fad751ee203baa5c10822f-116`. The runtime does not treat ETag as a
content digest; it verifies the configured SHA-256 after the full download.

## Render and promotion evidence

Deterministic smoke and production renders both produced bundle fingerprint
`ef6f52145e34c056c154f0d162dec47ec96d02d883f540ebdf6f793427801ec3`.
The parity checker accepted only the typed differences already authorized:
nodes/ranks, walltime, partition, and QoS. Both profiles contain the identical
immutable seed object and job-scoped materialization policy. Their serialized
input hashes differ because the approved typed profile fields differ:

- smoke `launch-inputs.json`: `70e01efed52520896fd610ae940a09e4f8be97d4b4884f1cfa9b75f160d1f3c3`
- production `launch-inputs.json`: `b6c9adb3364a604ac6d938efe5c8733317aef2250c829542ce544b0e5dc6aa3a`

No `promotion.json` was created or copied. The new seed participates in the
normalized bundle fingerprint, so old smoke/promotion evidence has a different
fingerprint and is rejected before the submission marker or `sbatch` execution.
No Slurm command was submitted by this task.

## Materialization and tests

`materialize_e97_s3_seed.py` fetches and parses both JSON authorities before
the checkpoint, requires every identity field, and rejects pointer/manifest
drift. It requires `SLURM_JOB_ID` in the destination path, rejects an existing
final file instead of reusing it, streams into an exclusive temporary file in
the same directory, verifies exact bytes and SHA-256, fsyncs, and uses
`os.replace` for atomic promotion. Failure removes the temporary file. It never
references or mutates the legacy shared seed directory. The runtime manifest
records the configured source identity, staged path/size/SHA, both parsed JSON
documents, and their content hashes before model load.

The focused suites cover absent fields, inaccessible authorities, pointer URI
drift, manifest disagreement, wrong size/hash/step/loss, partial transfer,
stale final files, non-job-scoped paths, atomic promotion, runtime identity,
render parity, mutation rejection, and stale promotion evidence.

## Primary-checkout preservation

Before edits, the user-owned untracked tree was inspected and hashed. A content
search found no occurrence of the final checkpoint URI, step, size, or SHA in
`.final_checkpoint_request` or `src/GatedDeltaNet-2/`; the former is a small
walltime sentinel and the latter is an intact nested NVIDIA reference clone
with its own `.git` directory. Neither is a canonical E97 handoff/launcher
addition, so neither was modified or staged. Their initial sentinel SHA-256 was
`26e1c1294fc6572ad2eb6b65e48eb806c75e9327a77fb4ca1831317569418f4a`;
the nested checkout and all unrelated primary-checkout state remained present.

## Resilient DiLoCo conformance

The conformance checklist in `RESILIENT_DILOCO_COMPUTE_POOL.md` was reviewed.
This change is limited to immutable initial-seed intake and preserves the
architecture boundaries. Applicable gap-matrix requirements are R07 (immutable
checkpoint/latest authority), R09 (verification before trainer model load),
R10 (no legacy Lustre seed mutation), R12 (explicit source identity), and R16
(no scale run before a fresh smoke). It also supports NDP15 (fenced read-only
checkpoint handoff) and NDP16 (source and runtime identity telemetry). No claim
is made that this seed-only integration closes any existing Frontier runtime
qualification gap.

