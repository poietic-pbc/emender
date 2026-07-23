# Final E97 submit-side seed bootstrap

Frontier job 5060027 exposed an invalid dependency: every compute node tried to
reach the S3 step manifest, latest pointer, and 7.7 GB checkpoint before model
load. Compute-node outbound access is not reliable and is no longer part of the
exact-two-node path.

The submission driver now resolves and retains the exact bytes of both
authority documents on the submit/login side, checks every immutable field,
downloads the checkpoint into a lock-protected temporary file, verifies size
and SHA-256, fsyncs it, and atomically publishes only
`sha256-0239706e....pt`. A locked cache hit is fully rehashed and both live
authorities are revalidated; a partial, stale, non-regular, multiply linked, or
incorrect entry is removed under the lock and reacquired. Authority drift
always fails before `sbatch`. The atomic attestation contains the authority
bytes, their SHA-256 digests, parsed documents, immutable seed identity, and
explicit cache-reuse status.

Inside the allocation, the batch script requires executable Slurm `sbcast`,
creates the exact `/tmp/emender-e97-seed-${SLURM_JOB_ID}` directory on every
allocated node, and broadcasts the verified checkpoint and pinned attestation.
One offline verifier per node independently checks the live job ID, node-local
mount, regular single-link checkpoint, exact size/SHA, attestation file digest,
authority-byte digests, parsed authority fields, and immutable seed identity.
Any missing binary, broadcast failure, stale directory, missing/corrupt copy,
attestation mismatch, or job-ID mismatch fails before the allocation
supervisor or model-bearing trainer starts. Error cleanup removes job-scoped
partial node copies. Trainers receive only the verified node-local checkpoint;
the shared cache is cold bootstrap input, never a trainer path.

The immutable identity remains step `2300930`, accepted tokens
`150793748480`, size `7719680116`, and SHA-256
`0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2`.
The submit request and scheduler evidence remain pinned separately to
`Partition=batch` and `QOS=debug`.

## Conformance

This change conforms to *Resilient DiLoCo Compute Pool*, version 1:

- **R01/R09:** immutable seed and offline evidence are verified before any
  model load; managers remain model-free.
- **R07:** temporary acquisition, fsync, atomic content-addressed publication,
  and atomic evidence publication prevent partial authoritative state.
- **R10:** shared storage is cold bootstrap/evidence only. Checkpoint
  consumption is node-local, and update, aggregate, heartbeat, membership, and
  redistribution hot paths are unchanged.
- **R13/R14:** scheduler-specific `sbcast` stays in the Frontier adapter and
  every bootstrap stage fails closed with retained per-node evidence.
- **R16:** the launcher remains the exact two-node acceptance rung and does not
  authorize 4+ nodes.
- **NDP13:** bootstrap failure is bounded before role startup and contained to
  the allocation.
- **NDP15:** read-only seed handoff is digest pinned; Python retains checkpoint
  admission policy while the native drain/publication protocol is unchanged.
- **NDP16:** attestations and per-node manifests retain source/cache/local
  digests, job identity, hostname, reuse state, and zero network fetch count.
- **NDP17:** this changes only the approved exact-two-node path after the
  retained full-layout G2 gate; it does not advance the scale ladder.

Validation uses bounded byte fixtures and mocks; it neither downloads the
production checkpoint nor submits Slurm work.
