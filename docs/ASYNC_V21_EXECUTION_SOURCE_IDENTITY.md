# Async-v2.1 execution-source identity

**Status:** Normative execution-identity authority for
`async-decoupled-v2.1-simple` qualification and scale submissions.

**Schema:** `emender-async-v21-execution-source-v1`.

This boundary supplements the scheduler fence, policy/schema identities,
native ABI/wire identities, and immutable input identities required by
[the compute-pool authority](RESILIENT_DILOCO_COMPUTE_POOL.md),
[the native data-plane authority](NATIVE_RESILIENT_DILOCO_DATAPLANE.md), and
[ADR-002](ASYNC_DECOUPLED_DILOCO_V2.md). It changes no R01–R16,
NDP01–NDP17, V21S01–V21S17, or ISP01–ISP07 requirement.

## Reviewed boundary

The execution-source digest is SHA-256 with domain
`emender-async-v21-execution-source-v1\0`. Entries are ordered by Git path.
For each included tracked regular file or symbolic link, the digest consumes
the path length, path bytes, content length, and tracked content bytes.

Exactly these path prefixes are evidence-only and excluded:

```text
docs/validation/
logs/
reports/
```

They are append-only qualification reports, scheduler logs, audit records,
and retained experiment evidence. A file under one of these prefixes MUST NOT
be imported, executed, parsed as policy/configuration/schema, or used as a
launcher, controller, native build input, dataset, tokenizer, or seed
authority. Moving an operational input under an excluded prefix is forbidden;
the fixed controller and launcher paths remain part of the digest.

Every other tracked byte is included. In particular, changes to Python,
shell, Slurm, C/C++, native sources, public headers, ABI/wire encodings,
schemas, authorities, policies, configurations, controllers, launchers,
training/data preparation, tokenizer configuration, or seed configuration
produce a different execution-source digest and require exact-identity
requalification. Adding a new tracked top-level evidence directory is not
implicitly excluded; changing the exclusion list changes the hashed
controller itself and requires review and requalification.

External inputs remain independently bound and are never made equivalent by
this boundary:

- the native build manifest, installed bundle SHA-256s, native source commit,
  and full-layout G2 artifact;
- the reviewed data-object identity;
- the p50k tokenizer path and SHA-256;
- the step-2300930 seed authority, size, accepted-token clock, attestation,
  and SHA-256
  `0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2`;
- the canonical training arguments, v2.1 policy digest, launcher digest,
  scale authorization, immediate predecessor, and V21S17 closure evidence.

A native build or G2 gate may name an earlier Git commit only when that
commit's recomputed execution-source digest is byte-identical to current clean
authoritative `main`. The native artifact and G2 identities must still match
each other exactly. The current checkout must still be clean
`HEAD = origin/main`; Git commit IDs remain provenance, while the immutable
execution-source digest determines whether an evidence-only commit changed
what executes.

## Durable scheduler transaction

The only valid model-submission ordering is:

```text
held payload submission
  -> atomic durable payload digest + model job identity
  -> scheduler-owned afterany:<model-job> collector registration
  -> atomic durable collector job/script/dependency identity
  -> model-job release
```

The model request is exact `Nodes=<reviewed rung>`,
`Partition=batch`, and `QOS=debug`; the two scheduler fields are retained
separately. A failed or ambiguous collector registration cannot release the
model job. Deterministic Slurm job names/comments allow reconciliation of a
scheduler side effect before any retry, and the durable payload state
recognizes held, queued, running, terminal, and retired identities without a
second payload or collector submission.

The collector is an independent standard-library Python program registered
with `afterany`. It requires neither WG nor Codex. It retains the literal
parent `sacct` row with separate `Partition` and `QOS`, `ExitCode` and
`DerivedExitCode`, stdout/stderr and SHA-256s, the exact payload/validator
inputs and SHA-256s, and one canonical machine verdict. An evidence-directory
lock and atomic final manifest make repeated collector execution idempotent.

This transaction does not authorize a Slurm job. Promotion remains
`two-node gates -> review -> 4 -> 8 -> 16 -> 32 -> 64 -> 256`; every scale
rung still requires its exact immediate predecessor and the reviewed
leased-READY V21S17 finite closure.
