# Async v2.1 8/32/128 scale-chain quality pass

Date: 2026-07-27
WG task: `.quality-pass-v21-scale-8-32-128`

## Scope and result

This was a graph/task-quality pass. It did not build code, run tests, submit
Slurm jobs, inspect live scheduler state, or execute any scale runner.

The three downstream task descriptions were edited in WG:

- `scale-v21-8n-clean`
- `scale-v21-32n-clean`
- `scale-v21-128n-clean`

The intended task interpretation is now explicit and unambiguous:

```text
current strict 2-node passed=true evidence
        + this quality pass
                    |
                    v
          exactly 8 nodes
                    |
        immutable passed=true
                    v
          exactly 32 nodes
                    |
        immutable passed=true
                    v
         exactly 128 nodes
```

The counts denote Frontier compute nodes, not GPUs, ranks, managers, a
maximum, or an elastic target. Each node is expected to run eight persistent
trainers, for 64, 256, and 1024 trainers respectively.

## Authorities reviewed

The quality pass read:

1. `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, including the required
   conformance checklist and R01-R16 semantics.
2. `docs/RESILIENT_DILOCO_GAP_MATRIX.md`, including the normative
   R01-R16, NDP01-NDP17, V21S01-V21S17, and ISP01-ISP07 crosswalks.
3. `docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md`, including the native
   NDP01-NDP17 contract, ordered-gate requirements, exact provider/identity
   evidence, bounded resource formulas, deadlines, telemetry, restart, and
   scale evidence.
4. `docs/ASYNC_DECOUPLED_DILOCO_V2.md` (ADR-002), including the v2.1
   compatibility boundary, exact-token/K40 semantics, immutable final E97
   seed, strict pass gates, and V21S17 scale-only closure.
5. The ISP01-ISP07 immutable-snapshot amendment in the gap matrix, including
   coherent ownership, immediate foreground resume after `OWNED`, immutable
   background work, bounded capacity, atomic apply, disjoint causal phase
   telemetry, and every-event tail rejection.

The live predecessor `qualify-v21-safe-boundary-2n` was inspected only through
`wg show` state, dependency, progress-log, and evidence references. No
predecessor artifact or runtime output file was opened for this quality pass.

## Live graph audit

WG state showed:

- `scale-v21-8n-clean` has exactly the two relevant blockers:
  `qualify-v21-safe-boundary-2n` and
  `.quality-pass-v21-scale-8-32-128`.
- `scale-v21-32n-clean` depends on `scale-v21-8n-clean`.
- `scale-v21-128n-clean` depends on `scale-v21-32n-clean`.
- All three scale tasks were Open but transitively blocked during the audit.
- `wg why-blocked` identified the live two-node runner and this quality pass as
  the root blockers; no scale task was prematurely ready.
- No dependency from the two-node/quality gates directly bypasses 8 nodes, no
  dependency bypasses 32 nodes, and no larger runner has an independent ready
  path.

Completion status alone is not treated as a scientific pass in the edited
tasks. Every next rung also requires its immediate predecessor's immutable
machine-readable `passed=true` verdict, exact identity binding, reviewed
promotion, and reviewed V21S17 closure authorization.

## Defects found and WG edits

### Exact-identity promotion hole

The original tasks allowed a runner to fix a defect, change source, rebuild,
and continue at its current scale. That could have promoted a source/native
identity that had never passed the immediate smaller rung.

All three tasks now say that any change to source, native binaries, policy,
schema, ABI/wire, launcher/controller, data, tokenizer, or seed invalidates
the predecessor authorization. A regression-first correction may be developed
and pushed, but the corrected identity must regain every required smaller
strict pass before a same- or larger-scale payload may be submitted. Otherwise
the task remains incomplete and its consumers remain blocked.

### Runner-versus-evaluator ambiguity

Each task now explicitly requires the runner to perform the concrete approved
activation, rebuild, preflight, scale gate, and clean-job commands once its
prerequisites pass. A runner may not convert the assignment into a report that
merely grades the absence of evidence it was assigned to create. Missing or
non-passing prerequisites remain fail-closed and authorize no submission.

### Canonical Frontier environment

Every runner now names:

```bash
source scripts/frontier/activate_emender_frontier.sh
"$EMENDER_PYTHON"
PYTHON_BIN="$EMENDER_PYTHON"
```

where the wrapper accepts `PYTHON_BIN`. No bare or guessed Python interpreter
is permitted.

### Exact immutable seed

Every stage now repeats the complete final E97 identity:

- step `2300930`
- accepted tokens `150793748480`
- size `7719680116`
- SHA-256
  `0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2`

The task text requires submit-side authority/attestation verification,
job-scoped `sbcast` staging, independent offline verification on every node
before model load, and `network_fetches=0`.

### Exact scheduler binding

Each runner now requires exact Nodes=8, Nodes=32, or Nodes=128 plus both
`Partition=batch` and `QOS=debug`. It retains those as separate fields with
literal queued/running `squeue` evidence and terminal `sacct` evidence. No
task may infer QoS from the partition column.

### Durable terminal ownership

The original wording required an `afterany`/equivalent terminal collector but
did not close the interval between payload submission and collector
registration. The edited tasks require the scheduler-owned collector or
equivalent durable controller transaction to be registered and durably bound
to the payload before the model job is released to run.

The collector must survive WG/Codex worker death and idempotently retain:

- terminal `sacct` and exit/derived-exit state;
- job logs and hashes;
- validator inputs;
- the machine-readable verdict; and
- its own collector/job identity bound to the payload digest.

A simulated monitoring-worker interruption must prove this behavior. Live LLM
monitoring is operational assistance, not the owner of terminal truth.

### No-duplicate reconciliation

Before every submission, the runner must reconcile WG state, the durable
controller ledger, retained attempt/payload records, `squeue`, and `sacct`.
Any digest that is or was queued, running, terminal, failed, timed out, or
incomplete is reconciled and never submitted unchanged. Failed payloads are
retired. Each stage permits exactly one active changed payload.

### Empirical V21S17 deadline chain

Deadline derivation is now explicitly stage-to-stage:

- 8 nodes derive from digested passing two-node per-event/per-phase
  distributions.
- 32 nodes derive from digested passing 8-node max/p99 distributions.
- 128 nodes derive from digested passing 32-node max/p99 distributions.

Each record must retain the chosen quantiles, maxima and hard-tail treatment,
safety margins, arithmetic, input digests, and resulting finite discovery,
leased-READY close, preparation, boundary-rendezvous, apply, cadence, and
enclosing bounds. An unexplained constant copied upward is invalid.

The close is immutable and finite over the leased READY open snapshot,
includes every complete admissible pre-close arrival, never depends on
launched ranks, is not an all-READY barrier, and cannot close early merely
because `Q_min=2` contributions arrived.

### Strict scientific pass

Each stage now requires:

- at least two warm-up and ten measured K40 windows for every trainer;
- at least ten immutable commits/checkpoints;
- independent fresh-process restart verification;
- preparation, boundary-ready, release, all-eight-per-node apply receipts,
  and exactly one node-applied authority for every node and accepted
  transaction;
- no partial apply, stale/superseded identity, double correction, or
  generation gap;
- separate causal `freeze_snapshot`, `snapshot_admission`,
  `publish_network`, `aggregation`, `checkpoint`, `result_wait`, and
  `apply_swap` evidence plus K40 cadence and total foreground idle;
- zero foreground result wait, median cadence no more than 1.25 times raw
  K40, foreground idle below 0.10, and every-event maximum/p99/hard-tail
  enforcement;
- retained bounded resident-byte, immutable-buffer/mailbox, registered-slot,
  credit, replay/reassignment, receipt, exact-token, and full-release proof;
  and
- zero/absent Python or Lustre dense bytes, shared SQLite/locks, disk replay,
  central full-model broker, MPI/all-rank collective, and foreground capacity
  wait.

Only a literal machine-readable `passed=true` verdict permits `wg done`.
Failure, timeout, incomplete/missing evidence, identity drift, or a terminal
collector gap requires the runner to retain evidence and use
`wg incomplete`/`wg fail` as appropriate. That leaves every larger stage
blocked.

## Validation performed

- Re-read each edited WG description after `wg edit`.
- Confirmed each description contains the compute-pool checklist obligation
  and complete R01-R16, NDP01-NDP17, V21S01-V21S17, ISP01-ISP07 citation.
- Confirmed exact 8/32/128 node, partition, QoS, immutable identity, seed,
  K40/commit, atomic receipt, restart, resource, tail, no-duplicate,
  durable-terminal, empirical-deadline, and `passed=true` requirements.
- Ran `wg why-blocked` for every scale task and confirmed the serial blocking
  chain and absence of premature readiness.
- Recorded the WG edits in the quality-pass task log.
- Intentionally ran no build, test, Python, native executable, Slurm
  submission, or live scheduler command.
