# Native G2 artifact-root ownership correction

Task: `fix-g2-artifact-root-ownership`
Date: 2026-07-29
Release-candidate commit: the single commit containing this record; its exact
SHA and `origin/main` equality are recorded in the WG task log after push.
Scheduler mutations: **none**

## Result

The job5109414 pre-dataplane collision is fixed in the canonical native G2
submit and batch paths. `NDP_ARTIFACT_ROOT` now contains an immutable
`emender-native-g2-artifact-ownership-v1` schema. Callers provide actor and
scheduler identities to
`scripts/frontier/native_g2_artifact_namespace.py`; they never provide an
arbitrary destination path.

| Owner | Exclusive retained namespace | Enforced publication |
|---|---|---|
| submit/controller | `controller/<payload-job-id>/scheduler-evidence/` | Complete canonical records are content addressed and atomically hard-linked without replacement. Immediate capture runs `squeue -h -j <id> -o '%i\|%T\|%P\|%q'` and `scontrol show job -dd <id>`. |
| native G2 batch | `<payload-job-id>` | An atomic no-replace relative symlink publishes a batch-owned directory under `.batch-storage/` that already contains `.artifact-owner.json`. Any existing path exits 73 and remains byte-for-byte unchanged. |
| scheduler-owned `afterany` collector | `collectors/<collector-job-id>/payload-<payload-job-id>/` | Complete canonical collection records are content addressed and atomically hard-linked only in the collector namespace. The collector may read but cannot name or write the batch/controller root through this API. |

Frontier Lustre rejects Linux `renameat2(RENAME_NOREPLACE)` with `EINVAL`.
That behavior was exercised before commit. The reviewed implementation
therefore uses Lustre-supported atomic primitives: hard-link publication for
fully written/fsynced immutable files and `symlink(2)` no-replace publication
for the pre-marked batch-owned directory. The visible job root is never an
empty directory, so a competing directory rename cannot replace it. A
competing `mkdir`, another batch publisher, or a pre-existing failed artifact
can only win first and cause the batch to fail closed with exit 73.

## Exact historical reproduction and fixed flow

The permanent regression
`test_job5109414_legacy_order_reproduces_exit_73` performs the historical
ordering exactly:

1. Initialize retained G2 container `g2/`.
2. Reproduce the operator/monitor write
   `g2/5109414/scheduler-evidence/terminal.txt`.
3. Invoke the real batch namespace publisher for job `5109414`.
4. Observe exit-73 conflict and verify the historical failed evidence was not
   overwritten and no batch owner marker appeared.

This matches the retained scheduler history: fault G2 job 5109414 was
submitted, immediate operator evidence pre-created its final job directory,
and the old batch guard exited `73:0` in eight seconds before the dataplane.

The paired
`test_real_submit_immediate_observation_then_batch_guard_cannot_collide`
drives the actual submit shell with fake Slurm executables, captures the
rendered `sbatch` arguments, returns the historical numeric ID, performs the
submit script's immediate queue observation, and invokes the exact helper used
by the batch script:

```text
fake sbatch -> 5109414
  -> controller/5109414/scheduler-evidence/submitted-<digest>.json
  -> controller/5109414/scheduler-evidence/immediate-<digest>.json
  -> batch publish g2/5109414 -> .batch-storage/5109414.<nonce>.owned
  -> loader-preflight.txt (first dataplane-setup write)
```

The rendered request remains exact `-p batch --qos=debug -N 2 -t 00:20:00`.
The immediate observation retains `5109414|PENDING|batch|debug` and the
separate `QOS=debug` and `Partition=batch` facts from `scontrol`. Neither
controller record creates `g2/5109414`; the batch publishes its complete owner
marker and proceeds to the next dataplane-setup write.

## Reconciliation, conflict, and ordering coverage

The permanent test module also proves:

- equal controller or collector observations reconcile to the same digest
  path with one file, while a later scheduler state appends a distinct
  immutable record;
- existing failed controller evidence is legitimate and never blocks batch
  publication;
- collector registration evidence can precede the batch and terminal
  collection can follow it without either actor entering the batch root;
- a genuine existing authoritative batch directory and its failed evidence
  remain untouched while the new batch exits 73;
- duplicate batch publication has exactly one winner;
- batch publication racing a `mkdir` has exactly one winner, and a foreign
  winner remains an empty conflicting directory rather than being overwritten;
- batch publication racing a directory `rename` either publishes the
  pre-marked batch target and blocks the rename or preserves the foreign target
  and returns exit 73;
- controller/collector symlink redirection into a batch root is rejected
  before any evidence write; and
- the CLI returns literal exit 73 and the historical refusal message on an
  existing authoritative root.

## Compute-pool conformance checklist

Authorities read in full:

- `RESILIENT_DILOCO_COMPUTE_POOL.md`, version 1, including its mandatory
  conformance checklist;
- `RESILIENT_DILOCO_GAP_MATRIX.md`, including the only normative
  V21S01–V21S17 and ISP01–ISP07 crosswalks; and
- the G2 and durable-publication portions of
  `NATIVE_RESILIENT_DILOCO_DATAPLANE.md` and
  `ASYNC_V21_EXECUTION_SOURCE_IDENTITY.md`.

This is retained-evidence ownership, not a change to membership, numerical
aggregation, snapshot overlap, or scale policy. The applicable maps below name
the direct proof and explicitly state which invariants are preserved rather
than falsely claiming new dataplane or training evidence.

### R01–R16

| ID | Conformance for this change |
|---|---|
| R01 | **Direct.** Numeric Slurm job ID, immutable run/payload identity, create-once batch owner marker, and exclusive writer namespaces fail closed before dataplane mutation. No shared database or mutable lock is introduced. |
| R02 | Membership lifecycle is unchanged; the artifact guard runs before native DISCOVER/BOOT/SYNC/READY and cannot manufacture a peer or incarnation. |
| R03 | Active-world semantics are unchanged; queue node count is retained evidence only, never membership authority. |
| R04 | **Direct.** Canonical content-addressed observations make identical replay idempotent; reserved identity conflicts and existing authoritative roots are rejected without mutation. |
| R05 | No aggregate bytes, weights, denominator, or arithmetic path changes. |
| R06 | **Direct for bounded control.** Each publication is a finite syscall sequence with no polling/retry loop; conflicts terminate deterministically at exit 73. Training quorum/deadline policy is unchanged. |
| R07 | **Direct.** Complete fsynced records publish exactly once; no conflicting current artifact is overwritten. This is retained scheduler/G2 evidence, not a substitute for peer commit receipts. |
| R08 | No dense broker/chunk/replay path changes. Controller/collector records are bounded metadata; batch artifacts keep their existing checksums and release behavior. |
| R09 | No manager/trainer state ownership changes. |
| R10 | **Direct boundary preservation.** The only Lustre writes are append-only retained scheduler/job/collector evidence. No live membership, heartbeat, aggregate, redistribution, or shared database is added. |
| R11 | **Direct for observation retry.** A repeated monitor/collector invocation reconciles idempotently. Native disappearance/rejoin behavior is unchanged. |
| R12 | **Direct for retained failure/retry evidence.** Historical failure bytes and immutable identities are never overwritten; a later changed job ID obtains a disjoint root. Checkpoint/outer-state recovery is unchanged. |
| R13 | The helper is scheduler-adapter metadata and introduces no backend assumption into the native protocol. Slurm-specific capture remains under `scripts/frontier/`. |
| R14 | **Direct.** Immediate evidence names commands, return codes, stdout/stderr, and separate Partition/QOS fields; conflicts have explicit terminal reasons. No overlap timing claim is made. |
| R15 | Numerical/reference behavior is unchanged. The checksum-linked native reference record was refreshed solely because its normative document is digest-bound. |
| R16 | **Direct.** The hard G2 artifact is now publishable only by its exact two-node batch actor; legitimate monitoring cannot fabricate or block the G2 job root, and conflicting artifacts still block promotion. No 4+ work is authorized. |

### NDP01–NDP17

| ID | Conformance for this change |
|---|---|
| NDP01 | **Direct.** Python remains the scheduler/evidence adapter and the batch remains the sole native G2 artifact owner; no Python dense control or shared database is added. |
| NDP02 | No MPI, collective, rank barrier, or shutdown path is introduced. |
| NDP03 | Exact `cxi`, two-node service, and provider requirements remain in the rendered `sbatch`; this test uses fake Slurm and does not claim live provider evidence. |
| NDP04 | No snapshot/handoff bytes change; controller evidence cannot enter the batch storage directory through the API or a symlink redirection. |
| NDP05 | Exact native arithmetic and layout are unchanged. |
| NDP06 | **Direct.** Batch marker binds job/run/payload; controller/collector records bind payload and collector job IDs under versioned schemas. |
| NDP07 | Endpoint exchange and AV routing are unchanged. |
| NDP08 | Metadata records and namespace paths are bounded; this change allocates no dense buffers or unbounded queue. |
| NDP09 | Fabric credits and foreground progress are unchanged. |
| NDP10 | **Direct.** Immutable records are idempotent; reserved-field reuse, symlink redirection, and batch-root conflicts fail closed without once-only state replacement. |
| NDP11 | Dense replay/reassignment is unchanged. Failed artifact evidence remains available for operator reconciliation. |
| NDP12 | Redistribution is unchanged. |
| NDP13 | **Direct for the pre-dataplane deadline boundary.** Namespace conflict is route-local, immediate, and exit 73; no wait or whole-allocation side effect is added. |
| NDP14 | Native ABI/seqpacket code is unchanged; the ownership helper is standard-library scheduler metadata. |
| NDP15 | **Direct for durable G2 handoff.** Only complete pre-marked batch storage is made visible, while controller and collector use complete immutable file publication. Checkpoint/apply behavior is unchanged. |
| NDP16 | **Direct.** Scheduler commands/results, owner, schema, job IDs, evidence kind, and publication policy are machine readable. Missing scheduler data is retained honestly rather than treated as a pass. |
| NDP17 | **Direct.** The real G2 submit/render/batch guard is the regression surface. The artifact remains exact-code/two-node/batch/debug gated, and no Slurm or later rung is run here. |

### V21S01–V21S17

| ID | Conformance for this change |
|---|---|
| V21S01 | **Direct identity boundary.** The new G2 ownership/evidence schemas are versioned and do not relabel v1/v2.0/v2.1 payloads. |
| V21S02 | Lag clocks/drop/defer/foreground catch-up behavior is unchanged. |
| V21S03 | Exact-token-only weight/quorum math is unchanged. |
| V21S04 | K40 and eta-one outer update are unchanged. |
| V21S05 | **Direct.** Job, run, payload, and collector identities are fixed strings before publication; exact two-node batch/debug render remains enforced. |
| V21S06 | Mutable trainer/snapshot ownership is unchanged; no background actor gains access to batch or trainer state. |
| V21S07 | Atomic result apply behavior is unchanged. |
| V21S08 | **Direct analogy at the evidence boundary.** Old/conflicting records cannot replace a canonical record; capacity-one result mailbox semantics are unchanged. |
| V21S09 | Dense resident/credit/replay/mailbox bounds are unchanged; metadata publication is finite. |
| V21S10 | Leased READY membership and one-node-authority prohibition are unchanged; scheduler `Nodes=2` is evidence, not membership. |
| V21S11 | Eight-trainer apply/recovery is unchanged. |
| V21S12 | Native CXI/memfd/point-to-point transport is unchanged and the ownership helper carries no dense bytes. |
| V21S13 | Immediate Partition/QOS evidence is improved, but no new phase timing/overlap pass is claimed. |
| V21S14 | **Direct.** Immutable G2/controller/collector identities and failed evidence are durable and retry-safe; model/outer/token/seed restore is unchanged. |
| V21S15 | **Direct.** G2 retained evidence can no longer be invalidated by legitimate monitoring; exact separate Partition/QOS evidence is captured. This is not a five-gate v2.1 pass. |
| V21S16 | A collision/failure remains blocking and cannot advance promotion; no scale authorization or submission occurs. |
| V21S17 | Scale close policy is unchanged; no constant, arrival distribution, or 4+ render is introduced. |

### ISP01–ISP07

| ID | Conformance for this change |
|---|---|
| ISP01 | Snapshot coherence/live-state ownership is unchanged; the helper handles scheduler metadata only. |
| ISP02 | No trainer foreground path calls this helper; snapshot/OWNED pause behavior is unchanged. |
| ISP03 | No checkpoint, hash, or background dense input changes. |
| ISP04 | The new metadata paths are finite/content addressed; this is not evidence for snapshot/mailbox capacity behavior. |
| ISP05 | Result apply visibility/timing is unchanged. |
| ISP06 | Scheduler identity telemetry is retained separately, but no missing phase telemetry is added or claimed. |
| ISP07 | No checkpoint-count, median-only, cadence, or overlap claim is made. The 200-second tail-stall gate remains independent. |

Minimum progress remains v2.1 `Q_min=2`, `T_min=3,934,080` for its exact
two-node training profile. G2 itself loads no model and this metadata fix does
not change any admission floor, READY set, numerical state, or deadline.

## Validation commands and results

All Python/native commands were run only after:

```bash
source scripts/frontier/activate_emender_frontier.sh
```

The activated interpreter was Python 3.12.13 and every Python command used
`"$EMENDER_PYTHON"`; the canonical build used
`PYTHON_BIN="$EMENDER_PYTHON"`.

Regression first:

```bash
"$EMENDER_PYTHON" -m pytest -q tests/test_native_g2_artifact_namespace.py
```

Before implementation: **7 failed**, all because the ownership helper did not
exist. After implementation and expanded coverage: **9 passed**.

Frontier Lustre-local regression (not `/tmp`):

```bash
mkdir -p build/g2-ownership-lustre-tmp
TMPDIR="$PWD/build/g2-ownership-lustre-tmp" \
  "$EMENDER_PYTHON" -m pytest -q \
  tests/test_native_g2_artifact_namespace.py
```

Result: **9 passed**. This run first detected unsupported
`RENAME_NOREPLACE`, then passed after the hard-link/symlink handoff correction.

Canonical native build/preflight:

```bash
PYTHON_BIN="$EMENDER_PYTHON" BUILD_JOBS=8 \
  scripts/frontier/build_native_resilient_dataplane.sh
```

Result on the exact narrow `origin/main` candidate:
configure/build/install succeeded; **11/11 CTests passed**, including
coordination kernel, protocol/owner, fabric multiprocess, RPC, ABI, provider
fail-closed, and production-environment gates. The installed manifest was
recorded under ignored `build/native-resilient-dataplane/`.

Focused shell/Python/native gate:

```bash
bash -n scripts/frontier/submit_native_dataplane_2n_gate.sh
bash -n scripts/frontier/native_dataplane_2n_gate.sbatch
"$EMENDER_PYTHON" -m py_compile \
  scripts/frontier/native_g2_artifact_namespace.py
"$EMENDER_PYTHON" -m pytest -q \
  tests/test_native_g2_artifact_namespace.py \
  tests/test_validate_native_dataplane_2n_gate.py \
  tests/test_native_artifact_attestation.py \
  tests/test_native_dataplane_2n_controller.py \
  tests/test_native_dataplane_reference.py \
  tests/test_native_dataplane_failure.py \
  tests/test_native_dataplane_abi.py
```

Result on the exact narrow `origin/main` candidate after the canonical build:
**53 passed**. `git diff --check` also passed.
No `sbatch` command reached Frontier: the live-equivalent test prepends fake
`git`, `squeue`, `scontrol`, and `sbatch` executables and asserts the rendered
request and retained scheduler output. This repository has no
`tests/smoke/manifest.toml`; the permanent focused module is therefore the
available no-Slurm smoke/regression gate for this historical collision.
