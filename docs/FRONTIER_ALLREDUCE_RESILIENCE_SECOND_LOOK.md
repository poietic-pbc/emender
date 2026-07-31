# Frontier all-reduce resilience: a second look

**Status:** Research assessment and recommendation; no production-code change,
Slurm submission, or scale authorization
**Date:** 2026-07-30
**Authorities reviewed:** [Compute Pool v1](RESILIENT_DILOCO_COMPUTE_POOL.md),
[ADR-002 async v2.1](ASYNC_DECOUPLED_DILOCO_V2.md), the
[gap matrix](RESILIENT_DILOCO_GAP_MATRIX.md), and
[ADR-003 thousand-GPU topology](ADR_THOUSAND_GPU_RESILIENT_DILOCO_TOPOLOGY.md)

## Reading labels

This report deliberately separates four kinds of statements:

- **Verified fact** — supported by a linked primary source or by an exact check
  of the currently approved Frontier environment.
- **Repository evidence** — supported by a named, committed Emender source,
  test, report, or retained run artifact. It is not automatically a general
  result.
- **Hypothesis** — a falsifiable interpretation that still needs an experiment.
- **Recommendation** — an architecture or experiment proposed here; it is not
  current conformance or authorization.

## Executive answer

**Second-look finding:** Emender was right to reject one failure-sensitive
collective spanning the entire elastic membership, but it over-generalized that
constraint into a ban on collectives in the production data plane. The project
did not establish that its proven bucketed RCCL data plane was intrinsically
unusable. It established that an *existing communicator* cannot silently skip a
dead rank and that launched-rank membership is the wrong global resilience
contract. Those are different claims.

The evidence is unusually strong on both sides:

1. **Repository evidence — the collective fast path worked.** Bucketed global
   RCCL completed eight 32-node/256-rank ScheduleFree x/z merges in
   `2.650–2.729 s` (mean `2.684625 s`). The hierarchical
   reduce/root-allreduce/broadcast path completed four 128-node/1,024-rank
   merges in `6.380–6.629 s` (reported mean `6.5296 s`). A later fixed-world
   compiled Cray MPICH hierarchy reduced a 2,048-rank, 5,506,770,496-byte
   update in exactly `5.304643992334604 s` at 256 nodes. These are different
   paths and payload representations, so the timings are references, not a
   head-to-head benchmark.
2. **Repository evidence — the resilience stack solved real integrity
   problems.** Scheduler fencing, immutable receipts/checkpoints, exact-token
   math, duplicate/stale/corrupt rejection, bounded buffers/replay, and
   recovery identities are valuable and should survive any redesign.
3. **Repository evidence — most repeated two-node liveness failures were above
   the transport.** Snapshot-admission accounting, compounded deadlines,
   stale progress-stage labels, serial or over-concurrent all-eight result
   preparation, and supervisor role-scope mistakes repeatedly killed healthy
   cohorts. None of those failures demonstrates that bucketed RCCL reduction
   was the cause.
4. **Verified fact — transparent survivor continuation still does not exist.**
   A rank that dies during ordinary RCCL/NCCL all-reduce poisons that operation
   and parent communicator. Timeouts, watchdogs, async-error polling, debug
   logging, and RAS improve detection/diagnosis or abort; they do not make the
   missing contribution disappear safely.
5. **Verified fact — explicit survivor communicator creation now exists.** The
   approved environment currently exposes Cray MPICH 9.1
   `MPIX_Comm_revoke/shrink/agree`, RCCL 2.27.7 `ncclCommShrink`, and PyTorch
   2.10 `distributed_c10d.shrink_group`. These create a **new** communicator
   after excluding named ranks. They do not repair the in-flight all-reduce,
   reconstruct its output, restore model/optimizer state, or prevent Slurm from
   killing the step/allocation. Their runtime behavior on Frontier under an
   actual rank/node loss has not been qualified by this project.

**Recommendation:** evolve toward fixed, independently restartable **eight-GPU
islands/cells**. Keep fast bucketed RCCL inside a cell (and retain the existing
hierarchical path as a larger-cell control); publish one complete immutable
cell delta asynchronously between cells through the fenced native service.
Failure discards the cell's in-flight update, aborts/recreates only its
communicator, and restarts or replaces that cell. A missing cell is an
availability omission at a finite close, never checksum/data corruption.

This is consistent with ADR-003, which selects no production architecture and
permits only two research prototypes: P1 transactional sharded asynchronous
fragments (`Q_min=1`, explicitly not v2.1) and P2 finite-quorum streaming owner
trees over eight-GPU local-SGD islands. It conflicts with ADR-003 if interpreted
as permission to keep an outer collective across cells, scale a full cohort,
weaken atomic all-eight island apply, or claim that P1/P2 is already selected.

Two operational conclusions are intentionally conservative. First, launching a
new `srun` step inside an allocation that remains alive is a plausible cell
restart mechanism, but **no repository artifact or experiment reviewed here
verifies intra-allocation relaunch after a Frontier rank or node loss**. Slurm's
step and `--no-kill` facilities do not supply replacement hardware or repair a
communicator. Second, the measured `2.650–6.629 s` collective merges do not by
themselves justify the elaborate overlap machinery that later produced
approximately 200-second stalls. Retain coherent `OWNED` handoff and background
inter-cell publication as a resilience boundary, but treat communication
*overlap* as an optional optimization: E0 must show material cadence/tail value
against the simpler synchronous cell baseline.

## What was proven, exactly

| Artifact | Exact observation | What it proves | What it does not prove |
|---|---|---|---|
| [32-node merge diagnostic](../reports/frontier/e97-1p3b-32n-diloco-merge-debug-20260627.md) | Job `4908087`; 32 nodes, 256 ranks; 64M-element buckets; eight merges `2662, 2650, 2693, 2713, 2660, 2691, 2729, 2679 ms`; ranks 0, 1, and 255 entered/exited all 20 sf_x and 20 sf_z first-merge buckets | Bucketization avoided the prior monolithic in-training timeout and repeatedly moved full ScheduleFree x/z state through RCCL | Rank-failure tolerance; clean terminal job (it was intentionally cancelled after evidence); superiority to every other topology |
| [128-node hierarchical smoke](../reports/frontier/run-128n-e97-hierarchical-smoke-20260627.md) | Job `4908849`; 128 nodes, 1,024 ranks; group size 4; 64M-element buckets; four merges `6380, 6629, 6528, 6582 ms`; `DILOCO_SYNC_AVG_MS=6529.6`; final consensus checkpoint; `COMPLETED 0:0` | Two-level RCCL reduce → root all-reduce → broadcast ran repeatedly at 1,024 ranks | Recovery after a rank loss; long-duration reliability; direct comparability with the later MPICH representation |
| [256-node exact smoke](../reports/frontier/e97-async-256-rerun-job4974616-20260712.md) | Job `4974616`; 256 nodes, 2,048 ranks; compiled Cray MPICH reduction; all 2,048 updates accepted; `5,506,770,496` aggregate update bytes; merge `5.304643992334604 s`; `COMPLETED 0:0` | A compiled hierarchical fixed-world MPI path could aggregate the full launched cohort quickly and publish a reload-checked checkpoint | Elasticity: its quorum was exactly all 2,048 launched ranks and the helper used failure-sensitive collectives |

The requested “around 2.65–2.73 s,” “around 6.53 s,” and “around 5.30 s”
figures are therefore accurate with the qualifications above. The arithmetic
mean of the four millisecond-rounded printed values is `6.52975 s`; the source
report independently records its underlying aggregate as `6.5296 s`.

## Source map

### Proven/legacy collective data plane

| Location | Role |
|---|---|
| `train.py:374-411` | Bucket size, global/hierarchical topology, group size, construction pacing, completion-barrier, and debug controls. |
| `train.py:1823-1886` | Builds consecutive local groups and a root group. Every rank participates in deterministic group construction. |
| `train.py:1889-1909` | Exact hierarchy: local `dist.reduce(SUM)`, root `dist.all_reduce(SUM)`, divide by global world size, then local `dist.broadcast`; accelerator synchronization separates communicators. |
| `train.py:1923-1958` | Monolithic or bucketed global/hierarchical sum-then-divide. Global buckets call `dist.all_reduce`; hierarchy delegates each bucket to the two-level helper. |
| `train.py:1961-2078` | `diloco_merge` flattens and merges ScheduleFree `x`, `z`, optional `y`, or ordinary parameters; a hierarchical completion barrier prevents later collective overlap. |
| `train.py:2780-2888` | Initial rank-0 broadcast, hierarchical group construction/warm-up, and optional small DDP islands. |
| `tests/test_diloco_merge.py` and `tests/test_diloco_hierarchical_math.py` | Numerical, bucket, ScheduleFree, and hierarchical controls for the collective implementation. |
| `scripts/frontier/compiled_mpich_dense_helper.cpp:638-748` | Fixed-world reference hierarchy: node communicator, leader communicator, bucket-shape agreement, node `MPI_Reduce`, leader `MPI_Allreduce`, node publication/barrier. |
| `tests/test_async_diloco_compiled_mpich.py:47-68` | Asserts the helper actually uses bucketed collective reduction rather than root point-to-point fan-in. |

The implementation is already a **cell primitive**: its groups are explicit,
its state is bucketed, and its hierarchy is exact for unequal final group sizes
because it sums before the single world-size division. The global initialization,
warm-up, and completion barriers are the portions that cannot span a changing
elastic world.

### Native resilient and Python orchestration stack

| Location | Role |
|---|---|
| `docs/RESILIENT_DILOCO_COMPUTE_POOL.md` | Normative v1 lifecycle, fencing, READY membership, failure semantics, overlap contract, and conformance checklist. |
| `docs/RESILIENT_DILOCO_GAP_MATRIX.md` | Normative R/NDP/V21S/ISP crosswalk and explicit current gaps. |
| `docs/ASYNC_DECOUPLED_DILOCO_V2.md` | v2.1 K40, exact-token eta-one math, four lag clocks, bounded mailbox, atomic all-eight apply, and qualification policy. |
| `docs/ADR_THOUSAND_GPU_RESILIENT_DILOCO_TOPOLOGY.md` | Research-only P1/P2 topology assessment; explicitly retains synchronous eight-GPU island collectives and rejects a full-cohort outer collective. |
| `ndm/async_diloco_v2.py:68-1573` | v2.1 policy/identity, exact aggregation, mailbox, worker lane, descriptor service, commit authority, and checkpoint restore. |
| `ndm/async_diloco_v2.py:1618-2240` | Safe-boundary rendezvous and `AtomicEightTrainerApply`. |
| `ndm/async_diloco_real.py:1324` | Persistent asynchronous real-training lane. |
| `ndm/native_pool_runtime.py:247-789` | `NativeManagerSession`, bounded local buffers, transport, freeze, redistribution, checkpoint proposal, commit, and telemetry. |
| `ndm/native_e97_runtime.py:34,484-851` | ABI v2.1 identity and trainer-facing direct native data plane. |
| `ndm/resilient_pool_runtime.py:926-1924` | Native-backed control server/client for READY, generation, contribution, commit, recovery, apply, and owner loss; Python TCP server remains debug-only. |
| `scripts/frontier/resilient_e97_role.py:2213-3084` | Native manager orchestration, owner exchange/reduction, checkpoint publication, peer apply, and node marker. |
| `scripts/frontier/resilient_e97_role.py:3411-4873` | Trainer load, persistent K40 windows, immutable handoff, result materialization, and safe-boundary apply. |
| `scripts/frontier/resilient_e97_allocation_supervisor.py:246-999` | Per-role progress deadlines, cohort stop/restart, and allocation lifecycle. |
| `src/native_resilient_dataplane/src/ndp.cpp:1777` | Persistent service invokes the pure native coordination transition. |
| `native/dataplane/src/fabric.cpp:134-154` | Bounded libfabric `FI_EP_RDM` endpoint selection; Frontier production requires `cxi`. |

### Tests that exclude collectives by construction

The native evidence is strong evidence for a point-to-point backend, but it is
not a controlled comparison showing that collectives inside a bounded failure
domain fail:

- `tests/test_native_dataplane_abi.py:22-45` fails if the native elastic library
  exports any `MPI_`/`PMPI_` symbol.
- `tests/test_native_dataplane_reference.py:134-148` asserts
  `all_rank_collective_allowed=false`, `mpi_allowed_in_elastic_binary=false`,
  and labels the fixed-world reference incompatible.
- `tests/test_resilient_e97_runtime.py:1742-1800` is explicitly
  `test_two_model_free_managers_exchange_without_collective` and scans the role
  source for absence of `mpi4py`, `TCPStore`, `RCCL`, and `all_reduce`.
- `scripts/frontier/validate_native_dataplane_2n_gate.py:167-168,403-404`
  hard-codes accepted counters `mpi_collectives=0` and
  `all_rank_barriers=0`; the production launcher prints `collective=none`.

**Second-look conclusion:** these tests faithfully implement NDP02/V21S12's
chosen boundary. They cannot serve as an experiment asking whether a small,
restartable cell collective is useful.

## Why the architecture declared collectives nonconforming

The Compute Pool requirement was not “all-reduce is slow.” It was:

- active membership is leased READY peers, not launched ranks (R03);
- ordinary node loss must not force whole-allocation restart while the progress
  floor remains;
- no failure-sensitive all-rank operation may appear in elastic membership,
  endpoint exchange, redistribution, recovery, or shutdown (NDP02);
- v2.1 dense inter-node traffic is bounded point-to-point CXI, not MPI or a
  Python/Lustre broker (V21S12).

A `train.py` world process group violates those requirements because its rank
set is fixed, every collective requires matching participation/order, and the
hierarchical helper includes default-world construction/warm-up/completion
barriers. The compiled helper is even more explicit: it checks descriptor
counts/layout with `MPI_Allreduce` over `MPI_COMM_WORLD` before reducing.

That proves **nonconformance as a global elastic backend**. It does not prove
nonconformance inside a cell whose failure contract is “abort this cell,
discard its uncommitted delta, and recreate the cell.” ADR-003 already makes
this distinction: NDP02 remains point-to-point between islands while RCCL/FSDP
is the fast local island implementation.

## The liveness failures were mostly contract/orchestration failures

The following retained failures occurred with collectives deliberately absent:

| Attempt/report | Failure layer | Exact mechanism |
|---|---|---|
| [5078907](validation/qualify-simple-async-v21-2n-clean-attempt-5078907.md) | Overlap/validator and live-state ownership | Runtime made 12 commits, but alternating ~0 s/~199–212 s gaps yielded foreground idle `0.575375`; the endpoint path also reread live GPU model state instead of the frozen endpoint. |
| [5079966](validation/qualify-simple-async-v21-2n-clean-attempt-5079966.md) | Snapshot-admission contract | All captures met 1 s, but a redundant full-state digest and telemetry I/O were charged before `OWNED`, exhausting the same deadline. |
| [5080178](validation/qualify-simple-async-v21-2n-clean-attempt-5080178.md) | Host-memory/copy orchestration | Eight simultaneous pageable GPU→CPU full-state copies contended; one trainer exceeded the 1 s synchronous admission budget while 15 remained healthy. |
| [5080070](validation/qualify-simple-async-v21-2n-clean-attempt-5080070.md) | Deadline composition | Peers started a 180 s wait before independently bounded readiness, 42.85 s materialization, and checkpoint write; the valid composite took 198.76 s. |
| [5080289](validation/qualify-simple-async-v21-2n-clean-attempt-5080289.md) | Supervisor progress labeling | Native owner transport completed healthy in ~130 s, then checkpoint/apply remained labeled `owner_transport`; the supervisor killed both cohorts at ~181 s. |
| [5080469](validation/qualify-simple-async-v21-2n-clean-attempt-5080469.md) | All-eight preparation/release contract | A one-reader chain prepared ranks serially, but the 60 s apply clock began when rank 0 became ready. Later ranks could not possibly finish. |
| [5080730](validation/qualify-simple-async-v21-2n-clean-attempt-5080730.md) | Capacity/read scheduling | Removing the one-reader credit let all 16 trainers materialize the same 5.5 GB result concurrently; 15 took ~262–308 s and failed a 60 s SLO. |
| [5080902](validation/qualify-simple-async-v21-2n-clean-attempt-5080902.md) | Authority semantics | The allocation supervisor treated later `published_node_applied` as the only proof of the already-published committed latest and killed both managers mid-apply. |
| [5081182](validation/qualify-simple-async-v21-2n-clean-attempt-5081182.md) | Role scope | The allocation-wide first-commit predicate was also run against trainer progress documents that intentionally lacked manager commit authority, stopping healthy cohorts. |

The later [safe-boundary qualification](validation/qualify-v21-safe-boundary-2n-20260727.md)
(job `5099195`) eventually recorded ten commits, 160 applied receipts, zero MPI
collectives/all-rank barriers, cadence `1.0042x` raw K40, and a passing machine
verdict. That is evidence that the stack can be made to work. It also shows how
much liveness machinery was spent coordinating a local all-eight transaction.
It is not evidence that every component is unnecessary.

**Hypothesis:** the project paid a complexity tax twice: once for correct global
failure isolation and again for local coordination that an optimized cell
collective plus cell-level restart could simplify.

## What actually happens when a rank dies

### Ordinary RCCL/NCCL collective

**Verified fact:** the in-flight operation cannot produce a valid “all-reduce
minus the dead rank.” The surviving ranks may hang until transport/watchdog
failure or receive a remote/system/asynchronous error. Output from an aborted
operation must be discarded. The parent communicator must be aborted/revoked,
and any continuation uses a newly created communicator or restarted worker
group.

Current [NCCL communicator documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#error-handling-and-communicator-abort)
classifies system/transport errors as fatal for the communicator and says to
abort and recreate it. Its [fault-tolerance section](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#fault-tolerance)
adds `ncclCommShrink(..., NCCL_SHRINK_ABORT)` for a **new** communicator that
excludes failed ranks. The current [RCCL 2.27.7 API](https://rocm.docs.amd.com/projects/rccl/en/docs-7.1.1/api-reference/api-library.html#_CPPv414ncclCommShrink10ncclComm_tPiiP10ncclComm_tP12ncclConfig_ti)
likewise exposes `ncclCommShrink`, reorders surviving ranks contiguously, reports
`ncclRemoteError` for a dead remote/network error, and exposes
`ncclCommGetAsyncError` and `ncclCommAbort`.

PyTorch 2.10 documents
[`distributed_c10d.shrink_group`](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.distributed_c10d.shrink_group):
only non-excluded ranks call it; `SHRINK_ABORT` attempts to terminate parent
operations; shrinking the default group replaces it and destroys other groups
because rank reassignment makes them inconsistent. That is explicit survivor
continuation **after communicator replacement**, not transparent continuation
inside the failed collective.

### Which settings only detect, diagnose, or abort?

| Mechanism | Provides | Does not provide |
|---|---|---|
| `torch.distributed` process-group `timeout` | Bounds collective wait (documented NCCL default: 10 minutes); PyTorch aborts collectives asynchronously and crashes the process because continuing after failed async GPU work is unsafe | A reduced survivor result or repaired process group |
| [`TORCH_NCCL_ASYNC_ERROR_HANDLING`](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html) | Modes that abort communicator and/or tear down the process when the watchdog observes an error | Membership selection, checkpoint restoration, or survivor agreement |
| `TORCH_NCCL_ENABLE_MONITORING`, heartbeat timeout, trace/dump settings | Kills a process when the watchdog stalls and retains diagnostics | Recovery |
| `ncclCommGetAsyncError` / `ncclCommAbort` | Polls asynchronous communicator error; aborts in-flight calls and destroys resources | Reconstructed collective output |
| `NCCL_DEBUG` / RCCL debug subsystems | Logs initialization, collectives, transport, and errors | Fault tolerance |
| NCCL RAS (`NCCL_RAS_ENABLE`, timeout factor, status client) | Current NVIDIA NCCL health/hang/crash diagnostics | Automatic shrink or model recovery; AMD RCCL 2.27.7 documents RAS as a debug subsystem, not the full NVIDIA RAS control service |
| Frontier `FI_CXI_RDZV_PROTO`, CQ/TX sizes, GDR/plugin variables | Rendezvous, transport, and performance behavior used by passing runs | Dead-rank semantics |
| `MPI_ERRORS_RETURN` | Allows an MPI operation to return an error to application code | Standard-MPI guarantee that the old communicator remains usable |
| `MPIX_Comm_revoke/shrink/agree` | ULFM-style error propagation, survivor communicator creation, and survivor agreement | Automatic application-state rollback or replacement ranks |

### Frontier and the approved environment

**Verified official fact:** the [Frontier user guide](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html#mpi)
says Frontier's MPI is GPU-aware Cray MPICH. Its known-issue guidance associates
fatal MPICH/OFI `UNDELIVERABLE` failures with node failure and tells users to
check Slurm `NODE_FAIL`; its signal guidance makes checkpointing application
work.

**Verified environment fact (2026-07-30, no Slurm launch):** after sourcing
`scripts/frontier/activate_emender_frontier.sh`, the canonical environment
loaded CPE 26.03, `cray-mpich/9.1.0`, ROCm 7.1.1, RCCL 2.27.7, and
`torch 2.10.0+rocm7.1`.

- Cray MPICH's `mpi_proto.h` declares `MPIX_Comm_revoke`,
  `MPIX_Comm_shrink`, `MPIX_Comm_agree`, failure acknowledge/query APIs, and
  the process-failed/revoked error classes; `libmpi_gnu.so.12` exports them.
  A compile/link-only program taking those three function addresses succeeded.
- HPE's current [Cray MPICH 9 `MPIX_Comm_shrink`](https://cpe.ext.hpe.com/docs/latest/mpt/mpich9/MPIX_Comm_shrink.html)
  page says it creates a new communicator excluding failed processes; HPE also
  documents [`MPIX_Comm_revoke`](https://cpe.ext.hpe.com/docs/latest/mpt/mpich9/MPIX_Comm_revoke.html)
  and [`MPIX_Comm_agree`](https://cpe.ext.hpe.com/docs/latest/mpt/mpich9/MPIX_Comm_agree.html).
- `/opt/rocm-7.1.1/include/rccl/rccl.h` declares RCCL 2.27.7
  `ncclCommShrink`, `NCCL_SHRINK_ABORT`, async error, and abort; a
  compile/link-only check against `librccl.so.1` succeeded.
- The installed PyTorch module exposes `distributed_c10d.shrink_group`,
  `SHRINK_ABORT`, and `ProcessGroupNCCL.shrink`.

This corrects any blanket statement that “ULFM/shrink is unavailable on
Frontier.” **Compile/link exposure is verified. Operational support is not.**
No rank was killed, no `srun` survival behavior was tested, no GPU-aware ULFM
collective was recovered, and no PyTorch/RCCL cell was shrunk in this task.
OLCF confirmation plus a controlled small test is required before relying on
it. The comparable [Open MPI ULFM documentation](https://docs.open-mpi.org/en/main/features/ulfm.html)
warns why: schedulers/launchers often clean up the entire application on a
process failure, so runtime support and launcher policy matter as much as API
symbols.

### Slurm boundary

The normal Frontier pattern remains fixed resources and gang-oriented steps.
Slurm's [`--no-kill`](https://slurm.schedmd.com/srun.html#OPT_no-kill) can keep
an allocation after a node failure, but the application must tolerate the lost
resource; it does not add a replacement node. `--kill-on-bad-exit` controls
whether one bad task kills the step, not communicator repair. Slurm
[`--requeue`](https://slurm.schedmd.com/sbatch.html#OPT_requeue) restarts the
batch script; the application must load its own checkpoint. Preallocated spare
nodes/ranks or a later allocation are required for replacement capacity.

A fresh `srun` can generally create another step on resources that remain in a
live allocation, so process-only cell restart on healthy nodes is a reasonable
prototype. It is **not yet a verified Frontier recovery path in Emender**: this
review did not inject a failure, test whether Frontier's step/launcher policy
preserves the allocation, or relaunch any rank. After physical node loss the
failed node is unavailable and Slurm does not grow the allocation, so relaunch
requires surviving reserved resources (usually a preallocated spare cell) or a
new/requeued allocation. E1 therefore treats allocation survival and relaunch
as measured outcomes, not assumptions.

## Recovery choices

| Approach | Simplicity | Failure blast radius | No-fault performance | Maturity on Frontier/PyTorch | Principal cost/risk |
|---|---|---|---|---|---|
| Whole-job immutable checkpoint + requeue/resubmit | Highest | Entire allocation; work since checkpoint lost | Fast collective path unchanged | Most conventional and operationally mature | Large restart time and correlated I/O; high expected waste at thousands of GPUs |
| TorchElastic all-worker restart and process-group recreation | High/medium | Entire worker group restarts even for one failed worker | Fast during steady state | Mature PyTorch semantics; Frontier multi-node rendezvous/job-manager integration still must be engineered | All survivors killed; rank/world-size changes; checkpoint required |
| Fixed stable cells + asynchronous inter-cell publication | Medium | One cell's in-flight interval/update; healthy cells continue | Keeps RCCL fast path within cell; removes global collective tail | Architecture hypothesis; directly aligned with ADR-003 P1/P2 but not yet implemented/qualified | New statistical policy, cell scheduler/supervisor, and durable publication protocol |
| Preallocated spare cell/ranks | Low simplicity | One cell if replacement is cell-scoped | Idle reserve reduces utilization; steady path stays fast | Common HPC technique in principle, custom in Emender | Scheduler allocation cannot grow; state transfer and rank/incarnation remap must be fenced |
| ULFM Cray MPICH survivor communicator | Low simplicity | Communicator can shrink; application chooses rollback scope | Standard collectives retained; detection/rebuild overhead only on fault | APIs are exposed in approved Cray MPICH 9.1; Emender runtime behavior untested | Application recovery, GPU/PyTorch integration, launcher survival, and changed numerical world remain its responsibility |
| RCCL/PyTorch `shrink_group` | Medium/low simplicity | Selected process group/cell if groups were isolated correctly | Native GPU collectives retained | Very new but present in RCCL 2.27.7/PyTorch 2.10; no Emender fault test | Must identify exclusions out of band, discard in-flight output, destroy inconsistent groups/DDP state, and decide whether reduced-size training is valid |

Whole-job restart is the conservative fallback. True in-allocation survivor
continuation is possible only with explicit application/runtime cooperation;
there is no setting that turns an arbitrary current `train.py` world group into
that system.

## Evolutionary architecture

### Boundary

```text
one stable cell (initially one Frontier node / eight GPUs)
  trainers + RCCL local SGD / bucketed model-state merge
  -> coherent immutable complete cell delta + exact tokens
  -> native fenced point-to-point publish (P1 direct owners or P2 owner tree)
  -> finite accepted set / one immutable global commit
  -> capacity-one verified result
  -> atomic cell apply at a safe boundary, or restart the cell
```

### Keep

1. **Bucketed collective math and instrumentation.** Start with the proven
   64M-element global sum/divide inside a cell. Retain `sf_x/sf_z` semantics,
   merge debug, exact full-fresh parity, and the existing hierarchical
   reduce/root-allreduce/broadcast implementation as a larger-cell control.
2. **Potential reduce-scatter/all-gather optimization.** It can reduce peak
   replicated traffic for a sharded cell, but is not the proven path. Add it
   only after all-reduce parity and rank-loss behavior are baselined.
3. **Native integrity core.** Keep scheduler fence/incarnation, versioned
   identities, CRC/SHA/nonfinite validation, exact-token deterministic
   weighting, idempotent receipts, immutable manifest/checkpoint lineage,
   bounded credits/replay/mailbox, and fresh-fence recovery.
4. **Immutable overlap contract.** Snapshot one coherent boundary, return
   `OWNED`, perform network/aggregation/checkpoint in background, and apply only
   a complete verified result at a later cell boundary.
5. **Causal telemetry.** Preserve ISP01–ISP07 and report every tail, including
   communicator abort/rebuild and discarded-cell work.

### Change

1. Scope a collective to a **cell communicator**, never the leased global
   membership. Avoid a default world process group for cell data.
2. Give each cell an independent supervisor/step. On one rank loss: revoke or
   abort the cell communicator, discard the incomplete bucket/update, stop the
   remaining cell ranks, and recreate all eight from the latest verified cell/global
   anchor. Shrink-to-seven is a research arm, not the default, because it changes
   token rate, optimizer dynamics, and device topology.
3. Publish at most one complete immutable delta per cell transition. A P1/P2
   manifest either accepts the entire cell transaction or none of it.
4. Replace “all cohort members must arrive/apply” at global scope with a finite
   close over complete eligible cell arrivals. Missing cells are listed as
   omissions; corrupt means a present payload violated identity/checksum/layout/
   finiteness.
5. Retain a positive token/diversity floor where the algorithm requires it, but
   do not use quorum as a proxy for payload integrity. If the floor is absent,
   publish no commit; healthy local training may continue only within its bounded
   lag policy.

### Failure sequence

1. A collective timeout/async error or supervisor exit identifies the failed
   **cell**, not a valid partial merge.
2. Fence the old cell incarnation and abort/revoke its communicator. No bucket
   or cell delta from the interrupted transition is admissible.
3. Other cells continue K windows and background publication. The finite close
   does not extend for the failed cell.
4. Restart all ranks of the failed cell on the same healthy resources, a
   preallocated spare cell, or a later allocation. Create a new incarnation and
   new communicator; verify the authoritative immutable result before READY.
5. The repaired cell joins a later transition. No old-incarnation replay is
   accepted unless it was already frozen under its exact identity.

## Requirement disposition

This recommendation is a **research specialization**, not current v1/v2.1
conformance. The Compute Pool v1
[conformance checklist](RESILIENT_DILOCO_COMPUTE_POOL.md#conformance-checklist-required-in-every-implementationrunnerscale-task-validation)
remains mandatory: cite all applicable authorities/IDs; prove leased READY
membership and bounded waits;
keep the compute closure SQLite/database/lock-free; show fencing, deterministic
math, idempotence, rejection, and atomic evidence; keep bounded non-Lustre
transport and no full-model broker; exercise failure/recovery and state the
floor; retain exact commands/artifacts; and for async work prove ISP01–ISP07
causal overlap and hard tails.

“Retire” below means retire the old mechanism from a newly versioned cell
prototype, **not** erase the historical requirement or relabel the prototype
v2.1.

### R01–R16

| ID | Disposition for a cell prototype |
|---|---|
| R01 | **Retain:** scheduler-fenced claim, newer fence before load, no database. |
| R02 | **Retain:** lifecycle, stable cell/worker identity, new incarnation. |
| R03 | **Retain:** global active set is leased READY cells, never launched ranks. |
| R04 | **Retain/revise policy:** fencing/idempotence unchanged; bounded lag only under a new schema. |
| R05 | **Retain:** exact-token deterministic accepted-set math; local collective must match its reference. |
| R06 | **Revise:** finite close and positive floor remain, but missing cells are omission, not corruption; P1 `Q=1` is research-only. |
| R07 | **Retain:** exact-once immutable receipt/checkpoint chain. |
| R08 | **Retain/revise transport:** bounded owners/replay remain inter-cell; allow collectives only inside a restartable cell. |
| R09 | **Retain:** trainers own mutable state; services consume immutable state. |
| R10 | **Retain:** no Lustre/database/dense-Python hot path. |
| R11 | **Retain:** discard unfinished work; new-incarnation catch-up/rejoin. |
| R12 | **Retain:** authoritative outer/token state and fresh-allocation resume. |
| R13 | **Retain:** backend-neutral global protocol; cell collective is an adapter. |
| R14 | **Retain/extend:** add communicator detection/abort/rebuild and discarded-work timing. |
| R15 | **Retain:** high-precision parity, participation accounting, separate convergence. |
| R16 | **Revise gate only:** preserve exact-source small fault gates; the old v2.1 ladder does not authorize a new cell policy. |

### NDP01–NDP17

| ID | Disposition for a cell prototype |
|---|---|
| NDP01 | **Retain:** native global authority and metadata/C++ dense boundary. |
| NDP02 | **Revise wording:** retire the blanket collective ban; prohibit collectives **between cells**, permit them inside a contained restartable cell. |
| NDP03 | **Retain:** persistent C++17 `FI_EP_RDM`/`cxi` inter-cell service. |
| NDP04 | **Retain:** coherent sealed memfd/XPMEM snapshot; no background live read. |
| NDP05 | **Retain:** deterministic binary64 global baseline and exact cell-reference parity. |
| NDP06 | **Retain/extend:** bind cell communicator/incarnation/transition identity. |
| NDP07 | **Retain:** current-fence endpoint routes; communicator membership is cell-local. |
| NDP08 | **Retain:** pre-registered finite buffers and nonblocking exhaustion. |
| NDP09 | **Retain:** background credits distinct from foreground and collective completion. |
| NDP10 | **Retain:** checksums, finiteness, exact-once apply, idempotent receipts. |
| NDP11 | **Retain:** bounded replay/reassignment; add bounded spare-cell activation. |
| NDP12 | **Retain:** one shared complete node/cell result, not eight full copies. |
| NDP13 | **Retain/extend:** route/cell-local containment and absolute recovery deadline. |
| NDP14 | **Retain:** new ABI/wire identity for cell lifecycle metadata. |
| NDP15 | **Retain:** immutable background checkpoint and later atomic cell apply. |
| NDP16 | **Retain/extend:** cell collective/rebuild, omission, and lost-work telemetry. |
| NDP17 | **Revise gate:** add no-fault collective parity and controlled cell rank-loss before the existing CXI/fresh-recovery ladder; no scale inheritance. |

### V21S01–V21S17

| ID | Disposition for a cell prototype |
|---|---|
| V21S01 | **Retain:** new policy/schema/ABI; never relabel P1/P2 or cell mode v2.1. |
| V21S02 | **Retain/revise:** keep four clocks/nonblocking lag disposition; define cell restart interaction. |
| V21S03 | **Retain:** exact tokens remain sole quantitative weight/clock. |
| V21S04 | **Retain as parity arm:** K40 eta-one; any alternative is a convergence sweep. |
| V21S05 | **Retire outside its gate:** exact two-node `Q=2/T_min` remains v2.1-only; new cell floors require a new policy. |
| V21S06 | **Retain:** coherent immutable capture and immediate resume. |
| V21S07 | **Retain:** complete once-only ScheduleFree correction at a safe boundary. |
| V21S08 | **Retain:** verified capacity-one mailbox and nonblocking replacement/defer. |
| V21S09 | **Retain/revise:** finite state remains; “one immutable + one mutable” is per cell. |
| V21S10 | **Retain:** leased cells/new incarnation; missing is neither wait nor corruption. |
| V21S11 | **Revise mechanism:** retire the orchestration-heavy global interpretation; atomic apply/restart remains one eight-GPU cell transaction. |
| V21S12 | **Revise wording:** retire blanket “no MPI/all-rank”; require no **inter-cell** collective while allowing contained RCCL in the island. CXI remains inter-cell. |
| V21S13 | **Retain/extend:** add collective/rebuild causal phases and tails. |
| V21S14 | **Retain:** fenced immutable bundle and newer-fence recovery. |
| V21S15 | **Revise gate:** cell policy needs its own two-node clean/rank-loss/fresh-recovery qualification; old pass cannot be relabeled. |
| V21S16 | **Retire as authorization for this mode:** replace with a new immediate-predecessor ladder only after the cell policy passes; no current scale approval. |
| V21S17 | **Retain for P2/revise for P1:** finite READY-cell close for P2; P1 has one transaction and no peer wait. Neither may key off launched ranks. |

### ISP01–ISP07

| ID | Disposition for a cell prototype |
|---|---|
| ISP01 | **Retain:** coherent immutable cell snapshot and no live-state background access. |
| ISP02 | **Retain:** bounded `OWNED`, immediate next K, no wait on failed/missing cells. |
| ISP03 | **Retain:** only immutable admitted bytes reach collective publication/checkpoint workers. |
| ISP04 | **Retain/extend:** exhaust buffers, credits, communicator-rebuild queue, and spare slots without foreground wait. |
| ISP05 | **Retain:** complete atomic cell apply or whole-cell restart; no mixed anchor. |
| ISP06 | **Retain/extend:** all existing phases plus detect/abort/recreate/reload. |
| ISP07 | **Retain:** raw hard tails; restart success/medians cannot hide ~200 s stalls. |

The intentional NDP02/V21S12 change requires a reviewed new ADR and schema
before implementation can claim conformance. ADR-003's P1/P2 research boundary
is the current safe home for the idea.

## Smallest next experiments — plans only

No command below was run and no Slurm job was submitted by this task.

### E0: no-fault parity and performance replay

1. **Local deterministic replay:** use identical ScheduleFree x/z tensors,
   exact tokens, bucket order, and eta-one update. Compare:
   - current bucketed `dist.all_reduce` within one eight-rank cell;
   - current hierarchical reduce/root-allreduce/broadcast in a bounded
     multi-node cell;
   - native exact-token point-to-point reference;
   - optional reduce-scatter/all-gather only as a later arm.
2. Require full-fresh equal-token numerical agreement at the stated tolerance,
   identical x/z translation, no scalar collectives, and one complete cell
   contribution or none.
3. On a separately authorized smallest Frontier run, replay the known 64M
   bucket settings and record setup, per-bucket, total merge, snapshot,
   publication, result, and apply tails. The success question is parity and
   whether the collective removes local orchestration cost—not whether it beats
   the unmatched 32/128/256 historical numbers.

**Stop condition:** any unexplained numerical difference, mismatched collective
order, hidden world barrier, foreground result wait, or tail regression.

### E1: controlled rank-loss matrix

Use one isolated cell, an independent Slurm step/allocation policy that is
explicitly verified not to gang-kill unrelated cells, and a checkpoint older
than the injected transition. Inject one rank exit (a) before a bucket, (b)
mid-all-reduce, and (c) after collective completion but before cell publication.
Run three recovery arms:

1. baseline detection/abort + restart all cell ranks;
2. RCCL/PyTorch `shrink_group(..., SHRINK_ABORT)` survivor arm;
3. Cray MPICH `MPIX_Comm_revoke/shrink/agree` helper arm.

For every arm require:

- the interrupted output and cell delta are rejected;
- no old communicator is reused and no old-incarnation contribution commits;
- healthy cells continue K windows with zero foreground wait;
- restart/shrink creates a new communicator and verifies the authoritative
  anchor before READY;
- the global manifest records the omitted cell, exact tokens, failure stage,
  detection/abort/rebuild/reload times, and lost work;
- Slurm `Partition` and `QOS` are retained separately and the actual step/node
  survival semantics are captured; and
- no claim is made that a seven-rank survivor has the same training semantics
  as a restored eight-rank cell.

**Decision:** prefer whole-cell restart unless shrink shows a material recovery
advantage and the reduced-size convergence/token-accounting arm passes. Do not
add spares until restart correctness passes without them. Then test exactly one
preallocated spare cell, with a new incarnation and no dynamic-allocation claim.

### E2: P1/P2 protocol choice

After E0/E1, follow ADR-003 rather than inventing a third outer topology:

- P1 tests the simplest one-cell transactional publication and is rejected if
  its lag/eta convergence sweep fails.
- P2 tests finite READY-cell closure and bounded-degree owner trees, with the
  existing synchronous cell as its numerical/control baseline.
- Moshpit/random groups remain a convergence/failure control, not durable
  authority.

Only a separately reviewed new policy can choose among them or authorize the
existing `2 -> 8 -> 32 -> 128` systems ladder.

## Final assessment

The proven collective plane was discarded **too broadly**, not necessarily
unnecessarily. A global launched-rank collective is the wrong resilience
boundary. A bucketed RCCL collective inside an eight-GPU failure-isolated cell
is a different architecture and is already the local topology favored by
ADR-003.

The smallest credible system therefore combines rather than chooses between
the two generations of work:

- legacy RCCL/MPICH evidence for fast dense arithmetic inside a stable cell;
- native fencing, identity, deterministic math, bounded point-to-point
  publication, immutable checkpointing, and recovery between cells; and
- a simpler availability contract in which missing cells are omitted at a
  finite close and corrupt present data is rejected.

The critical correction is semantic: **detection/abort is not continuation;
shrink is new-communicator continuation, not in-collective survival; and cell
restart is not global restart.** The next decision should be made by the
no-fault parity replay and controlled rank-loss matrix, not by another blanket
ban or an unsupported claim that a dead rank can be ignored.

## Primary external sources

- OLCF, [Frontier User Guide](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html)
  (Cray MPICH, GPU-aware MPI, Slurm signals, and node-failure diagnostics).
- HPE CPE 26.03, [Cray MPICH 9 `MPIX_Comm_shrink`](https://cpe.ext.hpe.com/docs/latest/mpt/mpich9/MPIX_Comm_shrink.html),
  [`MPIX_Comm_revoke`](https://cpe.ext.hpe.com/docs/latest/mpt/mpich9/MPIX_Comm_revoke.html), and
  [`MPIX_Comm_agree`](https://cpe.ext.hpe.com/docs/latest/mpt/mpich9/MPIX_Comm_agree.html).
- AMD ROCm 7.1.1, [RCCL 2.27.7 API](https://rocm.docs.amd.com/projects/rccl/en/docs-7.1.1/api-reference/api-library.html)
  (`ncclRemoteError`, async error, abort, and communicator shrink).
- NVIDIA, [NCCL communicator creation/error/fault-tolerance guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html)
  and [`ncclCommShrink` API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommshrink).
- PyTorch 2.10, [distributed communication package / `shrink_group`](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.distributed_c10d.shrink_group),
  [ProcessGroupNCCL environment variables](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html), and
  [TorchElastic failure/membership semantics](https://docs.pytorch.org/docs/2.9/elastic/run.html#failure-modes).
- SchedMD, [`srun`](https://slurm.schedmd.com/srun.html) (`--no-kill`,
  `--kill-on-bad-exit`) and [`sbatch`](https://slurm.schedmd.com/sbatch.html)
  (`--requeue`).
- Open MPI, [ULFM feature and scheduler notes](https://docs.open-mpi.org/en/main/features/ulfm.html)
  (comparable-system survivor continuation and launcher caveats).
