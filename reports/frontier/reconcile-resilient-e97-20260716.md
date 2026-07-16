# Resilient E97 recovery audit after the July 2026 filesystem outage

Audit task: `reconcile-resilient-e97`. Snapshot: 2026-07-16 11:31 EDT. This
was a read-only scheduler audit: the only Slurm commands used were `squeue` and
`sacct`. No job was submitted, cancelled, held, requeued, or modified.

## Resumption decision

The generation-9 checkpoint from job 5000436 and all three jobs' retained
evidence survived. Do not repeat jobs 5000869 or 5009365: both payloads started
16 full trainers and failed host OOM before any generation finalized. Do not
launch commit 642b1b6 as the required gate: it starts only two model-bearing
trainer/manager processes in a 2-node allocation and turns the other fourteen
Slurm ranks into sentinels. The resilient implementation is not on `main` or
`origin/main`; it is clean and pushed on an unmerged task branch.

## Git reconciliation

After `git fetch origin --prune`:

- local `main` has no tracked modifications but is dirty with many unrelated
  untracked WG/runtime/log paths at
  `d8422c21d1fc35f39ee3b721b969b73c01042e76`;
  `origin/main` is `f82e940d8d728aca7e3ae62fbc81bf21323894fc`.
  They diverge after `933323e`: local main has one unique commit and the remote
  has three. Thus local main is neither pushed nor a fast-forward of the remote.
- `wg/agent-1107/complete-resilient-e97` and
  `origin/wg/agent-1107/complete-resilient-e97` both equal clean HEAD
  `642b1b6f33e2d23c73ee43aefa476afa4ccca37e`. The worktree has no tracked or
  untracked changes, is 12 commits ahead of local main, and is not merged.
- None of `11d8c66`, `43853ca`, `df46fc6`, `954b68d`, `0133bff`, `e565534`,
  `3135a2f`, or `642b1b6` is reachable from either local main or origin/main;
  every one is reachable from the local and remote task branch. The other four
  task-only commits are evidence commits `89477ce`, `1abcec5`, `0bbf237`, and
  `4a2ff18`.
- The valid task-only delta is 22 files: the framed resilient transport and
  integration, mmap/released optimizer-state restart changes, runner/rank-lane
  changes, focused tests, four job-evidence logs, and five reports. It must be
  reviewed/rebased against the diverged main histories; logs that said "pushed"
  were correct only for the task branch, not evidence of a merge to main.

## Filesystem and scheduler evidence

Job 5000436's run root remains readable at:

`/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_quorum_b4k40_ladder_20260709/20260715/E97_1.3B_step1065000_async_quorum_b4k40_ladder_256n/5000436-20260715T064518Z`

It retains 10 atomic generation manifests (0--9), 2,048 progress files and
12,288 helper-trace files. Accounting still reports debug QoS, 256 nodes,
02:00:00 limit, 00:42:43 elapsed, and `CANCELLED by 19032`. The evidence shows
all 2,048 ranks reached generation 10 and wrote
`collective_reduce_complete`, with zero `collective_reduce_reduced` or
`return_written`; generation 10 never finalized. The strongest supported
boundary is after return from the blocking compiled collective and before any
rank consumed/published its reduced result. The evidence cannot distinguish
post-collective helper logic from MPI progress/finalization.

Job 5000869's readable run root is
`/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/resilient_e97_foundation_2n/20260715/E97_resilient_2n_20260715T083455Z/5000869-20260715T083737Z`.
Its manifest, command/environment, train log, 16-line rank-start ledger and 16
progress files survive. It has zero generation manifests and zero checkpoints.
Accounting reports debug QoS, two nodes, exactly 02:00:00, terminal `FAILED
90:0`; step `.0` is `OUT_OF_MEMORY 0:125`, MaxRSS 64,857,540K. Its exact last
boundary was generation-0 local training/update construction before manager
exchange, aggregation, apply, or publication.

Job 5009365's readable run root is
`/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/resilient_e97_foundation_2n/20260716/E97_resilient_2n_20260716T081224Z/5009365-20260716T082850Z`.
The same manifest/log/rank-start/progress classes survive (16 ranks and 16
progress files), again with zero generation manifests and zero checkpoints.
Accounting reports debug QoS, two nodes, exactly 02:00:00, terminal `FAILED
90:0`; step `.0` is `OUT_OF_MEMORY 0:125`, MaxRSS 64,590,084K. Retained stderr
identifies rank 4 on frontier06070. Its mmap payload did not cure eight full
model/optimizer workspaces per node.

The current `squeue` snapshot contains no job whose ID/name matches E97,
resilient, 5000436, 5000869, or 5009365. All three named jobs are terminal in
`sacct`; no related survivor is running or pending.

## Last valid checkpoint and handoff semantics

The last atomic checkpoint is:

`/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_quorum_b4k40_ladder_20260709/20260715/E97_1.3B_step1065000_async_quorum_b4k40_ladder_256n/5000436-20260715T064518Z/async_run/checkpoints/emender_E97_100m_20260715/checkpoint_step_1525400_loss_2.4184.pt`

- source job 5000436; generation 9; step 1,525,400; loss 2.41842;
- exact size 15,439,252,298 bytes;
- freshly recomputed with OpenSSL SHA-256 after the outage:
  `ee9d69d9c3efd5696042b30ad1ad57236d5035876bae5ce2e9cc2010e5017fd3`;
- PyTorch 2.10.0+rocm7.1 successfully loaded it using
  `torch.load(map_location="cpu", mmap=True, weights_only=False)`: it contains
  146 model tensors, a ScheduleFree optimizer mapping with 145 state entries,
  step/loss, and `checkpoint_metadata.kind=async_diloco_chain`, generation 9,
  local_steps 40, tokenizer p50k_base, and source step 1,525,000;
- it has no stored `diloco_outer_state`. It is compatible with model and inner
  optimizer restart and with the existing async-chain metadata/step loader,
  but a momentum/SFSGD outer optimizer continuation must use the code's explicit
  `from-loaded-model` bootstrap contract; the file does not prove restoration
  of an already-evolved outer state. The current average-outer path has no
  separate outer state to restore.

`continuation/last-valid.json` and immutable
`continuation/last-valid-20260715T072807Z.json` are byte-identical, readable
1,049-byte manifests pointing to that checkpoint with its size, checksum,
source job, and generation. Their `step` field is null, but the checkpoint
loader and filename both establish step 1,525,400. Cancellation produced no
new scheduler-exit checkpoint. The continuation records merely preserve the
already finalized generation-9 checkpoint. Jobs 5000869 and 5009365 produced
neither a handoff nor an eligible partial checkpoint.

## Reachable resilient behavior and missing topology

The following candidate behavior exists only on the unmerged task branch:

- `QuorumTransportServer` is a single coordinator server (on node rank 0), and
  each model-bearing rank is also its node manager/trainer. On the 642b1b6
  2-node runner this means 2 combined trainer/managers, not 2 managers plus 16
  trainers. All 16 Slurm ranks remain launched/accounted, but 14 are heartbeat
  sentinels with no model, optimizer, training, update, or trainer supervision.
- The data plane is framed point-to-point TCP with size and SHA-256 checks, not
  MPI/RCCL/TCPStore. Per-node buckets are disk-spooled with byte and generation
  retention bounds. Connect/socket, generation, heartbeat, sentinel and apply
  deadlines are bounded.
- The server fences run ID, generation, attempt, coordinator epoch and apply
  payload identity; it rejects late, duplicate/conflicting, corrupt,
  nonfinite, schema-incompatible and wrong-epoch payloads. It freezes the first
  complete quorum, computes an exact token/weight-weighted float64 mean, streams
  aggregate buckets back, and applies `base + eta_outer * mean_delta`.
- Incomplete pre-quorum members can be heartbeat-evicted. The disk spool allows
  bounded crash-surviving resend material and pruning, but the client exchange
  does not itself reconnect/replay after an interrupted exchange. There is no
  complete manager failover/election or demonstrated coordinator-epoch
  recovery. Catch-up/rejoin rules and local-trainer replay are not wired into
  this live runner. Apply acknowledgements exist server-side, but the shown
  client exchange releases its spool after receipt and does not call the
  acknowledgement API; this path is therefore not an end-to-end applied-commit
  proof.
- Each generation trains locally, obtains an aggregate, applies it, and the
  rank-0/global path emits atomic generation/latest/checkpoint artifacts.
  Restart selects the immutable model/optimizer checkpoint and advances its
  step. The two failed 16-trainer payloads proved that this ownership model was
  unbounded; 642b1b6 bounds it by deleting 14 trainers rather than supervising
  them independently.

The precise missing implementation is a process boundary on each physical
node: one lightweight manager owning coordination/spool/aggregate state, plus
eight separately supervised GPU trainer processes owning only their bounded
local model/optimizer state. The manager must collect and locally aggregate all
eight trainer updates, restart/evict an individual trainer without terminating
the node's other trainers, and participate as one node in the global network.
Manager loss must have bounded detection/fencing/recovery; replay, catch-up,
apply acknowledgement, atomic checkpoint selection, and resource bounds must
be end-to-end rather than isolated helpers. Sentinels cannot satisfy any of
those trainer requirements.

## Executable next-task specification

**Title:** Implement bounded E97 node managers with eight independently
supervised trainers and prove only the 2-node recovery gate

**Scope:** Start from the task-branch delta, reconcile it with current
`origin/main`, and implement one non-model-bearing manager per physical node
plus eight independent GPU trainer children per node. Preserve E97 data/model,
ScheduleFree inner optimizer, K=40, weighted outer apply, immutable checkpoint
identity and all 16 Slurm trainer ranks. Remove sentinels from the trainer gate.
Do not authorize a scale ladder, normal QoS, or production action.

**Required implementation/testing before any submission:**

1. Add failing process-level tests for eight trainer registrations per manager,
   local weighted aggregation, trainer crash/timeout restart or eviction while
   peers advance, manager crash fencing/recovery, reconnect/replay, stale and
   duplicate rejection, corrupt/nonfinite payload rejection, catch-up/rejoin,
   applied-commit acknowledgement, atomic checkpoint selection, and bounded
   model/optimizer/spool memory ownership. Then implement them and run the full
   focused resilient/checkpoint suite.
2. Add a launcher dry-run assertion that 2 nodes account for exactly two
   managers and 16 real trainers, each trainer has a distinct GPU and supervisor,
   and there are zero sentinels. Assert debug QoS and exact `02:00:00` walltime.
3. Commit and push the implementation and tests. Record a new code commit and
   payload hash distinct from jobs 5000869/5009365. Recompute the seed checksum
   and rerun its model/optimizer/step/async-chain loader check.

**Mandatory live acceptance gate after those tests:** submit one changed-payload
2-node/16-trainer job in debug QoS with exactly two hours. Record the exact
command, job ID, allocation, rank/manager/trainer ledger, payload hash, and
resource samples. Finalize at least two baseline generations; inject a
non-coordinator trainer failure and a manager/node-step failure; prove bounded
detection, eviction/recovery, fencing, aggregate apply, healthy-participant
accounting, and at least three newly finalized post-injection generations with
advancing step and finite loss. After an immutable checkpoint exists, perform a
controlled whole-allocation termination and, in a fresh 2-node debug allocation
also limited to exactly two hours, reload that exact checksum-verified handoff
and finalize at least two further generations. A failed attempt requires a
diagnosed, tested, committed, pushed changed payload before retry. No scale
ladder or production action follows automatically from success.
