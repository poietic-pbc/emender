# E97 node-memory repair and repeated 2-node smoke: job 4979704

Task: `fix-e97-per`

Date: 2026-07-13

## Result

PASS. The repaired compiled-MPICH path completed three generations and three
all-16-rank merges without host OOM or cross-node path failure. Slurm job
`4979704` then ran to its scheduler time limit (`TIMEOUT`, `00:20:01` for a
requested `00:20:00`) rather than terminating at a training cap. No production
or larger smoke job was submitted; production remains manually paused.

## Root cause and repair

Job `4979526` materialized the 5,506,770,496-byte aggregate separately on all
eight ranks of each node. Each Python rank then retained every aggregate bucket,
joined a second whole-payload byte copy, unpacked a complete delta, and allocated
a complete replacement state. Those copies overlapped the seed model and
optimizer reconstruction, producing task-12 MaxRSS `64683948K` and node OOM.

The repair in `dc372fc` makes four bounded changes:

1. Dense local updates serialize directly to bucket files, so a model-sized
   serialized envelope is never retained; the source delta is released before
   MPICH allocates reduction buffers.
2. `MPI_Comm_split_type(..., MPI_COMM_TYPE_SHARED, ...)` elects local rank zero
   as the node aggregate manager. Ranks reduce to that manager; only node
   leaders exchange global bucket sums; only one leader-owned aggregate
   workspace is written per node.
3. Python consumes the leader-owned aggregate one bucket at a time and updates
   the existing CPU base state in place. It does not retain complete aggregate
   bytes, aggregate delta, and old/new state simultaneously.
4. Prior rank-generation workspaces are replaced, the unused seed model copy is
   released, and checkpoint verification uses memory mapping.

Focused regressions cover the shared-node communicator topology, leader-owned
workspace, cross-node node-local paths, bounded streaming serialization,
in-place state identity, collective ordering, and repeated workspace cleanup.

## Exact submission identity

| Field | Retained value |
|---|---|
| Repair commit | `dc372fcdb2d4140f39ce311d169dd29c9e3ca93c` |
| Submitted HEAD | `9fff689c9f9252b6a264773c207f8f8ca8509666` |
| Job ID | `4979704` |
| Attempt submissions | exactly 1 |
| Nodes / ranks | 2 / 16 (8 ranks per node) |
| Partition / QoS | `batch` / `debug` |
| Walltime | `00:20:00` |
| Generations / steps / local steps | `1000000` / `40000000` / `40` |
| Seed step / SHA-256 | `1525000` / `1da27d2e...5b0dc5bca9` |
| Render parity fingerprint | `a4a493eb60c6425f3df2dea71436ded0b43fee282de6f2bac87a0a49e4f0ad5b` |

The rank ledger contains exactly 16 entries across `frontier03980` and
`frontier04048`. Runtime output records `git_commit=9fff689...`; repaired source,
helper, launcher, and payload hashes are retained in
`build/e97-256/repeated-smoke-4979704/repaired-source-hashes.sha256`.

## Repeated generation and memory evidence

Committed generation manifests show:

| Generation | Accepted ranks | Reduce duration | Merge duration | Tokens | Result |
|---:|---:|---:|---:|---:|---|
| 0 | 16 | 78.9659 s | 4.9249 s | 5,245,440 | advanced |
| 1 | 16 | 79.7293 s | 4.8343 s | 5,245,440 | advanced |
| 2 | 16 | 80.2543 s | 4.8963 s | 5,245,440 | advanced |

Each generation reduced 80 buckets totaling 5,506,770,496 aggregate bytes.
The final helper payload points to the node-leader workspace
`rank_00000/gen000002/aggregate.bucketNNNNN.bin`, and the generation workspace
is replaced on each iteration. All 16 ranks recorded collective completion and
local result handoff. MaxRSS plateaued at `61159724K`; it did not grow across
generations and remained below the failed run's `64683948K` peak. There is no
OOM record in terminal accounting or stderr.

Three same-sized output checkpoints were completed:

- step 1,525,040, 15,439,251,857 bytes;
- step 1,525,080, 15,439,251,857 bytes;
- step 1,525,120, 15,439,251,857 bytes.

## Scheduler, checkpoint, and seed validation

Terminal accounting records `TIMEOUT 0:0`, elapsed `00:20:01`, time limit
`00:20:00`; the training step was scheduler-cancelled at the boundary. This is
the required scheduler-controlled finalization, not a training-cap exit.

An independent
`torch.load(latest.pt, map_location="cpu", mmap=True, weights_only=True)`
successfully loaded the final generation-2 checkpoint at step 1,525,120, found
146 model tensors, optimizer state, and `async_diloco_chain` metadata. The five
pinned seed/pointer SHA-256 values exactly match the retained pre-run reference;
`seed-pointer-diff.txt` is empty.

## Evidence and disposition

The committed evidence bundle is
`build/e97-256/repeated-smoke-4979704/`, including render inputs, exactly-once
job guard, repaired hashes, rank ledger, per-generation manifests, terminal
accounting, full logs, helper traces, final metrics, checkpoint reload, and seed
pointer comparison. The durable attempt directory is:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/fix-e97-per/attempt-001
```

No production or larger job was submitted. Production remains paused pending
an explicit user resume.
