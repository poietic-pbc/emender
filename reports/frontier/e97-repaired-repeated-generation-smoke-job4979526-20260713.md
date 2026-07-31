# Repaired E97 repeated-generation 2-node smoke: job 4979526

Task: `run-fresh-2-node-20`

Date: 2026-07-13

## Result

The sole authorized fresh submission failed after the repaired compiled-MPICH
collective completed once on all 16 ranks but before generation 0 finalized.
It therefore does **not** validate repeated generations, two merges, bounded
repeated-generation scratch, scheduler-controlled finalization, or output
checkpoint reload, and it does not unblock the 256-node production launch.
No retry, larger smoke, or production job was submitted.

Slurm job `4979526` started exactly 16 ranks on two nodes. Every rank loaded
the pinned checkpoint, entered the repaired helper for generation 0, completed
the MPI collective reduce/broadcast path, and wrote a rank-local result. Rank
12 on `frontier07043` was then OOM-killed. Slurm terminated the step and
recorded `FAILED 90:0` after `00:05:05`, rather than allowing it to reach the
20-minute scheduler boundary.

## Exact submission identity

| Field | Retained value |
|---|---|
| Launch commit | `7c3be0e57b2363377c35486a63b1f89005da6332` |
| Job ID | `4979526` |
| Submission attempts | `1` |
| Prior failed job reused/resubmitted | no |
| Nodes / ranks | `2` / `16` |
| Partition / QoS | `batch` / `debug` |
| Walltime | `00:20:00` |
| Generations | `1000000` |
| Steps / local steps | `40000000` / `40` |
| Canonical fingerprint | `37ec44faa6d3337f6b04ab8b5f94619c5941bc9b7e02e9acc2c75164ef3ad645` |
| Helper source SHA-256 | `18f6ffe966b31d2c5bb628822157f9fb629fe109c903cb56594611a9c32aa45b` |
| Python payload bridge SHA-256 | `c09ff7186b0f79b665b6da4206f57a93bbdf9f14bd3ff5de552aee677e7b20fa` |
| Launcher SHA-256 | `106a4dde6b966b0af66a1ac92ea0f459c7a435f81f6e322d92e08f30a2cfad30` |
| Immutable seed SHA-256 | `1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9` |

The pre-submit gate was performed before the one submission: the repaired
dependency was merged and pushed; `HEAD` equaled `origin/main`; the tracked
worktree was clean; the task guard had no job ID; the seed was pinned to step
1,525,000; and the exact two-node, 16-rank, debug-QoS, 20-minute shape and
`1000000/40000000/40` caps were rendered. The durable job guard contains only
`4979526`. Job `4979251` was neither reused nor resubmitted.

## Runtime evidence

Slurm recorded:

```text
JobIDRaw|State|ExitCode|Elapsed|Timelimit|Start|End|NodeList
4979526|FAILED|90:0|00:05:05|00:20:00|2026-07-13T05:28:06|2026-07-13T05:33:11|frontier[07041,07043]
4979526.0|OUT_OF_MEMORY|0:125|00:04:35||2026-07-13T05:28:35|2026-07-13T05:33:10|frontier[07041,07043]
```

The step's recorded maximum RSS was `64683948K`. Stdout identifies task 12 on
`frontier07043` as the OOM-killed rank.

The rank-start ledger has exactly 16 records: ranks 0-7 on `frontier07041`
and ranks 8-15 on `frontier07043`. The helper trace contains, for every rank,
one each of `bridge_enter`, `mpi_initialized`, `request_parsed`,
`collective_reduce_complete`, and `result_written` for generation 0. This is
positive evidence that the repaired collective and rank-local aggregate
handoff crossed both nodes once.

Unlike failed job 4979251, no result refers a consumer to rank 0's node-local
tree. Each trace names only its own rank-local request under
`/tmp/.../ipc/rank_NNNNN/request.gen000000.json`; the helper completed and
materialized a local result for every rank. Thus no cross-node `/tmp`
reference was observed. Because only generation 0 was attempted, the run
cannot establish the required repeated-generation scratch bound.

The finalized run manifest reports:

- rank starts: `16 / 16`
- accepted updates: `0`
- completed generations: `0`
- completed merges: `0`
- output checkpoints: `0`
- validation: `fail`

There is no second generation, second merge, or output checkpoint to reload.
The early OOM, rather than a training cap or scheduler finalization, ended the
job.

## Seed and checkpoint validation

The immutable input checkpoint remains 7,719,679,924 bytes with SHA-256
`1da27d2e...5b0dc5bca9`. An independent post-run
`torch.load(..., map_location="cpu", mmap=True, weights_only=True)` succeeded
and returned the expected dictionary, step 1,525,000, loss
2.4378370428085328, model state, and optimizer state.

The step-1525000 pointer/manifest files retain their pre-job mtimes and the
same hashes recorded by the prior intake evidence. The stable seed pointer is
unchanged. This reload proves only that the immutable input remains healthy;
it is not substituted for the unmet output-checkpoint reload criterion.

## Retained artifacts

The committed evidence bundle is `build/e97-256/repeated-smoke-4979526/`.
It includes the render, launch inputs, source hashes, exactly-once submission
guard, terminal accounting, rank ledger, run manifest, independent seed
reload, pointer hashes, and all 16 per-rank compiled-helper traces. Complete
Slurm stdout/stderr are committed under
`logs/frontier/trainpy_async_quorum/`.

The durable pre-submit/submission directory is:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/run-fresh-2-node-20
```

The actual run directory is:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_quorum_b4k40_ladder_20260709/20260713/E97_1.3B_step1065000_async_quorum_b4k40_ladder_256n/4979526-20260713T092808Z
```

## Disposition

The exactly-once authority is exhausted. Production remains blocked. Any new
Slurm submission requires a separate task with fresh authority after the
per-rank host-memory OOM is diagnosed and repaired.
