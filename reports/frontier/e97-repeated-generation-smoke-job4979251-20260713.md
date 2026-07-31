# E97 repeated-generation 2-node smoke: job 4979251

Task: `run-exact-20-minute`

Date: 2026-07-13

## Result

The sole authorized smoke submission failed before completing generation 0.
It therefore does **not** validate repeated generations and does not unblock a
256-node production launch. No retry or 256-node job was submitted.

Slurm job `4979251` started all 16 requested ranks on two nodes, but the exact
compiled-MPICH path failed while collecting the first aggregate. The aggregate
result referred every consumer to rank 0's header under node-local `/tmp`:

```text
/tmp/emender-erikgarrison/trainpy_async_quorum/4979251-20260713T084120Z/ipc/rank_00000/gen000000/header.json
```

Ranks on the other node could not read that path and raised
`FileNotFoundError`. Slurm terminated the step for task failure. The job ended
`FAILED 90:0` after `00:04:10`, rather than remaining active until the
20-minute scheduler boundary.

## Exact submission identity

| Field | Retained value |
|---|---|
| Origin commit | `2b196d9f299b4714723ebc73b83ed08b05efa778` |
| Job ID | `4979251` |
| Submission attempts | `1` |
| Nodes / ranks | `2` / `16` |
| Partition / QoS | `batch` / `debug` |
| Walltime | `00:20:00` |
| Generations | `1000000` |
| Steps / local steps | `40000000` / `40` |
| Canonical fingerprint | `2d2f302ae30d55fc529e1fec79925fa731a25a2825d9e5abdcff83773c98fedd` |
| Rendered launcher SHA-256 | `106a4dde6b966b0af66a1ac92ea0f459c7a435f81f6e322d92e08f30a2cfad30` |
| Immutable seed SHA-256 | `1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9` |

The pre-submit gate verified a clean tracked worktree with `HEAD` exactly equal
to `origin/main`, an absent task-owned job/state directory, the immutable seed
size and SHA-256, byte-identical canonical launcher content, the 2-node smoke
profile, 16 derived ranks, debug QoS, 20-minute walltime, and the exact final
caps. The earlier temporary `10000/400000` candidate was discarded before any
submission after the authoritative task update; it was never passed to
`sbatch`.

## Runtime evidence

Slurm recorded:

```text
JobID|State|ExitCode|Elapsed|Timelimit|Start|End
4979251|FAILED|90:0|00:04:10|00:20:00|2026-07-13T04:41:16|2026-07-13T04:45:26
4979251.0|FAILED|1:0|00:03:41||2026-07-13T04:41:45|2026-07-13T04:45:26
```

The rank-start ledger contains exactly 16 records across
`frontier06025` and `frontier06026`. All ranks independently reached
`checkpoint_loaded` for generation 0 and then
`compiled_mpich_helper_send_starting`. All 16 compiled helpers reached
`collective_reduce_complete`; eight second-node consumers then reported the
missing rank-0 header. The first merge did not complete:

- completed generations: `0`
- latest generation: absent
- successful merges: `0`
- accepted global updates: no finalized metric
- latest run checkpoint: absent
- scheduler-controlled finalization: not reached

The repeated tracebacks terminate in
`ndm.async_diloco_compiled_mpich.load_aggregate_payload`, where the code reads
`source_header_path` relative to the configured IPC root. The canonical root
was `${TMPDIR:-/tmp}/emender-erikgarrison/...`; on this two-node allocation it
was node-local, not a shared namespace. This is the concrete blocker that must
be repaired and tested before another scheduled smoke is authorized.

## Seed and checkpoint validation

The immutable input checkpoint remained readable at 7,719,679,924 bytes with
SHA-256 `1da27d2e...5b0dc5bca9`. An independent post-run
`torch.load(..., map_location="cpu", mmap=True, weights_only=True)` succeeded
and returned the expected checkpoint dictionary, step 1,525,000, loss
2.4378370428085328, model state, and optimizer state. Before/after size,
nanosecond mtime, and SHA-256 records for `latest_emender_E97_1.3B.json`,
`latest_symlink_target.txt`, `manifest.json`, and `metadata_files.sha256` are
identical. The stable seed pointer was unchanged.

There is no output checkpoint to reload because generation 0 never finalized.
Consequently the task's independent latest-checkpoint reload criterion is
unmet, rather than silently substituted with a reload of the unchanged input
seed.

## Retained artifacts

The committed evidence bundle is
`build/e97-256/repeated-smoke-4979251/`. The durable submission guard,
preflight, and canonical render are retained at:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/run-exact-20-minute
```

The actual run directory is:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_quorum_b4k40_ladder_20260709/20260713/E97_1.3B_step1065000_async_quorum_b4k40_ladder_256n/4979251-20260713T084120Z
```

That run directory retains the complete environment, command, train log, run
manifest/summary, rank ledger, and all compiled-helper traces. The complete
Slurm stdout/stderr are committed under `logs/frontier/trainpy_async_quorum/`.

## Disposition

The exactly-once submission authority is exhausted. A follow-up must repair
the cross-node IPC artifact visibility (or make the aggregate self-contained),
add a multi-node regression that exercises the real filesystem topology, and
obtain fresh authority for any new Slurm submission. Production remains
blocked.
