# Gate actual MPI compiled-helper 256n debug smoke

Date: 2026-07-08
Task: `gate-actual-mpi`

## Verdict

Decision: **GO, then PASS for exactly one bounded 256n debug smoke**.

The go decision was based only on `run-actual-mpi`, not on failed
`run-mpi-async`. The upstream report
`reports/frontier/run-actual-mpi-20260708.md` records a task-owned
1n -> 8n -> 64n ladder with jobs `4959329`, `4959340`, and `4959370`, all
`COMPLETED` with exit `0:0`, `batch` / `debug`, `00:20:00`, and
`ASYNC_QUORUM_TRANSPORT=compiled-cray-mpich-helper-p2p`.

The upstream evidence satisfied the gate:

- 1n: `8 / 8` rank starts, `8` accepted, no missing/stale/late/timed-out/rejected updates.
- 8n: `64 / 64` rank starts, `64` accepted, no missing/stale/late/timed-out/rejected updates.
- 64n: `512 / 512` rank starts, `512` accepted, no missing/stale/late/timed-out/rejected updates.
- All rungs reported `mode=actual_multinode_compiled_mpich_quorum`,
  `async_quorum_transport=compiled-cray-mpich-helper-p2p`,
  `async_quorum_transport_actual=compiled-cray-mpich-helper-collective-reduce`,
  `transport.helper_result.reducer=mpi_reduce_bucketed_weighted_sum`,
  `transport.helper_result.mpi.collective=MPI_Reduce`, and
  `transport.tcp_dense_data_plane=false`.
- The production latest pointer was unchanged and production `last.pt` was
  absent before and after the ladder.

Therefore TCP was not used as the hot dense aggregation path in the qualifying
ladder, and the production latest/last guard was intact before the 256n debug
smoke.

## Submitted job

Exactly one 256n debug smoke was submitted for this gate:

| Job | Job name | State | Exit | Partition | QOS | Nodes | Requested walltime | Requested node-hours | Elapsed | Actual node-hours |
| --- | --- | --- | --- | --- | --- | ---: | --- | ---: | --- | ---: |
| `4959434` | `actual-mpi-compiled-256n-debug` | `COMPLETED` | `0:0` | `batch` | `debug` | 256 | `00:20:00` | `85.333333` | `00:07:14` | `30.862222` |

Slurm accounting snapshot:

```text
4959434|actual-mpi-compiled-256n-debug|COMPLETED|0:0|batch|debug|00:07:14|00:20:00|256|2026-07-08T16:40:59|2026-07-08T16:47:00|2026-07-08T16:54:14
4959434.batch|batch|COMPLETED|0:0|||00:07:14||1|2026-07-08T16:47:00|2026-07-08T16:47:00|2026-07-08T16:54:14
4959434.extern|extern|COMPLETED|0:0|||00:07:14||256|2026-07-08T16:47:00|2026-07-08T16:47:00|2026-07-08T16:54:14
4959434.0|bash|COMPLETED|0:0|||00:06:40||256|2026-07-08T16:47:34|2026-07-08T16:47:34|2026-07-08T16:54:14
```

The smoke used the run-local debug output root:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/gate_actual_mpi_256n_debug_20260708/20260708/E97_1.3B_step1065000_actual_mpi_compiled_helper_256n_debug/4959434-20260708T204704Z
```

Configuration:

```text
task_id=gate-actual-mpi
smoke_name=256n
run_id=gate-actual-mpi-256n-4959434-20260708T204704Z
scaleout_variant=E97_1.3B_step1065000_actual_mpi_compiled_helper_256n_debug
e97_checkpoint=/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt
slurm_nodes=256
trainpy_launched_ranks=2048
async_expected_ranks=2048
async_global_quorum=2048
async_expected_missing_updates=0
async_timeout_s=1200
async_diloco_quorum_mode=resilient_quorum
async_quorum_transport=compiled-cray-mpich-helper-p2p
async_quorum_transport_actual=compiled-cray-mpich-helper-collective-reduce
allow_frontier_tcp_scale_debug=0
tcp_scale_debug_override=0
requested_walltime=00:20:00
requested_node_hours=85.333333
bounded_debug_transport=actual_multinode_compiled_mpich_quorum
git_commit=dfa126e8a9f9b5824aae840439c3347831fb049b
```

Human approval record embedded in the job environment:

```text
WG gate-actual-mpi: one bounded 256-node MPI compiled-helper async quorum debug smoke after clean run-actual-mpi 1n/8n/64n jobs 4959329/4959340/4959370; batch/debug 00:20:00; run-local latest/checkpoint only; no production latest/last mutation; no 1h/12h/production launch.
```

## 256n validation

The 256n smoke summary reported `Validation: pass` and exit status `0`.

| Metric | Value |
| --- | ---: |
| Rank starts | `2048 / 2048` |
| Accepted updates | `2048` |
| Missing updates | `0` |
| Stale updates | `0` |
| Late updates | `0` |
| Timed-out updates | `0` |
| Rejected updates | `0` |
| Failed updates | `0` |
| Invalid updates | `0` |
| Quorum status | `advanced` |
| Latest advanced | `true`, generation `0` |
| Generation duration | `73.4914 s` |
| Merge duration | `5.253338 s` |
| Tokens per generation | `264,192` |
| Tokens/sec | `3594.8696` |
| Loss window | `loss=13.8485`, `loss_100=13.8485` |

Resilient-mode metrics:

- `mode=resilient_quorum`
- `requested_workers=2048`
- `participating_workers=2048`
- `quorum_size=2048`
- `quorum_threshold=2048`
- `catchup_events=[]`
- `staleness_distribution={}`

MPI/compiled-helper transport evidence:

- `mode=actual_multinode_compiled_mpich_quorum`
- `transport.name=compiled-cray-mpich-helper-collective-reduce`
- `transport.selector=compiled-cray-mpich-helper-p2p`
- `transport.actual=compiled-cray-mpich-helper-collective-reduce`
- `transport.helper_result.reducer=mpi_reduce_bucketed_weighted_sum`
- `transport.helper_result.mpi.collective=MPI_Reduce`
- `transport.helper_result.mpi.world_size=2048`
- `transport.helper_result.strict_collective_all_launched_ranks=true`
- `transport.filesystem_live_quorum=false`
- `transport.tcp_dense_data_plane=false`

Dense update bytes:

| Field | Bytes |
| --- | ---: |
| Accepted dense delta | `5,506,770,496` |
| MPI reduce aggregate | `5,506,770,496` |
| MPI reduce payload sent | `11,277,865,975,808` |

Bucketed reduce metrics:

- `bucket_count=80`
- bucket latency min / median / max: `0.037404 / 0.621829 / 5.408860 s`

This confirms TCP was **not** used as the hot aggregation path for the 256n
debug smoke.

## Latest and checkpoints

The smoke advanced only run-local latest to generation `0`:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/gate_actual_mpi_256n_debug_20260708/20260708/E97_1.3B_step1065000_actual_mpi_compiled_helper_256n_debug/4959434-20260708T204704Z/async_run/latest.json
```

Checkpoint/publication records:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/gate_actual_mpi_256n_debug_20260708/20260708/E97_1.3B_step1065000_actual_mpi_compiled_helper_256n_debug/4959434-20260708T204704Z/async_run/generations/gen_000000/manifest.json
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/gate_actual_mpi_256n_debug_20260708/20260708/E97_1.3B_step1065000_actual_mpi_compiled_helper_256n_debug/4959434-20260708T204704Z/async_run/recovery_checkpoints/gen_000000/initial.json
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/gate_actual_mpi_256n_debug_20260708/20260708/E97_1.3B_step1065000_actual_mpi_compiled_helper_256n_debug/4959434-20260708T204704Z/async_run/export_checkpoints/gen_000000/initial.json
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/gate_actual_mpi_256n_debug_20260708/20260708/E97_1.3B_step1065000_actual_mpi_compiled_helper_256n_debug/4959434-20260708T204704Z/async_run/recovery_checkpoints/gen_000000/walltime_finalization.json
```

The production latest/last guard remained intact. The production latest
dereferenced stat matches the `run-actual-mpi` snapshot, and production
`last.pt` remains absent:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt|7719679569|1782849877|'/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt'
/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/last.pt|ABSENT
```

The seed latest remained readable and unchanged relative to the upstream
snapshot:

```text
/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt|7719679924|1783330191|'/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt'
```

## Artifacts

- 256n summary:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/gate_actual_mpi_256n_debug_20260708/20260708/E97_1.3B_step1065000_actual_mpi_compiled_helper_256n_debug/4959434-20260708T204704Z/summaries/summary.md`
- 256n manifest:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/gate_actual_mpi_256n_debug_20260708/20260708/E97_1.3B_step1065000_actual_mpi_compiled_helper_256n_debug/4959434-20260708T204704Z/artifacts/manifest.json`
- 256n metrics:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/gate_actual_mpi_256n_debug_20260708/20260708/E97_1.3B_step1065000_actual_mpi_compiled_helper_256n_debug/4959434-20260708T204704Z/artifacts/metrics.json`
- 256n environment:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/gate_actual_mpi_256n_debug_20260708/20260708/E97_1.3B_step1065000_actual_mpi_compiled_helper_256n_debug/4959434-20260708T204704Z/artifacts/env.txt`
- 256n command:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/gate_actual_mpi_256n_debug_20260708/20260708/E97_1.3B_step1065000_actual_mpi_compiled_helper_256n_debug/4959434-20260708T204704Z/artifacts/command.txt`
- 256n rank starts:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/gate_actual_mpi_256n_debug_20260708/20260708/E97_1.3B_step1065000_actual_mpi_compiled_helper_256n_debug/4959434-20260708T204704Z/artifacts/rank-start.tsv`
- 256n train log:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/gate_actual_mpi_256n_debug_20260708/20260708/E97_1.3B_step1065000_actual_mpi_compiled_helper_256n_debug/4959434-20260708T204704Z/logs/trainpy_async_quorum.log`
- 256n Slurm logs:
  `logs/frontier/trainpy_async_quorum/actual-mpi-compiled-256n-debug-4959434.out`
  and
  `logs/frontier/trainpy_async_quorum/actual-mpi-compiled-256n-debug-4959434.err`

No 1h, 12h, or production job was submitted by this gate.
