# Async DiLoCo E97 32/64-Node Approval Gate

Task: `approve-async-diloco`
Date: 2026-07-05
Decision: `approved`

## Approval Scope

Human approval is recorded in the WG log for `approve-async-diloco` at
2026-07-05T15:02:27Z. The approval is limited to the next short async quorum
DiLoCo E97 Frontier configuration/debug step:

- One 32-node job and one 64-node job.
- Maximum walltime `00:30:00` per job.
- Total primary requested cap: 48 node-hours.
- Non-production run directory root:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/async_diloco/E97_1.3B_step483000_32n64n_config_20260705`
- Launch owner: WG task `async-diloco-e97-32n64n-config`.

This is not approval for 128-node, 256-node, 12-hour, or production runs. It
does not approve any production latest pointer advancement. GDN2/model-only
paths remain controls only and must not displace the E97 main research arm.

## Evidence Reviewed

### 1-Node Debug

Task `async-diloco-e97-1n-debug` produced report
`reports/frontier/async-diloco-e97-1n-debug-20260705.md` and metrics:

`/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97/20260705/4943217-20260705T142147Z/artifacts/async_diloco_e97_1n_metrics.json`

Summary:

- Slurm job `4943217`, state `COMPLETED`, elapsed `00:00:40`.
- Requested 1 node for `00:20:00`, 0.333333 node-hours.
- Conclusion `pass`.
- Configured quorum local/global: 8/1.
- Effective quorum local/global: 8/1.
- Source E97 checkpoint unchanged before/after.
- Debug run latest advanced only under the non-production debug run directory.
- Production latest guard changed: `false`.
- Generation manifest and checkpoint/latest metrics were machine-readable.

### 2/8-Node Debug Ladder

Task `async-diloco-e97-2n8n-debug` is done, evaluated, and recorded commit
`603d48e`. Its report is available in that commit as
`reports/frontier/async-diloco-e97-2n8n-debug-20260705.md`. Metrics artifacts:

- 2-node:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97_2n8n/20260705/4943251-20260705T144346Z/artifacts/async_diloco_e97_2n_metrics.json`
- 8-node:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97_2n8n/20260705/4943254-20260705T144529Z/artifacts/async_diloco_e97_8n_metrics.json`

2-node summary:

- Slurm job `4943251`, state `COMPLETED`, elapsed `00:00:44`.
- Requested 2 nodes for `00:20:00`, 0.666667 node-hours.
- Conclusion `pass`.
- Configured quorum local/global: 8/2.
- Effective quorum local/global: local 8 on both nodes, global 2.
- Update counts: local accepted 16, global accepted 2, stale/failed/invalid/timed-out all 0.
- Resume check tested generation 0 to generation 1 in the debug run directory.
- Production latest guard changed: `false`.

8-node summary:

- Slurm job `4943254`, state `COMPLETED`, elapsed `00:01:04`.
- Requested 8 nodes for `00:20:00`, 2.666667 node-hours.
- Conclusion `pass`.
- Configured quorum local/global: 8/6.
- Effective quorum local/global: local 8 on every node, global 6.
- Induced global dropped node IDs: 6 and 7.
- Update counts: local accepted 64; global accepted 6 and timed out 2; stale/failed/invalid 0.
- Resume check tested generation 0 to generation 1 in the debug run directory.
- Production latest guard changed: `false`.

The debug ladder therefore supports a bounded go decision for 32/64-node config
tests, but not production readiness.

## Required 32/64-Node Conditions

The approved 32/64-node runs must satisfy all of the following:

- Jobs are short debug/config tests only and write only under the approved
  non-production run directory root.
- No production latest pointer is advanced.
- Every Frontier job is launched by WG task `async-diloco-e97-32n64n-config`.
- Every job logs Slurm job ID, command, stdout/stderr paths, elapsed time,
  requested node-hours, and pass/no-go conclusion.
- Generation manifests are written every DiLoCo generation.
- Metrics artifacts are machine-readable.
- Metrics include configured quorum, effective quorum distribution, stale/drop
  and timeout counts, tokens/sec, generation duration, loss moving averages,
  checkpoint paths/latest behavior, recovery checkpoint write duration,
  checkpoint size, and checkpoint percent overhead.
- Recovery cadence is measured or modeled as N generations or wall-clock
  interval, whichever fires first. The later 256-node B4 K40 package must not
  inherit a fixed 20-30 minute recovery interval without measured checkpoint
  overhead.
- At least one resume-from-latest test is run across the 32/64-node jobs in the
  non-production run directory.

## Pass/No-Go

Pass for `async-diloco-e97-32n64n-config` to submit exactly the approved 32-node
and 64-node short E97 config/debug jobs under the constraints above.

No-go for any expansion to 128 nodes, 256 nodes, production duration, production
latest advancement, or displacement of E97 by GDN2/model-only controls without a
separate human approval gate.
