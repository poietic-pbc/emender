# Analyze B4 Four-Point 64n K Scan

Task: `analyze-b4-four`
Snapshot: `2026-06-29`

## Recommendation

Use `BATCH_SIZE=4`, `DILOCO_K=80`, hierarchical group size 4, 64M bucket, avg
outer optimizer for the 24h 64-node extended wrapper.

Reason: all four completed scans are in the same late-window loss band, so the
choice should not be made from a single final loss. K80 has the best average
throughput, the best late-window per-GPU throughput, and the lowest total sync
time while still completing 13 512-rank hierarchical merges and writing a final
consensus checkpoint. K20 has the lowest final checkpoint loss, but its broad
late-window loss is only 0.0042 below K80 and it costs 95.285 more sync seconds
in this bounded scan. K10 is slower and sync-heavy. K40 is not better on broad
loss or throughput.

The extended job was already submitted as `4911454` (`e97-s483-b4-k80-64n24h`).
Current scheduler state at this pass:

```text
squeue: 4911454|e97-s483-b4-k80-64n24h|PENDING|0:00|1-00:00:00|(Priority)|2026-06-29T14:52:00|2026-06-30T14:52:00
sacct: 4911454|e97-s483-b4-k80-64n24h|extended|PENDING|0:0|00:00:00|Unknown|Unknown|None assigned
```

Follow-up: monitor `4911454` after its estimated start window and compute the
same late-window loss / throughput / checkpoint / error checks for the long run.

## Slurm State

Recorded with:

```bash
sacct -j 4910758,4910757,4910912,4910913 --format=JobID,JobName%50,Partition,State,ExitCode,Elapsed,Start,End,NodeList%40 -P
squeue -j 4910758,4910757,4910912,4910913 -o '%i|%j|%T|%M|%L|%R'
```

`squeue` returned only the header for the four scan jobs, so none are active in
the queue. Top-level `sacct` rows:

| K | Job | Name | Partition | State | Exit | Elapsed | Start | End |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | `4910758` | `e97-s483-b4-k10-64n` | `batch` | `COMPLETED` | `0:0` | `00:35:34` | `2026-06-28T15:20:12` | `2026-06-28T15:55:46` |
| 20 | `4910757` | `e97-s483-b4-k20-64n` | `batch` | `COMPLETED` | `0:0` | `00:35:58` | `2026-06-28T15:20:12` | `2026-06-28T15:56:10` |
| 40 | `4910912` | `e97-s483-b4-k40-64n` | `batch` | `COMPLETED` | `0:0` | `00:35:53` | `2026-06-28T15:20:12` | `2026-06-28T15:56:05` |
| 80 | `4910913` | `e97-s483-b4-k80-64n` | `batch` | `COMPLETED` | `0:0` | `00:35:22` | `2026-06-28T15:29:56` | `2026-06-28T16:05:18` |

## Artifacts

All completed scan run roots contain `logs/train.log`,
`summaries/summary.md`, `artifacts/env.txt`, and `artifacts/manifest.json`.

| K | Job | Run root | Local Slurm logs |
| ---: | --- | --- | --- |
| 10 | `4910758` | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260628/E97_1.3B_step483000_b4_k10_64n_hier_g4_bucket64m_avg_scan/4910758-20260628T192016Z` | `.wg-worktrees/agent-420/logs/frontier/scaleout/e97-s483-b4-k10-64n-4910758.{out,err}` |
| 20 | `4910757` | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260628/E97_1.3B_step483000_b4_k20_64n_hier_g4_bucket64m_avg_scan/4910757-20260628T192016Z` | `.wg-worktrees/agent-420/logs/frontier/scaleout/e97-s483-b4-k20-64n-4910757.{out,err}` |
| 40 | `4910912` | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260628/E97_1.3B_step483000_b4_k40_64n_hier_g4_bucket64m_avg_scan/4910912-20260628T192016Z` | `logs/frontier/scaleout/e97-s483-b4-k40-64n-4910912.{out,err}` |
| 80 | `4910913` | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260628/E97_1.3B_step483000_b4_k80_64n_hier_g4_bucket64m_avg_scan/4910913-20260628T192958Z` | `logs/frontier/scaleout/e97-s483-b4-k80-64n-4910913.{out,err}` |

## Metrics

Metric method: parse rank-0 training metric rows from each full `logs/train.log`.
`first50` and `last50` are 50 logged rows each, equivalent to a broad 250-step
window because the scan logs every 5 steps. Per-GPU throughput is
`global_tok/s / 512` for the 64-node, 512-rank run.

| K | Rows | Final metric step | Final ckpt step | First50 loss | Last50 loss | Last20 loss | Avg global tok/s, all rows | Avg global tok/s, last50 | Per-GPU tok/s, last50 | Merges | Sync total s | Avg merge ms | Final ckpt loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 90 | 483450 | 483450 | 2.565150 | 2.537268 | 2.532045 | 1,372,426 | 1,375,671 | 2,686.9 | 45 | 298.055 | 6,623.4 | 2.5560 |
| 20 | 120 | 483600 | 483600 | 2.563998 | 2.547108 | 2.536745 | 2,035,688 | 2,039,359 | 3,983.1 | 30 | 216.282 | 7,209.4 | 2.5346 |
| 40 | 144 | 483720 | 483720 | 2.574278 | 2.561724 | 2.531630 | 2,388,663 | 2,377,072 | 4,642.7 | 18 | 157.025 | 8,723.6 | 2.5713 |
| 80 | 200 | 484000 | 484000 | 2.589152 | 2.551316 | 2.549900 | 2,519,888 | 2,492,814 | 4,868.8 | 13 | 120.997 | 9,307.5 | 2.5564 |

Interpretation:

- Broad late-window loss does not separate the runs strongly. K10 is best on
  `last50`, K20 is 0.0098 worse, K80 is 0.0140 worse, and K40 is 0.0245 worse.
  This spread is small relative to the logged loss noise, and K10 pays the
  largest sync cost by far.
- K80 is best on throughput: 2.520M global tok/s average over all rows and
  2.493M over the last 50 rows, or 4,868.8 tok/s per GPU in the late window.
- K80 has the lowest sync total, 120.997 s, versus 157.025 s for K40,
  216.282 s for K20, and 298.055 s for K10. Average merge latency rises with K,
  but total sync time falls because merge count falls.
- K20's final checkpoint loss is the lowest single finalization loss, but the
  task explicitly calls for broad late-window averages rather than single-point
  loss. On that basis K20 does not justify giving up K80 throughput and sync
  efficiency for the 24h wrapper.

## Checkpoint Status

All four runs entered walltime finalization, skipped the final merge because the
last step was already K-aligned and therefore already consensus, wrote a final
checkpoint, and updated `latest.pt`.

| K | Final checkpoint | `latest.pt` |
| ---: | --- | --- |
| 10 | `.../checkpoint_step_483450_loss_2.5560.pt` | `latest.pt -> checkpoint_step_483450_loss_2.5560.pt` |
| 20 | `.../checkpoint_step_483600_loss_2.5346.pt` | `latest.pt -> checkpoint_step_483600_loss_2.5346.pt` |
| 40 | `.../checkpoint_step_483720_loss_2.5713.pt` | `latest.pt -> checkpoint_step_483720_loss_2.5713.pt` |
| 80 | `.../checkpoint_step_484000_loss_2.5564.pt` | `latest.pt -> checkpoint_step_484000_loss_2.5564.pt` |

## Error Scan

Scanned each `logs/train.log` and `summaries/summary.md` for:

```text
Traceback|Exception|RuntimeError|FAILED|OOM|OutOfMemory|watchdog|segmentation fault|NCCL.*(error|Error)|RCCL.*(error|Error)|nan|non-finite
```

Result: zero severe matches for all four runs. The summaries contain repeated
non-fatal Triton 3.2.0 recommendation warnings; these are the same known warning
seen in earlier passes and not a runtime failure signature.

## Validation

- sacct/squeue state for all four jobs is recorded.
- Run roots and summaries/logs are located for all four completed jobs.
- Late-window average loss and average throughput are computed for all four K
  values.
- Recommendation for the 24h `BATCH_SIZE=4` extended wrapper is K80; current
  state of submitted extended job `4911454` is recorded above.
