# Resilient Quorum 1n/8n/64n Ladder - 2026-07-08

Task: `run-resilient-quorum-1n8n64n-ladder`

Conclusion: `stopped-at-1n-hard-failure`

## Summary

The ladder attempted the required first rung only. Slurm job `4956022` failed
before producing quorum metrics because `scripts/frontier/async_diloco_e97_2n8n_debug.sbatch`
delegated to `scripts/frontier/async_diloco_e97_multinode.py`, whose current
CLI no longer accepts the wrapper's older prototype arguments:

- `--worker-count-per-node`
- `--tokens-per-step`
- `--delta-scale`
- `--task-id`
- `--slurm-job-id`
- `--slurm-job-name`
- `--requested-walltime`
- `--requested-node-hours`
- `--command-file`
- `--stdout-path`
- `--stderr-path`
- `--training-target`
- `--resume-check`
- `--production-latest-path`

Per task policy, no 8n or 64n job was submitted after this first hard failure.
No 128n, 256n, or 1h job was submitted.

## Seed And Guard

Current verified checkpoint seed used for the attempted 1n rung:

`/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt`

Resolved seed target:

`/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/checkpoint_step_1065000_loss_2.5386.pt`

Seed target stat observed before launch:

```text
/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/checkpoint_step_1065000_loss_2.5386.pt|7719679924|-rw-r--r--|erikgarrison|bif148|1783330191
```

Production latest guard path:

`/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt`

Production latest resolved target before and after the failed job:

`/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260630/E97_1.3B_step483000_b4_k80_64n_hier_g4_bucket64m_avg_24h/4911454-20260630T164035Z/train/emender_E97_1.3B_20260630_124257/checkpoint_step_489920_loss_2.4894.pt`

Production latest stat before and after was unchanged:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt|231|lrwxrwxrwx|erikgarrison|bif148|1783092774
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260630/E97_1.3B_step483000_b4_k80_64n_hier_g4_bucket64m_avg_24h/4911454-20260630T164035Z/train/emender_E97_1.3B_20260630_124257/checkpoint_step_489920_loss_2.4894.pt|7719679569|-rw-------|erikgarrison|bif148|1782849877
```

No production `latest.pt` or `last` pointer was mutated by this task.

## 1n Rung

| Field | Value |
| --- | --- |
| Slurm job ID | `4956022` |
| Job name | `resilient-quorum-e97-1n` |
| Partition / QOS | `batch` / `debug` |
| Requested walltime | `00:20:00` |
| Requested node-hours | `0.333333` |
| Actual elapsed | `00:00:35` |
| Approx actual node-hours | `0.009722` |
| Nodes | `1`, `frontier05912` |
| State / exit | `FAILED`, `2:0` |
| Output root | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/resilient_quorum_1n8n64n_ladder_20260708` |
| Run root | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/resilient_quorum_1n8n64n_ladder_20260708/20260708/4956022-20260708T093551Z` |
| Metrics JSON | expected path only; not created |
| Summary | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/resilient_quorum_1n8n64n_ladder_20260708/20260708/4956022-20260708T093551Z/summaries/summary.md` |
| Command file | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/resilient_quorum_1n8n64n_ladder_20260708/20260708/4956022-20260708T093551Z/artifacts/command.txt` |
| Env file | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/resilient_quorum_1n8n64n_ladder_20260708/20260708/4956022-20260708T093551Z/artifacts/env.txt` |
| Run log | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/resilient_quorum_1n8n64n_ladder_20260708/20260708/4956022-20260708T093551Z/logs/async_diloco_e97_2n8n.log` |
| Slurm stdout | `logs/frontier/async_diloco_e97/resilient-quorum-e97-1n-4956022.out` |
| Slurm stderr | `logs/frontier/async_diloco_e97/resilient-quorum-e97-1n-4956022.err` |

Submit command:

```bash
REPO=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-882 \
WG_TASK_ID=run-resilient-quorum-1n8n64n-ladder \
TRAINING_TARGET=E97_1.3B_step1065000_resilient_quorum_1n8n64n_ladder_20260708 \
OUTPUT_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/resilient_quorum_1n8n64n_ladder_20260708 \
E97_CHECKPOINT=/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt \
PRODUCTION_LATEST_GUARD=/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt \
ASYNC_NODE_COUNT=1 \
ASYNC_WORKER_COUNT_PER_NODE=8 \
ASYNC_LOCAL_QUORUM=8 \
ASYNC_GLOBAL_QUORUM=1 \
ASYNC_LOCAL_STEPS=1 \
ASYNC_RESUME_CHECK=1 \
ASYNC_RECOVERY_EVERY_GENERATIONS=1 \
ASYNC_REUSE_REPRESENTATIVE_NODE=0 \
REQUESTED_WALLTIME=00:20:00 \
REQUESTED_NODE_HOURS=0.333333 \
sbatch --parsable -N 1 -J resilient-quorum-e97-1n --time=00:20:00 --export=ALL scripts/frontier/async_diloco_e97_2n8n_debug.sbatch
```

The run-local summary reported `no-go-missing-metrics` because the metrics JSON
was never written.

Exact Python error:

```text
async_diloco_e97_multinode.py: error: unrecognized arguments: --worker-count-per-node 8 --tokens-per-step 1024 --delta-scale 1.0e-8 --task-id run-resilient-quorum-1n8n64n-ladder --slurm-job-id 4956022 --slurm-job-name resilient-quorum-e97-1n --requested-walltime 00:20:00 --requested-node-hours 0.333333 --command-file /lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/resilient_quorum_1n8n64n_ladder_20260708/20260708/4956022-20260708T093551Z/artifacts/command.txt --stdout-path logs/frontier/async_diloco_e97/resilient-quorum-e97-1n-4956022.out --stderr-path logs/frontier/async_diloco_e97/resilient-quorum-e97-1n-4956022.err --training-target E97_1.3B_step1065000_resilient_quorum_1n8n64n_ladder_20260708 --resume-check --production-latest-path /lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt
```

## Metrics Status

No rung passed, so there are no live ladder values for:

- ranks started
- quorum accepted
- missing/stale/late/timed-out/rejected counts
- catchup events
- merge latency
- bytes
- loss window
- latest/checkpoint behavior

The 1n failure happened during argument parsing before the runner could start
the resilient quorum generation or write the metrics JSON.

## Failure-Injection Evidence

The upstream dependency `validate-resilient-quorum-failure-injection` passed in
commit `f15557a`. That validation was synthetic only and submitted no Slurm job.
It added `tests/test_async_diloco_failure_injection.py` and report
`reports/resilient-quorum-failure-injection-validation-20260708.md`.

Relevant covered scenarios:

- Missing/nonjoining and stuck ranks: 4 requested ranks, ranks 0 and 1 submit,
  ranks 2 and 3 are timed out, quorum 2 advances with `accepted_updates == 2`
  and `timed_out_updates == 2`.
- Late old-base policy: current-generation updates are accepted while an old
  `base_generation` update is rejected and counted as stale.
- Stale worker catchup: a behind worker loads run-local latest, rebases local
  schedule-free state, and resets the next update base generation.
- Run-local latest isolation: debug latest advances under the run directory
  while a simulated production `latest.pt` symlink remains unchanged.
- Strict collective path remains present as a separate tested path; the
  resilient dense transport uses nonblocking quorum collection rather than
  requiring unanimity.

No additional deliberate live missing/stuck/late-rank exercise was feasible in
this ladder because the first 1n job failed before any resilient quorum metrics
could be emitted.

## Recommendation

Do not proceed to a bounded 256n debug gate yet.

First fix the wrapper/entrypoint contract or explicitly set an entrypoint whose
CLI matches `async_diloco_e97_2n8n_debug.sbatch`, then rerun this ladder from
1n. The likely repair choices are:

- update `async_diloco_e97_2n8n_debug.sbatch` to translate its environment into
  the current `async_diloco_e97_multinode.py` CLI, or
- set `ASYNC_ENTRYPOINT=scripts/frontier/async_diloco_e97_2n8n_debug.py` for
  this prototype wrapper if the intended rung is the older simulated
  E97-sized quorum path.

Only after 1n passes should 8n be submitted with a bounded missing-node
exercise, and only after 8n passes should 64n be submitted.
