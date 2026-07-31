# Evaluation: MPI ladder and 256n debug gate

Date: 2026-07-07

Task: `evaluate-mpi-ladder`

Evaluated upstream task: `run-post-patch`

Evaluator: `agent-796`

## Verdict

Gate decision: **NO-GO for 256n**.

Reason: the post-patch MPI dense ladder did not pass cleanly through 64n. It
passed the adopted 1-node rung, then failed at the 2-node rung before recording
usable training metrics, latest advancement, or checkpoint metadata. The 8-node
and 64-node rungs were therefore not eligible and were not submitted.

No 256-node debug or production job was submitted from this evaluation.

Recommended next action: debug the 2-node Frontier Cray MPICH/OFI
initialization failure under the same launcher/module stack before retrying the
2n rung. Start with a minimal 2-node `mpi4py` `MPI.Init_thread` and send/recv
smoke using the same `srun -N 2 -n 16` shape, then compare required Frontier
network and MPICH environment settings against the train.py launcher.

## Calibrated Grade

Score for the completed upstream `run-post-patch` actor output: **0.94 / 1.00**.

Confidence: **0.91**.

Rubric underspecified: **no**. The task supplied an explicit validation
checklist and a clear conditional ladder policy.

The score is high because the actor followed the ladder gate, submitted only
the eligible 2n rung after 1n passed, stopped immediately on the first failure,
classified the failure, and wrote artifact-linked evidence. The main limitation
is that the report inherited one ambiguous phrasing around the failed 2n
`metrics.json`: the summary points at a metrics path, but the metrics file is
absent on disk; the summary itself records an empty metrics excerpt and the
validation error `metrics_json_missing_or_empty`. This does not change the
no-go decision.

## Evidence Reviewed

- WG task state and logs: `wg show run-post-patch`, `wg show evaluate-mpi-ladder`.
- Upstream report:
  `reports/frontier/run-post-patch-mpi-dense-smoke-ladder-20260707.md`.
- Prior evaluator artifact:
  `reports/evaluations/run-post-patch-evaluation-20260707.md`.
- Slurm accounting:
  `sacct -j 4953629,4953646 --format=JobID,JobName%30,Partition,QOS,State,ExitCode,Elapsed,NNodes,NodeList -P`.
- 1n summary:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_mpi_dense_1n_postpatch/4953629-20260707T202832Z/summaries/summary.md`.
- 1n manifest:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_mpi_dense_1n_postpatch/4953629-20260707T202832Z/artifacts/manifest.json`.
- 1n metrics:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_mpi_dense_1n_postpatch/4953629-20260707T202832Z/artifacts/metrics.json`.
- 1n rank starts:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_mpi_dense_1n_postpatch/4953629-20260707T202832Z/artifacts/rank-start.tsv`.
- 1n run-local latest:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_mpi_dense_1n_postpatch/4953629-20260707T202832Z/async_run/latest.json`.
- 2n summary:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_mpi_dense_2n_postpatch/4953646-20260707T203737Z/summaries/summary.md`.
- 2n manifest:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_mpi_dense_2n_postpatch/4953646-20260707T203737Z/artifacts/manifest.json`.
- 2n rank starts:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_mpi_dense_2n_postpatch/4953646-20260707T203737Z/artifacts/rank-start.tsv`.
- 2n train log:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_mpi_dense_2n_postpatch/4953646-20260707T203737Z/logs/trainpy_async_quorum.log`.
- 2n command and environment captures:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_mpi_dense_2n_postpatch/4953646-20260707T203737Z/artifacts/command.txt`,
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_mpi_dense_2n_postpatch/4953646-20260707T203737Z/artifacts/env.txt`.

## Rung Decisions

| Rung | Passed perfectly? | Evidence and decision |
| --- | --- | --- |
| 1n | Yes | Job `4953629` completed in `batch`/`debug` with exit `0:0`, elapsed `00:05:21`, one node. Summary validation is `pass`; rank starts were `8 / 8`; accepted updates `8`; timed-out updates `0`; tokens `1032`; run-local latest advanced; checkpoint manifests were recorded; dense MPI transport was enabled; live filesystem quorum collection was disabled. |
| 2n | No | Job `4953646` failed in `batch`/`debug` with job exit `90:0` and step exit `255:0`, elapsed `00:01:42`, two nodes. Summary validation is `fail`; rank starts were `16 / 16`, but accepted updates `0`, tokens `0`, no latest advancement, no checkpoint paths, and no usable metrics JSON. First failing rung. |
| 8n | No | Not submitted because the 2n rung failed. |
| 64n | No | Not submitted because the 2n rung failed. |
| 256n | No | Not submitted because the ladder did not pass through 64n. |

## First Failing Rung

First failing rung: **2n MPI dense post-patch**, Slurm job `4953646`.

Root-cause category: **runtime/MPI**.

Observed failure:

```text
Fatal error in PMPI_Init_thread
MPIDI_OFI_mpi_init_hook
open_fabric(1559): OFI fi_getinfo() failed (ofi_init.c:1559:open_fabric:No data available)
```

The 2n evidence shows the payload reached real train.py execution and all 16
launched ranks wrote rank-start/progress evidence. The per-rank progress files
reached `stage="mpi_dense_send_starting"` with `transport="mpi-dense"`,
`global_quorum=16`, and `mpi_bucket_bytes=67108864`, then Cray MPICH aborted
during MPI initialization before the smoke could produce accepted updates,
training tokens, latest metadata, checkpoint manifests, or MPI byte counters.

## Latest And Checkpoint Behavior

The 1n rung used the debug run root and advanced only its run-local latest:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_mpi_dense_1n_postpatch/4953629-20260707T202832Z/async_run/latest.json
```

The 1n summary and metrics record run-local generation, recovery, export, and
walltime-finalization checkpoint manifests under the same debug run root. The
upstream report identifies the E97 input checkpoint seed as:

```text
/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt
```

The failed 2n rung did not advance run-local latest and did not record
checkpoint paths. The upstream report states that no production latest path was
intentionally mutated. I found no contradictory evidence in the reviewed
artifacts.

## 256n Decision

Do not submit 256n now. The explicit policy requires a clean ladder through
64n before the first bounded 256-node debug smoke. That condition is false:
2n failed and 8n/64n were not run.

Because this is a no-go case, the 256n debug-result checklist is not
applicable: there is no 256n job id, command, rank-start set, quorum size,
timeout/stale-rank result, MPI byte/timing summary, loss window, or
latest/checkpoint behavior to summarize.

Recommendation after the next fix:

1. Run a minimal 2-node MPI/mpi4py diagnostic under the same Frontier module
   stack and launcher shape to isolate the OFI initialization failure.
2. Retry the 2n train.py MPI dense smoke only after the minimal diagnostic
   passes.
3. Resume the ladder sequentially: 2n, then 8n, then 64n. Only after a clean
   64n pass should a bounded 256n debug-QOS smoke be considered.
4. Do not submit a 256n x 1h job until a human reviews a passing short 256n
   debug result. A 256n x 1h job would request about 256 node-hours, before any
   early cancellation. Cancellation criteria should include missing rank starts,
   MPI initialization errors, stale or timed-out ranks, missing/empty metrics,
   non-finite loss, no quorum advancement, or any production-latest mutation.

## Dimension Scores

| Dimension | Score | Rationale |
| --- | ---: | --- |
| Artifact inspection and linkage | 0.96 | The actor report links the run roots, summaries, manifests, metrics/rank-start files, train logs, stdout/stderr, Slurm states, and submit command for every rung that ran. Direct inspection confirmed the key artifacts. |
| Ladder policy compliance | 1.00 | 2n was submitted only after 1n passed; 8n, 64n, and 256n were not submitted after 2n failed. |
| Rung pass/fail accuracy | 0.95 | 1n pass and 2n failure are correctly classified. 8n/64n are correctly marked not eligible. Minor imprecision: the failed 2n metrics path is referenced, but the file is absent on disk; the failure summary still correctly records missing/empty metrics. |
| Root-cause classification | 0.94 | Runtime/MPI is the right category based on repeated `PMPI_Init_thread` / `MPIDI_OFI_mpi_init_hook` / `OFI fi_getinfo()` failures after rank starts. |
| Latest/checkpoint and production guard | 0.90 | The report records run-local latest behavior and no intentional production latest mutation. The conclusion is well supported by the artifacts reviewed, though a dedicated production-latest before/after hash would have been stronger. |
| Next-step recommendation | 0.92 | The next action is concrete: minimal 2n mpi4py/OFI diagnosis before retrying 2n, then resume the sequential ladder. |

Overall grade: **0.94 / 1.00**.
