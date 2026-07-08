# Rerun resilient quorum 1n/8n/64n ladder

Task: `rerun-resilient-quorum-1n8n64n-ladder`

Date: 2026-07-08 UTC

Conclusion: all in-scope rungs passed. Proceed to the bounded 256n debug gate, with the same run-local output-root policy and no production latest mutation.

## Policy checks

- Transport/mode: all rungs used the explicitly resilient TCP debug path, `actual_multinode_tcp_quorum_debug`, with global generation metrics `mode=resilient_quorum`. The strict compiled MPICH/MPI_Reduce path was not used.
- Seed: all rungs used `/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt`.
- Output root: all rungs wrote under `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/rerun_resilient_quorum_1n8n64n_ladder_20260708`.
- Submission order: 1n completed and validated before 8n submission; 8n completed and validated before 64n submission.
- Scope: no 128n, 256n, 1h, 12h, or production job was submitted by this task.
- Wrapper/entrypoint fix: command captures show `srun ... "$PYTHON_BIN" -u "$ASYNC_ENTRYPOINT"` with `ASYNC_ENTRYPOINT=scripts/frontier/e97_async_diloco_train.py` and `--actual-multinode-tcp-quorum`; the stale `async_diloco_e97_multinode.py` CLI mismatch path was not used.
- Production latest/last: `latest.pt` and sibling `latest*`/`last*` listings under `/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg` were byte-for-byte identical before and after this ladder.

## Rungs

| Rung | Job | Slurm state | QOS | Walltime | Node-hours | Run root |
| --- | --- | --- | --- | --- | --- | --- |
| 1n | `4956437` | `COMPLETED 0:0` | `debug` | `00:20:00` | `0.333333` | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/rerun_resilient_quorum_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_resilient_quorum_rerun_1n/4956437-20260708T104400Z` |
| 8n | `4956445` | `COMPLETED 0:0` | `debug` | `00:20:00` | `2.666667` | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/rerun_resilient_quorum_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_resilient_quorum_rerun_8n/4956445-20260708T104614Z` |
| 64n | `4956459` | `COMPLETED 0:0` | `debug` | `00:20:00` | `21.333333` | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/rerun_resilient_quorum_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_resilient_quorum_rerun_64n/4956459-20260708T105539Z` |

## Resilient metrics

| Rung | Ranks started/joined | Quorum accepted | Missing | Stale | Late | Timed out | Rejected | Catchup events | Merge duration | Bytes | Loss window | Latest/checkpoint behavior |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1n | `8/8` | `8`, `advanced` | `0` | `0` | `0` | `0` | `0` | `0` | `0.0s` | `69772` TCP payload, `71037` node metadata | `loss=13.633476257324219`, `loss_100=13.633476257324219` | `latest_advanced=true`; `async_run/latest.json`; 4 checkpoint/publication records |
| 8n | `64/64` | `64`, `advanced` | `0` | `0` | `0` | `0` | `0` | `0` | `0.0s` | `558649` TCP payload, `569051` node metadata | `loss=13.835908219218254`, `loss_100=13.835908219218254` | `latest_advanced=true`; `async_run/latest.json`; 4 checkpoint/publication records |
| 64n | `512/512` | `512`, `advanced` | `0` | `0` | `0` | `0` | `0` | `0` | `0.0s` | `4471340` TCP payload, `4554757` node metadata | `loss=13.834494853392243`, `loss_100=13.834494853392243` | `latest_advanced=true`; `async_run/latest.json`; 4 checkpoint/publication records |

Notes:
- The metrics use `merge_duration_s`; this is the merge-latency field available in the current resilient-quorum metric schema.
- `catchup_events=[]` on all three rungs.
- All three runs used real token data from `/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt`; `synthetic_token_stream=false`.

## Artifacts

1n:
- Metrics: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/rerun_resilient_quorum_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_resilient_quorum_rerun_1n/4956437-20260708T104400Z/artifacts/metrics.json`
- Manifest: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/rerun_resilient_quorum_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_resilient_quorum_rerun_1n/4956437-20260708T104400Z/artifacts/manifest.json`
- Env/command: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/rerun_resilient_quorum_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_resilient_quorum_rerun_1n/4956437-20260708T104400Z/artifacts/env.txt`, `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/rerun_resilient_quorum_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_resilient_quorum_rerun_1n/4956437-20260708T104400Z/artifacts/command.txt`
- Rank starts: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/rerun_resilient_quorum_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_resilient_quorum_rerun_1n/4956437-20260708T104400Z/artifacts/rank-start.tsv`
- Summary: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/rerun_resilient_quorum_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_resilient_quorum_rerun_1n/4956437-20260708T104400Z/summaries/summary.md`

8n:
- Metrics: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/rerun_resilient_quorum_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_resilient_quorum_rerun_8n/4956445-20260708T104614Z/artifacts/metrics.json`
- Manifest: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/rerun_resilient_quorum_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_resilient_quorum_rerun_8n/4956445-20260708T104614Z/artifacts/manifest.json`
- Env/command: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/rerun_resilient_quorum_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_resilient_quorum_rerun_8n/4956445-20260708T104614Z/artifacts/env.txt`, `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/rerun_resilient_quorum_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_resilient_quorum_rerun_8n/4956445-20260708T104614Z/artifacts/command.txt`
- Rank starts: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/rerun_resilient_quorum_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_resilient_quorum_rerun_8n/4956445-20260708T104614Z/artifacts/rank-start.tsv`
- Summary: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/rerun_resilient_quorum_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_resilient_quorum_rerun_8n/4956445-20260708T104614Z/summaries/summary.md`

64n:
- Metrics: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/rerun_resilient_quorum_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_resilient_quorum_rerun_64n/4956459-20260708T105539Z/artifacts/metrics.json`
- Manifest: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/rerun_resilient_quorum_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_resilient_quorum_rerun_64n/4956459-20260708T105539Z/artifacts/manifest.json`
- Env/command: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/rerun_resilient_quorum_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_resilient_quorum_rerun_64n/4956459-20260708T105539Z/artifacts/env.txt`, `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/rerun_resilient_quorum_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_resilient_quorum_rerun_64n/4956459-20260708T105539Z/artifacts/command.txt`
- Rank starts: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/rerun_resilient_quorum_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_resilient_quorum_rerun_64n/4956459-20260708T105539Z/artifacts/rank-start.tsv`
- Summary: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/rerun_resilient_quorum_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_resilient_quorum_rerun_64n/4956459-20260708T105539Z/summaries/summary.md`

Local evidence files committed with this report:
- `reports/frontier/rerun-resilient-quorum-1n.jobid`
- `reports/frontier/rerun-resilient-quorum-8n.jobid`
- `reports/frontier/rerun-resilient-quorum-64n.jobid`
- `reports/frontier/rerun-resilient-quorum-1n8n64n-ladder-production-latest.before`
- `reports/frontier/rerun-resilient-quorum-1n8n64n-ladder-production-latest.after`
- `reports/frontier/rerun-resilient-quorum-1n8n64n-ladder-production-latest-last.before`
- `reports/frontier/rerun-resilient-quorum-1n8n64n-ladder-production-latest-last.after`

## 256n debug gate recommendation

Proceed with the bounded 256n debug gate. The prerequisites that failed earlier are now satisfied: the fixed train.py-backed entrypoint is actually used, 1n/8n/64n all completed cleanly in resilient TCP quorum mode, all ranks joined at each scale, quorum advancement and run-local latest/checkpoint publication are healthy, and production latest/last remained unchanged.

Recommended guardrails for the 256n gate:
- Keep the same non-production output-root and run-local latest behavior.
- Keep explicit `ASYNC_QUORUM_TRANSPORT=tcp` or otherwise explicitly label any control-path transport; do not fall back to strict collective.
- Preserve the command/env/rank-start/manifest/metrics capture requirements used here.
- Stop before any 1h/12h/production run until the 256n debug gate evaluator accepts the 256n evidence.
