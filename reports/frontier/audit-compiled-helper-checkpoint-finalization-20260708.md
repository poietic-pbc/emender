# Compiled-helper checkpoint finalization audit

Date: 2026-07-08
Task: `audit-compiled-helper-checkpoint-finalization`

## Verdict

Decision: **conditional go** for exactly one human-approved **256n x 1h
debug-QOS continuation** of the compiled-helper path.

Do not treat this as approval for 256n x 12h, production QOS, larger scale, or
any production `latest.pt` / `last.pt` / chain-pointer mutation. The current
evidence supports a longer 256-node debug stability check because the replacement
1n/2n rungs and the 8n/64n/128n/256n-debug ladder all loaded the verified seed,
started every expected rank, accepted every launched update, advanced run-local
latest/checkpoint state, and left production pointers untouched. The condition is
that the 1h run must keep the same run-local debug pointer policy, strict
full-rank gate, and live monitoring/stop rules listed below.

## Evidence reviewed

- WG task logs for `implement-compiled-mpich-2`: 1n job `4954252` and 2n job
  `4954257` were validated after the failed obsolete 1n attempt `4953961`.
- `reports/frontier/run-compiled-helper-20260708.md`: 8n job `4954290` and
  64n job `4954317`.
- `reports/frontier/run-compiled-helper-128n-20260708.md`: 128n job `4954539`.
- `reports/frontier/evaluate-compiled-helper-256n-debug-20260708.md`: bounded
  256n debug job `4954634`.
- `reports/frontier/synthesize-compiled-helper-scaleout.md`: integrated ladder
  synthesis and proposed 256n x 1h approval boundary.
- Launcher/script audit of
  `scripts/frontier/trainpy_async_quorum_smoke_common.sh`,
  `scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch`,
  `scripts/frontier/async_diloco_e97_256n12h_launch.sbatch`, and
  `scripts/frontier/async_diloco_e97_2n8n_debug.py`.

No Slurm job was submitted by this audit.

## Checkpoint load

Each accepted rung loaded the intended verified seed checkpoint:

```text
/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt
```

Evidence:

| Rung | Job | Load evidence |
| --- | ---: | --- |
| 1n | `4954252` | WG log for `implement-compiled-mpich-2` records pass with latest/checkpoint artifacts under run root; synthesis report says 1n command capture included the seed checkpoint above. |
| 2n | `4954257` | WG log for `implement-compiled-mpich-2` records pass; synthesis report says 2n command capture included the same seed checkpoint. |
| 8n | `4954290` | `run-compiled-helper-20260708.md` shared config lists that seed checkpoint and the 8n command/run root. |
| 64n | `4954317` | Same shared config and 64n command/run root in `run-compiled-helper-20260708.md`. |
| 128n | `4954539` | `run-compiled-helper-128n-20260708.md` lists the same seed checkpoint under submission. |
| 256n-debug | `4954634` | `evaluate-compiled-helper-256n-debug-20260708.md` lists the same seed and states `env.txt` records it as the only checkpoint input. |

The original `validate-compiled-helper` job `4953961` is not part of the
accepted ladder. It failed at 1n and did not advance latest/checkpoint; it was
superseded by the `implement-compiled-mpich-2` replacement ladder.

## Run-local latest and checkpoints

Run-local latest/checkpoint state advanced at every accepted rung. All artifacts
were written under debug run roots below:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/
```

| Rung | Job | Run root | Latest/checkpoint result |
| --- | ---: | --- | --- |
| 1n | `4954252` | `.../E97_1.3B_step1065000_trainpy_async_quorum_1n/4954252-20260708T005658Z` | `async_run/latest.json` advanced; four checkpoint/publication records. |
| 2n | `4954257` | `.../E97_1.3B_step1065000_trainpy_async_quorum_2n/4954257-20260708T010207Z` | `async_run/latest.json` advanced; four checkpoint/publication records. |
| 8n | `4954290` | `.../E97_1.3B_step1065000_trainpy_compiled_helper_8n/4954290-20260708T012807Z` | Latest advanced to generation 0; wrote generation manifest, recovery checkpoint, export checkpoint, and `walltime_finalization.json`. |
| 64n | `4954317` | `.../E97_1.3B_step1065000_trainpy_compiled_helper_64n/4954317-20260708T013528Z` | Same generation-0 latest and four publication records. |
| 128n | `4954539` | `.../E97_1.3B_step1065000_trainpy_compiled_helper_128n/4954539-20260708T025917Z` | Same generation-0 latest and four publication records. |
| 256n-debug | `4954634` | `.../E97_1.3B_step1065000_trainpy_compiled_helper_256n_debug/4954634-20260708T032415Z` | Same generation-0 latest and four publication records. |

For 8n and above the exact publication record set is:

- `async_run/generations/gen_000000/manifest.json`
- `async_run/recovery_checkpoints/gen_000000/initial.json`
- `async_run/export_checkpoints/gen_000000/initial.json`
- `async_run/recovery_checkpoints/gen_000000/walltime_finalization.json`

## Production latest/last mutation

No accepted rung was authorized to mutate production `latest.pt`, `last.pt`, or
a shared chain pointer, and the reports consistently state no such path was
passed or updated.

Evidence:

- `run-compiled-helper-20260708.md` says the only latest publications for 8n
  and 64n were run-local `async_run/latest.json` files and no production
  `latest.pt` / `last.pt` path was passed.
- `run-compiled-helper-128n-20260708.md` says no production `latest.pt` or
  `last.pt` path was passed to 128n and no shared production chain pointer was
  updated.
- `evaluate-compiled-helper-256n-debug-20260708.md` says the 256n debug human
  approval record explicitly denied production latest/last and shared pointer
  mutation; `manifest.json` records `latest_path` under the debug run root.
- The 256n-debug report also records `filesystem_live_quorum=false`; the dense
  data plane is the compiled MPICH reducer, not live Lustre update collection.

## Periodic and final save coverage

The short debug smokes exercised the run-local publish path and a forced/light
finalization record, not a full 1h walltime-pressure profile.

What is covered:

- `trainpy_async_quorum_smoke_common.sh` always passes recovery/export cadence
  arguments and `--walltime-remaining-s "$FINALIZATION_BUFFER_SECONDS"` to the
  async entrypoint, with default `FINALIZATION_BUFFER_SECONDS=1200`.
- Accepted rungs produced `walltime_finalization.json`, so the finalization
  record path and metrics publication path are covered.
- `scripts/frontier/async_diloco_e97_2n8n_debug.py` writes
  `checkpoint_cadence` and `checkpoint_finalization` metrics, including
  `latest_advanced`, `latest_path`, checkpoint paths/sizes, records, recovery
  records, total size bytes, overhead percent, and
  `debug_run_directory_only`.

What is only lightly exercised:

- Every accepted compiled-helper rung requested `00:20:00` debug walltime and
  ran only one generation. Runtime was roughly 3-7 minutes, so no rung proved
  repeated periodic checkpoints over a 1h window.
- Finalization is represented by the generated `walltime_finalization.json`
  records and forced debug finalization semantics, not by a real near-walltime
  Slurm deadline at 256 nodes.
- Checkpoint retention pressure is not proven: each accepted rung wrote a small
  fixed record set, not enough generations/exports to exercise pruning or
  retention under 1h output volume.

The 256n x 1h run should therefore be treated as the stability/checkpoint
coverage run, not as already validated production-duration behavior.

## Monitoring required for 256n x 1h

Require all of the following before and during the run:

- Human approval explicitly limited to one `batch` / `debug` 256n x 1h run, a
  new distinct variant such as
  `E97_1.3B_step1065000_trainpy_compiled_helper_256n_1h_debug`, the seed
  checkpoint above, run-local latest/checkpoint only, and no production pointer
  mutation.
- Confirm the job command captures `ASYNC_TRAINPY_RANKS=2048`,
  `ASYNC_EXPECTED_RANKS=2048`, `ASYNC_GLOBAL_QUORUM=2048`,
  `ASYNC_EXPECTED_MISSING_UPDATES=0`, compiled-helper transport, and the same
  seed/checkpoint paths.
- Watch Slurm state, elapsed time, stdout/stderr, and
  `logs/trainpy_async_quorum.log`.
- Check `artifacts/rank-start.tsv` reaches `2048/2048`; cancel if rank starts
  stall materially below 2048 or spread becomes an outlier relative to the 37 s
  256n-debug spread.
- Check `artifacts/metrics.json` appears and records accepted/quorum
  `2048/2048`, zero stale/failed/timed-out/invalid updates, finite loss, reduce
  bucket count `80`, expected aggregate bytes `5,506,770,496`, and stable
  per-bucket latency.
- Verify run-local `async_run/latest.json` advances and checkpoint records stay
  under the new debug run root only.
- Verify no production `latest.pt`, `last.pt`, or chain-pointer path changes.
- Check `checkpoint_finalization.records`, `recovery_records`,
  `total_size_bytes`, `overhead_percent`, and `duration_s` after every published
  generation/export.

Stop or fail the run if any of these occur:

- Slurm failure, timeout, node failure, helper abort, MPI abort, or stalled
  helper collective.
- Rank starts remain below 2048 after the expected startup window.
- Any stale, failed, timed-out, or invalid update is recorded. Current
  compiled-helper path is strict full-launched-rank collective; nonzero missing
  update counts are not validated success criteria.
- Loss is NaN/inf or metrics are missing/empty.
- Run-local latest/checkpoint publication does not occur after generation 0.
- Any latest/checkpoint artifact is written outside the debug run root, or any
  production latest/last/chain pointer would be touched.
- Checkpoint writes become long enough to threaten finalization reserve.

## Residual risks

- **Strict all-launched-rank collective:** The compiled-helper path is
  `mpi_reduce_bucketed_weighted_sum` with `collective=MPI_Reduce`,
  `filesystem_live_quorum=false`, and `strict_collective_all_launched_ranks=true`.
  It is not failure-tolerant async for non-joining ranks. A dead/non-started
  rank can still hang or fail the collective.
- **Short-window evidence:** 256n was proven for one bounded debug generation,
  not for sustained 1h cadence, repeated exports, or real near-walltime exit.
- **Checkpoint retention:** The accepted rungs wrote only a small fixed set of
  metadata/checkpoint records. Retention under many generations/exports remains
  unproven.
- **Production latest guard:** The scripts are fail-closed for production-style
  256n without human approval and gate artifacts, but the proposed 1h debug run
  must preserve non-production debug/run-local semantics. Any change to output
  root or chain update policy should be treated as out of scope.
- **Run-local vs production pointer handling:** Evidence is clean for
  run-local `latest.json`; it does not validate promotion from debug latest to
  production chain latest/last.
- **Missing long-run metrics:** Current reports include rank starts, quorum,
  reduce latency, dense bytes, checkpoint paths, loss window, and basic
  checkpoint overhead. The 1h run should additionally track per-generation
  checkpoint durations/bytes over time and verify metrics do not stop updating.

## Bottom line

The compiled-helper scaleout path is checkpoint/finalization-ready enough for a
single guarded 256n x 1h debug continuation. Approval should be conditional on
the monitoring and stop rules above, and should not authorize production latest
mutation or any longer production run.
