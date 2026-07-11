# E97 Async 256n12h Rank-Count Failure Handoff

Date: 2026-07-11

Audience: next WG/Codex agent working on Frontier E97 async DiLoCo production launch.

## Current State

Do not submit another production job until this handoff is addressed and the
production wrapper itself has passed a debug smoke.

The 256-node 12-hour production job did run, but failed before training:

- Slurm job: `4963853`
- Wrapper: `scripts/frontier/async_diloco_e97_256n12h_launch.sbatch`
- Started: `2026-07-11T07:55:37` America/New_York
- Ended: `2026-07-11T07:58:52` America/New_York
- Elapsed: `00:03:15`
- State: `FAILED`
- Exit: `143:0`
- Step: `4963853.0` was `CANCELLED`
- Requested shape: `256` nodes, `2048` Slurm tasks, `8` tasks per node
- Estimated cost: about `14` node-hours

No useful training happened:

- no loss
- no merge
- no metrics JSON
- no checkpoint
- no latest pointer advancement

The local seed/latest checkpoint is therefore still the input seed, not an
advanced Frontier checkpoint:

```text
/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260709_084606_step_1282500/latest.pt
```

## Evidence

Slurm accounting:

```text
4963853|async-diloco-e97-256n12h|batch|normal|FAILED|143:0|00:03:15|256||2026-07-09T14:13:18|2026-07-11T07:55:37|2026-07-11T07:58:52
4963853.batch|batch|||FAILED|143:0|00:03:15|1|1|2026-07-11T07:55:37|2026-07-11T07:55:37|2026-07-11T07:58:52
4963853.extern|extern|||COMPLETED|0:0|00:03:18|256|256|2026-07-11T07:55:37|2026-07-11T07:55:37|2026-07-11T07:58:55
4963853.0|python|||CANCELLED|0:15|00:02:35|256|2048|2026-07-11T07:56:17|2026-07-11T07:56:17|2026-07-11T07:58:52
```

Relevant log files:

```text
logs/frontier/async_diloco_e97/async-diloco-e97-256n12h-4963853.out
logs/frontier/async_diloco_e97/async-diloco-e97-256n12h-4963853.err
```

Run root:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/async_diloco/E97_1.3B_step1282500_async_quorum_b4_k40_256n12h/20260711/4963853-20260711T115539Z
```

The run root contains only env/command/helper-build artifacts and a few
`starting` heartbeats. It does not contain metrics/checkpoints.

## Root Cause

The failing exception is:

```text
ValueError: node_rank must be in [0, node_count)
```

The production wrapper launched `2048` processes:

```text
srun -N 256 -n 2048 --ntasks-per-node=8 ...
```

but passed this trainer configuration:

```text
--node-count 256 --worker-count 2048 --global-quorum 171
```

In `scripts/frontier/e97_async_diloco_train.py`, actual multinode mode derives
`node_rank` from `SLURM_PROCID` when `--node-rank` is not provided:

```python
node_rank = args.node_rank
if node_rank is None:
    node_rank = int(os.environ.get("SLURM_PROCID", os.environ.get("PMI_RANK", "0")))
```

Then `ndm/async_diloco_real.py` validates:

```python
if config.node_rank < 0 or config.node_rank >= config.node_count:
    raise ValueError("node_rank must be in [0, node_count)")
```

So ranks `256..2047` failed immediately because `node_count` was physical node
count (`256`), while `node_rank` was Slurm process rank (`0..2047`).

## Why The Smoke Ladder Did Not Catch This

The passing debug ladder did validate the trainer/transport at 256 nodes and
2048 GPU ranks, but it did not use the same production wrapper contract.

Passing 256n debug smoke:

- Report: `reports/frontier/async-quorum-b4k40-debug-ladder-20260709.md`
- Job: `4962400`
- Wrapper path: `scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch`
- Common runner: `scripts/frontier/trainpy_async_quorum_smoke_common.sh`
- Shape: `256` nodes, `2048` tasks, `8` tasks per node
- Result: `COMPLETED 0:0`, accepted `2048/2048`, loss `2.52083`

The smoke common runner used rank-count semantics:

```bash
--worker-count "$ASYNC_TRAINPY_RANKS"
--node-count "$ASYNC_EXPECTED_RANKS"
--global-quorum "$ASYNC_GLOBAL_QUORUM"
...
srun -N "$SMOKE_NODE_COUNT" -n "$ASYNC_TRAINPY_RANKS"
```

For the 256n smoke, `ASYNC_TRAINPY_RANKS=2048`,
`ASYNC_EXPECTED_RANKS=2048`, and `ASYNC_GLOBAL_QUORUM=2048`.

The production wrapper diverged by treating `ASYNC_NODE_COUNT=256` as both
physical Slurm node count and trainer `--node-count`. That was the invalid
contract.

## Required Fix

Fix `scripts/frontier/async_diloco_e97_256n12h_launch.sbatch` so production
uses the same participant-count semantics as the passing smoke, or refactor the
Python API to make the distinction explicit.

Minimal safe patch for the current trainer API:

- Keep physical Slurm node count as `ASYNC_NODE_COUNT=256`.
- Keep launched worker/rank count as `ASYNC_WORKER_COUNT=ASYNC_NODE_COUNT * 8`.
- Pass trainer `--node-count "$ASYNC_WORKER_COUNT"`, not
  `--node-count "$ASYNC_NODE_COUNT"`.
- Pass `--worker-count "$ASYNC_WORKER_COUNT"`.
- Do not keep `ASYNC_GLOBAL_QUORUM=171` unless deliberately changing the
  algorithm. For the current compiled helper collective path, the only fully
  smoke-tested 256n behavior is rank-count quorum, i.e. `2048/2048`.
- If using a lower quorum, compute it in rank-count terms, not physical-node
  terms. For example, `ceil(2/3 * 2048) = 1366`. This must be separately
  tested before production.

Recommended durable patch:

- Rename variables in the production wrapper:
  - `ASYNC_PHYSICAL_NODE_COUNT`
  - `ASYNC_RANKS_PER_NODE`
  - `ASYNC_PARTICIPANT_COUNT`
- Add a fail-closed pre-submit check:

```bash
if [[ "$ASYNC_PARTICIPANT_COUNT" != "$ASYNC_WORKER_COUNT" ]]; then
  fail
fi
```

- Ensure generated `command.txt` clearly records:
  - Slurm `-N`
  - Slurm `-n`
  - trainer `--node-count`
  - trainer `--worker-count`
  - trainer `--global-quorum`
- Update the production report when this is fixed.

## Validation Gate Before Any Production Resubmission

Do not rely on the previous ladder alone. The next validation must exercise
the production wrapper or a mechanically identical debug mode.

Required debug ladder:

1. Run the production wrapper in debug/QOS-debug with small node counts
   by overriding `ASYNC_NODE_COUNT` and the Slurm node count together.
2. Confirm the captured `artifacts/command.txt` has:
   - `srun -N <physical_nodes>`
   - `-n <physical_nodes * 8>`
   - `--node-count <physical_nodes * 8>`
   - `--worker-count <physical_nodes * 8>`
   - sane `--global-quorum`
3. Run at least:
   - `1n`
   - `2n`
   - `8n`
   - `64n`
   - `256n` debug if available
4. For each rung, require:
   - all ranks start
   - no `node_rank must be in [0, node_count)` errors
   - loss in sane range around the seed, roughly `2.4..2.7`
   - at least one merge completes
   - metrics JSON written
   - checkpoint/latest behavior matches intended run-local policy

Only after the production wrapper itself passes these checks should a new
256n12h production job be queued.

## Current Open Questions

The current code path is one rank per GPU. It does not implement true
physical-node-level local aggregation before global aggregation. Therefore:

- `local_quorum=6` is not enough to make physical nodes into islands unless
  the implementation explicitly groups ranks per node.
- `global_quorum=171` is a physical-node quorum and is not equivalent to the
  rank-count quorum used by the trainer.
- The compiled MPICH helper path is currently collective-reduce based in the
  passing evidence. Treat it as an all-launched-ranks path until a lower-quorum
  compiled-helper design is explicitly implemented and tested.

If the research goal is true node-island DiLoCo, implement that as a separate
workstream:

- group 8 GPU ranks per physical node
- locally aggregate or average within node
- submit one node-level update per physical node
- then global quorum can reasonably be `ceil(2/3 * physical_nodes)`

That is not what the failed 256n12h wrapper actually did.

## WG Operational Notes

The original `submit-and-monitor` task is not safe to resume unchanged. It still
contains submit-oriented language and previously spawned an agent that began
preflight again. Keep it paused or rewrite it to monitor-only/failure-summary
before dispatch.

For follow-up work, create new WG tasks with explicit scopes:

1. Fix production wrapper rank-count semantics.
2. Run production-wrapper debug smoke ladder.
3. If and only if the debug ladder passes, submit a new 256n12h production job.

Do not queue another production job from a generic "submit and monitor" task
until the wrapper fix and validation report are committed and pushed.

## Files To Inspect First

Start with these files, not broad run-tree scans:

```text
scripts/frontier/async_diloco_e97_256n12h_launch.sbatch
scripts/frontier/e97_async_diloco_train.py
scripts/frontier/trainpy_async_quorum_smoke_common.sh
ndm/async_diloco_real.py
reports/frontier/async-quorum-b4k40-debug-ladder-20260709.md
reports/frontier/e97-async-256n12h-production-run-20260709.md
logs/frontier/async_diloco_e97/async-diloco-e97-256n12h-4963853.err
logs/frontier/async_diloco_e97/async-diloco-e97-256n12h-4963853.out
```
