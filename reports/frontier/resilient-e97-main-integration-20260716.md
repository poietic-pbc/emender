# Recovered resilient E97 main integration — 2026-07-16

## Scope and safety

This was a Git-only integration. No Slurm job was submitted, cancelled, held,
requeued, or otherwise mutated. No checkpoint, `latest` pointer, promotion
pointer, or production pointer was changed. The immutable generation-9
checkpoint from job 5000436 remains external evidence.

The authoritative configured remote is `origin`, URL
`git@github.com:spinozans/emender`, and its integration target is `main`.

## Pre-merge state

After `git fetch --all --prune`, the relevant refs were:

- local `main`: `d8422c21d1fc35f39ee3b721b969b73c01042e76`
- assigned integration branch: `d8422c21d1fc35f39ee3b721b969b73c01042e76`
- `origin/main`: `f82e940d8d728aca7e3ae62fbc81bf21323894fc`
- recovered source `wg/agent-1107/complete-resilient-e97` and its upstream:
  `642b1b6f33e2d23c73ee43aefa476afa4ccca37e`
- source worktree: clean, at `642b1b6f33e2d23c73ee43aefa476afa4ccca37e`
- assigned integration worktree: clean

Local `main` had one concurrent commit not in `origin/main`; `origin/main` had
three concurrent commits not in local `main`. Both histories were retained.
The recovered branch had exactly 12 commits unreachable from both local main
and `origin/main` before integration (`git rev-list --count` returned 12).

## Ordered recovered source set

Oldest to newest, as recorded before merging with
`git log --reverse --format='%H %s' origin/main..wg/agent-1107/complete-resilient-e97`:

1. `11d8c666bcc30ce1ef6ca4700c57223c2c857940` — harden resilient node protocol
2. `43853cae4d30ac3b258520fbfbdaef617ba27e8c` — expose resilient transport in Frontier runner
3. `89477cea79f15b0104a18ebd3d268bfd43c0b733` — record resilient E97 foundation gate
4. `df46fc68ee8f22c24c65824f2f9828f72966ede1` — derive resilient gate step count
5. `1abcec577c5655723c71e3ad4e3373e7f42af551` — preserve failed resilient gate evidence
6. `954b68daa0e120db038388e16b7a771eb7541fb8` — allow resilient manager scale path
7. `0133bffc207879ad2a79eeb0b08044c537c2abe0` — bound resilient restart memory
8. `0bbf237aa292c2c24acfbb2934df9f393c2b2cfc` — register resilient retry 5009365
9. `4a2ff18960a45ea42ad8406865690f80fd8f604f` — checkpoint pending resilient gate monitor
10. `e565534f45aaf5017c7c4b9cfdb68b3ffb5c1609` — diagnose resilient retry OOM
11. `3135a2f21ea296f7245ce347c8def8f95baabbb3` — release consumed E97 optimizer state
12. `642b1b6f33e2d23c73ee43aefa476afa4ccca37e` — bound resilient E97 node rank lanes

## Integration and review decisions

First, `origin/main` was merged with `--no-ff`, producing
`c203417fd6700be1e02b283e8ea4f795635af85d`. Then the recovered source was
merged with `--no-ff`, producing the integration merge
`c30c4451160fa8393ee69013e47e1e11e21968d3`. Both merges completed without
conflicts, so no side was manually selected and no source commit was omitted or
replaced by a patch-equivalent commit. The recovered tip is a direct parent of
the integration merge, preserving all 12 original commit identities.

The combined diff was reviewed by name/status and content. Durable source code,
tests, reports, and the four pairs of captured failure-evidence logs were kept.
No WG task-state file, transient test output, checkpoint payload, pointer, core
dump, or unrelated local file was added. Absolute paths appearing inside the
captured logs/reports are historical evidence, not active configuration. The
only cleanup was removal of trailing whitespace/extra EOF blank lines from the
durable log files so `git diff --check` passes.

The bounded rank-lane helper added by `642b1b6` creates sentinel lanes; those
sentinels are not independently supervised trainers. This integration does not
claim otherwise.

## Validation

Commands and results:

- `python3 -m pytest -q <focused files>` using system Python 3.6: collection
  failed because that interpreter has no PyTorch. This was an environment
  failure, not a test failure; the complete suite was rerun in the established
  project runtime below.
- `/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python -m pytest -q tests/test_resilient_node_quorum.py tests/test_resilient_node_transport.py tests/test_resilient_e97_rank_lane.py tests/test_async_diloco_real_trainer.py tests/test_async_diloco_worker_supervisor_prototype.py tests/test_async_diloco_checkpoint_manager.py tests/test_async_diloco_local_simulation.py tests/test_async_diloco_core.py tests/test_diloco_merge.py tests/test_trainpy_async_quorum_smoke_launchers.py tests/test_checkpoint_finalization.py tests/test_walltime_final_checkpoint.py`: **115 passed in 232.92s**. This covers protocol/network fencing and deadlines; stale, duplicate, corrupt and nonfinite rejection; replay/catch-up; supervision; real trainer; aggregation/apply; checkpoint/finalization; runner topology; and bounded-memory behavior.
- `python -m compileall -q ndm scripts/frontier train.py tests`: recorded after the report commit.
- `bash -n scripts/frontier/trainpy_async_quorum_smoke_common.sh`: recorded after the report commit.
- `git diff --check origin/main...HEAD` and `git diff --check`: recorded after whitespace cleanup and report commit.
- Direct `git merge-base --is-ancestor` checks for all 12 source commits against
  final local HEAD and fetched `origin/main`: recorded after push.

## Remaining gap and non-authorization

This is valuable implementation and evidence WIP, not a completed resilient
foundation. True bounded one-manager-per-physical-node plus independently
supervised trainers is still not proven. The live 2-node injection/restart gate
with 16 independently supervised trainers is still not proven and is not
claimed complete. No debug scale ladder or production action follows
automatically from this merge.

The pushed final main commit ID, fetched remote equality, and complete
reachability result are recorded in the WG task log and artifact registration
after the final report commit, because those facts only exist after pushing and
fetching the commit that contains this report.
