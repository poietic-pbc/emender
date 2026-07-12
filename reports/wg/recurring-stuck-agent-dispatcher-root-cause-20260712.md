# Recurring stuck agents and frozen dispatcher ticks — incident report

Date: 2026-07-12 (evidence timestamps are UTC). Task: `root-cause-recurring`.
Scope: WG coordinator/evaluation lifecycle only. No E97 job, launcher, config, or
submission state was modified.

## Executive finding

The confirmed dispatcher freeze is a synchronous, unbounded maintenance call on
the daemon's sole poll-loop thread. It is not an inline-evaluator pipe wait.
Every coordinator tick calls worktree cleanup before lifecycle reconciliation and
dispatch. That cleanup synchronously traverses worktrees and invokes `git` via
`Command::output()` without a deadline. When one filesystem or Git operation
wedges, the service PID remains alive, `last_tick` remains at the last completed
tick, and all unrelated dispatch/reconciliation stops.

Inline FLIP/evaluator processes are independently spawned with stdin/stdout/stderr
set to null and `setsid()`. The coordinator does not wait for their exit. Thus a
hung evaluator consumes an agent slot and can leave its source in `PendingEval`,
but it does not by itself block the loop unless all slots are exhausted. The
observed correlation with evaluator spawn occurs because the next tick enters the
same synchronous maintenance path and can wedge there.

A second, contributing lifecycle defect is that `PendingEval -> Done` is only
performed by a later coordinator tick. Recording a score and marking the evaluator
Done are separate writes; no completion transaction also promotes the source.
Therefore an evaluator can be Done, and its evaluation can exist, while its parent
remains `PendingEval` indefinitely whenever the poll loop is frozen. Restart makes
the next maintenance/reconciliation tick run and normally repairs it.

## Evidence timeline

| UTC | Task / agents | Evidence and interpretation |
|---|---|---|
| 08:41:16–10:20:20 | `make-successful-256-node`, agent-968 | Worker reported done at 08:51:22 but registry completion was not recorded until 10:20:20. This is stale process/registry state after task completion, not proof that stdout blocked the daemon. |
| 10:26:16.592 | `.flip-make-successful-256-node`, agent-970 | Inline FLIP spawned (PID 3737475). Daemon log records tick #1 complete at 10:26:16.798, proving spawn returned and did not wait for evaluator exit. |
| 10:26:16.852–10:28:58.653 | same | Tick #2 starts, prints worktree-sweep entries through agent-44, and never completes. A new daemon starts at 10:28:58.653. This is conclusive localization inside `sweep_cleanup_pending_worktrees`, before evaluation reconciliation/dispatch. |
| 10:27:13.721 | agent-970 | Registry says FLIP completed while tick #2 was frozen. The child could finish and update state; the parent loop was independently wedged. |
| 10:28:58.803 | agent-969 | Restart tick detects/cleans the dead earlier FLIP agent. This demonstrates restart reconciliation. |
| 13:34:49–14:48:22 | `rerun-exact-proven-256`, agent-976 | Worker remained registered until explicit cleanup despite completion evidence. |
| 14:48:32.476–14:49:59.084 | FLIP agent-977 | Spawn tick completed at 14:48:32.602; service was restarted at 14:51:20.710. Again, the spawn itself returned. |
| 14:51:28.490–14:52:25.671 | evaluator agent-978 | Evaluation completed, but service did not restart until 15:38:09.982; source reconciliation was delayed. |
| 15:39:31–16:55:12 | `quality-pass-e97-1525000`, agent-979 | Task called `wg done` at 15:52:28 and became `PendingEval`; process/registry stayed live until killed before the 16:55 restart. |
| 16:55:22.130–16:56:08.801 | FLIP agent-980 | FLIP score 0.82 recorded and task Done. Ticks continued, disproving OpenRouter warning causality. |
| 16:56:08.618–16:57:33.038 | evaluator agent-981 | Score 0.91 recorded. Source promotion log is 16:56:40.966, performed by a later tick. |

Primary runtime evidence is retained in `.wg/service/daemon.log` and rotated
`.wg/service/daemon.log.1`; durable task transitions are in
`.wg/log/operations.jsonl`, `.wg/graph.jsonl`, registry state, and agent output.

## Precise code-path trace

The inspected WG source is the installed source checkout at
`/autofs/nccs-svm1_home1/erikgarrison/wg`; it is not part of this emender Git
repository, so this task does not modify it.

1. The daemon calls `coordinator_tick` synchronously and updates `ticks` and
   `last_tick` only after it returns
   (`src/commands/service/mod.rs:3029-3056`). Any unbounded operation inside the
   tick exactly produces “PID alive, frozen last_tick.”
2. `coordinator_tick` calls task cleanup/reaping, then
   `sweep_cleanup_pending_worktrees` before pending-eval reconciliation
   (`src/commands/service/coordinator.rs:4546-4665`).
3. The sweep scans every marked worktree synchronously
   (`src/commands/service/worktree.rs:957-1044`). It calls
   `find_branch_for_worktree`, `is_safe_to_reap`, and possibly removal on the
   poll-loop thread.
4. Safety checking invokes unbounded `git rev-parse` and `git merge-base` using
   `Command::output()` (`worktree.rs:111-135`). Removal recursively calculates
   directory size, removes large target trees, and invokes unbounded `git
   worktree remove` and `git branch -D` (`worktree.rs:178-208,211-314`). The
   branchless fallback also uses unbounded `Command::output()` and recursive
   removal (`worktree.rs:1071-1088`). There is no timeout, process-group kill,
   or off-loop worker.
5. Inline evaluation claims atomically, builds a shell wrapper, redirects all
   three standard streams to null, calls `setsid()`, uses `Command::spawn()`,
   registers the PID, and returns (`coordinator.rs:3214-3434`). There is no
   `wait`, `wait_with_output`, pipe reader, or PTY in this path. Unix zombie
   collection uses nonblocking `waitpid(..., WNOHANG)`
   (`service/mod.rs:1495-1508`).
6. A normal task's `wg done` chooses `PendingEval` while `.evaluate-X` is
   nonterminal; dot/system tasks bypass the gate
   (`src/commands/done.rs:1370-1390`). The transition is an atomic graph
   modification, but evaluator completion and source completion are not one
   transaction.
7. A later maintenance phase scans `PendingEval` tasks and promotes them when
   `.evaluate-X` is terminal (`coordinator.rs:883-929`). The scan runs inside a
   single `modify_graph` call, so it is idempotent/atomic once reached, but a
   frozen tick prevents it from being reached. It also promotes when the eval is
   missing (`:896-899`), a fail-open policy that should be reconsidered.
8. Registry liveness requires status, PID, and a fresh heartbeat
   (`src/service/registry.rs:98-132`), yet coordinator slot counting uses only
   alive status plus PID, not heartbeat (`coordinator.rs:112-123`). A
   completion-reporting process that stays alive can therefore retain a slot
   indefinitely until task status becomes terminal and a later tick reaps it.
   The task-aware reaper sends TERM with a five-second grace
   (`coordinator.rs:74-109`), but this too runs only on a healthy tick.

## Classification

### Confirmed root cause

Unbounded synchronous Git/filesystem worktree maintenance on the sole dispatcher
thread. Production log boundaries locate the freeze within the sweep, and the
source contains multiple blocking calls with no deadline. The minimal harness at
`scripts/wg/reproduce_dispatcher_blocking_child.sh` deterministically models the
same invariant: daemon alive, synchronous child wedged, `last_tick` unchanged.

### Contributing defects

- Parent reconciliation is deferred to the next tick rather than committed with
  evaluator completion. A valid evaluation plus evaluator Done is insufficient
  until `resolve_pending_eval_tasks` runs.
- Worker/evaluator registry completion is not a lease: a live PID with stale
  heartbeat can retain capacity. Inline agents get no periodic heartbeat.
- Task-aware reaping applies TERM to a PID, not an explicit process group; a CLI
  descendant can survive. The inline spawn does create a new session, so cleanup
  should target `-pid`/the process group and escalate to SIGKILL.
- Worktree sweeping is O(all retained marked worktrees) every tick and emits a
  line for each. The large retained population raises latency and exposure to one
  slow path even when no operation fully hangs.
- `last_tick` means “last completed tick”; there is no persisted phase/deadline or
  watchdog thread able to interrupt a currently running tick.

### Unrelated or disproven causes

- Chat session-lock warnings are unrelated: the failure reproduced with the chat
  coordinator disabled, and the task poll loop still wedged.
- OpenRouter registry-refresh errors are unrelated: dispatch is Codex, and logs
  show later ticks and successful FLIP/evaluator work after identical warnings.
- Child stdout/pipe backpressure is not present in inline evaluation: all streams
  are null and output is file redirection inside the shell.
- A stale installed binary increased incident frequency but is not sufficient:
  the structural unbounded calls and deferred reconciliation remain in the July
  12 source. Note that the final `wg dev-check` during this investigation again
  warned the installed binary predated emender HEAD; that comparison concerns
  different repositories and should not be treated as source freshness proof.

## Required durable fixes, ordered by impact

1. **Remove maintenance from the poll-loop critical path.** Run worktree sweep in
   a bounded background worker with at most one in flight. The dispatcher should
   continue readiness/reconciliation ticks if cleanup is slow or fails.
2. **Bound every subprocess and recursive cleanup.** Spawn each Git command in a
   new process group/session; deadline it (suggest 5–15 seconds for read-only Git,
   configurable longer for removal), send TERM to the group, wait a short grace,
   then SIGKILL the group and reap it. Recursive size/removal needs a deadline or
   a background job too. Never use bare `Command::output()` on the daemon thread.
3. **Add a tick watchdog.** Persist `tick_started_at`, phase, and operation; a
   separate supervisor thread must abort an overdue child/maintenance operation.
   Updating only `last_tick` after return cannot self-heal a blocked tick.
4. **Make evaluator completion reconcile atomically and idempotently.** In the
   graph transaction that marks `.evaluate-X` terminal, inspect its source: if
   source is `PendingEval`, load the matching durable non-FLIP evaluation and
   promote/reject it in the same locked graph write. Repeating the transaction
   must be a no-op. Keep the tick scan as restart repair. Do not infer “passed”
   solely from terminal evaluator status; require a matching score/verdict.
5. **Enforce agent leases and process-group cleanup.** Count an agent live only
   with a fresh heartbeat. Give inline agents a wrapper heartbeat or deadline.
   On task terminal, timeout, stale lease, or restart recovery, TERM then KILL the
   entire session/process group and mark the registry terminal atomically.
6. **Bound evaluator execution.** Apply a role-specific hard timeout to FLIP and
   evaluator commands even if provider timeouts fail. A timeout should record a
   loud terminal evaluator failure and allow retry/fail-closed policy without
   blocking unrelated work.

## Regression plan for the WG source repository

The source repository should add focused integration tests (not added here because
WG source is outside emender):

1. `child_reports_done_but_stays_alive`: child marks source PendingEval then
   ignores TERM; next tick remains responsive, kills the process group after
   grace, and retains unrelated dispatch capacity.
2. `inline_evaluator_hangs`: fake `wg evaluate` never exits; poll count continues
   increasing, unrelated ready task dispatches, evaluator reaches hard timeout,
   descendants are absent afterward.
3. `evaluator_records_score_then_fails_to_exit`: durable score is written and
   evaluator graph task becomes Done while wrapper remains alive; source is
   promoted in the same transaction and wrapper group is reaped.
4. `restart_reconciles_pending_eval_idempotently`: fixture starts with source
   PendingEval, terminal evaluator, and matching score. First restart promotes
   exactly once; second restart produces no duplicate log or state change.
5. `worktree_git_hang_does_not_freeze_tick`: inject a fake `git` that ignores
   TERM. Verify tick heartbeat/readiness continue, timeout and SIGKILL occur, and
   cleanup is retried later.
6. `maintenance_filesystem_walk_deadline`: inject/abstract a stalled directory
   walk and verify it cannot execute on or block the poll-loop thread.

The included harness covers the essential frozen-tick and hard-timeout shape:

```bash
bash scripts/wg/reproduce_dispatcher_blocking_child.sh
```

Expected: one `PASS frozen-tick` line followed by one `PASS bounded-tick` line.

## Operational mitigation until fixed

- Alert when `now - last_tick > 2 * poll_interval`, not merely when PID is dead.
- Capture `ps -ef --forest`, `/proc/$PID/wchan`, child process groups, daemon log
  tail, registry, graph, and open locks before recovery.
- Kill only the offending completed/hung agent process group, then restart the
  service. Immediately verify ticks advance and run/read-only reconciliation
  (`wg sweep --dry-run`, task status, evaluation record) before redispatch.
- Set bounded task/evaluator timeouts and keep agent concurrency above one so a
  single evaluator cannot exhaust capacity.
- Temporarily disable automatic worktree cleanup or reduce retained marked
  worktrees if configuration permits; perform cleanup manually outside the daemon.
- Do not “fix” this by changing chat locks, OpenRouter credentials, or E97 task
  state. Those are outside the causal path.

## Validation performed

- Correlated daemon log, rotated log, graph, registry, operations, and nine named
  agent outputs without modifying E97 state.
- Inspected dispatcher, inline spawn, nonblocking reap, worktree cleanup,
  heartbeat/liveness, `PendingEval`, FLIP/evaluator, and restart reconciliation
  paths in WG source commit `c264b1f3`.
- Ran the deterministic frozen-tick/hard-timeout harness.
- No WG source change was made because WG source is not in this repository; this
  is therefore a report-plus-reproducer commit, as permitted by the task.
