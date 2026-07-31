# WG dispatcher recovery for the E97 parity batch

Date: 2026-07-11

Task: `restore-wg-dispatcher`

## Outcome

The dispatcher is healthy and dispatched `audit-e97-async-256` itself. No E97
task was claimed manually, and the human-paused incident
`wg-dispatcher-assignment-crash-20260709` remains paused.

## Diagnosis

The apparent daemon crash was a slow first scheduling tick combined with
repeated forced restarts, not an assignment panic. The graph contains about 900
tasks and the first tick performs assignment scaffolding, agent reconciliation,
and a large worktree-retention sweep. Earlier starts were replaced before the
first tick reached its completion log:

- PID 2734424 started at `2026-07-11T17:11:10.135Z`.
- PID 2735762 replaced it at `2026-07-11T17:11:26.454Z`, only 16 seconds later.
- PID 2749852 started at `2026-07-11T17:14:14.540Z`; its tick entered the same
  long reconciliation path.
- PID 2761206 started at `2026-07-11T17:17:13.734Z` and was left uninterrupted.
  Its first tick completed at `2026-07-11T17:18:25.356Z`, about 72 seconds
  later, with `agents_alive=2`, `tasks_ready=2`, and `spawned=2`.

The exact diagnostic at the previously silent spawn boundary is now present in
`.wg/service/daemon.log`:

```text
[spawn] Worktree creation failed for agent-940, falling back to shared working directory: Failed to symlink .wg into worktree
[dispatcher] Spawned agent-940 (PID 2774389)
2026-07-11T17:18:25.356Z [INFO] Coordinator tick #1 complete: agents_alive=2, tasks_ready=2, spawned=2
```

Thus the worktree setup problem is recoverable and is not fatal to the daemon.
The installed `wg` binary (`/ccs/home/erikgarrison/.cargo/bin/wg`, timestamp
`2026-07-11T16:44:03Z`, reported healthy by `wg dev-check`) emits the failure
and falls back safely. The unrelated OpenRouter registry-refresh error is also
nonfatal: the dispatcher uses the configured Codex route and continued polling
after the registry refresh entered cooldown.

This is related to the July 9 report because that report also inferred a crash
from a log ending at `Assignment path ...` without a panic, stack trace, or
terminal error. The present run supplies the missing observation: on this large
graph, assignment and reconciliation can substantially exceed the nominal
five-second poll interval, and a forced restart truncates the log at whichever
substep is active. This task did not resume or modify the paused July 9
incident; its separate code-level hardening criteria remain valid.

## Corrective action

1. Verified the current installed binary with `wg dev-check` and configuration
   with `wg config lint`.
2. Started the service through `wg service start` and did not issue another
   forced start while tick 1 was running.
3. Allowed the first reconciliation tick to finish. No graph-state workaround,
   manual claim, task resume, or E97 configuration change was made.
4. Used `wg service status`, `wg status`, `wg show`, and the daemon log to verify
   dispatcher ownership and continued liveness.

Operationally, a successful `wg service start` should be followed with
`wg service status` and the daemon's `tick #1 starting/complete` diagnostics;
do not treat the nominal poll interval as a first-tick deadline or repeatedly
use `--force` while that tick is still reconciling a large graph.

## Validation evidence

- Dispatcher spawn: `audit-e97-async-256` changed from ready to in-progress at
  `2026-07-11T17:18:24.862857642Z`, assigned to live `agent-940`. Its task log
  records `Spawned by coordinator --executor codex --model gpt-5.6-sol`.
- Worker response: at `2026-07-11T17:18:36.610611444Z`, the worker logged
  `Starting audit...`, demonstrating that it is alive and processing the task.
- Poll continuity: the same PID completed ticks 1 through 8 by
  `2026-07-11T17:19:34.376Z`, exceeding the required three poll intervals.
- Service status after validation reported PID 2761206 running, two live
  workers, and zero ready tasks.
- Paused incident preservation: `wg show wg-dispatcher-assignment-crash-20260709`
  continued to report `Status: open (PAUSED)` throughout diagnosis.
- Graph preservation: no `wg claim`, `wg resume`, dependency edit, retry,
  requeue, or manual spawn command was used.

