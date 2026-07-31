# WG Dispatcher Assignment Crash: debug-async-quorum

Date: 2026-07-09

## Summary

The WG dispatcher service repeatedly exited while trying to dispatch the full-agent task
`debug-async-quorum`. The service became stable again after that task was paused.

This caused the chat agent to bypass normal WG dispatch and perform work manually. That
workaround was incorrect for this project because it made job submission/monitoring less
durable and harder to audit from the WG graph.

## Impact

- `debug-async-quorum` could not be dispatched normally by `wg service`.
- The service was down or immediately exiting until the triggering task was paused.
- The work was partially recorded through `wg log debug-async-quorum`, but Slurm job
  submission and polling were performed by the chat agent instead of by WG worker tasks.
- The task is now paused to keep the service alive.

## Triggering Task

- Task: `debug-async-quorum`
- Title: `Debug async DiLoCo with exact B4/K40 smoke parity`
- Status after mitigation: `open (PAUSED)`
- Dependency state: `.assign-debug-async-quorum` had been marked `done`.
- Downstream: `.flip-debug-async-quorum` still open.

## Observed Service Behavior

Before mitigation, the daemon log ended around:

```text
[dispatcher] Assignment path for 'debug-async-quorum': Learning (total_assignments=239)
```

After that line, `wg service status` reported:

```text
Service: not running
```

No clear panic, stack trace, or terminal error was emitted in the tail of
`.wg/service/daemon.log`.

After pausing `debug-async-quorum` and restarting/starting the service, status became:

```text
Service: running (PID 626062)
Dispatcher: enabled, max_agents=8, poll_interval=5s, executor=codex, model=codex:gpt-5.5
Last tick: 2026-07-09T11:46:47.662273139+00:00
ready: No tasks ready
```

The remaining service-status error is unrelated registry refresh noise:

```text
No API key found for provider 'openrouter'
```

That did not appear to kill the dispatcher once `debug-async-quorum` was paused.

## Earlier Related Issue

Before this crash, WG also warned about one corrupt agency cache YAML:

```text
.wg/agency/cache/agents/79ba3db989872efc9eba2fab3611f28d3d518e782e53684539c3212bb661a9f8.yaml:
YAML error: did not find expected key at line 294 column 1
```

That file ended with:

```yaml
attractor_weight: 0.7
: 0.7
```

It was moved to:

```text
.wg/agency/cache/quarantine/79ba3db989872efc9eba2fab3611f28d3d518e782e53684539c3212bb661a9f8.yaml.corrupt-20260709T1127Z
```

After quarantining that file, the YAML warning disappeared, but the dispatcher still exited
when trying to assign `debug-async-quorum`.

## Reproduction Sketch

1. Ensure `debug-async-quorum` is open and unpaused.
2. Start the service:

```bash
wg service start
```

3. Wait for one dispatcher tick.
4. Observe `.wg/service/daemon.log`.

Expected failing log tail:

```text
[dispatcher] Assignment path for 'debug-async-quorum': Learning (total_assignments=239)
```

Then:

```bash
wg service status
```

reports `Service: not running`.

## Expected Behavior

One of:

- Dispatch `debug-async-quorum` normally.
- If assignment fails, leave the service alive and mark/log the task as blocked, failed,
  or pending manual assignment with a clear error.
- Emit a stack trace or structured error that identifies the failed assignment primitive,
  model/provider/config issue, or malformed cache record.

The dispatcher should not silently exit.

## Mitigation Applied

```bash
wg pause debug-async-quorum
wg service start
```

The service is currently stable with no ready tasks.

## Process Lesson

The chat agent should not bypass WG dispatch for long-running or coordination-sensitive
work when the service fails. It should pause the triggering task, restore service health,
and create a WG/system bug report before continuing.

