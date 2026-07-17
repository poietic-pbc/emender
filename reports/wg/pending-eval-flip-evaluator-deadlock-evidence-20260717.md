# Raw evidence: PendingEval / FLIP / evaluator lifecycle stall

This artifact was captured on 2026-07-17. Production graph commands were read-only except for the already-recorded manual recovery performed before this investigation. The minimal graph lived under `/tmp/wg-pending-eval-repro.K39L2d` and did not touch training tasks.

## Installed identity and routing

```text
$ command -v wg
/ccs/home/erikgarrison/.cargo/bin/wg

$ wg --version
wg 0.1.0

$ wg dev-check
wg binary: /autofs/nccs-svm1_home1/erikgarrison/.cargo/bin/wg (2026-07-16T15:20:21Z)

$ strings "$(command -v wg)" | grep graphwork
WG_UPGRADE_SOURCE_URLhttps://github.com/graphwork/wg.git
wg/0.1.0 (+https://github.com/graphwork/wg)

$ git ls-remote https://github.com/graphwork/wg.git HEAD
ca35e2cc976e1062895930833e8fcd10348a702a HEAD
```

Relevant effective configuration:

```text
[agency]
auto_evaluate = true
auto_assign = true
eval_gate_threshold = 0.7
flip_enabled = true

[models]
evaluator.model = "codex:gpt-5.4-mini"
flip_inference.model = "codex:gpt-5.4-mini"
flip_comparison.model = "codex:gpt-5.4-mini"
```

The live `wg config --show` also warned that registry model `gpt-5.4-mini` has no slash, but reported the role route as `codex:gpt-5.4-mini`.

## Disposable graph commands and output

```text
$ scratch=$(mktemp -d /tmp/wg-pending-eval-repro.XXXXXX)
$ cd "$scratch" && git init -q
$ env -u WG_DIR -u WG_PROJECT_ROOT -u WG_WORKTREE_PATH -u WG_BRANCH \
    -u WG_WORKTREE_ACTIVE wg init --route codex-cli
Initialized WG at /tmp/wg-pending-eval-repro.K39L2d/.wg

$ env ... wg config --auto-assign false --auto-evaluate true --flip-enabled true --no-reload
Set agency.auto_evaluate = true
Set agency.auto_assign = false
Set agency.flip_enabled = true

$ env ... wg add 'Minimal parent' --id minimal-parent --no-place \
    -d $'## Validation\n- [ ] lifecycle reaches Done automatically'
Added task: Minimal parent (minimal-parent)

$ env ... wg claim minimal-parent
Claimed 'minimal-parent'

$ env ... wg done minimal-parent
Marked 'minimal-parent' as done

$ env ... wg service tick
Running single coordinator tick (max_agents=8, executor=codex, model=codex:gpt-5.5)...
[eval-scaffold] Created FLIP task '.flip-minimal-parent' blocked by 'minimal-parent'
[eval-scaffold] Created evaluation task '.evaluate-minimal-parent' blocked by 'minimal-parent'
[dispatcher] Priority dispatch order: [.flip-minimal-parent:5(d0)]
[dispatcher] Spawning eval inline for: .flip-minimal-parent - FLIP: minimal-parent (model: gpt-5.4-mini)
[dispatcher] Failed to spawn eval for .flip-minimal-parent: invalid invocation-scoped evaluator route "gpt-5.4-mini"
Tick complete: 0 alive, 1 ready, 0 spawned
```

Serialized FLIP row (long evaluator description omitted):

```json
{"kind":"task","id":".flip-minimal-parent","status":"open","after":["minimal-parent"],"exec":"wg evaluate run minimal-parent --flip","model":"gpt-5.4-mini","provider":"codex","exec_mode":"bare","spawn_failures":1}
```

This minimal run proves the route failure and the zero-run pre-claim behavior. The production run proves the `PendingEval` form because its pipeline was eagerly scaffolded before `wg done`.

## Live affected-task state

`wire-split-role` completion and recovery log:

```text
2026-07-17T08:06:42.607958331Z Task pending eval (agent reported done; awaiting `.evaluate-*` to score)
2026-07-17T08:14:16.978441977Z Manual lifecycle recovery after confirmed PendingEval/FLIP deadlock.
2026-07-17T08:15:11.213126221Z PendingEval -> Done (evaluator passed; downstream unblocks)
```

FLIP log:

```text
08:06:47 Spawn failed (attempt 1/5): invalid invocation-scoped evaluator route "gpt-5.4-mini"
08:06:56 Spawn failed (attempt 2/5): invalid invocation-scoped evaluator route "gpt-5.4-mini"
08:07:05 Spawn failed (attempt 3/5): invalid invocation-scoped evaluator route "gpt-5.4-mini"
08:07:13 Spawn failed (attempt 4/5): invalid invocation-scoped evaluator route "gpt-5.4-mini"
08:07:22 Spawn failed (attempt 5/5): invalid invocation-scoped evaluator route "gpt-5.4-mini"
08:07:22 Circuit breaker tripped ... Task marked incomplete for evaluator review.
```

Before recovery, `.flip-wire-split-role` was `Incomplete`, with zero operations and zero agent runs; `.evaluate-wire-split-role` was `Open`, also with zero operations and zero agent runs. After recovery, FLIP trace still shows zero agent runs and only the manual retry/done operations, proving no evaluator process caused the transition.

Read-only status for all known recurrences:

```text
reconcile-resilient-e97:
  worker completion 2026-07-16T15:35:40.676Z
  PendingEval -> Done 2026-07-16T20:57:36.968Z

integrate-recovered-resilient:
  worker completion 2026-07-16T15:48:17.309Z
  PendingEval -> Done 2026-07-16T20:57:25.433Z

wire-split-role:
  worker completion 2026-07-17T08:06:42.608Z
  PendingEval -> Done 2026-07-17T08:15:11.213Z, after manual hidden-task recovery
```

Dispatcher health during the stall:

```text
Service: running (PID 3273908)
Uptime: 11h 21m
Dispatcher: enabled, max_agents=8, poll_interval=5s
Last tick: 2026-07-17T08:15:01.438Z (#3123, agents_alive=1/8, tasks_ready=1, spawned=0)
```

## Readiness, diagnostics, and cycles

Current upstream source at `ca35e2c` implements the system bypass in:

- `src/query.rs::blocker_satisfied_for_dependent`
- `src/query.rs::ready_tasks_cycle_aware`
- `src/query.rs::ready_tasks_with_peers_cycle_aware`
- `src/commands/done.rs` manual system-task blocker filtering

The key predicate is equivalent to:

```rust
if dependent_is_system && blocker_pending_eval {
    return true;
}
```

Thus the dispatcher's selection of `.flip-minimal-parent` is expected even when a parent is soft-done. Generic diagnostics do not share this predicate. Before recovery:

```text
.flip-wire-split-role
 \-- blocked by: wire-split-role (status: PendingEval) <-- ROOT CAUSE

.evaluate-wire-split-role
 \-- blocked by: .flip-wire-split-role (status: Incomplete) <-- ROOT CAUSE
```

`wg cycles` output:

```text
No cycles detected in after edges.
```

That is correct for the acyclic edge chain. The indefinite wait is a lifecycle failure, not an SCC.

## Source-grounded route failure

`src/commands/eval_scaffold.rs::scaffold_flip_task` stores:

```rust
model: Some(flip_resolved.model),
provider: flip_resolved.provider,
```

`src/commands/service/coordinator.rs::spawn_eval_inline` then does:

```rust
if let Some(route) = evaluator_model {
    execution_system_key(route)
        .with_context(|| format!("invalid invocation-scoped evaluator route {route:?}"))?;
    (handler_for_model(route).as_str().to_string(), route.to_string())
}
```

The call site passes `task.model.as_deref()` but not `task.provider`. This exactly explains the observed rejection of `gpt-5.4-mini` despite serialized `provider:"codex"`.

## Direct evaluation and recovery observations

- Current upstream includes `tests/smoke/scenarios/evaluate_accepts_pending_eval.sh`; direct evaluation accepts PendingEval/FailedPendingEval. Therefore status precondition is not causal.
- A direct evaluation is not an atomic repair for the hidden FLIP -> evaluator chain and can leave task statuses unchanged.
- Restart/reconciliation reuses the invalid task model and cannot repair it.
- Retry resets the circuit breaker but repeats the invalid route unless configuration/task serialization is normalized.
- Manual hidden-task completion recovered `wire-split-role`, but its FLIP trace retained zero agent runs, so it bypassed intended semantics and is unsuitable as supported recovery.

## Broken pipes

Live daemon errors included:

```text
2026-07-17T08:06:56.720Z Error handling connection: Broken pipe (os error 32)
2026-07-17T08:13:36.518Z Error handling connection: Broken pipe (os error 32)
2026-07-17T08:14:41.900Z Error handling connection: Broken pipe (os error 32)
2026-07-17T08:14:52.007Z Error handling connection: Broken pipe (os error 32)
2026-07-17T08:15:01.525Z Error handling connection: Broken pipe (os error 32)
```

The causal errors are explicit pre-spawn route-validation failures. Broken pipes occur after and outside that sequence while the daemon continues ticking. The disposable one-shot tick reproduces route rejection without relying on daemon IPC. Conclusion: broken pipes are unrelated client-disconnect observability noise.

## Verified routing / maintainer handoff

Owning repository and intake are writable over the configured GitHub SSH identity:

```text
$ ssh -T git@github.com
Hi ekg! You've successfully authenticated, but GitHub does not provide shell access.

$ git ls-remote git@github.com:graphwork/wg.git HEAD
ca35e2cc976e1062895930833e8fcd10348a702a HEAD
```

If issue creation cannot be automated, ingest this report into the upstream WG graph with:

```bash
cd /path/to/graphwork/wg
wg add 'Fix: inline FLIP evaluator loses provider and strands PendingEval' \
  --id fix-inline-eval-qualified-route \
  -d "$(cat /path/to/emender/reports/wg/pending-eval-flip-evaluator-deadlock-20260717.md)"
```

The upstream issue URL/identifier is recorded in the task log and appended to the concise report after filing.

API issue filing was unavailable because this environment has GitHub SSH access
but no HTTPS/API credential. The report was therefore routed through the writable
upstream source repository as required:

```text
branch: report/pending-eval-flip-inline-route
commit: 80bcbfabec53f0ad5fe4bc3f11c9cfe27dd86716
intake: https://github.com/graphwork/wg/pull/new/report/pending-eval-flip-inline-route
```
