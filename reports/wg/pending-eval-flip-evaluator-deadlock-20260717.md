# WG bug: FLIP evaluator route failure strands PendingEval lifecycle

Date: 2026-07-17  
Reporter graph: `spinozans/emender`  
Affected WG: `wg 0.1.0`, binary `/ccs/home/erikgarrison/.cargo/bin/wg`, mtime `2026-07-16T15:20:21-04:00`  
Owning repository: `https://github.com/graphwork/wg.git` (`main` inspected at `ca35e2cc976e1062895930833e8fcd10348a702a`)

Upstream routing completed: branch `report/pending-eval-flip-inline-route`, commit
`80bcbfabec53f0ad5fe4bc3f11c9cfe27dd86716`. Maintainer intake/PR URL:
`https://github.com/graphwork/wg/pull/new/report/pending-eval-flip-inline-route`.

## Summary

With `agency.auto_evaluate=true`, `agency.flip_enabled=true`, and the standard `codex-cli` route, a completed task can remain `PendingEval` indefinitely. Its `.flip-<id>` task repeatedly fails before claim with:

```text
invalid invocation-scoped evaluator route "gpt-5.4-mini"
```

After five failures the spawn circuit breaker marks FLIP `Incomplete`; the `.evaluate-<id>` task remains open behind FLIP and the parent cannot be promoted to `Done`. No evaluator agent or evaluation operation is created. Restart/reconciliation does not repair the invalid serialized route. Manual completion/retry of hidden system tasks has been required.

This looks like a dependency deadlock in `wg why-blocked`, but source and dispatcher evidence show the normal parent-to-FLIP edge is not the root defect. Current readiness code deliberately lets dot-prefixed system tasks cross a `PendingEval` or `FailedPendingEval` source. Generic `why-blocked` does not apply that system-task exception and therefore reports a false blocking chain. The dispatcher does select FLIP and then fails route validation.

## Minimal reproduction

Run in a disposable directory; do not use a production graph:

```bash
scratch=$(mktemp -d /tmp/wg-pending-eval-repro.XXXXXX)
cd "$scratch"
git init -q
env -u WG_DIR -u WG_PROJECT_ROOT -u WG_WORKTREE_PATH \
  -u WG_BRANCH -u WG_WORKTREE_ACTIVE wg init --route codex-cli
env -u WG_DIR -u WG_PROJECT_ROOT -u WG_WORKTREE_PATH \
  -u WG_BRANCH -u WG_WORKTREE_ACTIVE \
  wg config --auto-assign false --auto-evaluate true --flip-enabled true --no-reload
env -u WG_DIR -u WG_PROJECT_ROOT -u WG_WORKTREE_PATH \
  -u WG_BRANCH -u WG_WORKTREE_ACTIVE \
  wg add 'Minimal parent' --id minimal-parent --no-place \
  -d $'## Validation\n- [ ] lifecycle reaches Done automatically'
env -u WG_DIR -u WG_PROJECT_ROOT -u WG_WORKTREE_PATH \
  -u WG_BRANCH -u WG_WORKTREE_ACTIVE wg claim minimal-parent
env -u WG_DIR -u WG_PROJECT_ROOT -u WG_WORKTREE_PATH \
  -u WG_BRANCH -u WG_WORKTREE_ACTIVE wg done minimal-parent
env -u WG_DIR -u WG_PROJECT_ROOT -u WG_WORKTREE_PATH \
  -u WG_BRANCH -u WG_WORKTREE_ACTIVE wg service tick
```

Observed tick:

```text
[eval-scaffold] Created FLIP task '.flip-minimal-parent' blocked by 'minimal-parent'
[eval-scaffold] Created evaluation task '.evaluate-minimal-parent' blocked by 'minimal-parent'
[dispatcher] Priority dispatch order: [.flip-minimal-parent:5(d0)]
[dispatcher] Spawning eval inline for: .flip-minimal-parent - FLIP: minimal-parent (model: gpt-5.4-mini)
[dispatcher] Failed to spawn eval for .flip-minimal-parent: invalid invocation-scoped evaluator route "gpt-5.4-mini"
Tick complete: 0 alive, 1 ready, 0 spawned
```

The serialized FLIP task contains `"model":"gpt-5.4-mini","provider":"codex","exec_mode":"bare"`. `spawn_eval_inline` validates only the bare `model` via `execution_system_key(route)` and rejects it before using the provider or role-resolved route. A project initialized with the documented `--route codex-cli` therefore scaffolds a task that its own inline evaluator cannot dispatch.

To reproduce the exact soft-done state, eagerly scaffold `.flip-minimal-parent` and `.evaluate-minimal-parent` (or allow one dispatcher tick before the worker calls `wg done`), then complete the parent. The parent transitions to `pending-eval`. `wg ready`/the dispatcher treats `.flip-minimal-parent` as ready through the system-task bypass, while `wg why-blocked .flip-minimal-parent` incorrectly labels the parent as its root blocker. Dispatch then fails on the bare route above.

Full commands and captured output are in `reports/wg/pending-eval-flip-evaluator-deadlock-evidence-20260717.md`.

## Production evidence and impact

The same lifecycle stall recurred for:

- `reconcile-resilient-e97`: worker completed `2026-07-16T15:35:40Z`; parent was only promoted `2026-07-16T20:57:36Z` after recovery.
- `integrate-recovered-resilient`: worker completed `2026-07-16T15:48:17Z`; parent was only promoted `2026-07-16T20:57:25Z` after recovery.
- `wire-split-role`: worker completed `2026-07-17T08:06:42Z`; parent entered `PendingEval`. FLIP failed at `08:06:47`, `08:06:56`, `08:07:05`, `08:07:13`, and `08:07:22`, then the circuit breaker marked it incomplete. Before manual recovery, FLIP and evaluator had zero operations and zero agent runs. The parent was promoted only at `08:15:11Z` after hidden-task repair.

For `wire-split-role`, the daemon remained healthy and ticked with free capacity but spawned no evaluator after the circuit breaker. Downstream `run-resilient-e97-2` stayed gated. This makes successful implementation work appear unfinished and requires operators to know hidden task IDs and manually mutate lifecycle state.

## Root cause and diagnostic gaps

1. `eval_scaffold::scaffold_flip_task` resolves the evaluator role, then stores the resolved model and provider in separate fields.
2. `spawn_eval_inline` receives only `task.model` as `evaluator_model` and calls `execution_system_key(route)` on the bare string. `gpt-5.4-mini` has no handler prefix, so the documented standard Codex route is rejected even though `task.provider` is `codex` and the project role is `codex:gpt-5.4-mini`.
3. Repeated pre-claim failures trip the generic spawn circuit breaker. `Incomplete` is not repaired by PendingEval reconciliation; the evaluator remains behind FLIP.
4. `wg why-blocked` uses generic dependency semantics, unlike dispatcher readiness. It reports `PendingEval` as an unsatisfied parent even for hidden system dependents, masking the fact that dispatcher readiness already bypasses it.
5. `wg cycles` correctly reports no cycles: the serialized graph is acyclic (`parent -> FLIP -> evaluator`). The deadlock is a lifecycle/state-machine wait plus an execution-route failure, not a structural strongly connected component. Cycle detection should not be broadened to call this an edge cycle; lifecycle health analysis should report it separately.

Internal system-task edges do need special semantics, but current `query::{ready_tasks,ready_tasks_cycle_aware,ready_tasks_with_peers_cycle_aware}` already implement the required PendingEval bypass. The missing consistency is in diagnostics and route reconstruction.

## Broken-pipe causality

Daemon `Broken pipe (os error 32)` entries are unrelated noise, not the cause of this stall:

- They occur on many ordinary CLI/status interactions, including at `08:13:36`, `08:14:41`, `08:14:52`, and `08:15:01`, after the decisive FLIP route failures.
- The five causal failures have explicit route-validation errors and no agent PID/run.
- Source logs connection-handler write failures generically in `commands/service/mod.rs`; client disconnects can therefore emit Broken pipe after the request outcome is already available.
- The daemon continued ticking and servicing commands throughout. The route failure reproduces in a one-shot disposable `wg service tick` with no long-lived IPC client and without a causal broken-pipe event.

Broken pipes should ideally be downgraded/ignored for peer-closed connections, but that is a separate observability issue.

## Direct evaluation, restart, retry, and recovery

- `wg evaluate run <parent>` accepts `PendingEval` in current source, so the old evaluation precondition bug is not the cause. Direct evaluation may record work, but it does not repair an incomplete FLIP task or atomically advance the hidden chain; it is not a supported end-to-end recovery.
- Service restart only reloads the same scaffolded bare model and repeats the same validation. It cannot normalize the historical task route.
- `wg retry .flip-<id>` resets `Incomplete` to `Open`, but retries are not useful until routing is fixed; they are safe only if evaluation result recording is idempotent.
- Manual `wg done` on FLIP/evaluator can release the parent but bypasses intended evaluation and must not be the normal repair.

## Expected behavior and proposed resolution

Use the canonical role/task dispatch resolver for inline evaluator tasks. Do not validate a separately serialized bare model without its provider. Either persist a handler-qualified route such as `codex:gpt-5.4-mini`, or reconstruct the route from task provider/model/profile using the same single dispatch planner used elsewhere. Role resolution must remain correct for Codex, Claude, Nex/OpenRouter, custom handlers, per-task profiles, and historical rows.

Add reconciliation that detects hidden FLIP/evaluator tasks whose route is invalid or whose pre-claim circuit breaker fired, normalizes the route from current configuration/profile, resets them once, and resumes the lifecycle. It must preserve an already recorded evaluation and never duplicate a semantic verdict.

Make `why-blocked` use the same dependent-aware predicate as dispatcher readiness, or explicitly state: “generic dependency unsatisfied, but system PendingEval bypass makes this task dispatcher-ready.” Add a lifecycle-health diagnostic for `PendingEval` parents whose hidden chain is incomplete/failed/unroutable.

## Acceptance criteria / regression coverage

- [ ] A real daemon flow with `auto_evaluate=true`, `flip_enabled=true`, and `wg init --route codex-cli` automatically reaches `parent PendingEval -> FLIP -> evaluator -> parent Done`.
- [ ] FLIP and evaluator each have one successful semantic execution; no bare-route validation failure and no duplicate verdict occurs.
- [ ] A no-agent-run hidden task that reaches `Incomplete` is automatically and idempotently repaired.
- [ ] Restart between parent completion, FLIP, and evaluator resumes exactly once at every boundary.
- [ ] `wg evaluate run` on a `PendingEval` parent does not strand or duplicate the scaffolded lifecycle, and a supported repair command/reconciler completes the chain.
- [ ] Retry/reset is idempotent after pre-claim failure, post-claim failure, and result-record-before-done failure.
- [ ] Ordinary downstream tasks remain blocked until the evaluator gate passes, then unblock automatically.
- [ ] Existing historical `PendingEval` graphs with bare `model` plus provider metadata are migrated/reconciled without manual `wg done` on hidden tasks.
- [ ] Readiness and `why-blocked` agree about the PendingEval system-task bypass.
- [ ] Tests cover Codex plus at least one non-Codex handler/profile so the fix does not hard-code a provider.
- [ ] `wg cycles` remains structural; a separate lifecycle diagnostic identifies this non-SCC wait.
