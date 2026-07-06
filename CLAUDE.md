<!-- WG-managed -->
# WG (project-specific guide)

This file is the **layer-2** project guide for agents working in this
WG project. It is NOT the universal chat-agent / worker-agent
contract — that is bundled inside the `wg` binary and emitted by:

```
wg agent-guide
```

Run `wg agent-guide` at session start (or read its output from a previous
session) to get the universal role contract: chat agent vs dispatcher vs worker
distinction, `## Validation` requirement, smoke-gate, cycle handling, git
hygiene, worktree isolation, "no built-in Task tool" rules, etc.

This file only covers things specific to this project. Add project-specific
build commands, test commands, architecture notes, and service recipes here.

## Runner / intake / implementation tasks

For this project, workers assigned to runner, intake, or implementation tasks
must perform the requested work before judging whether evidence exists. Do not
turn a runner task into an evaluator pass that merely reports missing artifacts.
For example, if a task asks you to launch the approved 32/64-node Slurm jobs,
submit or attempt the launch commands instead of grading the absence of prior
Slurm output; if a task asks you to create, download, or verify a refreshed seed
manifest, run the concrete manifest commands instead of only noting that no seed
manifest evidence is present yet.

Mark a runner/intake/implementation task incomplete only after attempting the
concrete commands available to you and logging the exact blocker, command, or
external dependency that prevented completion. This note is project-specific
mitigation for recent failures in `async-diloco-e97-32n64n-config` and
`register-refreshed-e97-seed-latest`, and is meant to reinforce, not replace,
the universal WG guidance from `wg agent-guide`.

**At the start of each session, run `wg quickstart` in your terminal to orient yourself.**
Use `wg service start` to dispatch work — do not manually claim tasks.

This guide is written to both `CLAUDE.md` and `AGENTS.md` and kept in
lock-step. The two files exist because Claude Code and Codex CLI look for
different filenames, but they should never drift in content. Any divergence is
a bug. Update both together.
