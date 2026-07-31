# Pipelined generation-start lifecycle fix

Date: 2026-07-20  
Task: `fix-pipelined-generation`  
Production reproducer: Frontier job `5037971`, authoritative source `87365c5f`

## Root cause

The native trainer emitted `native_generation_pipeline` telemetry after it had
received the frozen generation-0 result, but the telemetry duration referenced
`generation_started`, a local that was initialized only in the manager loop.
Every trainer taking the native `commit_ready` path therefore raised
`NameError` before applying the result and before the leader could publish the
atomic checkpoint.  The retained generation record proves the failure occurred
after the two-node floor was met (`Q_min=2`, 5,245,440 accepted tokens, result
root `63b3fef285173902e0ee4b54f4e7cab61fac8860c7427d9ef2c3750b9c641477`) but
before any authoritative generation-0 commit.

The fix initializes the monotonic origin at the single trainer-generation loop
entry.  That entry is shared by fresh starts, immutable-handoff resumes,
node-local recovery, and supervisor restarts.  It precedes publication, delayed
result receipt, rejection, safe-boundary admission, apply, and telemetry, so
all result outcomes observe a defined timestamp from the same monotonic clock.
No timestamp is synthesized on receipt and no deadline is reset.

## Failure-chain audit

The other terminal errors are consequences of the trainer exception, not an
independent clean-path lifecycle defect:

1. Both managers had already completed their first `FREEZE`, and the pool had
   durably recorded `commit_ready`.
2. Trainer leaders then raised the undefined-local exception.  They could not
   apply or publish the checkpoint release needed by their followers.
3. Supervision restarted roles while the persistent native service remained in
   the already-frozen attempt.  A restarted trainer's seal/submit consequently
   encountered route failure `-12`; restarted managers attempted `FREEZE` on
   the same frozen lifecycle and correctly received invalid-state `-3`.
4. The bounded supervisor exhausted its permitted retries and failed closed.

Thus removing the initiating exception lets the existing attempt continue
through safe-boundary apply and atomic commit; it does not weaken native state
validation to make repeated `FREEZE` or invalid routes succeed.  Focused tests
retain rejection/error propagation and exercise generation-0 `commit_ready`
through boundary admission followed immediately by nonblocking generation-1
handoff.

## Architecture conformance

This change conforms to *Resilient DiLoCo Compute Pool*, version 1, and checks
the required lifecycle/commit/deadline rows **R04, R06, R07, R11, R12, R14,
R16** and native rows **NDP06, NDP10, NDP13, NDP15, NDP16, NDP17** in
`docs/RESILIENT_DILOCO_GAP_MATRIX.md`:

- leased READY membership, the frozen Q/T floor, and bounded waits are
  unchanged;
- fenced identities, strict rejection, native lifecycle validation, and
  atomic checkpoint publication remain fail closed;
- the timestamp uses `time.monotonic()` at actual generation entry and does
  not change dense transport, backpressure, ownership, or filesystem use;
- no Slurm job was submitted.  A clean authoritative replacement run remains
  the downstream two-node rung required before scale.

## Validation

The focused regression anchors the trainer timer above every native result
lifecycle branch.  The deterministic pipeline regression uses the production
generation-0 accepted-token count, admits `commit_ready` at the safe boundary,
and reserves/hands off generation 1 without foreground blocking.  Canonical
runtime/launcher Python suites, the native build, and CTest are recorded in the
WG task log for the committed change.
