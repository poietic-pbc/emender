# Fixed async-v2 exact-two-node clean admission rerun

Date: 2026-07-24

Task: `re-run-fixed`

Status: **CHANGED-PAYLOAD CLEAN JOB FAILED AFTER GENERATION 0 — sequence
stopped**

## Authority and scope

This serialized rerun conforms to
`docs/RESILIENT_DILOCO_COMPUTE_POOL.md` version 1,
`docs/RESILIENT_DILOCO_GAP_MATRIX.md`,
`docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md`, and ADR-002 in
`docs/ASYNC_DECOUPLED_DILOCO_V2.md`. The directly applicable requirements
are **R04–R08/R14–R16**, **NDP05–NDP10/NDP15–NDP17**, and
**V2A03–V2A07/V2A13–V2A18**.

Only a refreshed full-layout G2 followed by `clean-overlap` is authorized by
this task. A clean failure stops the sequence. No fault, rejection,
publication-failure, restart, 4+ node, or scale job has been submitted.

## Source and native attestation

The fixed WG source was reconciled with authoritative `origin/main`, committed,
and pushed:

```text
source commit  7a46b8d653d3fa6dc7c651b67c5f474b66d501cf
snapshot       /lustre/orion/bif148/proj-shared/emender/source-snapshots/emender-7a46b8d6-main
branch         main
remote main    7a46b8d653d3fa6dc7c651b67c5f474b66d501cf
```

The source includes the `fix-async-v2-2` integration and its regression for
the job-5066162 envelope. The authoritative snapshot was clean before build
and submission.

Every Python and native command was run after:

```bash
source scripts/frontier/activate_emender_frontier.sh
```

The native data plane was configured, built, installed, and passed all 10
CTests. Its manifest is:

```text
/lustre/orion/bif148/proj-shared/emender/validation/re-run-fixed-7a46b8d6-20260724/native-g2-install/native-artifacts.json
```

The serial acceptance controller independently rebuilt the same source and
reproduced native bundle SHA-256
`9884a02d84bd9560a15314c26e386868350b865e688cc4c701c802f4f686227a`.

## Refreshed exact-current-source G2

The authoritative current-main full-layout G2 was job `5066460`. It passed:

```text
5066460|native-ndp-g2-clean|COMPLETED|0:0|2|batch|debug|2026-07-24T10:44:27|2026-07-24T10:44:27|2026-07-24T10:44:53|2026-07-24T10:48:05|00:03:12|00:20:00|None|frontier[08046-08047]
```

Queued evidence separately reported
`State=PENDING Nodes=2 Partition=batch QOS=debug`. Terminal accounting
explicitly reports `NNodes=2`, `Partition=batch`, and `QOS=debug`; QoS is not
inferred from the partition field.

The immutable passing gate is:

```text
/lustre/orion/bif148/proj-shared/emender/validation/re-run-fixed-7a46b8d6-20260724/native-g2-evidence/5066460/full-layout-gate.json
```

It binds `gate=G2`, `status=passed`, `nodes=2`, provider `cxi`, source commit
`7a46b8d6...`, and bundle `9884a02d...`. This refreshed G2 passed before the
real clean admission was submitted.

An earlier G2 `5066437` passed against pre-reconciliation source
`a4c5fc4d...`; it was deliberately superseded and was not used for the
current-main clean admission.

## Serialized clean admission

After rebuilding the authoritative stage, verifying the exact G2 bundle, and
attesting the final immutable seed, the serial controller submitted only
phase zero:

```text
SUBMITTED phase=clean-overlap job_id=5066495
```

Initial queue and accounting evidence is:

```text
JobID=5066495 State=PENDING Nodes=2 Partition=batch QOS=debug Reason=(Priority)
5066495|resilient-e97-true-2n|PENDING|0:0|2|batch|debug|2026-07-24T10:51:16|...
```

`scontrol show job -dd 5066495` independently reports:

```text
Account=bif148 QOS=debug
JobState=PENDING Reason=Priority
Partition=batch
NumNodes=2-2
ReqTRES=cpu=2,mem=1000G,node=2,billing=2
```

At 11:01 EDT the scheduler estimated a 16:00 EDT start. The controller state
retains `active.phase=clean-overlap`, `active.job_id=5066495`,
`next_phase=0`, and an empty history, preventing a duplicate or later-phase
submission.

## Required terminal checks

On terminal continuation, retain:

1. running and terminal scheduler evidence explicitly naming `Nodes=2`,
   `Partition=batch`, and `QOS=debug`;
2. generation-zero result-root evidence preserving exact
   `accepted_tokens=5,245,440` separately from frozen lag-zero aggregation
   weight `36,718,080`;
3. the immutable commit/checkpoint and clean SLO verdict if the phase passes;
4. the first failing invariant and unchanged serial state if it fails.

If clean fails, stop immediately. Under no outcome does this task authorize a
4+ node submission or scale promotion.

## Terminal harvest and changed-payload correction

Job `5066495` ran on `frontier[09215-09216]` and terminated after 27 minutes
37 seconds:

```text
5066495|resilient-e97-true-2n|FAILED|1:0|00:27:37|2026-07-24T12:17:04|2026-07-24T12:44:41|batch|debug|frontier[09215-09216]
```

This is terminal accounting evidence for exactly two nodes (`NNodes=2` in the
retained job request), `Partition=batch`, and `QOS=debug`. The run reached and
committed generation zero. Its pool-control record preserved
`accepted_tokens=5245440`, while both node result roots independently
preserved:

```json
{
  "exact_tokens": 5245440,
  "weight": 5245440,
  "aggregation_weight": 36718080,
  "global_weight": 36718080,
  "result_root": "7636bfeba03dc38b23a829e850bad29a6e63e31b561002410fae757ab11406e1"
}
```

Thus the `fix-async-v2-2` accounting correction passed its live production
case: exact accepted tokens are not substituted for the frozen numerical
aggregation weight.

The first independent post-commit failure was at the next real trainer
boundary:

```text
ValueError: ScheduleFree z point is missing or malformed
```

ScheduleFree initializes per-parameter state lazily. The full-model async-v2
translation therefore encountered a parameter not touched by the first sparse
window and lacking a resident `z` point. Later manager submission timeouts and
generation-one reconnect identity errors were cascades after trainers failed.

The smallest correction initializes only missing ScheduleFree `z` and
`exp_avg_sq` entries from the resident model when a persistent real session is
bootstrapped. It preserves restored/live state and train mode. The focused
regression `test_persistent_real_worker_materializes_lazy_schedulefree_z`
reproduces a mixed initialized/lazy optimizer and verifies the complete x/z
translation. Activated validation passed that regression and the current
native build passed all 10 CTests.

No unchanged resubmission was made. Any subsequent admission must be a
source-pinned changed payload, must refresh G2 for that source, and remains
restricted to the clean phase on exactly two `batch`/`debug` nodes. Pending
state is checked at most 30 minutes apart; running state is checked every 2–5
minutes with generation/deadline progress inspection and immediate terminal
harvest.

## Changed-payload G2 and clean terminal result

The correction commit was `92e398c2914194847e2aa460f2f7bd0482f2fad4`.
It was published to authoritative `main`, snapshotted cleanly, rebuilt after
canonical Frontier activation, and passed the focused Python suites (28/28)
and all native CTests (10/10).

The required new exact-source full-layout G2 ran before any changed-payload
clean admission:

```text
5068822|native-ndp-g2-clean|COMPLETED|0:0|2|batch|debug
```

Queued/running scheduler evidence named `Nodes=2`, `Partition=batch`, and
`QOS=debug`. Terminal accounting independently named `NNodes=2`,
`Partition=batch`, and `QOS=debug`. The immutable G2 gate is:

```text
/lustre/orion/bif148/proj-shared/emender/validation/re-run-fixed-92e398c2-20260724/native-g2-evidence/5068822/full-layout-gate.json
```

Only after that pass, the serialized controller submitted the changed-payload
clean phase as job `5068873`. Its terminal accounting is:

```text
5068873|resilient-e97-true-2n|FAILED|1:0|00:27:50|2|batch|debug|2026-07-24T19:44:12|2026-07-24T20:12:02|frontier[02299,02304]
```

The retained `scontrol show job -dd 5068873` record also explicitly reports
`NumNodes=2`, `Partition=batch`, `QOS=debug`, and the changed source snapshot
and G2 gate paths in its immutable submit line.

Generation zero again reached an atomic handoff. The pool-control freeze
preserves `accepted_tokens=5245440` for exactly the two READY node
contributions. Both native node result identities preserve the separate
lag-zero numerical denominator:

```json
{
  "exact_tokens": 5245440,
  "weight": 5245440,
  "aggregation_weight": 36718080,
  "global_weight": 36718080,
  "result_root": "04dfbf27ebfe52371e56142e55ddbbae93588440c2906744720d4cf6d6bbab3d"
}
```

The immutable generation-one handoff additionally binds that result root,
`accepted_tokens=5245440`, source commit `92e398c2...`, exact provider `cxi`,
and native bundle `9884a02d...`.

The ScheduleFree lazy-state failure did not recur. The first independent
post-commit failure was failure to complete every node-1 generation-zero
apply marker before its absolute deadline. Trainers 4–7 reported missing
`native-result-applied-00000000-{03,04,05,06}.json`; peer trainers then
reported checkpoint-leader/apply deadlines. Consequently the next manager
attempt waited for generation-one native submissions that could not complete:

```text
TimeoutError: native metadata deadline expired for
.../node-0/control/native-submit-00000001-01.json

TimeoutError: native metadata deadline expired for
.../node-1/control/native-submit-00000001-04.json
```

Supervisor restarts then encountered stale/invalid generation metadata; those
identity errors and local-reduction deadlines are downstream cascades, not
evidence of a successful clean admission.

The serialized controller remains fail-closed with
`active.phase=clean-overlap`, `active.job_id=5068873`, `next_phase=0`, and an
empty history. In conformance with R04–R08/R14–R16,
NDP05–NDP10/NDP15–NDP17, ADR-002, and
V2A03–V2A07/V2A13–V2A18, execution stopped after this clean failure. No later
phase, duplicate unchanged payload, 4+ node submission, or scale
authorization was made.
