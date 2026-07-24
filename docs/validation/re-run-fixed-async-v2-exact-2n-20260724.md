# Fixed async-v2 exact-two-node clean admission rerun

Date: 2026-07-24

Task: `re-run-fixed`

Status: **CLEAN JOB QUEUED — terminal validation pending scheduler start**

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
