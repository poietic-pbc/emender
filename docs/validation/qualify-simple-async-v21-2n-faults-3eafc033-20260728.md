# Async v2.1 exact-two-node fault/restart qualification: fixed-source retained failure

Date: 2026-07-28

WG task: `qualify-simple-async-v21-2n-faults`

Status: **INCOMPLETE — the fixed-source fault-rejoin phase failed; fresh
recovery, convergence, and scale remain blocked.**

The machine-readable companion is
`docs/validation/qualify-simple-async-v21-2n-faults-3eafc033-20260728.json`.
Its top-level `full_pass` and `passed` values are both `false`. This report
supersedes the result of the earlier `a71939c8` attempt without changing that
historical report.

## Terminal outcome

The exact fixed source passed every required prerequisite and the
two-generation fault baseline. The next serialized phase failed closed:

- model job `5108175` ended `FAILED|1:0|2|batch|debug` after `00:23:32`,
  from `2026-07-28T16:47:06-04:00` to
  `2026-07-28T17:10:38-04:00`;
- the job requested `02:00:00` only as a safety ceiling and drained
  immediately when the route-local restart budget was exhausted;
- durable `afterany` collector `5108176` independently ended
  `COMPLETED|0:0|1|batch|normal` in four seconds;
- its immutable terminal verdict has `passed=false`, SHA-256
  `cea268abb7281d644049993455cc427c8b3dfdf4f4855bf50f2ecebaac39663d`;
- only generations 3 and 4 of the required g2→g6 phase committed;
- the node-0 native-service injection was never reached; and
- the controller did not submit fresh-allocation recovery, convergence, or
  any four-or-more-node job.

The first current-run, uninjected error is retained in
`fault/faults/logs/node-1-manager.err` (SHA-256
`7b8fb2f784d5508ccefb4be6c4fac867f5ec9972277e89af963c0e74581e2d51`):

```text
RuntimeError: generation is not open
```

Node 1 tried to contribute to generation 3 after that generation had closed
while node 0 was recovering from the intended trainer loss. The RPC error
escaped the manager and caused a whole node-1 cohort restart. After the
intended generation-4 manager loss, its first replacement failed:

```text
ValueError: native peer recovery handshake disagrees with manifest
```

The following replacement failed recovery with:

```text
RuntimeError: conflicting recovery incarnation rejected
```

The supervisor then emitted `restart_exhausted`. This is a live qualification
failure of the generation-closed/recovery-incarnation path, not a soak
timeout.

## Authorities and immutable identity

Before Python, tests, builds, preflight, and submission, the runner sourced
`scripts/frontier/activate_emender_frontier.sh` and reviewed:

- `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, including the conformance
  checklist and R01–R16;
- `docs/RESILIENT_DILOCO_GAP_MATRIX.md`, including R01–R16, NDP01–NDP17,
  and V21S01–V21S17;
- `docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md`, including G0–G6 and
  NDP01–NDP17; and
- accepted ADR-002 in `docs/ASYNC_DECOUPLED_DILOCO_V2.md`, plus
  `docs/ASYNC_V21_EXECUTION_SOURCE_IDENTITY.md`.

The immutable snapshot was clean and equal to pushed `origin/main` at
submission. The fixed-source preflight passed `140` with one expected skip,
the dependency-focused validation passed `80/80`, native CTest passed
`10/10`, and post-run controller/pool/launcher validation passed `113/113`
under activated Python 3.12.13.

| Identity | Exact value |
|---|---|
| source commit | `3eafc033a707614e1ed4e56fe6cb350400193295` |
| execution-source digest | `5e94459d306f004c327aeec4f3cd5850b088154fb944cccec16e9232add2f004` |
| native manifest | `562d99c574d59d6a318f47309b12780addff540ae798bf779d576bbd515422eb` |
| native bundle | `f19e10be9987cfdb551a8dd75c5c88145c3cf35b73c54d3898fe562ce4182441` |
| policy | `async-decoupled-v2.1-simple` |
| policy digest | `fa9def95daf7bce25f1b962ca5437e7a76317b94ccfb9a710fbf126a344e7d98` |
| launcher | `278fae0985efd302542424e6feb1d584812a0cffe5c03cb282861a90b355dad1` |
| train args | `afc2a65fd8c73499e74e21cb9531c978206c3a9c898e42d18cc58bb93eb9fe9c` |
| data | `91321b2b90bb159f3aa73881455778f10e8df588edd526b1066281fa72997962` |
| tokenizer | `94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069` |
| campaign | `f6cdac1c6feba428c239eef7447fd974842d3941bbf88d5b5df9671fd6fdaa82` |
| fault-rejoin payload | `8c2d4e2bb5d61f38cda18177a9aca17f7cf76319fcb2c485e43cee5c874f7334` |

## Exact seed chain

Every phase remained bound to:

```text
step:     2300930
tokens:   150793748480
bytes:    7719680116
SHA-256:  0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2
```

Submit-side cache and authority documents agreed under attestation SHA-256
`27e234891df02b64b9db77fc784c341e5a3ae6e87418b8f1af167776d1d710bb`.
Job `5108175` used job-scoped `sbcast` to
`/tmp/emender-e97-seed-5108175/checkpoint-step-2300930.pt`.
`frontier08881` and `frontier08885` independently retained exact offline
receipts (SHA-256 `9c004ce8...1c3a0` and `ad0d556a...7ffa4`) with
`network_fetches=0`. No compute-node network fetch occurred.

## Pre-submission phase table and serialization

This exact table was logged before fault-rejoin submission. `02:00:00` was
never a requested soak duration.

| Phase | Exact injection | Required pre/post evidence | Immediate success | Expected | Actual |
|---|---|---|---|---|---|
| fault-baseline | none | g0→g2; 2 commits/checkpoints; 4 node applies; 32 receipts | g2 checkpoint plus both eight-trainer applies | 15–25 min | pass, 17:07 |
| fault-rejoin | node-1 READY +45 s at g2; node-0 trainer lane 3 after applied; node-0 native service at g4 `owner_transport`; node-1 manager after g4 `published_node_applied` | g2→g6; 4 commits/checkpoints; 8 node applies; 64 receipts; serialized loss/rejoin/replay | g6 checkpoint plus both all-eight applies after every recovery | 40–60 min | **failed, 23:32** |
| fresh-allocation-recovery | none; strictly newer fence | reload g6 model/outer/token; exactly 5 more K40 windows; at least 3 commits; 10 node applies; 80 receipts | g11 checkpoint plus both node applies | 40–50 min | not submitted |

At most one debug/model job was active. Every collector completed or remained
the sole durable dependent before another model could be considered. Payload
digests `3359249c...a5b` and `8c2d4e2b...7334` were each submitted exactly
once. No unchanged digest was resubmitted.

## Exact scheduler evidence

Queued and running checks used:

```text
squeue -h -j "$JOB_ID" -o '%i|%T|%P|%q'
```

Terminal checks used:

```text
sacct --format=JobIDRaw,State,ExitCode,NNodes,Partition,QOS,Start,End,Elapsed -P
```

| Job | Purpose | State/exit | Nodes | Partition | QOS | Start | End | Elapsed |
|---:|---|---|---:|---|---|---|---|---|
| 5106323 | native clean G2 | COMPLETED/0:0 | 2 | batch | debug | 11:47:02 | 11:49:58 | 00:02:56 |
| 5106331 | clean model | COMPLETED/0:0 | 2 | batch | debug | 13:02:19 | 14:05:06 | 01:02:47 |
| 5106332 | clean collector | COMPLETED/0:0 | 1 | batch | normal | 14:06:12 | 14:06:17 | 00:00:05 |
| 5107408 | native fault G2 | COMPLETED/0:0 | 2 | batch | debug | 14:11:19 | 14:13:15 | 00:01:56 |
| 5107452 | fault baseline | COMPLETED/0:0 | 2 | batch | debug | 15:16:02 | 15:33:09 | 00:17:07 |
| 5107453 | baseline collector | COMPLETED/0:0 | 1 | batch | normal | 15:33:43 | 15:33:48 | 00:00:05 |
| 5108175 | fault rejoin | **FAILED/1:0** | 2 | batch | debug | 16:47:06 | 17:10:38 | 00:23:32 |
| 5108176 | failure collector | COMPLETED/0:0 | 1 | batch | normal | 17:16:20 | 17:16:24 | 00:00:04 |

Terminal `scontrol show job -dd 5108175` retains `Account=bif148`,
`QOS=debug`, `Partition=batch`, `NumNodes=2`,
`NodeList=frontier[08881,08885]`, the exact payload/source/build/G2/policy/
launcher/data/tokenizer/seed identities, `Q_min=2`, both two-window lag
bounds, all serialized injection triggers, and the `02:00:00` ceiling.
Collector `5108176` retains `Account=bif148`, `Partition=batch`,
`QOS=normal`, `Nodes=1`, and scheduler-owned `afterany:5108175`.
The collector retained model accounting, payload logs, payload input, and
`sacct` independently; it failed because the model exit was `1:0` and the
required semantic verdict was absent. It retired the digest with
`verdict=failed`, cleared `active_job`, and left the user v2.1 queue empty.

## Passed prerequisites and baseline

- Clean G2 `5106323` passed, artifact SHA-256
  `3cee4f30af86c3a63047e306dae87a47b6e7ff271e5a3a3e694d7a897ad27cb1`.
- Clean model `5106331` and collector `5106332` passed, terminal SHA-256
  `8e88589282102e9dcb6dd11aa14a4341067a318f8685e66b80bf67ce429c1f70`.
  It stopped after exactly 10 commits/checkpoints, 20 node applies, and 160
  trainer receipts.
- Fault G2 `5107408` passed, artifact SHA-256
  `5cfe92e36de8548a84bf56f94651737db19db45f26f373e8528601c292d5cf64`.
  It proved CXI/FI_EP_RDM, owner epoch 1→2, new incarnation, old-epoch
  rejection, one reassignment, 134217728 replay bytes, atomic/no partial
  commit, bounded release, and zero MPI collectives, all-rank barriers,
  Python dense socket bytes, full-copy handoff bytes, disk replay bytes, or
  trainer spool bytes.
- Fault baseline `5107452` and collector `5107453` passed, payload
  `3359249ccded953c04f52c691e19d320f365bf499ebb002db186a1ae58880a5b`,
  terminal SHA-256
  `080f849459f5812d8f756807fa6d8cff578b8eb4e000a133115a7b8cc6b2c4a5`.
  It retained 2 commits/checkpoints, 4 node applies, 32 receipts,
  `one_node_commit_authority=false`, and passing probes for lag 0–2,
  lag-3 drop/catch-up, identical duplicate idempotence, conflicting identity,
  checksum, nonfinite, wrong fence, failed-publication invisibility,
  capacity-one mailbox replacement, and local OWNED timeout/release.

## Fault-rejoin causal evidence

1. Node 1 completed the prescribed 45-second READY delay. Generation 2 then
   closed only on the exact two-member READY snapshot
   `node-0:bfc50aef...` plus `node-1:1169c928...`, with
   `required_contributions=2`, `accepted_tokens=5,245,440`, and
   `reason=accepted_floor_met` (record SHA-256
   `b95767cbec76c8fd8c2512777f36bb8165ebb69cc6efe5f111d3a6cdbc040e5d`).
2. Generation 3 committed as receipt `a33e2359...6de7`, checkpoint
   `25edf959...638c`, outer step 3, accepted tokens `150809484800`. Both node
   apply authorities contained exactly eight trainer receipts.
3. The intended node-0 trainer-3 loss fired at
   `17:00:39.835413-04:00`, only after applied authority. The complete node-0
   cohort failed at `17:00:56.855041` and reconstructed under new node
   incarnation `33a526f8...` at `17:01:15.024550` (18.170 seconds), with
   eight new trainer incarnations.
4. At `17:02:44.553627`, uninjected node 1 failed on
   `generation is not open`. It reconstructed under incarnation
   `b664f205...` at `17:03:04.757934` (20.204 seconds).
5. No commit occurred with one cohort. Generation 4 closed only on the two
   replacement incarnations `33a526f8...` and `b664f205...`,
   `required_contributions=2`, and `accepted_tokens=5,245,440` (record
   SHA-256
   `88b312a59a7e96336204a4fc35a8112bd5cfc41b184dc65fc44a19c627871213`).
6. Generation 4 committed as receipt `6ec906ac...d1dc`, checkpoint
   `cb8a5695...b8ae`, outer step 4, accepted tokens `150814730240`. Both node
   apply authorities again contained exactly eight receipts.
7. The intended node-1 manager loss fired at
   `17:09:18.228626`, after `published_node_applied`. The cohort failed at
   `17:09:35.248030` and reconstructed at `17:09:53.437903`; that replacement
   then failed the peer recovery handshake at `17:09:57.473911`. A final
   replacement was rejected as a conflicting recovery incarnation, and
   `restart_exhausted` was emitted at `17:10:15.521770`.
8. The planned node-0 native-service loss at generation-4
   `owner_transport` was never reached. The allocation drained; no all-rank
   abort primitive was needed.

Within this failed phase, the retained authorities prove no one-node commit,
no partial eight-trainer node apply, no stale-incarnation commit, and no
observed double x/z correction. Those safety properties do not convert the
phase into a pass. The supervision stream SHA-256 is
`96b747b0844e50c495fc1abb7aeaa1a0b5585425c545270d4676c406655f9755`.

## Conformance checklist mapping

The machine report contains one entry for every required ID. This report
records the same controlling disposition.

| ID | Evidence/disposition |
|---|---|
| R01 | Exact allocation fences passed; required fresh newer fence was not reached. |
| R02 | Stable worker identities and new incarnations retained; recovery handshake failed. |
| R03 | Every live close used two leased READY members, never launched-rank authority. |
| R04 | Baseline lag/identity/corruption probes passed; live closed-generation catch-up failed. |
| R05 | Exact token-weighted reference and native G2 passed. |
| R06 | `Q_min=2`, `T_min=3934080`, 420-second deadline, and two-member closes retained. |
| R07 | Immutable g1–g4 receipt/checkpoint chains retained; required restart chain incomplete. |
| R08 | Bounded chunks/checksums/credits/replay/reassignment/release passed G2/baseline. |
| R09 | Model-free managers and immutable local handoffs ran; every fault did not complete. |
| R10 | Compiled memfd/CXI path and zero Python/Lustre dense path passed. |
| R11 | **Failed:** closed-generation and peer-handshake errors caused uninjected restarts. |
| R12 | Model/outer/token advanced through g4; fresh allocation was blocked. |
| R13 | Backend isolation retained; no scale-adapter claim. |
| R14 | Causal timestamps and deadlines retained; terminal phase success absent. |
| R15 | Exact η=1 token math/reference roots passed through g4. |
| R16 | Safety boundary passed: no convergence or 4+ job launched. |
| NDP01 | Native peer control/fencing ran; full recovery campaign failed. |
| NDP02 | Zero MPI collective/all-rank barrier/abort path retained. |
| NDP03 | Persistent C++17 CXI/RDM service ran; its planned loss was not reached. |
| NDP04 | Producer-direct memfd and zero full-copy path passed. |
| NDP05 | Exact-token deterministic native reference passed. |
| NDP06 | Fenced identity plus stale/conflict/checksum rejection passed. |
| NDP07 | Current-fence leased endpoint routes retained. |
| NDP08 | Resident/transport/timeout bounds and release passed. |
| NDP09 | Credits bounded; baseline timeout/lag probes did not pause foreground work. |
| NDP10 | CRC/SHA/idempotence/corruption/nonfinite rejection passed baseline. |
| NDP11 | Reassignment, old-epoch rejection, bounded replay, and zero disk replay passed G2. |
| NDP12 | g3/g4 each had two atomic eight-receipt node applies. |
| NDP13 | **Failed:** route containment worked, but recovery errors exhausted restart. |
| NDP14 | Versioned C ABI and metadata-only control passed build/G2. |
| NDP15 | Immutable g1–g4 checkpoints/applies retained; restart/fresh chain incomplete. |
| NDP16 | Raw causal telemetry retained; full passing semantic/tail verdict absent. |
| NDP17 | Clean/fault G2 order passed before model work; larger rung never launched. |
| V21S01 | Exact policy/schema/source/bundle/launcher/payload admission passed. |
| V21S02 | **Fault anchor:** lag 0–2/lag-3 probes passed; live catch-up hit a closed generation. |
| V21S03 | Exact positive token counts remained the sole quantitative weight. |
| V21S04 | K40/η=1 outer/token state advanced immutably through g4. |
| V21S05 | **Fault anchor:** exact identities/Q/T/deadlines passed; restart phase failed. |
| V21S06 | OWNED timeout/release baseline passed; full campaign incomplete. |
| V21S07 | **Fault anchor:** g3/g4 all-eight node apply was atomic. |
| V21S08 | **Fault anchor:** mailbox replacement and invalid/failed-publication rejection passed. |
| V21S09 | **Fault anchor:** resident/queue/credit/replay/mailbox/timeout bounds passed prerequisites. |
| V21S10 | **Fault anchor:** `Q_min=2` prevented one-node commits, but uninjected peer restart occurred. |
| V21S11 | **Fault anchor:** atomic reconstruction occurred, then recovery handshake/restart failed. |
| V21S12 | **Fault anchor:** compiled CXI/memfd and zero MPI/Python/Lustre dense path retained. |
| V21S13 | **Fault anchor:** timestamped triggers/detection/release retained; passing terminal tail absent. |
| V21S14 | **Restart anchor:** seed and g3/g4 checkpoint state retained; fresh-fence restore not run. |
| V21S15 | Fault/restart gate is not a full pass. |
| V21S16 | Promotion and scale authorization remain blocked. |
| V21S17 | READY/open/freeze/deadline evidence exists, but failed fault evidence cannot authorize scale closure. |

## Final disposition

`full_pass=false`. The fixed-source payload was submitted exactly once and
failed on current-run recovery errors. The retained failing phase blocks
fresh recovery and all downstream convergence/scale work. `wg done` is not
allowed; this task must be marked incomplete after collector finalization.
