# Async v2.1 exact-two-node fault qualification: retained G2 runner failure

Date: 2026-07-28

WG task: `qualify-simple-async-v21-2n-faults`

Status: **INCOMPLETE — the refreshed native fault G2 prerequisite failed
closed before dataplane launch; every fault model phase, fresh recovery,
convergence, and scale remain blocked.**

The machine-readable companion is
`docs/validation/qualify-simple-async-v21-2n-faults-b756c14f-g2-failure-20260728.json`.
Its top-level `passed` and `full_pass` values are both `false`. This report
does not claim a fault qualification pass and does not call `wg done`.

## Terminal outcome and root cause

The exact fixed source passed its refreshed native clean G2 and mandatory
ten-generation clean model gate. The next prescribed serialized prerequisite,
native fault G2 job `5109414`, ended:

```text
5109414|FAILED|73:0|2|batch|debug|2026-07-28T19:41:51|2026-07-28T19:41:59|00:00:08
```

The batch stderr ended with:

```text
refusing to overwrite retained job evidence
```

This was a runner/operator error, not a source or native dataplane result.
After `sbatch` returned job ID `5109414`, the runner prematurely created
`$NDP_ARTIFACT_ROOT/5109414/scheduler-evidence` to retain queued `scontrol`.
The batch script correctly checked that its job-scoped artifact directory did
not already exist:

```bash
ARTIFACT_DIR="$NDP_ARTIFACT_ROOT/$SLURM_JOB_ID"
[[ ! -e $ARTIFACT_DIR ]] ||
  { echo "refusing to overwrite retained job evidence" >&2; exit 73; }
```

It therefore exited before loader preflight, membership establishment, any
fault injection, native transport, or gate validation. No
`failure-injection-gate.json` exists for `5109414`.

The failed directory, queued/terminal scheduler records, stdout, and stderr
were retained unchanged. Per the task's first-failure rule, no retry—changed
or unchanged—was submitted. No fault model job, durable fault-model
collector, convergence job, or four-or-more-node job was submitted.

## Reviewed authorities and immutable execution identity

Before Python, build/preflight, and submission, the runner sourced
`scripts/frontier/activate_emender_frontier.sh` and reviewed:

- `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, including its conformance
  checklist and R01–R16;
- `docs/RESILIENT_DILOCO_GAP_MATRIX.md`, including R01–R16, NDP01–NDP17,
  V21S01–V21S17, and ISP01–ISP07;
- `docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md`, including G0–G6,
  fault semantics, bounds, and NDP01–NDP17; and
- accepted ADR-002 in `docs/ASYNC_DECOUPLED_DILOCO_V2.md`.

All Python work used the activated Python 3.12.13 environment. Before this
attempt, the recovery-focused regression suite passed `83/83`, the official
current-source preflight passed `142/142`, and native CTest passed `10/10`.

| Identity | Exact value |
|---|---|
| immutable source/snapshot | `b756c14fa3a7495e733c6498f69730c3e115f281` |
| execution-source digest | `95cc5018d948e844091bef0d96c5b07605385b0bcedff653760b70b200964b8b` |
| native build manifest SHA-256 | `8cd0f836e9f798a48e72aef7c60591d6d939abe2b1b3cdb8b06d4c30502fe00c` |
| native bundle SHA-256 | `f19e10be9987cfdb551a8dd75c5c88145c3cf35b73c54d3898fe562ce4182441` |
| clean G2 SHA-256 | `68fff9f44d066aee80a2fc32830e0b7a4352b72bbdc107508c4a29cb0743ae26` |
| passed clean launch manifest SHA-256 | `cda81bb450aff6fe34feaf1fbddbb0037da7278c4c9d0741f715525211ca4747` |
| policy ID | `async-decoupled-v2.1-simple` |
| policy digest | `fa9def95daf7bce25f1b962ca5437e7a76317b94ccfb9a710fbf126a344e7d98` |
| launcher digest | `278fae0985efd302542424e6feb1d584812a0cffe5c03cb282861a90b355dad1` |
| train-args digest | `afc2a65fd8c73499e74e21cb9531c978206c3a9c898e42d18cc58bb93eb9fe9c` |
| data identity digest | `91321b2b90bb159f3aa73881455778f10e8df588edd526b1066281fa72997962` |
| tokenizer digest | `94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069` |
| clean model payload digest | `6f02204cdddc6512d05d8cbeaaeb8d9a08952ffb9837fe825bea347d0269e0d7` |
| fault G2 run ID | `native-g2-fault-b756c14fa3a7-20260728T234058Z` |
| fault G2 payload ID | `b756c14fa3a7-e97-g2-fault-20260728T234058Z` |

The immutable snapshot was clean and equal to pushed `origin/main` when the
clean and fault-G2 payloads were prepared. Later movement of `origin/main`
does not change the snapshot, build manifest, or submitted payload identity.

## Exact seed chain and passed clean prerequisite

The clean prerequisite remained bound to:

```text
step:     2300930
tokens:   150793748480
bytes:    7719680116
SHA-256:  0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2
```

Submit-side authority agreed under attestation SHA-256
`27e234891df02b64b9db77fc784c341e5a3ae6e87418b8f1af167776d1d710bb`.
Clean model job `5109029` used job-scoped `sbcast` to
`/tmp/emender-e97-seed-5109029/checkpoint-step-2300930.pt`.
`frontier01875` and `frontier04065` retained offline verification receipts
with exact bytes and SHA-256, and each recorded `network_fetches=0`
(receipt SHA-256 values `0f42de71...c4b` and `63fc88a1...9a5`).

Native clean G2 job `5108993` passed at exact `2|batch|debug`; its gate
artifact is the `68fff9f4...ae26` identity above. Mandatory clean model
`5109029` exited immediately after its exact success condition:

- 10 immutable commits;
- 10 immutable checkpoints;
- 20 node-apply authorities;
- exactly eight trainer receipts in every node apply, 160 total;
- 16 measured trainers and 10 measured K40 windows per trainer;
- zero restarts and zero injections;
- maximum commit/anchor lag 1 and result/speculative lag 0;
- capacity-one result mailbox and sealed descriptor high-water marks;
- native OWNED maximum 0.017903 seconds;
- exact eta=1 policy math; and
- passing correctness, training-lane, pause-tail, and forbidden-path gates.

Its scheduler-owned `afterany` collector `5109030` independently retained a
literal passing semantic verdict (SHA-256
`e70ff5808c345fdcc35f2ffe72ce13c07a716f1a8a87513b589bacaf0a7b7b64`)
and passing terminal verdict (SHA-256
`5cbd1091083e4fb8ddcc1b507e0b3e9f40e092dbd3408f48f1247acb033bb555`).

Those clean results are prerequisites only. They do not substitute for fault
G2 or any required injected fault/restart evidence.

## Serialized phase plan and actual disposition

No fault model was submitted, so the user-required pre-model phase table was
not used to authorize a model job. The controller's reviewed plan is retained
here to make the blocked work explicit. `02:00:00` would have been only a
safety ceiling.

| Phase | Exact injection | Minimum pre/post evidence and immediate terminal condition | Expected | Actual |
|---|---|---|---|---|
| fault-baseline | none | g0→g2; 2 commits/checkpoints; 4 node applies; 32 receipts; stop after both g2 all-eight applies | 15–25 min | not submitted; fault G2 blocked |
| fault-rejoin | node-1 READY +45 s at g2; node-0 trainer lane 3 after g3 apply; node-0 native service at g4 `owner_transport`; node-1 manager after g4 `published_node_applied` | g2→g6; 4 commits/checkpoints; 8 applies; 64 receipts; every serialized loss detected/rejoined before next injection; stop at g6 all-eight applies or first failure | 40–60 min | not submitted |
| fresh-allocation recovery | no injection; strictly newer fence | reload g6 model/outer/token state; exactly 5 additional K40 windows; at least 3 commits; 10 applies; 80 receipts; stop at final all-eight applies | 40–50 min | not submitted |

At most one active job was maintained. Fault G2 was submitted only after clean
collector `5109030` had a literal terminal pass. After `5109414` failed, the
queue was empty and remained empty.

## Exact scheduler and retained artifact evidence

| Job | Purpose | State/exit | Nodes | Partition | QOS | Start | End | Elapsed |
|---:|---|---|---:|---|---|---|---|---|
| 5108993 | refreshed native clean G2 | COMPLETED/0:0 | 2 | batch | debug | retained in clean G2 | retained | 00:03:00 |
| 5109029 | mandatory clean model | COMPLETED/0:0 | 2 | batch | debug | 18:35:28 | 19:38:47 | 01:03:19 |
| 5109030 | durable clean collector | COMPLETED/0:0 | 1 | batch | normal | 19:39:53 | 19:39:58 | 00:00:05 |
| 5109414 | refreshed native fault G2 | **FAILED/73:0** | 2 | batch | debug | 19:41:51 | 19:41:59 | 00:00:08 |

Queued and terminal `scontrol -dd` for `5109414` retain
`Account=bif148`, `QOS=debug`, `Partition=batch`, exact
`NumNodes=2`, terminal nodes `frontier[00769-00770]`, `TimeLimit=00:20:00`,
the snapshot, build manifest, clean gate, run/payload IDs, native layout,
weights, CXI variables, and full submit line. Terminal `sacct` explicitly
retains both `Partition` and `QOS`.

Key artifact hashes:

| Retained artifact | SHA-256 |
|---|---|
| `native-ndp-g2-fault-5109414.err` | `68ae288930f4ff97502ee25bf34b533e9d53ad34f787f65e55c05a55ebc20af1` |
| queued `scontrol-5109414` | `02d6d9c8bfdfaa65e1322d9303651590fd36744d1d49e093660fe5a1c623b790` |
| terminal `scontrol-5109414` | `b74a013ade1594c5bd4de2b062be2d2f7703aeb5ebce8fabb8e27b64c1e4e6ca` |
| terminal `sacct-5109414` | `e8600f5e9358451d78947b6a4567d98b414d17e2e02415fb7a79c6d12b202079` |

## Compute-pool conformance checklist: R01–R16

| ID | Evidence and disposition |
|---|---|
| R01 | Clean exact allocation fence passed. Fault and fresh-allocation fences were not exercised; qualification blocked. |
| R02 | Stable identities and clean incarnations passed. Loss/new-incarnation rejoin was not exercised. |
| R03 | Clean two-member READY closure passed. Missing/late/rejoin population closure was not exercised. |
| R04 | Clean bounded-version behavior passed; lag-2 acceptance and lag-3 live drop/catch-up probes were not run. |
| R05 | Exact deterministic eta=1/native clean math passed. Fault-side rejection/reference probes were not run. |
| R06 | Clean `Q_min=2`, token threshold, and deadline policy passed. Node-loss no-one-node authority was not exercised. |
| R07 | Clean immutable commit/checkpoint chain through generation 10 passed. Fault/restart and fresh-reload chains are absent. |
| R08 | Clean bounded chunk/checksum/credit transport passed. Fault replay/reassignment/rejection evidence is absent. |
| R09 | Clean model-free managers/local handoff passed. Trainer, service, and manager losses were not exercised. |
| R10 | Clean compiled CXI/memfd and forbidden-path verdict passed. Fault transport was not launched. |
| R11 | **Blocked:** no injected worker/node loss, reconstruction, catch-up, or drain/rejoin evidence exists. |
| R12 | Clean model/outer/token checkpointing passed. Fresh-allocation recovery was not submitted. |
| R13 | Backend isolation passed for the clean two-node rung only; no scale claim is made. |
| R14 | Clean causal timing/deadline telemetry passed. Required fault detection/release timestamps are absent. |
| R15 | Clean exact-token eta=1 math passed. Fault/fresh checkpoint state continuity is absent. |
| R16 | Safety boundary passed: fail closed, retain evidence, and submit no convergence or 4+ job. Overall fault qualification remains incomplete. |

## Native dataplane conformance: NDP01–NDP17

| ID | Evidence and disposition |
|---|---|
| NDP01 | Exact clean native G2 passed; fault native peer-control campaign did not launch. |
| NDP02 | Clean point-to-point CXI path passed; loss/rejoin route behavior is blocked. |
| NDP03 | Clean persistent service path passed; native-service loss/restart is blocked. |
| NDP04 | Clean memfd handoff passed; fault-side ownership timeout/release is blocked. |
| NDP05 | Clean deterministic sharded reference passed; rejection/fault references are blocked. |
| NDP06 | ABI/wire/policy identities are exact; fault validator produced no gate. |
| NDP07 | Clean leased routes passed; exclusion and new-incarnation rejoin are blocked. |
| NDP08 | Clean capacity bounds passed; replay/reassignment fault bounds are blocked. |
| NDP09 | Clean credit/OWNED high-water evidence passed; injected OWNED timeout is blocked. |
| NDP10 | Clean retry-safe path passed; duplicate/conflict/corruption fault matrix is blocked. |
| NDP11 | Clean direct owner redistribution passed; owner-loss replay/reassignment is blocked. |
| NDP12 | Clean all-eight node apply passed; partial-apply prevention during loss is blocked. |
| NDP13 | **Blocked:** no trainer/native-service/manager loss, route-local containment, or bounded rejoin evidence exists. |
| NDP14 | Native ABI `131073` and wire 2.1 are bound; no passing fault G2 artifact exists. |
| NDP15 | Clean checkpoint/apply identities passed; fault/new-fence identity continuity is blocked. |
| NDP16 | Clean semantic/tail verdict passed; fault semantic/tail verdict is absent. |
| NDP17 | Clean G2 gate ordering passed. Fault G2 failed before validation, so no model or larger rung was authorized. |

## Async-v2.1 conformance: V21S01–V21S17

| ID | Evidence and disposition |
|---|---|
| V21S01 | Exact source/policy/bundle/launcher/data/tokenizer/seed identities passed clean admission. Fault-model admission was never attempted. |
| V21S02 | **Fault anchor blocked:** lag 0–2 admission, lag-3 pre-mutation drop, and peer catch-up were not exercised in this attempt. |
| V21S03 | Clean exact-token eta=1 accumulation passed; injected delayed/missing contribution behavior is absent. |
| V21S04 | Clean bounded local speculation passed; node-loss two-window pause/catch-up behavior is absent. |
| V21S05 | **Fault anchor blocked:** clean `Q_min=2` and deadlines passed, but timestamped delayed/missing/rejoin membership and no-one-node authority under loss are absent. |
| V21S06 | Clean checkpoint version/token chain passed; fault/fresh chain is absent. |
| V21S07 | **Fault anchor blocked:** clean atomic x/z/interval apply passed, but loss-time no-double-correction evidence is absent. |
| V21S08 | **Fault anchor blocked:** failed-publication invisibility, mailbox replacement, duplicate/conflict/checksum/nonfinite/wrong-fence rejection were not exercised. |
| V21S09 | **Fault anchor blocked:** clean capacity high-water marks passed, but injected queue/credit/replay/reassignment/deadline bounds are absent. |
| V21S10 | **Fault anchor blocked:** clean two-node quorum passed; exclusion/new-incarnation rejoin and no one-node authority during node loss are absent. |
| V21S11 | **Fault anchor blocked:** clean all-eight receipt gates passed; trainer-loss and partial-apply prevention evidence is absent. |
| V21S12 | **Fault anchor blocked:** clean CXI/memfd and zero Python/Lustre dense paths passed; the fault transport campaign did not launch. |
| V21S13 | **Fault anchor blocked:** no timestamped injection, detection, route-local containment, release, or recovery latency exists. |
| V21S14 | **Fault/restart anchor blocked:** exact seed/offline clean bootstrap passed, but fresh newer-fence model/outer/token reload and five additional K40 windows are absent. |
| V21S15 | Clean exact identity and immutable reports passed; no full-pass fault verdict exists. |
| V21S16 | Controller/runner serialization passed: one job, first-failure stop, no unchanged resubmit, no convergence/scale launch. |
| V21S17 | Missing/late/rejoin timing relative to READY snapshot/open/freeze/deadline was not generated, so bounded non-blocking scale closure cannot be derived. |

## Immutable snapshot protocol: ISP01–ISP07

| ID | Evidence and disposition |
|---|---|
| ISP01 | Clean immutable snapshot admission and exact source identities passed. |
| ISP02 | Clean frozen interval descriptors remained immutable and bounded. |
| ISP03 | Clean ownership transfer/commit publication ordering passed. |
| ISP04 | Clean result staging/mailbox replacement bounds passed. |
| ISP05 | Clean all-eight apply receipts gated each next READY version. |
| ISP06 | Clean checkpoint publication and exact-token state were immutable. |
| ISP07 | Clean semantic/tail validator passed; fault/restart ISP evidence is blocked with the fault campaign. |

## Validation and terminal condition

- Machine report parses as JSON and enumerates all 16 R, 17 NDP, 17 V21S,
  and 7 ISP requirement IDs.
- Terminal accounting explicitly records `2|batch|debug` for the failed
  fault G2.
- The failed artifact path remains retained; no deletion or overwrite was
  performed.
- The user queue was empty after the failed job.
- No unchanged payload was resubmitted.
- No fault model, fresh recovery, convergence, or four-or-more-node job was
  submitted.

The task's full-pass terminal condition was not met. Downstream convergence
and scale authorization must remain blocked. A future attempt must use a new
payload identity and retain scheduler evidence outside the batch script's
reserved `$NDP_ARTIFACT_ROOT/$SLURM_JOB_ID` directory until the job creates
that directory itself.
