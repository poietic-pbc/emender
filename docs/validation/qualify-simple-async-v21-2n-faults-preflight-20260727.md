# Async-v2.1 exact-two-node fault gate: superseded-source preflight

Date: 2026-07-27

WG task: `qualify-simple-async-v21-2n-faults`

Status: **INCOMPLETE — stopped before every `sbatch`; no native-fault,
model-fault, convergence, promotion, or scale job was submitted.**

## Outcome

The prescribed preflight against the then-qualified source
`46c8043791e9b14c4cb3376c1fb03ebe7fe6932f` completed successfully, but the
source was withdrawn before the native fault prerequisite could be submitted.
WG message `#1` from `scale-v21-8n-clean`, received at
`2026-07-27T23:50:10.059327681Z`, required an immediate stop because the exact
source lacks the mandatory durable held-job/`afterany` terminal collector and
corrected execution-source identity. The graph now makes this task depend on:

```text
fix-v21-durable-collector-source-identity
    -> requalify-v21-durable-collector-2n-clean
    -> qualify-simple-async-v21-2n-faults
```

At capture time, `fix-v21-durable-collector-source-identity` was in progress
and `requalify-v21-durable-collector-2n-clean` was open. The new clean
qualification must pass on the corrected execution identity before this task
may build, submit G2 fault, or submit the model fault/restart controller.

This record is deliberately a machine-checkable non-pass. It does not reuse,
promote, or reinterpret the passing `46c80437` clean artifact. It does not call
`wg done`.

## Authorities read

Before preflight, the runner read the complete current versions of:

- `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, including the required conformance
  checklist and R01–R16;
- `docs/RESILIENT_DILOCO_GAP_MATRIX.md`, including R01–R16, NDP01–NDP17,
  V21S01–V21S17, and ISP01–ISP07;
- `docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md`, including the G0–G6 order,
  NDP01–NDP17, failure semantics, exact native protocol, and telemetry
  contract; and
- accepted ADR-002 in `docs/ASYNC_DECOUPLED_DILOCO_V2.md`, including the
  exact-two-node policy, Q/T floor, lag semantics, atomic all-eight apply,
  newer-fence recovery, and no-one-node-authority rule.

The intended fault/restart anchors were V21S02, V21S05, and V21S07–V21S14:
distinct lag clocks and lag-3 rejection; full fenced contribution identity and
the exact `Q_min=2`, `T_min=3,934,080` floor; exact-once atomic correction;
verified capacity-one mailbox replacement; finite resident/credit/replay
bounds; leased READY expiry/new-incarnation rejoin without one-node commit
authority; atomic all-eight apply/recovery; compiled CXI/memfd transport;
causal phase telemetry; and immutable newer-fence model/outer/token restart.
None of those live fault anchors is claimed by this preflight.

## Immutable predecessor that was verified, then superseded

The prior exact-two-node clean pass was:

| Identity | Value |
|---|---|
| source commit | `46c8043791e9b14c4cb3376c1fb03ebe7fe6932f` |
| source tree | `660f817c0419943d33d07f91573a6abc933bab7a` |
| source digest | `553bc996723bc1698a37f847c981b1b3864d260d5d2d0cc70cb314f5c46e0184` |
| native bundle | `f19e10be9987cfdb551a8dd75c5c88145c3cf35b73c54d3898fe562ce4182441` |
| native build manifest | `d0f05e6ea15f38e72950680d710d70229b94eee1dfbc8b3468f33431318b82e3` |
| clean G2 | job `5099135`, gate SHA-256 `bc67ed30791c46892aa1d787f50a99b25b9f97bdead69e95a5678ad7cacfe660` |
| clean model gate | job `5099195`, payload `46f0ad69d07dffdf277f25d321051b765befa7f034d0462f4bbdae1082b454cf` |
| clean manifest | `qualify-v21-safe-boundary-2n-20260727.json`, SHA-256 `aaaf19a80f85d6d783267fa430966f5ae589b42be693d2fa9df098b70e32ab10` |
| policy | `async-decoupled-v2.1-simple`, digest `fa9def95daf7bce25f1b962ca5437e7a76317b94ccfb9a710fbf126a344e7d98` |
| launcher | `70b96385b5ec0795d2d1c6b6495846b20e94fe53e5256e9c53c824b65c223fb7` |
| data | `91321b2b90bb159f3aa73881455778f10e8df588edd526b1066281fa72997962` |
| tokenizer | `94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069` |

The complete predecessor `FINAL-SHA256SUMS.txt` passed `sha256sum -c`.
The compact manifest parsed as `passed=true`, with clean job `5099195` and G2
job `5099135`. The exact snapshot was clean on local branch `main` with
`HEAD=origin/main=46c80437` in its sealed clone. The production native
attestation verifier accepted the manifest, bundle, exact source, and clean G2
gate:

```text
status=attested
backend=native-cxi
production=true
full_layout=true
source_commit=46c8043791e9b14c4cb3376c1fb03ebe7fe6932f
bundle_sha256=f19e10be9987cfdb551a8dd75c5c88145c3cf35b73c54d3898fe562ce4182441
```

This only proves that the predecessor files remain internally immutable. The
new dependency expressly says that this execution source may no longer
authorize a fault submission.

## Canonical activation and exact seed

The standalone snapshot has no private `.envs` directory. Its default
snapshot-relative activation therefore failed closed before Python ran. The
runner then used the exact approved environment override recorded by the
passing clean report and sourced the canonical activation:

```bash
export EMENDER_CONDA_ENV=\
/lustre/orion/bif148/scratch/erikgarrison/emender/\
.envs/olcf-rocm711-torch210-py312
source "$SNAPSHOT/scripts/frontier/activate_emender_frontier.sh"
```

The activated interpreter was Python `3.12.13`. All subsequent Python
preflight used `"$EMENDER_PYTHON"`.

The verified predecessor manifest still binds the exact cold seed:

```text
step:     2300930
tokens:   150793748480
bytes:    7719680116
SHA-256:  0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2
```

The prior clean pass contains submit-side authority agreement, job-scoped
`sbcast`, two node-local offline receipts, and `network_fetches=0`. Because
this fault task submitted no allocation, it makes no new seed-materialization
claim.

## Submission boundary and scheduler state

Immediately before the intended native-fault submission, the user queue was
empty. The stop message arrived before executing
`submit_native_dataplane_2n_gate.sh fault`.

Consequently:

- native fault G2 job IDs: none;
- async-v2.1 model fault/restart job IDs: none;
- fresh-allocation recovery job IDs: none;
- convergence job IDs: none;
- 4+ job IDs: none;
- payload digests submitted by this attempt: none;
- queue/running/terminal evidence for this attempt: not applicable; and
- unchanged payload resubmissions: zero.

There is no failing runtime phase to retain because scheduler mutation never
occurred. The retained failure is the fail-closed pre-submit authorization
boundary.

## Requirement disposition

Every live requirement remains blocked, not passed:

| Namespace | Disposition |
|---|---|
| R01–R16 | Authorities read; old clean chain verified; no corrected-source fault/restart run exists. |
| NDP01–NDP17 | Old exact native bundle/G2 attested; no corrected-source native fault prerequisite exists. |
| V21S01–V21S17 | Policy reviewed; no live fault injections, lag/rejoin/apply/restart timings, or V21S17 arrival evidence exist for the corrected source. |
| ISP01–ISP07 | Prior clean evidence is immutable but superseded as launch authority; no new fault-specific snapshot/apply evidence exists. |

In particular, this attempt does **not** claim delayed/missing contribution,
lag-2 acceptance, lag-3 drop/catch-up, duplicate/conflicting identity,
checksum/nonfinite/wrong-fence rejection, local OWNED timeout, trainer/native
service/manager loss, owner reassignment, failed-publication invisibility,
mailbox replacement, partial eight-trainer apply prevention, or
fresh-allocation recovery. It also does not claim bounded detection,
route-local containment, prompt buffer release, all-eight receipt closure, or
five additional K40 windows/three commits after recovery.

## Required resume point

Resume only after `requalify-v21-durable-collector-2n-clean` produces an
immutable `passed=true` manifest for the corrected execution source and the
graph releases this dependency. Then:

1. verify the new clean source/manifest, durable collector registration, native
   bundle, clean G2, policy, launcher, data, tokenizer, and exact seed;
2. use the corrected clean snapshot and rebuild/attest as required;
3. submit one new-digest native fault G2 on exact
   `Nodes=2, Partition=batch, QOS=debug`;
4. only after that prerequisite passes, submit the corrected controller's
   serialized fault/restart plan with the new clean manifest;
5. monitor and retain literal queued/running/terminal Partition and QOS
   evidence plus every required fault/recovery anchor; and
6. call `wg done qualify-simple-async-v21-2n-faults` only if the complete
   machine verdict is `passed=true`.

Until then, `qualify-simple-async-v21-2n-convergence`,
`authorize-simple-async-v21-scale`, and every scale rung remain blocked.
