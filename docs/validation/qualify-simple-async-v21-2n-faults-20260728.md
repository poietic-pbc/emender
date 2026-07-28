# Async v2.1 exact-two-node fault/restart qualification: retained failure

Date: 2026-07-28

WG task: `qualify-simple-async-v21-2n-faults`

Status: **INCOMPLETE — fault-rejoin failed; fresh recovery, convergence, and
scale remain blocked.**

The machine-readable verdict is
`docs/validation/qualify-simple-async-v21-2n-faults-20260728.json`. Its
top-level `full_pass` and `passed` fields are both `false`. This report does
not call `wg done`.

## Outcome

The exact-source clean prerequisites, native fault G2, and two-generation
fault baseline passed. The next serialized phase failed closed:

- model job `5105811` ended `FAILED 1:0` after `00:24:43`;
- terminal scheduling is exactly `NNodes=2`, `Partition=batch`, `QOS=debug`;
- durable `afterany` collector `5105812` independently completed `0:0`;
- its immutable terminal verdict is `passed=false`, SHA-256
  `3496142919fd3aea359f4e14051a57bfda6d54bd57d4599964b01cdce7049721`;
- the phase completed generations 3 and 4, only two of four required
  transitions to generation 6; and
- the controller retained the payload as `verdict=failed`, cleared its active
  job, and did not submit fresh-allocation recovery.

The primary unexpected error is in retained `node-1-manager.err` (SHA-256
`729226bcfa5a8b3197b9c13d6dccff98158ce27825a5370f5743ca63e3461c0a`):

```text
RuntimeError: generation is not open
```

This was not an injection. Node 1 attempted its generation-3 contribution
while the peer was catching up from the planned trainer loss; native control
had already closed that generation. The RPC error escaped the manager and the
supervisor atomically restarted the entire node-1 cohort. The later planned
node-1 manager-loss injection correctly fired only after generation 4
`published_node_applied`, but the extra restart had consumed the recovery
budget. The supervisor then emitted `restart_exhausted` at
`2026-07-28T11:05:39.275391800-04:00`, before it could reconstruct that
planned loss.

This is a real qualification failure, not a two-hour timeout. The model job
terminated immediately on the first terminal failure, 95 minutes before its
`02:00:00` safety ceiling.

## Authorities and exact execution identity

Before builds, tests, and submission, the runner used the canonical Frontier
activation and reviewed:

- `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, including its conformance
  checklist and R01–R16;
- `docs/RESILIENT_DILOCO_GAP_MATRIX.md`, including R01–R16, NDP01–NDP17,
  V21S01–V21S17, and ISP01–ISP07;
- `docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md`, including the G0–G6 order,
  fault semantics, bounds, and NDP01–NDP17; and
- accepted ADR-002 in `docs/ASYNC_DECOUPLED_DILOCO_V2.md`.

All Python/test/build/preflight work used the activated Python 3.12.13
environment. The source regression suite passed `195` with `1` expected
skip, the exact official preflight passed `139` with `1` expected skip, and
the native build passed CTest `10/10`.

| Identity | Exact value |
|---|---|
| source commit | `a71939c8e3a518f1cb5084941b82a27972443ba9` |
| execution-source digest | `25d31aea87518aa7ce91376ec30a2f34550c39a64650a5f0852a71ff4c35a948` |
| native manifest | `187266f6b8f9ed8b39d3d481166f692f5921df8a6afb506244dd540d77a726d6` |
| native bundle | `f19e10be9987cfdb551a8dd75c5c88145c3cf35b73c54d3898fe562ce4182441` |
| policy | `async-decoupled-v2.1-simple` |
| policy digest | `fa9def95daf7bce25f1b962ca5437e7a76317b94ccfb9a710fbf126a344e7d98` |
| launcher | `278fae0985efd302542424e6feb1d584812a0cffe5c03cb282861a90b355dad1` |
| train args | `afc2a65fd8c73499e74e21cb9531c978206c3a9c898e42d18cc58bb93eb9fe9c` |
| data | `91321b2b90bb159f3aa73881455778f10e8df588edd526b1066281fa72997962` |
| tokenizer | `94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069` |
| campaign | `0accc5427f12bb77a6771af1a6f02457e65ef0ce95063368a8e293e81e591604` |

The clean snapshot was fetched, clean, and equal to pushed `origin/main` when
the payloads were created. Subsequent movement of `origin/main` cannot change
the immutable snapshot or any submitted digest.

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
Job `5105811` used job-scoped `sbcast` to
`/tmp/emender-e97-seed-5105811/checkpoint-step-2300930.pt`.
`frontier01059` and `frontier01060` independently retained offline receipts
with exact bytes/SHA and `network_fetches=0` (receipt hashes
`fbaf7b1c...f0f54` and `3d41bb72...e39a`).

## Phase plan and terminal scheduler evidence

The phase table was recorded before the first fault model submission. The
two-hour Slurm request was only a safety ceiling; each payload rendered the
exact required generation count and exited at evidence completion or first
failure.

| Phase | Exact injections | Required pre/post evidence | Immediate success condition | Expected | Actual |
|---|---|---|---|---|---|
| fault-baseline | none | g0→g2; 2 commits; both node applies | g2 checkpoint and 16 final trainer receipts | 15–25 min | pass, 17:40 |
| fault-rejoin | node-1 READY +45 s at g2; node-0 trainer 3 after g3 apply; node-0 native service at g4 `owner_transport`; node-1 manager after g4 `published_node_applied` | g2→g6; 4 commits; every loss contained/rejoined | g6 checkpoint and 16 final trainer receipts | 40–60 min | **fail**, 24:43 |
| fresh-allocation-recovery | none | newer fence, g6→g11; 5 K40 windows; ≥3 commits | exact model/outer/token reload plus both g11 node applies | 40–50 min | not submitted |

Raw terminal accounting was queried with:

```text
sacct --format=JobIDRaw,State,ExitCode,NNodes,Partition,QOS,Start,End,Elapsed -P
```

| Job | Purpose | State/exit | Nodes | Partition | QOS | Start | End | Elapsed |
|---:|---|---|---:|---|---|---|---|---|
| 5105172 | native clean G2 | COMPLETED/0:0 | 2 | batch | debug | 07:55:41 | 07:58:40 | 00:02:59 |
| 5105214 | clean model | COMPLETED/0:0 | 2 | batch | debug | 08:01:16 | 09:03:28 | 01:02:12 |
| 5105215 | durable collector | COMPLETED/0:0 | 1 | batch | normal | 09:04:11 | 09:04:16 | 00:00:05 |
| 5105527 | native fault G2 | COMPLETED/0:0 | 2 | batch | debug | 09:07:15 | 09:09:15 | 00:02:00 |
| 5105554 | fault baseline | COMPLETED/0:0 | 2 | batch | debug | 09:34:29 | 09:52:09 | 00:17:40 |
| 5105555 | durable collector | COMPLETED/0:0 | 1 | batch | normal | 09:54:27 | 09:54:32 | 00:00:05 |
| 5105811 | fault rejoin | **FAILED/1:0** | 2 | batch | debug | 10:41:18 | 11:06:01 | 00:24:43 |
| 5105812 | durable collector | COMPLETED/0:0 | 1 | batch | normal | 11:06:46 | 11:06:50 | 00:00:04 |

The terminal `scontrol -dd` record for `5105811` retains
`Account=bif148`, `QOS=debug`, `Partition=batch`, `NumNodes=2`,
`NodeList=frontier[01059-01060]`, `TimeLimit=02:00:00`, the complete
submission line, payload digest, policy, source, bundle, G2, launcher, data,
tokenizer, seed, deadline, injection, and prior-gate identities. Collector
`5105812` retains `afterany:5105811`, `Account=bif148`,
`Partition=batch`, `QOS=normal`, and the independent terminal-collector
command.

At most one model/debug payload was active. The baseline collector completed
before fault-rejoin was submitted. No payload digest was resubmitted, and the
queue was empty after `5105812`.

## Passing prerequisites and baseline

Native clean G2 `5105172` passed with gate SHA-256
`e0a00464074ffd21f34a848ca86296af9b27be779bf0c3833b251e600bb9dffb`.
Mandatory clean model `5105214` plus collector `5105215` passed with terminal
verdict SHA-256
`28b4a390e15e9a33f5c815ffe704a5b915148241e4f13233b5ab7df55284bb6a`.
It exited after exactly 10 commits/checkpoints, 20 node applies, and 160
trainer receipts.

Native fault G2 `5105527` passed with gate SHA-256
`49e972b3c9c86a7386c3f2ef3b55eada00856b9bd2330a092d454156fbb12f72`.
It proved exact `cxi`/`FI_EP_RDM`, one owner reassignment, new incarnation,
old-epoch rejection, no partial commit, bounded replay/release, and zero MPI
collectives, all-rank barriers, Python dense socket bytes, full-copy handoff
bytes, disk replay, or trainer spool bytes.

Fault baseline `5105554` passed with payload
`7431ce96acb237ea729790f693621d9793a5325fbe6e890f97b632858e5d2ff3`
and terminal verdict SHA-256
`69542c0a5929ea812cb2172867d14867ac39293ca01d70c6b43cd3d4aa3e613d`.
It completed g0→g2, two commits/checkpoints, four node applies, and 32
trainer receipts with `one_node_commit_authority=false`.

Its timestamped semantic probes passed:

- lag 0, 1, and 2 admission;
- lag-3 drop/catch-up without foreground pause;
- identical duplicate idempotence and conflicting identity rejection;
- checksum, nonfinite, and wrong-fence rejection without accumulator mutation;
- failed-publication invisibility;
- capacity-one mailbox replacement; and
- local OWNED timeout, prompt release, bounded high water, and no foreground
  pause.

The baseline g2 checkpoint is
`4f27ba2837af7e963d8e17e4be8e376b6001b07841cfee08692e9b59c2422683`,
outer step 2, accepted tokens `150804239360`.

## Fault-rejoin timeline and safety invariants

The retained supervision stream has SHA-256
`ad7f00da7598a8058f2d2da55326330dd3998974070b1db197628d783608a321`.

1. Node 1 completed the prescribed 45-second delayed READY for generation 2.
   The immutable delay receipt is SHA-256
   `4965c17ef872c5fb55472dcd4d05c1f94a6af8422b60fdab6e8926a18da45fa4`.
2. The generation-2 READY/commit record contains both original incarnations,
   accepted tokens `5,245,440`, required contributions `2`, and
   `reason=accepted_floor_met`. Nodes froze at 10:52:08/09, entered owner
   transport at 10:52:28/29, and checkpointed at 10:54:37/38—inside the
   420-second absolute deadline.
3. After generation 3 committed and both eight-trainer node applies were
   immutable, the planned node-0 trainer-3 injection fired at
   10:56:26.305979700. The supervisor failed the complete node-0 cohort at
   10:56:43.326269900 and reconstructed it with a new node incarnation plus
   eight new trainer incarnations at 10:57:01.498537500 (18.172 s).
4. While node 0 was catching up, the uninjected node-1 manager encountered
   `generation is not open`. The supervisor contained that failure to node 1,
   failed its whole cohort at 10:58:30.562201000, and reconstructed a new
   node incarnation plus eight trainers at 10:58:48.736864000 (18.175 s).
5. No new commit occurred while only one cohort was usable. Generation 4
   committed only after the next READY/commit record contained both new
   incarnations, again with exactly two contributions and tokens `5,245,440`.
6. The planned manager injection fired at
   11:05:04.205066400, immediately after node 1’s generation-4
   `published_node_applied` marker. The cohort failed at
   11:05:21.226303600, but reconstruction exhausted the restart budget at
   11:05:39.275391800. The planned node-0 native-service injection had not
   yet reached its generation-4 `owner_transport` trigger.

Two current-fence commit receipts exist:

| Generation | Receipt | Checkpoint | Outer step | Accepted tokens | Applies/receipts |
|---:|---|---|---:|---:|---|
| 3 | `bdd0303f...f9802` | `3f4232b2...f32fd` | 3 | 150809484800 | 2 nodes / 16 trainers |
| 4 | `47d6ac63...c995a` | `5b2266b1...a746c` | 4 | 150814730240 | 2 nodes / 16 trainers |

Each node-apply authority has exactly eight trainer receipts. Both
generation commit-ready records have two current-fence READY identities.
Therefore, within the retained failed run:

- no one-node commit authority was observed;
- no partial eight-trainer node apply was published;
- no stale-incarnation commit was observed;
- no double x/z correction is claimed or indicated;
- no all-rank collective/abort primitive was invoked; and
- bounded local work stopped at the fail-closed restart boundary.

These safety observations do not turn the failed phase into a pass.

## Conformance mapping

The detailed one-ID-per-entry mapping is in the machine-readable report.
This table gives the required complete crosswalk and the controlling
disposition.

| Requirements | Evidence and disposition |
|---|---|
| R01–R03 | Exact Slurm fences and two-member leased READY snapshots retained. No one-node authority. Fresh newer-fence recovery was not reached. |
| R04–R06 | Baseline lag/identity/corruption probes and exact Q/T/deadline records passed. Live generation-close recovery failed. |
| R07–R10 | Immutable g1–g4 commit/apply chains plus bounded native memfd/CXI, replay/release, and zero Python/Lustre dense path retained. |
| R11 | **Failed:** planned node-0 rejoin succeeded, but an uninjected node-1 generation-close race caused an extra restart. |
| R12–R15 | Model/outer/token state is immutable through g4; exact-token math and telemetry exist. Fresh-allocation restoration and a terminal semantic verdict are absent. |
| R16 | Safety boundary passed: no convergence or 4+ job was submitted and downstream remains blocked. |
| NDP01–NDP07 | Exact fenced native peer control, point-to-point cxi, persistent service, memfd, deterministic reference, wire identities, and leased routes retained. Recovery campaign failed. |
| NDP08–NDP12 | Bound/credit/replay/reassignment/corruption and direct redistribution evidence passed prerequisites/baseline; g3/g4 node applies are atomic eight-receipt records. |
| NDP13 | **Failed:** route-local containment worked, but the generation-closed RPC escaped as a fatal manager failure and exhausted restart. |
| NDP14–NDP16 | Versioned C ABI and immutable checkpoint/apply evidence retained; no passing fault semantic/tail verdict exists. |
| NDP17 | Clean and fault exact-source G2 passed before model work; no larger rung launched. |
| V21S01 | Exact policy/schema/source/bundle/launcher/payload identities passed admission. |
| V21S02 | **Explicit anchor:** lag 0–2 and lag-3 baseline probes passed; live catch-up failed on `generation is not open`. |
| V21S03–V21S04 | Exact tokens, K40, eta-one outer step/token state passed through g4. |
| V21S05 | **Explicit anchor:** full identities, `Q_min=2`, `T_min=3,934,080`, deadlines, and one contribution per worker retained; restart failed. |
| V21S06 | Baseline OWNED timeout/release was bounded and nonblocking; full campaign incomplete. |
| V21S07 | **Explicit anchor:** g3/g4 atomic all-eight applies retained; complete fault phase absent. |
| V21S08 | **Explicit anchor:** mailbox replacement and invalid/failed/fence rejection probes passed. |
| V21S09 | **Explicit anchor:** resident/credit/replay/mailbox/deadline bounds passed prerequisite/baseline evidence. |
| V21S10 | **Explicit anchor, failed:** Q floor prevented one-node commits and first rejoin worked, but an uninjected peer restart occurred. |
| V21S11 | **Explicit anchor, failed:** atomic cohorts reconstructed twice; planned manager loss ended in `restart_exhausted`. |
| V21S12 | **Explicit anchor:** compiled cxi/memfd path; zero MPI, Python dense, Lustre dense, or central broker. Native-service fault itself was not reached. |
| V21S13 | **Explicit anchor:** causal READY/freeze/transport/commit/apply/failure timestamps retained; terminal semantic/tail verdict missing. |
| V21S14 | **Explicit anchor, incomplete:** exact seed and g3/g4 model/outer/token checkpoints retained; newer-fence fresh allocation not submitted. |
| V21S15 | Fault/restart gate failed; no five-gate promotion claim. |
| V21S16 | Promotion and scale remain blocked. |
| V21S17 | Delayed arrival, both READY snapshots, freeze, owner transport, checkpoint, and 420-second deadline inputs are retained. Because the fault artifact failed, they cannot derive or authorize a scale close. |

## Immutable evidence roots and next action

The authoritative external evidence root is:

```text
/lustre/orion/bif148/scratch/erikgarrison/emender-qualification/
qualify-simple-async-v21-2n-faults/
a71939c8e3a518f1cb5084941b82a27972443ba9
```

Important immutable hashes:

| Artifact | SHA-256 |
|---|---|
| controller state | `fdfd1f2745c6ce90a86ac7858047844e724aa9f86e73e1730a50317e265b1871` |
| final rendered fault manifest | `7626fc191a639a3d856a955c01a5617d7cd9f37bfa08d93fad379ae96d539090` |
| fault-rejoin terminal verdict | `3496142919fd3aea359f4e14051a57bfda6d54bd57d4599964b01cdce7049721` |
| model stdout | `cdf2d43874d449bc08313dc82da49ff394bbf763a2d7fe1796cc03207c9e665e` |
| model stderr | `a646298b10637e7c2a9d2b1b01a8f135f3705c4ed2fe5f2c6f96320e5297e2d0` |
| node-1 manager error log | `729226bcfa5a8b3197b9c13d6dccff98158ce27825a5370f5743ca63e3461c0a` |
| supervision events | `ad7f00da7598a8058f2d2da55326330dd3998974070b1db197628d783608a321` |

The required correction is to treat the generation-closed contribution race
as a fenced catch-up/reload outcome rather than an unplanned fatal manager
exit, with a regression that reproduces concurrent peer reconstruction. Any
retry requires a changed source/payload digest, full test/build/push,
refreshed exact-source clean and fault G2, a re-passed clean two-node gate,
and then a new serialized campaign. The failed digest
`4ae37668...97cbf` must never be resubmitted.
