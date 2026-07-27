# Async v2.1 exact-eight-node runner: fail-closed prerequisite record

Date: 2026-07-27

WG task: `scale-v21-8n-clean`

Status: **BLOCKED — no Slurm job was submitted.**

Machine record:
`docs/validation/scale-v21-8n-clean-20260727.json`

## Outcome

The exact-eight-node runner performed the prerequisite verification and the
canonical, activated, non-submitting scale preflight. The preflight rejected
the run before it could render or submit a payload:

```text
ValueError: signed scale authorization and immediate predecessor pass are required
```

That rejection is correct. The immutable two-node artifact is a literal
`passed=true` clean/performance result, but its own scope statement says that
it authorizes no fault, convergence, promotion, or scale execution. The
separate scale-chain quality pass was only a WG task-description review; it
explicitly ran no build, test, scheduler query, or Slurm job, did not open the
two-node artifacts, and did not issue the signed promotion or V21S17 closure
required by ADR-002.

Three additional independent fail-closed conditions were found:

1. The exact controller pins the normative predecessor map
   `4 <- 2`, `8 <- 4`, `16 <- 8`, `32 <- 16`, `64 <- 32`,
   `256 <- 64`. An eight-node submission therefore requires an immutable
   passed four-node manifest; none exists.
2. The exact controller has no held-job plus scheduler-owned
   `afterany:<jobid>` collector transaction, no pre-release durable collector
   identity, and no idempotent terminal `sacct`/verdict reconciliation. A
   payload could run after the WG/Codex worker died without an independent
   terminal owner, which this task explicitly forbids.
3. The current source digest covers every tracked file while submission
   requires a clean `main == origin/main`. Evidence-only commits after the
   qualifying execution changed `origin/main` and its tree. Without a reviewed
   executable-source identity boundary, using current `origin/main` changes
   the predecessor identity while using the old execution checkout violates
   the newly required fetched-remote equality.

The runner did not guess a closure, fabricate a signature, bypass the
four-node rung, rebuild a different identity, register a collector after
release, or call `sbatch`.

## Authorities read

The following authorities were read in full from the fetched authoritative
history. Their content SHA-256 values at the audit point were:

| Authority | SHA-256 |
|---|---|
| `docs/RESILIENT_DILOCO_COMPUTE_POOL.md` | `db5a487c9f0a6e398e669b8de320f1a73171b28fe165a230ac2c50505aa9f07b` |
| `docs/RESILIENT_DILOCO_GAP_MATRIX.md` | `a4bb7b0b5e993e0d0c6f1c89e1b0122d5fcd3ac71dea5dfe7803217a3f3ce935` |
| `docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md` | `7fdc6c890cab5d7e8785485cc18d6140c0887610bd2d6f3cd1b566991d20b2d3` |
| `docs/ASYNC_DECOUPLED_DILOCO_V2.md` (ADR-002) | `0a1c580c7ad30a28837042af8e1fef2a66c208a3ce535e76847263b8b3eb4052` |
| ISP01–ISP07 amendment | Normative rows in the gap matrix above |
| two-node report | `a26c45c59680ddef0d329e50fe18c75c1b711e56eb8faac38c18f77e9662deaa` |
| two-node compact verdict | `aaaf19a80f85d6d783267fa430966f5ae589b42be693d2fa9df098b70e32ab10` |
| scale-chain task-quality report | `d95bd9e64b2a06ba000dc61e13cb9d1cf98f8f063e6e38c55b0eeaab833a50eb` |

The compute-pool conformance checklist is applied below. This is a blocker
record, not an eight-node acceptance record.

## Immutable predecessor verification

The retained two-node root is:

```text
/lustre/orion/bif148/scratch/erikgarrison/emender-qualification/
  qualify-v21-safe-boundary-2n/
  46c8043791e9b14c4cb3376c1fb03ebe7fe6932f/
```

`sha256sum --check FINAL-SHA256SUMS.txt` returned `OK` for all 19 entries,
including source identity, build log, native manifest, G2 gate, controller
plan/state, both seed receipts, semantic verdict, queued/running/terminal
scheduler evidence, live monitor, checkpoint hashes, the 1,322-file retained
control manifest, and the independent final audit.

Verified predecessor facts:

| Field | Immutable value |
|---|---|
| execution commit | `46c8043791e9b14c4cb3376c1fb03ebe7fe6932f` |
| execution tree | `660f817c0419943d33d07f91573a6abc933bab7a` |
| source digest | `553bc996723bc1698a37f847c981b1b3864d260d5d2d0cc70cb314f5c46e0184` |
| native bundle | `f19e10be9987cfdb551a8dd75c5c88145c3cf35b73c54d3898fe562ce4182441` |
| native manifest | `d0f05e6ea15f38e72950680d710d70229b94eee1dfbc8b3468f33431318b82e3` |
| G2 gate | `bc67ed30791c46892aa1d787f50a99b25b9f97bdead69e95a5678ad7cacfe660` |
| policy | `async-decoupled-v2.1-simple` |
| policy schema | `emender-async-policy-v2.1` |
| policy digest | `fa9def95daf7bce25f1b962ca5437e7a76317b94ccfb9a710fbf126a344e7d98` |
| launcher | `70b96385b5ec0795d2d1c6b6495846b20e94fe53e5256e9c53c824b65c223fb7` |
| clean payload | `46f0ad69d07dffdf277f25d321051b765befa7f034d0462f4bbdae1082b454cf` |
| data | `91321b2b90bb159f3aa73881455778f10e8df588edd526b1066281fa72997962` |
| tokenizer | `94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069` |
| training arguments | `afc2a65fd8c73499e74e21cb9531c978206c3a9c898e42d18cc58bb93eb9fe9c` |
| semantic verdict | `8f48b370aa83a8e38b052d7e28871c9b3cb44a919150ded15c943578997dd947` |
| independent audit | `c7acbda4cf24dce9f257f367bec7a358c3257a59b6d8780eb0bb102c5162e1bd` |
| final manifest | `9e75284859f42c067eadb0446046a92c7b53ebedbcdf677bd90febe413eb8c07` |
| machine verdict | `passed=true`, clean-only |

The seed identity was also verified exactly:

```text
step=2300930
accepted_tokens=150793748480
bytes=7719680116
sha256=0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2
attestation=27e234891df02b64b9db77fc784c341e5a3ae6e87418b8f1af167776d1d710bb
two-node network_fetches=0
```

These are predecessor facts only. No eight-node seed materialization was
created because no eight-node allocation was authorized.

## Missing promotion and V21S17 closure

The full retained root was searched for promotion, authorization, closure,
scale, distribution, quantile, and deadline artifacts. No top-level immutable
artifact matching either required schema exists:

```text
emender-async-v21-scale-authorization-v1
emender-v21s17-scale-closure-v1
```

The two-node report says:

```text
This report creates no promotion authorization or 4+ node rung pass.
Scale-only leased-READY finite-close logic is unchanged and unexercised.
```

The task-quality report says:

```text
This was a graph/task-quality pass.
It did not build code, run tests, submit Slurm jobs, inspect live scheduler
state, or execute any scale runner.
```

The exact two-node correction also changed source identity. Exact-identity
fault/restart, deterministic-replay, and three-seed convergence passes were
not produced after that correction. ADR-002 requires all five two-node gates
before a separate promotion review can issue the first four-node
authorization. Consequently no reviewer could validly sign an eight-node
authorization from the available root.

No V21S17 arithmetic was guessed. In particular, this runner selected no
quantile, tail treatment, margin, discovery bound, leased-READY close,
preparation bound, boundary-rendezvous bound, apply bound, cadence bound,
enclosing deadline, or exact-token scale floor.

## Canonical preflight attempt

The runner sourced the required Frontier environment and used the activated
interpreter:

```bash
export EMENDER_CONDA_ENV=\
/lustre/orion/bif148/scratch/erikgarrison/emender/\
.envs/olcf-rocm711-torch210-py312
source "$EXACT_REPO/scripts/frontier/activate_emender_frontier.sh"
```

It then invoked the exact qualified controller without `--submit`:

```bash
"$EMENDER_PYTHON" \
  "$EXACT_REPO/scripts/frontier/run_async_v21_qualification.py" \
  --gate scale \
  --nodes 8 \
  --repo "$EXACT_REPO" \
  --state /tmp/scale-v21-8n-clean-preflight-state.json \
  --evidence-root "$EVIDENCE_ROOT" \
  --source-digest \
    553bc996723bc1698a37f847c981b1b3864d260d5d2d0cc70cb314f5c46e0184 \
  --policy-digest \
    fa9def95daf7bce25f1b962ca5437e7a76317b94ccfb9a710fbf126a344e7d98 \
  --bundle-digest \
    f19e10be9987cfdb551a8dd75c5c88145c3cf35b73c54d3898fe562ce4182441 \
  --launcher-digest \
    70b96385b5ec0795d2d1c6b6495846b20e94fe53e5256e9c53c824b65c223fb7 \
  --parameters-json \
    '{"close_on_q_min":false,"uses_launched_ranks":false,"wait_for_all_ready":false}' \
  --dry-run
```

Result: exit code `1`, before payload rendering, state mutation, or scheduler
submission:

```text
ValueError: signed scale authorization and immediate predecessor pass are required
```

The controller source independently proves:

```text
SCALE_RUNGS = (4, 8, 16, 32, 64, 256)
PREDECESSOR = {4: 2, 8: 4, 16: 8, 32: 16, 64: 32, 256: 64}
```

The approved rebuild, full-layout eight-node scale gate, collector
registration, and model submission were not run because the preconditions
that authorize them failed. Running those commands after this rejection would
have violated R16, NDP17, V21S16, V21S17, and the explicit task boundary.

## Durable terminal-owner implementation gap

Source inspection of the exact controller found:

- no `afterany` dependency;
- no held model submission or release transaction;
- no scheduler-owned terminal collector;
- no durable collector job/transaction identity;
- no terminal `sacct` reconciliation;
- no derived-exit/log/hash/validator-input/verdict collector; and
- the active payload record is written only after the model `sbatch` command
  returns.

The only scheduler reconciliation in `submit_plan` is a user-wide `squeue`
emptiness check. It neither reconciles historical `sacct` payloads nor closes
the worker-death window. Therefore the new terminal-ownership requirement
cannot be met by configuration or a manifest alone; it requires a tested
source correction followed by exact-identity requalification.

## Scheduler and retained-attempt reconciliation

The runner performed read-only reconciliation:

```bash
ls -1 "$QUAL_ROOT" | rg -i 'scale.*v21|v21.*scale'
squeue -u "$USER" -h -o '%i|%j|%T|%D|%P|%q|%V|%S' |
  rg -i 'async|v21|scale'
sacct -S 2026-07-25 -u "$USER" -X -n -P \
  --format=JobIDRaw,JobName,State,ExitCode,NNodes,Partition,QOS,Submit,Start,End,Elapsed |
  rg -i 'async|v21|scale'
```

No matching eight-node v2.1 scale evidence root, queued/running job, or
terminal scale job was found. Because the authorization was absent, no
eight-node payload digest could be validly rendered. There was therefore no
payload to retire, reconcile, or resubmit. No `sbatch`, `srun`, `salloc`,
`scontrol release`, or `scancel` command was issued by this runner.

## WG graph repair

The task had become ready directly after the clean two-node pass plus a task
quality review, bypassing the normative two-node fault/convergence/promotion
and four-node gates. The following graph corrections were recorded:

- `qualify-simple-async-v21-2n-faults` no longer depends on the failed
  historical clean task.
- A tested implementation prerequisite
  `fix-v21-durable-collector-source-identity` was added.
- A fresh exact-identity clean requalification prerequisite
  `requalify-v21-durable-collector-2n-clean` was added after that correction.
- Fault and promotion review now depend on the new clean requalification.
- This task now depends on `scale-simple-async-v21-4n` and the corrected clean
  requalification.
- The pre-existing eight-node task was ordered after this primary runner and
  was told to reconcile/adopt its future verdict rather than submit a
  duplicate payload.
- `scale-v21-32n-clean` was also ordered after the canonical 32-node chain,
  preventing a direct `8 -> 32` submission.

At record time, `wg why-blocked scale-v21-8n-clean` showed the actionable root
as `fix-v21-durable-collector-source-identity`, followed by clean
requalification, fault, convergence, promotion review, and the exact
four-node rung.

## Compute-pool conformance checklist

| Checklist obligation | Runner result |
|---|---|
| Cite the authorities and complete requirement sets | Complete in this record; R01–R16, NDP01–NDP17, V21S01–V21S17, and ISP01–ISP07 are enumerated below. |
| Peer-owned READY membership, bounded waits, no launched-rank invariant | Verified only in the two-node predecessor; no eight-node plan was authorized. The attempted parameters explicitly rejected launched ranks, all-READY wait, and Q-min early close. |
| No SQLite/database/lock/metadata-heartbeat compute closure | Verified only in the immutable two-node audit; no eight-node roles launched. |
| Fenced identities, deterministic math, idempotence, rejection, atomic evidence | Verified only for the two-node clean predecessor. Signed promotion, four-node predecessor, and eight-node artifacts are absent. |
| Bounded native hot path, release, no broker | Verified only for the two-node G2/clean predecessor. No eight-node native/model job ran. |
| Failure/deadline/recovery path and minimum floor | Exact-identity two-node fault/restart is missing; scale closure/floor arithmetic is consequently unauthorized. |
| Exact commands and immutable checkpoint artifacts; pass prior rungs | Preflight command and predecessor hashes are retained. Required fault/convergence/promotion/four-node rungs do not pass, so submission stopped. |
| V2.1 clocks and causal overlap/tail evidence | Verified only in the two-node clean predecessor. No eight-node phase events, maxima, p99, cadence, idle, or hard-tail evidence exists. |
| Durable terminal owner before payload release | Not implemented in the exact source. No payload was released. Corrective prerequisite created. |
| No duplicate payload | No eight-node payload was rendered or submitted. Existing duplicate WG runners were ordered to reconcile instead of resubmit. |

## Requirement-by-requirement eight-node status

Every row below maps the requirement to the exact reason that no immutable
eight-node artifact exists. The common command artifact is the canonical
preflight above; the common machine result is the companion JSON with
`passed=false`.

### R01–R16

| ID | Eight-node status |
|---|---|
| R01 | Blocked: no signed current-fence allocation claim may be created before promotion, four-node predecessor, and collector registration pass. |
| R02 | Blocked: no eight-node DISCOVER/BOOT/SYNC/READY identities or incarnations were launched. |
| R03 | Blocked safely: the attempted closure parameters reject launched ranks, but no reviewed leased-READY close exists. |
| R04 | Blocked: no eight-node contribution identities, replay receipts, or stale/conflict evidence exist. |
| R05 | Blocked: predecessor exact-token/native math is verified, but no authorized eight-node frozen set or reduction ran. |
| R06 | Blocked: V21S17 close, diversity predicate, exact-token floor, and derived deadlines are unsigned and therefore absent. |
| R07 | Blocked: no eight-node immutable commit/checkpoint/receipt chain exists. |
| R08 | Blocked: no eight-node owner map, bounded replay, receipts, or release evidence exists. |
| R09 | Blocked: no 64-trainer immutable snapshot or model-free manager/service ownership evidence exists. |
| R10 | Blocked: predecessor zero-Lustre/SQLite evidence is valid only at two nodes; no eight-node hot path ran. |
| R11 | Blocked: exact-identity two-node fault/rejoin and four-node predecessor passes are missing. |
| R12 | Blocked: no eight-node outer/token/fence/result/apply restart bundle exists. |
| R13 | Blocked: no eight-node backend execution or endpoint topology exists. |
| R14 | Blocked: no derived eight-node stage deadlines or causal phase/tail telemetry exists. |
| R15 | Blocked: no eight-node accepted-token or numerical artifact exists. |
| R16 | Passed as a fail-closed control only: missing fault/convergence/promotion/four-node evidence prevented scale submission. It is not an eight-node scientific pass. |

### NDP01–NDP17

| ID | Eight-node status |
|---|---|
| NDP01 | Blocked: no eight-node native peer-control session or claim/fence execution exists. |
| NDP02 | Blocked: predecessor proves zero collectives; no eight-node symbol/runtime evidence was produced. |
| NDP03 | Blocked: no eight persistent C++ services or exact-CXI endpoint records exist. |
| NDP04 | Blocked: no 64-trainer producer-direct coherent immutable handoff evidence exists. |
| NDP05 | Blocked: exact predecessor arithmetic is verified, but no eight-node result roots exist. |
| NDP06 | Blocked: no eight-node fenced frame/contribution/receipt identities exist. |
| NDP07 | Blocked: no eight-node leased endpoint exchange or current-fence AV routes exist. |
| NDP08 | Blocked: no eight-node registered-slot, snapshot/result pool, or resident-byte high-water evidence exists. |
| NDP09 | Blocked: no eight-node credit/application-receipt/foreground-progress evidence exists. |
| NDP10 | Blocked: no eight-node CRC/SHA/once-only receipt set exists. |
| NDP11 | Blocked: no eight-node replay/reassignment bounds or clean zero-replay evidence exists. |
| NDP12 | Blocked: no eight-node owner-direct redistribution or shared node aggregate exists. |
| NDP13 | Blocked: no empirically derived eight-node absolute deadlines or expiry outcomes exist. |
| NDP14 | Blocked: predecessor ABI is pinned, but no eight-node ABI/control session ran. |
| NDP15 | Blocked: no eight-node immutable publication, all-eight apply, or node receipt chain exists. |
| NDP16 | Blocked: no eight-node provider/byte/release/causal telemetry exists. |
| NDP17 | Passed as a fail-closed gate only: exact-source G2 is verified, but the required promotion and immediate four-node scale pass are absent. |

### V21S01–V21S17

| ID | Eight-node status |
|---|---|
| V21S01 | Blocked: predecessor policy/schema/digests are pinned, but no signed scale payload binds them to eight nodes. |
| V21S02 | Blocked: no eight-node commit/anchor/result/speculative lag events exist. |
| V21S03 | Blocked: no eight-node exact-token frozen set, clock, numerator, or denominator exists. |
| V21S04 | Blocked: no 64-trainer K40/eta-one execution exists. |
| V21S05 | Blocked: no signed eight-node contribution/closure identity or scale token floor exists. |
| V21S06 | Blocked: no 64-trainer exclusive mutable ownership/snapshot evidence exists. |
| V21S07 | Blocked: no eight-node ScheduleFree x/z/interval atomic correction receipts exist. |
| V21S08 | Blocked: no eight-node verified-latest mailbox evidence exists. |
| V21S09 | Blocked: no eight-node resident/slot/credit/replay/mailbox formula observation exists. |
| V21S10 | Blocked: no eight-node leased-READY/rejoin behavior exists. |
| V21S11 | Blocked: no eight-node preparation/boundary/release/apply/node-marker transactions exist. |
| V21S12 | Blocked: no eight-service CXI/memfd point-to-point scale artifact exists. |
| V21S13 | Blocked: no eight-node causal phases, cadence, idle, maximum, p99, or hard-tail result exists. |
| V21S14 | Blocked: no eight-node checkpoint/restart bundle or eight offline seed receipts exist. |
| V21S15 | Blocked: the exact corrected identity has clean-only evidence; fault/replay/convergence remain missing. |
| V21S16 | Passed as a fail-closed control only: absent signed promotion and four-node pass were rejected. |
| V21S17 | Passed as a fail-closed control only: absent empirical closure was rejected; no constant, launched-rank close, all-READY barrier, or Q-min early close was substituted. |

### ISP01–ISP07

| ID | Eight-node status |
|---|---|
| ISP01 | Blocked: no 64-trainer coherent immutable snapshot evidence exists. |
| ISP02 | Blocked: no eight-node snapshot/admission events or next-K overlap exists. |
| ISP03 | Blocked: no eight-node immutable-only publish/hash/aggregate/checkpoint trace exists. |
| ISP04 | Blocked: no eight-node capacity-edge/high-water/nonblocking foreground evidence exists. |
| ISP05 | Blocked: no eight-node all-eight-per-node bounded atomic apply evidence exists. |
| ISP06 | Blocked: no eight-node causal freeze/admission/network/aggregation/checkpoint/result/apply/idle telemetry exists. |
| ISP07 | Blocked: no eight-node every-event tail distribution or approximately-200-second stall gate exists. |

## Validation

- [x] Required authorities and the compute-pool checklist were read.
- [x] All 19 predecessor manifest entries passed `sha256sum --check`.
- [x] The immutable two-node machine verdict parses with `passed=true`.
- [x] The two-node report and task-quality report were checked for scope.
- [x] The retained root was searched for signed promotion and V21S17 closure
      artifacts; none exists.
- [x] Canonical Frontier activation was sourced and `"$EMENDER_PYTHON"` was
      used for the non-submitting exact-eight-node preflight.
- [x] The exact controller rejected missing authorization/predecessor before
      payload rendering or scheduler mutation.
- [x] The controller’s `8 <- 4` predecessor requirement was verified from
      exact source.
- [x] The missing durable terminal-collector implementation was verified from
      exact source.
- [x] Read-only retained-attempt, `squeue`, and `sacct` reconciliation found no
      matching eight-node payload/job.
- [x] No Slurm submission, allocation, release, cancellation, fault job,
      convergence job, 32-node job, or 128-node job was performed.
- [x] The WG graph was repaired to add tested implementation, fresh clean
      requalification, fault, convergence, promotion, and four-node
      prerequisites while preventing duplicate runner submissions.
- [x] R01–R16, NDP01–NDP17, V21S01–V21S17, and ISP01–ISP07 are explicitly
      mapped to the blocked eight-node artifact surface.
- [ ] No eight-node `passed=true` verdict exists; `wg done` is forbidden.

## Final disposition

This rung is not complete. The evidence is retained, but there is no
eight-node job ID, payload, collector, scheduler sample, commit/checkpoint,
metric set, or literal `passed=true` verdict. The correct WG disposition is
`incomplete`, leaving the 32-node and 128-node tasks blocked.
