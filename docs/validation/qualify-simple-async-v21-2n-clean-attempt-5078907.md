# Async v2.1 exact-two-node clean qualification: non-passing attempt 5078907

## Verdict

`passed=false`. Frontier job `5078907` was a concrete, exact-two-node clean attempt, but it did not pass the strict clean/performance gate. It is not a clean qualification artifact, does not authorize the fault gate or scale, and must not be represented as success.

The exact terminal accounting record is:

```text
5078907|FAILED|1:0|2|batch|debug|2026-07-26T02:45:52|2026-07-26T04:01:11|01:15:19
```

The payload digest
`47e005065166865b66bd6945d4a55672897291f9c8a45d3c0e4b5b0e60593a09`
is retired. No unchanged-payload retry and no replacement submission were
performed.

The machine-readable companion is
[`reports/frontier/qualify-simple-async-v21-2n-clean-attempt-5078907.json`](../../reports/frontier/qualify-simple-async-v21-2n-clean-attempt-5078907.json).

## Exact terminal cause and preliminary performance metrics

The job's first terminal exception was:

```text
ValueError: missing exact async-v2.1 policy declaration
```

That was an evidence-reader defect, not an absent runtime declaration. Both
nodes retained exact `async_v21_policy` declarations in standalone
`retained-evidence/node-{0,1}/control/production-pipeline-*.json` files, while
the strict reader scanned only JSONL timing files.

After a regression test and the narrow reader correction, a postmortem
validation of the unchanged retained job evidence reached the actual
performance gate and failed correctly:

```text
ValueError: training-lane foreground idle 0.575375 is not below 0.10
```

The preliminary interleave/performance measurements are:

| Measurement | Observed | Gate |
|---|---:|---:|
| Real trainers | 16 | exactly 16 |
| Windows retained per trainer | at least 12 | 2 warm-up + 10 measured |
| Atomic commits | 12 | at least 10 |
| Median raw K40 | 67.4206138329464 s | reference |
| Median steady K-boundary cadence | 68.23619746020995 s | at most 1.25 × raw K40 |
| Cadence/raw multiple | 1.01209694751941 | at most 1.25 |
| Aggregate foreground idle | 0.5753750134096515 | less than 0.10 |
| Maximum inter-window idle | 212.31781229004264 s | diagnostic |

Every trainer showed the same alternating pattern: an approximately
0.001–0.003-second gap followed by an approximately 199–212-second gap. The
postmortem selected measured windows 13–22 for all 16 trainers. No
unhealthy/backpressured interval was omitted to improve the result.

The runtime produced 12 immutable checkpoints, 12 handoff generation manifests,
12 immutable commit receipts, and 24 node apply receipts. There was no manager
restart, progress-deadline failure, or runtime exception before strict
validation. Those facts are useful diagnostic evidence, but they do not
override the failed foreground-idle gate.

## Scheduler, payload, and immutable identity

The exact submission argv is the `/command` array in:

```text
/lustre/orion/bif148/scratch/erikgarrison/emender-qualification/qualify-simple-async-v21-2n-clean/5cb2e73c3528787c99c3805b10b6c6ca436679e2/clean-qualification-plan.json
SHA-256 82619e343840857aa28057ed3b07ee2d61313aed5fe4f9c88d29504edaa3e072
```

That array binds `--nodes=2`, `--partition=batch`, `--qos=debug`,
`--time=02:00:00`, the controller deadline
`RESILIENT_E97_PROGRESS_DEADLINE_S=2700`, source commit
`5cb2e73c3528787c99c3805b10b6c6ca436679e2`, the clean gate, native
bundle, G2, policy/schema, launcher, payload, data, tokenizer, training
arguments, and seed identities. It is the authoritative invoked argv; this
report does not reconstruct or abbreviate it.

Queued/running evidence used:

```bash
squeue -h -j 5078907 -o '%i|%T|%P|%q'
scontrol show job -dd 5078907
```

Terminal evidence used:

```bash
sacct -n -X -j 5078907 --format=JobIDRaw,State,ExitCode,NNodes,Partition,QOS,Start,End,Elapsed -P
```

Twenty-four scheduler transcripts retain the monitored states. The terminal
transcript is:

```text
clean/clean-overlap/scheduler-evidence/monitor-20260726T040435-0400.txt
SHA-256 ff9549fd2e0e94d02d35d81f9155019969ec2748478ee7d8e823d40aee7e9ef9
```

The exact-source G2 prerequisite was job `5078760`. It passed with:

```text
5078760|COMPLETED|0:0|2|batch|debug|2026-07-26T02:03:23|2026-07-26T02:06:16|00:02:53
```

Its full-layout gate SHA-256 is
`93bc8854a0a161285b3cdde21f40d3b1936b6a7efcf10d37a39f43b40ecdbd6a`;
the bound build-manifest SHA-256 is
`77b69837a22f34cbe3071390e7926ba3b52d42c6af942b5b04184cadbc73bc6b`;
and the native bundle is
`f19e10be9987cfdb551a8dd75c5c88145c3cf35b73c54d3898fe562ce4182441`.
G2's median duration was `23.084823993` seconds and its retained speedup was
`4.2868616455`.

The clean payload binds:

- policy `async-decoupled-v2.1-simple`, schema
  `emender-async-policy-v2.1`, digest
  `fa9def95daf7bce25f1b962ca5437e7a76317b94ccfb9a710fbf126a344e7d98`;
- launcher digest
  `70b96385b5ec0795d2d1c6b6495846b20e94fe53e5256e9c53c824b65c223fb7`;
- source-tree digest
  `3c24462d7fed5678571b995172d291d1b6c36000c4104bec4ac2f82aec4357db`;
- data identity
  `91321b2b90bb159f3aa73881455778f10e8df588edd526b1066281fa72997962`;
- tokenizer SHA-256
  `94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069`;
  and
- training-arguments SHA-256
  `afc2a65fd8c73499e74e21cb9531c978206c3a9c898e42d18cc58bb93eb9fe9c`.

The seed is exactly immutable step `2300930`, accepted tokens
`150793748480`, byte size `7719680116`, and SHA-256
`0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2`.
The submit-side attestation SHA-256 is
`27e234891df02b64b9db77fc784c341e5a3ae6e87418b8f1af167776d1d710bb`.
Both nodes staged and reverified it offline at
`/tmp/emender-e97-seed-5078907/checkpoint-step-2300930.pt`; retained
`network_fetches=0`.

## Corrective change and validation

Commit `9dee67f6605ad68e591082f9009c32471482c6db`, pushed to `origin/main`,
contains two narrow changes:

1. The validator reads standalone JSON only when it is the exact
   `async_v21_policy` record. Unrelated control JSON cannot masquerade as
   timing telemetry.
2. The persistent trainer publishes its already-frozen immutable endpoint via
   `publish_state_delta` rather than rereading the live GPU model via
   `publish_model_delta` before `OWNED`. It also records an honest endpoint
   snapshot interval and the actual native direct-memfd duration.

The policy regression was written failing-first and failed with the exact
missing-policy exception before the reader fix. The endpoint regression was
also written failing-first and failed because the production path did not use
`publish_state_delta`.

All Python/test/build families used:

```bash
source scripts/frontier/activate_emender_frontier.sh
```

and then `"$EMENDER_PYTHON"` or `PYTHON_BIN="$EMENDER_PYTHON"`.

Validation retained:

- validator suite: 12/12 passed;
- focused corrected runtime/validator tests: 15/15 passed;
- stale path/fake-plane regressions: 3/3 passed;
- broad selected suite: 154 passed and three path-only failures because the
  WG worktree made the AF_UNIX socket path exceed `sun_path`;
- the exact three path-only cases rerun with a short `/tmp` `--basetemp`: 3/3
  passed; and
- canonical native build CTest: 10/10 passed.

Thus the selected logical suite is 157/157 across the transparent short-path
rerun; it is not misreported as one 157-test invocation.

After the push, the canonical current-source native rebuild bound clean commit
`9dee67f6605ad68e591082f9009c32471482c6db`, retained
`source_tree_dirty=false`, passed 10/10 native CTests, and wrote
`build/native-resilient-dataplane/native-artifacts.json` with SHA-256
`869d8592815abea58fedbd769cb1321ce0c8123851229a417ee46d797398a5d6`.
The native bundle remains
`f19e10be9987cfdb551a8dd75c5c88145c3cf35b73c54d3898fe562ce4182441`.

This corrective change has not been live requalified. Therefore it is a tested
candidate fix, not evidence that the clean performance gate now passes.

## Authority conformance and requirement mapping

This review used the normative
[`RESILIENT_DILOCO_COMPUTE_POOL.md`](../RESILIENT_DILOCO_COMPUTE_POOL.md)
conformance checklist, the
[`RESILIENT_DILOCO_GAP_MATRIX.md`](../RESILIENT_DILOCO_GAP_MATRIX.md), the
[`NATIVE_RESILIENT_DILOCO_DATAPLANE.md`](../NATIVE_RESILIENT_DILOCO_DATAPLANE.md),
and the accepted
[`ASYNC_DECOUPLED_DILOCO_V2.md`](../ASYNC_DECOUPLED_DILOCO_V2.md).

The mapping below distinguishes observed evidence from a passing claim. The
final result remains non-passing wherever one strict requirement is unresolved.

### Compute-pool requirements R01–R16

| ID | Attempt evidence and status |
|---|---|
| R01 | Scheduler-fenced job, exact source/payload identities, and no shared control database were retained; observed, not independently sufficient for pass. |
| R02 | Stable workers/incarnations and READY control telemetry were retained; no unintended restart occurred. |
| R03 | The live two-node READY cohort ran, but the observed lane interleave did not establish the required clean performance closure. |
| R04 | Fenced generation/contribution identities and immutable receipts advanced through generation 12. |
| R05 | Exact-token/native reduction was bound by the passed exact-source G2 and clean payload. |
| R06 | `Q_min=2`, `T_min=3,934,080`, exact two-node admission, and absolute deadlines were bound in the exact argv. |
| R07 | Twelve immutable commit/checkpoint chains and 24 node-apply receipts were retained. |
| R08 | Native bounded transport was bound, but all runtime high-water claims remain subordinate to the failed strict gate. |
| R09 | One model-free manager/service and eight trainer-owned models per node were observed without central model brokering. |
| R10 | Node-local native transport and `network_fetches=0` were retained; no clean pass is inferred from those facts. |
| R11 | No disappearance/rejoin occurred in this clean job; fault behavior is outside this task and unclaimed. |
| R12 | Immutable outer/token/result/fence state advanced through 12 checkpoints; fresh-process correctness cannot promote a failed clean run. |
| R13 | The Frontier adapter and exact `cxi` native bundle were bound; no hyperscale or scale claim is made. |
| R14 | Honest stage/K/idle evidence exposed the 0.575375 idle failure. Requirement not passed for this attempt. |
| R15 | Exact accounting artifacts exist, but training-lane performance/correctness qualification is not passed. |
| R16 | G2 preceded the real job, but the real two-node clean gate failed; 4+ remains blocked. |

### Native data-plane requirements NDP01–NDP17

| ID | Attempt evidence and status |
|---|---|
| NDP01 | Native peer control and the no-shared-database path were bound. |
| NDP02 | No MPI/all-rank wait was introduced or observed. |
| NDP03 | One persistent C++17 service per node and exact `cxi` were bound. |
| NDP04 | Producer-direct native buffer paths and zero Python/Lustre dense-byte policy were bound; strict performance still failed. |
| NDP05 | Bitwise native reference arithmetic was covered by the passed G2 and 10/10 CTests. |
| NDP06 | Fenced command/frame/contribution/receipt identities were bound to the payload. |
| NDP07 | Leased endpoint exchange and current-fence routes were retained. |
| NDP08 | Fixed preflight byte/slot bounds were bound; observed clean acceptance is unclaimed. |
| NDP09 | Capacity/credit bounds were configured; no passing high-water verdict can be issued from a failed run. |
| NDP10 | Checksums, roots, and apply receipts were retained without reported corrupt ingress. |
| NDP11 | No replay/reassignment was requested in this clean attempt; fault behavior is unclaimed. |
| NDP12 | Service-owned result redistribution and all-eight-trainer apply receipts were retained. |
| NDP13 | Absolute deadlines were retained, but the clean performance-stage result failed. |
| NDP14 | Versioned C ABI and descriptor-only local control were bound. |
| NDP15 | Twelve immutable checkpoint/commit handoffs and 24 node apply receipts were retained; the overall clean gate remains failed. |
| NDP16 | Required telemetry found a real idle violation after the reader fix; requirement is not a passing run artifact. |
| NDP17 | Source `5cb2e73c...` had a passing exact-source G2. Current corrected source `9dee67f...` has no refreshed G2, so no replacement may launch from this record. |

### Async v2.1 requirements V21S01–V21S17

| ID | Attempt evidence and status |
|---|---|
| V21S01 | Exact v2.1 policy/schema/digest were retained; the original reader failed to ingest their standalone JSON layout. |
| V21S02 | Four-clock/lag telemetry was retained without a reported lag-3 admission; no pass is inferred after the lane failure. |
| V21S03 | Exact tokens were the sole quantitative weight and accepted-token clock. |
| V21S04 | Exact K40 and `eta_outer=1.0` were payload-bound. |
| V21S05 | Full fenced identity, `Q_min=2`, `T_min=3,934,080`, zero retry, and ADR deadlines were exact-argv bound. |
| V21S06 | Resident workers produced at least 12 windows each, but the alternating long gaps violated the intended overlap behavior. |
| V21S07 | Atomic apply receipts advanced; no partial/double correction was reported, but failed clean performance prevents qualification. |
| V21S08 | Capacity-one verified-latest policy was bound; strict clean acceptance remains absent. |
| V21S09 | One sealed/one mutable and bounded memory/credit/replay policy was bound; no passing live high-water verdict is asserted. |
| V21S10 | Exact two-node leased membership remained intact; fault/rejoin was not submitted. |
| V21S11 | All eight trainers on each node produced node-level apply receipts; no partial-apply fault was submitted. |
| V21S12 | Native `cxi`, point-to-point, zero Python/Lustre dense-byte, no all-rank-wait policy was exact-payload bound and locally tested. |
| V21S13 | Honest K/cadence/idle evidence exposed 0.575375 idle. The original policy-reader defect is fixed, but this attempt does not pass V21S13. |
| V21S14 | Exact seed/offline staging and immutable checkpoints were retained; the clean attempt's overall checkpoint-correctness/performance gate did not pass. |
| V21S15 | Exact two-node `batch`/`debug` was proven, but the required clean/performance gate failed. Fault, replay, and convergence jobs were not submitted. |
| V21S16 | No promotion or 4+ authorization is permitted. |
| V21S17 | Raw peer/arrival/freeze/commit/K timing records remain diagnostic input, but no passing two-node evidence exists from which to derive or authorize a scale closure. |

## Immutable artifact hashes

| Artifact | SHA-256 |
|---|---|
| Clean qualification plan / exact argv | `82619e343840857aa28057ed3b07ee2d61313aed5fe4f9c88d29504edaa3e072` |
| Controller state | `2a03b28468f23816b421085e301f2d430aa2bd23d5ef1a38202ba9499bd80f8e` |
| Job stdout | `9d2748fc1ce732e341334c48254335933ce135163142bbaed420fd21a35236cb` |
| Job stderr | `6075a599157675c6a56d67150a7944668b13dd5cef04ffbdaac5214ccd262a3c` |
| Final scheduler transcript | `ff9549fd2e0e94d02d35d81f9155019969ec2748478ee7d8e823d40aee7e9ef9` |
| G2 full-layout gate | `93bc8854a0a161285b3cdde21f40d3b1936b6a7efcf10d37a39f43b40ecdbd6a` |
| Current-source native manifest | `869d8592815abea58fedbd769cb1321ce0c8123851229a417ee46d797398a5d6` |

## Disposition

No resubmission was performed. A future explicitly authorized attempt must:

1. fetch a clean current `origin/main`;
2. use canonical Frontier activation;
3. rebuild from that exact source;
4. refresh source/bundle attestations and exact-source full-layout G2;
5. produce a changed payload digest; and
6. submit only one exact-two-node `Partition=batch`, `QOS=debug` clean job.

Until that changed-source live attempt passes every clean/performance and
checkpoint-correctness criterion, this task remains incomplete and all
downstream fault and scale work remains blocked.
