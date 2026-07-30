# Fixed async-v2.1 RC two-node requalification: terminal clean failure

Task: `requalify-v21-fixed-rc-2n`

Date: 2026-07-30

Machine certificate: [`requalify-v21-fixed-rc-2n-20260730.json`](requalify-v21-fixed-rc-2n-20260730.json)

## Decision

The fixed RC did **not** qualify. The immutable machine certificate has literal
`full_pass=false` and `passed=false`. Scale is not authorized.

The exact-source native gates passed, and the receipt repair passed both its
local regression and the physical cold-start recovery boundary. The subsequent
real-model clean phase failed at its first global close. Per the stop rule, no
real-model fault campaign and no newer-fence fresh-allocation recovery were
submitted. The failed payload was not retried. Historical jobs 5111908 and
5111909 were never resubmitted.

## Frozen identity

One fetched `origin/main` was used throughout:

- Git commit: `7139cc5bf497e183d784bf276fb6bc55a4ff31bd`
- execution-source schema: `emender-async-v21-execution-source-v1`
- execution-source digest:
  `a97ac00bbcf21b08a858e6be67098ed2936aef30dc0a751929c2c90b81bb34bc`
- policy: `async-decoupled-v2.1-simple`, schema
  `emender-async-policy-v2.1`, digest
  `fa9def95daf7bce25f1b962ca5437e7a76317b94ccfb9a710fbf126a344e7d98`
- contribution/manifest schemas:
  `emender-native-e97-submission-v2.1` and
  `emender-native-e97-generation-v2.1`
- rebuilt native bundle:
  `6e962075594cf2db36280b55e05a35fde1965e67d8beefb40a3fec776b26d908`
- local/transport ABI: `65536`; async-v2.1 ABI: `131073` (`0x00020001`);
  wire: `2.1`
- launcher SHA-256:
  `1f2f1027e0dcc3e7e0940a20f6f0d0c469b670429e1928a94be6c51f3542fb55`
- runtime SHA-256:
  `7c395b23d4493982246bcc37a333cb994467a6f8945a4eab4ee2a6f46cdf5915`
- data identity:
  `91321b2b90bb159f3aa73881455778f10e8df588edd526b1066281fa72997962`
- tokenizer SHA-256:
  `94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069`
- final seed: step `2300930`, accepted tokens `150793748480`, bytes
  `7719680116`, SHA-256
  `0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2`.

The model allocation used job-local `sbcast` paths on both nodes. Both offline
node records verified the exact size/SHA/step/accepted-token authority and
reported `network_fetches=0`.

## Local regression and exact-code native gates

The local reproducer showed that the old literal generation-zero comparison
rejects the manifest empty string versus native 64-zero sentinel. The fixed
validator accepted exactly the two canonical no-prior-commit forms and rejected
four noncanonical spellings. The focused recovery suite passed 7/7.

The exact rebuilt native bundle passed all 11 CTests. The two physical native
jobs then passed sequentially:

| Phase | Job | Scheduler | Semantic result |
|---|---:|---|---|
| clean G2 | 5120760 | Nodes=2, Partition=batch, QOS=debug, COMPLETED 0:0 | exact CXI/FI_EP_RDM G2 passed |
| fault G2 | 5120892 | Nodes=2, Partition=batch, QOS=debug, COMPLETED 0:0 | `G2-fault-rejoin-replay` passed; reassignment=1, new incarnation=true, old epoch rejected, partial commit=false |

The clean model trace then accepted two `recover-peer` and two leased `READY`
events before K40 progress. All 16 trainers reached one coherent immutable
snapshot boundary, transferred `OWNED`, resumed the next local window, and
emitted finite loss observations. This physically clears the original
empty-versus-zero receipt failure: it did not recur before READY or K40.

## Durable scheduler transaction

The controller submitted model job 5120935 held, durably recorded payload
`cf065a49813efd552a935221c2837cbeefc060c148787aa34ff0eefd8003c769`,
registered scheduler-owned `afterany:5120935` collector 5120936, recorded the
collector script/command/dependency identity, and only then released the model
job. Controller, batch, and collector roots were disjoint.

| Actor | Job | Scheduler / result |
|---|---:|---|
| clean model | 5120935 | Nodes=2, Partition=batch, QOS=debug; FAILED 1:0 after 00:08:44 |
| durable collector | 5120936 | Nodes=1, Partition=batch, QOS=normal; COMPLETED 0:0; semantic `passed=false`, `verdict=failed` |

Queued/running/terminal evidence retains `Partition` and `QOS` as separate
fields. The collector independently retained the literal parent `sacct` row,
payload, stdout/stderr, hashes, and terminal verdict.

## First terminal semantic failure

Node 1 submitted the first complete node contribution at generation zero. It
was valid but alone could not satisfy `Q_min=2` and `T_min=3,934,080`. The
native kernel correctly returned a non-mutating finite disposition:

```text
{'attempt': 1, 'disposition': 'deferred',
 'generation': 0, 'status': 'deferred'}
```

`_native_manager` immediately converted that nonterminal response into:

```text
TimeoutError: native global freeze failed: {...}
```

instead of retaining/polling the open contribution through the immutable
420-second close deadline. The peer then expired. Node 0's later
`ndp_coord_step_v1: route failure (-12)` occurred during teardown and is
secondary.

The final native authority remained open at generation zero with one retained
contribution, `accepted_token_clock=0`, `result_receipt_count=0`, an all-zero
commit receipt/result, and zero commits. No checkpoint, mailbox result, partial
apply, publication, or double correction occurred.

Consequently the clean gate has only one initial K40 boundary per trainer and
zero atomic commits, not the required at least five windows and three commits.
The required `pipelined-performance.json` is absent, so no complete ISP01–ISP07
phase/tail, zero-foreground-result-wait, max/p99 pause, or anti-200-second-stall
claim is made.

## Conformance checklist and requirement map

The controlling checklist is **Resilient DiLoCo Compute Pool, Version 1**, the
“Conformance checklist (required in every implementation/runner/scale task
Validation)” section of
[`RESILIENT_DILOCO_COMPUTE_POOL.md`](../RESILIENT_DILOCO_COMPUTE_POOL.md).
The native specialization is
[`NATIVE_RESILIENT_DILOCO_DATAPLANE.md`](../NATIVE_RESILIENT_DILOCO_DATAPLANE.md),
and bounded async semantics are ADR-002 in
[`ASYNC_DECOUPLED_DILOCO_V2.md`](../ASYNC_DECOUPLED_DILOCO_V2.md).
The companion matrix is
[`RESILIENT_DILOCO_GAP_MATRIX.md`](../RESILIENT_DILOCO_GAP_MATRIX.md).

All applicable IDs are mapped individually in the machine certificate:

- **R01–R16**: scheduler fencing, leased membership, exact-code G2, coherent
  snapshot/finite loss, and fail-closed no-result behavior were observed;
  **R06** fails because deferred finite closure became terminal, **R12** was not
  reached, and **R16** fails the current-source qualification ladder.
- **NDP01–NDP17**: exact-code clean/fault G2 passed the compiled point-to-point
  CXI/ABI/wire/math/bounds/replay/rejection gates; **NDP13** fails the bounded
  nonterminal close handling and **NDP15** was not reached (with no partial
  result).
- **V21S01–V21S17**: identities, exact-token eta_outer=1 policy, two-node floor,
  final seed, and physical READY/snapshot startup were bound; **V21S10** and
  **V21S15** fail at closure/qualification, while **V21S16–V21S17** remain
  unauthorized.
- **ISP01–ISP07**: one-window physical ISP01–ISP04 evidence exists; ISP05 was
  not reached and failed closed; ISP06–ISP07 cannot pass without the complete
  performance verdict and every-event tail record.

The overall result is deliberately `not_conformant_terminal_clean_failure`,
not a partial authorization.

## Validation commands

Every Python, build, preflight, and submission command followed:

```text
source scripts/frontier/activate_emender_frontier.sh
"$EMENDER_PYTHON" -m pytest -q <focused recovery tests>
PYTHON_BIN="$EMENDER_PYTHON" scripts/frontier/build_native_resilient_dataplane.sh
scripts/frontier/submit_native_dataplane_2n_gate.sh clean
scripts/frontier/submit_native_dataplane_2n_gate.sh fault
"$EMENDER_PYTHON" scripts/frontier/run_async_v21_qualification.py \
  --gate clean --nodes 2 --repo <frozen-main> \
  --native-build-manifest <exact-manifest> \
  --full-layout-gate <job-5120760-gate> \
  --run-root <batch> --evidence-root <collector> \
  --state <controller-state> --output <clean-plan> --submit
sacct -j 5120760,5120892,5120935,5120936 -X -n -P \
  --format=JobIDRaw,JobName,State,ExitCode,DerivedExitCode,Elapsed,AllocNodes,Partition,QOS,NodeList,Submit,Start,End
```

No unchanged payload retry, real-model fault phase, fresh recovery allocation,
or scale job followed the terminal failure.
