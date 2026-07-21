# Pipelined native DiLoCo two-node gate attempt

Date: 2026-07-20  
Task: `validate-pipelined-native-2`  
Result: **supervisor-fix replacement K40 job 5043045 queued; monitoring sole exact two-node allocation**

## Supervisor-fix retry (active)

Exact-source G2 refresh job `5042988` completed successfully (`0:0`) on
exactly two nodes, `frontier[05335,05337]`, in 3:04.  Its passing
correctness/integrity gate is retained at
`g2-artifacts/5042988/full-layout-gate.json` beneath the immutable controller
root, with SHA-256
`f9acb5af938df5f8b47798d08dc67442a4d71986bb557a6acd61245e24042a6f`.
The throughput fields remain telemetry rather than K40 admission evidence.

After G2 was terminal, a fresh scheduler query proved the user queue empty.
The serial fail-closed launcher rebuilt the exact source, passed native CTest
10/10, and attested bundle digest
`59fa632b98999e522be6fee3cda98d095a0fc4c85b0b3a95286b0eb61c19fa6d`.
The acceptance native manifest SHA-256 is
`c0efedd312b27a19fd0ab911e2e21934359404a9359f0ba691f630726c4b059c`;
the immutable acceptance manifest SHA-256 is
`d1ebb1c2c172cc3e81f06774a94be9303f131071b79db17d070eb8f4342e10b1`.

At 2026-07-21 08:24 EDT the controller submitted exactly one clean-overlap
replacement, job `5043045`, requesting exactly two nodes, five K40
generations, and a two-hour bound.  It is the sole user job and is currently
`PENDING (Priority)` with no scheduler start estimate.  No duplicate, later
serial phase, or four-node-or-larger job has been submitted.  The task remains
active through terminal completion; live overlap, cadence, fault/rejection,
and restart claims remain pending rather than inferred from queue state.

Authoritative `origin/main` resolves to
`084f36f05cfa66ed8b2c416941824d80b88cbdf9`, containing the reviewed
supervisor generation-progress recovery fix. A fresh clean `main` checkout at
`/lustre/orion/bif148/scratch/erikgarrison/emender-exact2n-supervisor-fix-20260721T114200Z/source`
exactly matches that pushed identity. In the canonical Frontier environment,
its native bundle built successfully and CTest passed 10/10; the immutable
preflight bundle is retained under the adjacent
`native-stage/preflight/install` directory.

The user Slurm queue was empty before submission. Because the source identity
changed from the job-5042670 gate, the fail-closed path submitted exact-source
G2 correctness/integrity refresh job `5042988` at 2026-07-21 07:43 EDT for
exactly two nodes. At this checkpoint it is the sole user allocation and is
`PENDING (Priority)` with no scheduler start estimate. No K40 replacement,
duplicate, fault phase, or four-node-or-larger job has been submitted. The task
remains active through terminal G2 and, only after a passing gate and another
empty-queue check, the single authorized exact two-node K40 replacement.

Conformance is checked against *Resilient DiLoCo Compute Pool* v1 R01-R16 and
*Native resilient DiLoCo data plane* v1 NDP01-NDP17. The exact pushed source,
clean native build, fixed two-node capacity, no-overlap guard, source/bundle
fencing, and fail-closed K40 admission conform to R10/R13/R14/R16 and
NDP02/NDP03/NDP13/NDP16/NDP17. Live five-generation, overlap/cadence,
loss/rejoin, rejection, failed-publication retention, and fresh-restart
evidence remains pending rather than claimed.

## Generation-identity-fix retry

Authoritative `origin/main` resolves to
`dd50c5123f72d91f0618059f6689c2df9ea36233`, containing the reviewed
generation-identity recovery fix. A clean `main` clone at
`/lustre/orion/bif148/scratch/erikgarrison/emender-exact2n-generation-identity-20260721T081000Z/source`
exactly matches that pushed commit. Its canonical Frontier native build passed
CTest 10/10; the immutable preflight installation is under the adjacent
`native-stage/preflight/install` directory.

The user Slurm queue was empty immediately before submission. Because this
source identity differs from the retained job-5039234 G2 artifact, the
fail-closed launcher submitted exact-source G2 refresh job `5042670` at
2026-07-21 04:09 EDT for exactly two nodes. It completed `0:0` in 2:54 on
`frontier[04129,04132]` and published the passing correctness/integrity gate at
`g2-artifacts/5042670/full-layout-gate.json` (SHA-256
`c5a95ee303271686bbd2b6051829854e680287bdb3ae1a4f019fcdd7160804cc`).

After G2 left the queue, the serial fail-closed controller rebuilt and
re-attested the exact source. The acceptance native manifest SHA-256 is
`bdc72415338b74112087756762aa3819ee6762dd70a271878bd31b5f3b433ac0`,
and its recorded native bundle digest is
`59fa632b98999e522be6fee3cda98d095a0fc4c85b0b3a95286b0eb61c19fa6d`.
The immutable acceptance manifest SHA-256 is
`10d2167fc9836b9223a6e1dbafbe0b3c807ab20dfaa9ed39c34c368027ad1ee3`.
The adjacent empty-queue guard then submitted exactly one authorized K40
replacement, clean-overlap job `5042682`, for exactly two nodes and five K40
generations. Slurm records that it ran on `frontier[05631,08192]` from
2026-07-21 04:51:41 through 05:05:59 EDT and terminated `FAILED 1:0` after
14:18; its Python step failed after 13:32. No duplicate or
four-node-or-larger job was submitted.

The replacement completed and safely applied generations 0 and 1 on both
nodes. `handoff/latest.json` points to the finalized generation-2 manifest,
which records 10,490,880 accepted tokens and manifest SHA-256
`fc98357c2377a94f785f4776a4eba636e90235780cd91c6fe74c4ffd2fd1bf97`;
both generation-1 and generation-2 restartable checkpoint payloads remain on
disk. All 16 trainers report two handoffs and two applied results, queue
high-water marks of one, no replacements, and zero stale or rejected results.
For node-0 trainer 0, the generation pipeline intervals were 291.345 and
337.965 seconds, foreground waits were 9,619 and 22,243 ns, and both stayed
inside the 420-second bound. Generation-1 manager telemetry separately records
17.560 seconds local reduction, 35.036 seconds owner contribution, 22.066
seconds owner redistribution, and 65.028 seconds total redistribution, with
receive queue high-water one, zero Python dense-socket bytes, zero native
retries/CQ/route errors, and roughly 82.35 MB/s observed native throughput.

Trainer timestamps prove real generation-1 optimizer work through step 79
while generation-0 background collection/reduction/redistribution and apply
completed, and generation-2 safe-boundary redistribution/application began.
This is direct g-background/g+1-compute overlap evidence and keeps observed
foreground control-plane idle far below 10%. It is not the requested
five-generation steady-state sample, so the full cadence gate is not claimed.

The terminal defect is in supervision rather than data-plane integrity. During
the long, still-progressing generation-2 per-trainer redistribution/apply
sequence, both managers were killed for `progress_deadline`, their native
services were rejoined, and the replacement managers restarted at
`runtime_import`. The supervisor then immediately killed them for
`first_atomic_generation_deadline`, even though two atomic generations and a
finalized generation-2 checkpoint already existed, and exhausted their restart
budgets. Thus the clean phase failed after two of five generations and the
serial controller correctly did not submit the fault/rejoin, invalid-result,
checkpoint-publication-failure, or fresh-restart phases.

Conformance is checked against *Resilient DiLoCo Compute Pool* v1 R01-R16 and
*Native resilient DiLoCo data plane* v1 NDP01-NDP17. Exact pushed source,
clean native build, refreshed G2 correctness/integrity, empty-queue guard,
fixed two-node capacity, no-scale guard, R04/R06 safe application, R07/R11
bounded latest-only queues, and NDP10/NDP13/NDP15 integrity/ownership behavior
pass in retained evidence. The premature manager eviction/restart accounting
violates the required R12/R14/R16 and NDP16/NDP17 resilient progress gate;
five-generation cadence plus the named fault, rejection, publication-failure,
and fresh-restart phases remain unproven.

## Lifecycle-fix replacement (terminal harvest)

Authoritative `main` and `origin/main` resolve to
`3cfed722beb086d015cff254f473af2a63eaa492`, which contains the reviewed
generation-start lifecycle repair.  A clean authoritative clone was built in
the canonical Frontier environment.  The native build passed CTest 10/10 and
recorded bundle digest
`59fa632b98999e522be6fee3cda98d095a0fc4c85b0b3a95286b0eb61c19fa6d`.

Because this source differs from the historical job-5037939 source, G2 was
refreshed.  Job `5039166` failed in batch preflight before launching the native
payload because the clean clone did not contain the default clone-local Python
environment; it produced no gate.  After explicitly binding the canonical
approved Python 3.12 interpreter and confirming the user queue was empty, G2
retry `5039234` completed `0:0` in 3:01 on exactly
`frontier[05865,05870]`.  Its exact-source correctness/integrity attestation
passed.  The gate is retained at
`/lustre/orion/bif148/scratch/erikgarrison/emender-exact2n-lifecycle-fix-20260720T202500Z/g2-artifacts/5039234/full-layout-gate.json`
with SHA-256
`d6300536df887cf71f4393214707d9ffab68c89527ce31e981d043d5e304d449`.
G2 telemetry reports 22.792855478, 22.901945645, and 23.460527776 seconds,
961,799,592.29 logical B/s, and 4.321x the retained Python baseline; this ratio
is telemetry rather than K40 admission.

After job 5039234 cleared and a second empty-queue check, the exact serial
launcher rebuilt and re-attested the same source/bundle (CTest 10/10) and
submitted the single authorized replacement K40 clean-overlap job `5039258`.
The immutable controller root is
`/lustre/orion/bif148/scratch/erikgarrison/emender-exact2n-lifecycle-fix-20260720T202500Z/`.
Its acceptance manifest requires exactly two nodes, five K40 generations,
background-g/compute-(g+1) overlap, foreground idle below 10%, and cadence no
worse than 1.25x raw K40 compute when background work fits.

Slurm accounting records that the sole replacement started at
`2026-07-20T17:59:06` on exactly `frontier[06619-06620]` and terminated
`FAILED 143:0` at `18:09:09` after `00:10:03`; its Python step failed `1:0`
after `00:08:49`. No duplicate K40 job and no 4-node-or-larger job was
submitted.

The replacement materially advanced beyond job 5037971. Generation 0
completed K40 on all 16 trainers, accepted both nodes and 5,245,440 tokens,
produced identical 5,506,770,496-byte result root
`ede07c3e8d55504bcf189551f5f5cc5fba4f6b6e3bb10fccc4016765bf3a27a3`
on both managers, applied the result at the safe boundary, and atomically
published the finalized generation-1 checkpoint. The retained handoff is
7,899,873,331 bytes with SHA-256
`bf013bed934da7c54c339163bddd204c34457df5e326e851fb01778772242151`;
`latest.json` points only to that finalized manifest. Generation 1 then began
real optimizer/forward/backward work. This proves one atomic generation and
bounded safe handoff/application, but not the required five generations or
generation-g background overlap with generation-(g+1) compute: the observed
checkpoint commit preceded the recorded generation-1 training stages.

Generation-0 per-trainer pipeline elapsed times ranged from 288.477 to
300.584 seconds. All 16 records report one handoff and one applied result,
handoff and result queue high-water marks of one, zero replacements, zero
stale/rejected results, and a maximum foreground wait of 14,058 ns (0.0049%
of the fastest recorded pipeline interval). The two manager contribution
phases took 34.986/35.017 seconds and redistribution took 22.009/22.068
seconds, with receive queue high-water one and no Python dense-socket bytes.
These are useful bounded-queue, background-phase, and foreground-idle
measurements. They are not a five-generation steady-state cadence result, and
the generic performance validator refuses the incomplete artifact layout with
`no real trainer step telemetry`.

The terminal defect is a generation-identity handoff mismatch exposed during
trainer-7 recovery on both nodes: `GenerationMetadata.from_json()` rejected
`native-generation-00000001.json` as `native E97 generation identity is
invalid`. Both trainer-7 roles exhausted their single restart; other trainers
then encountered native buffer/operation release route failures (`-12`) as
the allocation step was torn down. Consequently the clean phase stopped in
generation 1 and the serial controller did not submit fault/rejoin,
invalid-result, checkpoint-publication-failure, or fresh-restart phases.

Conformance for this replacement is checked against *Resilient DiLoCo Compute
Pool* v1 R01-R16 and *Native resilient DiLoCo data plane* v1 NDP01-NDP17.
Exact-source identity, G2 correctness/integrity, two-node admission,
point-to-point CXI selection, and no-overlap/no-scale guards currently pass.
Exact identity, G2 integrity, two-node admission, one atomic safe-boundary
commit, checkpoint publication, and bounded latest-only queue behavior have
live evidence. The replacement does not satisfy R07/R11/R12/R14/R16 and
NDP10/NDP13/NDP15/NDP16/NDP17 acceptance as a whole: five generations,
steady-state overlap/cadence, loss/rejoin, invalid-result rejection,
failed-publication retention, and fresh restart were never reached. No scale
ladder job is admitted from this result.

## Terminal harvest of job 5037971

The existing job was monitored without a duplicate submission.  Slurm accounting
records that `5037971` left the queue, started at `2026-07-20T14:15:45` on
exactly two nodes (`frontier[02939-02940]`), and terminated `FAILED 1:0` at
`14:24:26` after `00:08:41`.  The Python step ran for `00:07:53`; therefore the
job did execute and does not satisfy the narrowly authorized condition for a
replacement submission.

The immutable execution root is
`/lustre/orion/bif148/scratch/erikgarrison/emender-exact2n-final-artifacts-20260720T181000Z/`.
Its retained pool-control record proves that generation 0 froze both READY node
incarnations, met `Q_min=2`, and reached `commit_ready` with 5,245,440 accepted
tokens.  Both node result records agree on result root
`63b3fef285173902e0ee4b54f4e7cab61fac8860c7427d9ef2c3750b9c641477`.
Trainer timestamps include generation-0 K40 work through step 39 on
`node-0-trainer-0` (final timestamp `1784571845.7019308`, loss
`2.5740623474121094`, 8,196 tokens for that step).  This is useful live K40
evidence, but generation 0 never became an atomic committed generation and no
generation 1 training exists.  It consequently cannot establish g/g+1 overlap,
steady-state foreground idle or cadence, five atomic generations, or the later
fault/rejection/restart phases.

The decisive failure sequence is retained in the per-role stderr and
supervision event stream:

- trainer leaders raised `NameError: name 'generation_started' is not defined`
  while calculating the post-generation deadline;
- another trainer failed `ndp_buffer_seal_v1` with route failure `-12`, while
  followers later expired waiting for checkpoint-leader apply release;
- both managers then failed `ndp_control_v1(FREEZE)` with invalid lifecycle
  state `-3`; supervisor restart attempts were exhausted.

No later phase and no 4-node-or-larger job was submitted.  The existing G2 job
`5037939` remains the reused exact-source correctness/integrity and throughput
telemetry gate because submission source identity did not change.

Key immutable SHA-256 identities are:

- acceptance manifest: `adbe2c8108bdfe4a4b538cf0a562e06dfea1dcf97535e5ef5b79f0a380ea14a7`
- runtime identity: `5c0fa27593fa1b173a57512eaee13e8dcb8bfadfb1c1b3783fae4457e02c6805`
- native launch attestation: `a50087225c6b85be4b6d7304582605385ae63de448e3438e9d97eac76d3d4153`
- generation-0 pool-control record: `832467298478d6669d0722268ab6e5cbc454925f029d0b46380e6458236d1440`

Conformance was checked against *Resilient DiLoCo Compute Pool* v1 R01-R16
and *Native resilient DiLoCo data plane* v1 NDP01-NDP17.  The live run confirms
exactly-two-node capacity, leased READY membership, fenced generation identity,
bounded execution, exact-source native launch attestation, contribution freeze,
and fail-closed behavior without an all-rank wait.  It does not satisfy R07,
R11, R12, R14, R16 or the corresponding NDP10/NDP13/NDP15/NDP16/NDP17 live
acceptance evidence because no atomic commit, safe apply, later generation,
failure phase, or fresh restart completed.

## Seventh concrete attempt: refreshed G2 passed and real K40 queued

Authoritative source `87365c5f846a950d7aaa01fec441982c79fc5e50` is merged
and pushed. Its canonical native build passed CTest 10/10. Exact-source G2 job
`5037939` completed `0:0` in 2:51 on exactly `frontier08123` and
`frontier08127`. The retained gate is
`/lustre/orion/bif148/scratch/erikgarrison/emender-exact2n-retry-stage-20260720T180500Z/g2-artifacts/5037939/full-layout-gate.json`.

G2 exact-reference correctness/integrity passed with zero CQ/route errors and
zero retries. It reported 44,322,599,424 useful TX and RX bytes, 44,323,138,304
wire TX and RX bytes, and timed intervals 23.353392370, 22.742834860, and
23.261179257 seconds. Median native logical throughput was 946,946,057.23 B/s
versus retained Python 222,582,457.59 B/s (4.254x). The ratio is telemetry, not
K40 admission; correctness/integrity admitted the real phase.

Initial K40 job `5037915` failed closed before model load because the serial
renderer used a caller-relative batch path and omitted the G2/runtime exports.
The controller now uses an absolute authoritative launcher and `--chdir`, and
binds source, seed, data, tokenizer, bundle, G2, and fence identities. The
focused canonical suite passed 62/62; fix `d3ee23f2` was pushed and merged into
the authoritative commit above.

The corrected controller submitted clean-overlap job `5037971` for exactly two
nodes and five K40 generations. It remains `PENDING (Priority)` at handoff, so
no K40 overlap/cadence claim is made. Controller state and the immutable source
inventory/acceptance manifest are under
`/lustre/orion/bif148/scratch/erikgarrison/emender-exact2n-final-artifacts-20260720T181000Z/`.
No overlapping, later-phase, or 4-node-or-larger job was submitted.

Conformance: *Resilient DiLoCo Compute Pool* v1 R01-R16 and *Native resilient
DiLoCo data plane* v1 NDP01-NDP17. G2 establishes exact fenced identities,
bounded point-to-point CXI, exact weighted math, integrity/rejection, bounded
release, and no MPI/all-rank or Python-dense fallback. The queued K40 phase
retains `Q_min=2` and `T_min=3,934,080`; live timing, loss/rejoin,
invalid-result rejection, and checkpoint failure/restart remain pending.

## Sixth concrete attempt: repaired linkage, controller fix, and live G2 performance blocker

Authoritative `main` first resolved to `692d0292`.  From a clean clone, the
canonical Frontier activation built the exact native bundle and passed CTest
10/10.  Exact-source G2 job `5036978` completed on exactly two nodes in 2:57
and produced a passing full-layout CXI gate.  The serial acceptance controller
then failed closed before `sbatch` because its plan omitted the top-level
`authoritative_stage` consumed by `advance()`.  A regression fix was committed
as `bb5cd54f`, cherry-picked and pushed to authoritative main as `09eac436`;
the focused launcher/runtime suite passed (26 collected, plus 8 passed/1
platform skip in the native gate selector).

Because that source change invalidated the earlier gate identity, the exact
`09eac436` bundle was rebuilt (CTest 10/10) and G2 job `5037046` was concretely
submitted on exactly two nodes.  Both native node payloads passed, but the
authoritative validator rejected publication because observed clean throughput
did not reach 4x the retained Python gate.  Slurm records `FAILED 1:0`, 3:00,
two nodes (`frontier[07388,07408]`).  Therefore no real K40 phase was admitted:
five generations, overlap/idle timing, fault/rejoin, invalid-result rejection,
and checkpoint restart remain unproven.  No 4-node-or-larger job was submitted.

Conformance was checked against Compute Pool v1 R01-R16 and native data plane
v1 NDP01-NDP17.  The source/bundle/G2 fences and exactly-two-node/no-overlap
guards behaved fail closed; NDP17 blocks scale-out until the exact-source G2
performance gate passes.

## Fifth concrete attempt: authoritative rebuild exposes G2 runtime-linkage blocker

On reassignment after the serial launcher fix merged, authoritative `main` and
`origin/main` resolved to
`d25c78b4b414e1a41c5ec2764f6307f93f64b316`.  Because the shared main checkout
contains retained untracked run evidence, the submission was prepared from a
fresh local `main` clone at:

```text
/lustre/orion/bif148/scratch/erikgarrison/emender-exact2n-20260720T142140Z/source
```

The clone was clean, on branch `main`, and exactly matched `origin/main`.  The
canonical `scripts/frontier/activate_emender_frontier.sh` activation selected
the approved Python 3.12 environment and Frontier GNU 14.2/Cray MPICH 9.1
module stack.  The reviewed exact acceptance renderer was invoked with
`--submit`, `--state`, and `--native-stage-root`.  Its first canonical native
build initially failed closed because inherited `FI_PROVIDER=cxi` forced five
local socket CTests onto CXI.  The build was repeated without fabric override
variables; all 10 native CTests then passed and the immutable bundle was
installed at:

```text
/lustre/orion/bif148/scratch/erikgarrison/emender-exact2n-20260720T142140Z/native-stage/d25c78b4b414e1a41c5ec2764f6307f93f64b316/install/native-artifacts.json
```

Production attestation correctly rejected the retained job-5033120 G2 gate:

```text
source_commit: 85cf5a09... != d25c78b4...
bundle_sha256: f2ac884e... != 7b4c7769...
```

No real-model job was admitted with mismatched evidence.  To concretely obtain
the required exact-source precursor, the canonical G2 launcher was then run
against the rebuilt bundle:

```text
NDP_BUILD_MANIFEST=.../d25c78b4.../install/native-artifacts.json \
NDP_ARTIFACT_ROOT=.../g2-artifacts \
bash scripts/frontier/submit_native_dataplane_2n_gate.sh clean
```

Slurm accepted job `5036108` on exactly two nodes.  It ran on
`frontier[09074-09075]` and terminated `FAILED 1:0` after 80 seconds, before a
G2 generation or full-layout gate was produced.  Both node tasks reported:

```text
ndp_frontier_2n_gate: error while loading shared libraries:
libamdhip64.so.7: cannot open shared object file: No such file or directory
```

The scheduler record and `g2-artifacts/5036108/submission.json` are retained
under the execution root above.  The exact K40 acceptance remains fail-closed:
without an exact-source G2 gate, five live generations and the fault/restart
phases cannot be admitted.  A follow-up must make the canonical native G2
batch runtime expose the approved ROCm 7.1 library path (or install a complete
self-contained runtime), rerun the two-node G2 clean gate, then resume the
serial acceptance controller.  No 4-node-or-larger job was submitted.

Conformance checked against *Resilient DiLoCo Compute Pool* version 1,
requirements R01-R16, and native data plane requirements NDP01-NDP17.  This
attempt demonstrates fail-closed source/bundle/G2 identity enforcement
(R13/R16, NDP13/NDP17), bounded exactly-two-node scheduler mutation, and no
acceptance claim from a partial or unattested run.  Runtime-only overlap,
timing, rejection, loss/rejoin, and atomic restart criteria remain unproven.

## Fourth concrete attempt: reviewed exact acceptance launcher

After the prerequisite launcher merged, authoritative `main` and
`origin/main` both resolved to
`08a26a0f0bc71c7e480010ba1ecb4e024ce37e36`.  The per-user Slurm queue was
empty at `2026-07-20T13:12:07Z`.  From the clean authoritative checkout, after
canonical Frontier activation, the new exact submit path was invoked with:

```text
$EMENDER_PYTHON scripts/frontier/render_resilient_e97_exact_2n_acceptance.py \
  --repo /lustre/orion/bif148/scratch/erikgarrison/emender \
  --native-build-manifest /lustre/orion/bif148/scratch/erikgarrison/emender/build/native-resilient-dataplane/native-artifacts.json \
  --full-layout-gate /lustre/orion/bif148/scratch/erikgarrison/emender/reports/frontier/native-dataplane/5033120/full-layout-gate.json \
  --run-root /lustre/orion/bif148/scratch/erikgarrison/emender/reports/frontier/pipelined-native-2-20260720T131330Z/phases \
  --output /lustre/orion/bif148/scratch/erikgarrison/emender/reports/frontier/pipelined-native-2-20260720T131330Z/acceptance.json \
  --submit
```

All mandatory immutable E97 seed, data, tokenizer, approved training-config,
native-CXI, and fenced-run inputs were exported before this command.  Slurm
accepted the first phase as job `5035685`:

```text
clean-overlap=5035685
5035685|resilient-e97-true-2n|RUNNING|2|02:00:00|frontier[03756-03757]
```

Thus this attempt concretely submitted exactly two nodes.  It submitted no
4-node-or-larger job.  Job `5035685` reached the approved Python/Torch/ROCm
runtime identity check, then failed closed after 2:48 with exit `1:0`, before
model admission or any K40 generation:

```text
ValueError: native build does not match the launched source commit
```

This is the expected enforcement of the source/bundle mismatch noted below;
no runtime-only acceptance claim can be made from this job.

The renderer immediately attempted to enqueue `fault-rejoin` behind the clean
job.  Slurm rejected that second submission before assigning a job ID:

```text
sbatch: error: QOSMaxSubmitJobPerUserLimit
sbatch: error: Batch job submission failed: Job violates accounting/QOS policy
```

The renderer consequently exited 64 and did not attempt the remaining three
phases.  The immutable render and submission transcript are retained under
`reports/frontier/pipelined-native-2-20260720T131330Z/` in the authoritative
checkout.  The clean job must be allowed to finish and harvested; subsequent
phases must then be submitted serially as the QoS slot becomes free.  The
current all-at-once dependency-chain implementation cannot establish the full
acceptance sequence on a QoS configured with a one-job-per-user submit limit.

Two additional lineage issues were exposed by the attempt: the installed
native manifest records source `176ae0bc...` rather than the submitted source
`08a26a0f...`, which caused job `5035685` to fail attestation, and the
renderer's phase-specific
`run_dir`/`restart_from` values are recorded in JSON but are not exported as
`RUN_DIR`/`RESILIENT_E97_RESUME_HANDOFF` to each `sbatch` command.  Runtime
attestation and restart evidence must not be claimed unless the job artifacts
prove those bindings.  A retry requires a canonical native rebuild from the
authoritative commit and a serial phase submit/harvest path compatible with
the one-job debug-QoS limit.

Retry note (agent-1347): at `2026-07-20T12:50:12Z`, a third concrete attempt
fetched `origin/main`, confirmed that the authoritative checkout's `HEAD`,
`main`, and `origin/main` still all resolved to
`176ae0bc11db5bf1cad51008d6891d209867004c`, and activated the canonical
Frontier environment.  The activation selected the current canonical module
stack (including `PrgEnv-gnu/8.7.0`, `cray-mpich/9.1.0`, and
`gcc-native/14.2`).  The exact approved clean two-node command was then invoked
again with the retained native manifest and `$EMENDER_PYTHON`; it exited 69
with `refusing to overlap another user allocation` before `sbatch`.

Immediately after that attempt, job `5035539` remained RUNNING on 32 nodes at
12:01 elapsed of its 30-minute bound.  The authoritative source had not gained
a reviewed real-K40 acceptance submitter: the clean G2 launcher still fixes
three synthetic generations, while `resilient_e97_true_2n.sbatch` still admits
only 20/30-minute gates and therefore cannot safely cover five independently
bounded 420-second K40 windows plus the required control/checkpoint phases.
No job was submitted by this retry, and the live acceptance criteria remain
unclaimed.

Retry note (agent-1346): the retry on the same date re-read both normative
design authorities, verified that the authoritative checkout still resolved
`HEAD`, `main`, and `origin/main` to `176ae0bc11db5bf1cad51008d6891d209867004c`,
and performed a fresh scheduler preflight.  Job `5035539` was still RUNNING on
32 nodes (`validate-native-pool-32n-late-ready-fixed`), so the no-overlap
condition had not changed and no `sbatch` call was permissible.
The exact canonical activation and approved clean-gate command were invoked
again; the launcher exited 69 with `refusing to overlap another user
allocation`, proving that the retry also stopped before `sbatch`.

## Concrete attempt

The canonical Frontier environment was activated with:

```text
source scripts/frontier/activate_emender_frontier.sh
```

The submission source was the authoritative checkout
`/lustre/orion/bif148/scratch/erikgarrison/emender` on branch `main`.
At the attempt, `HEAD`, `main`, and `origin/main` all resolved to:

```text
176ae0bc11db5bf1cad51008d6891d209867004c
```

The exact source native bundle was rebuilt before submission:

```text
PYTHON_BIN="$EMENDER_PYTHON" BUILD_JOBS=8 \
  bash scripts/frontier/build_native_resilient_dataplane.sh
```

All 10 native CTests passed. The retained build manifest was:

```text
/lustre/orion/bif148/scratch/erikgarrison/emender/build/native-resilient-dataplane/native-artifacts.json
sha256 cad13e0ebb93761f2bd9e8217ded24cb1592d9f9e59367a93dda6ca9c0a51fda
```

At `2026-07-20T12:41:36Z`, the exact approved submission command was
invoked from that authoritative `main` checkout:

```text
NDP_BUILD_MANIFEST=/lustre/orion/bif148/scratch/erikgarrison/emender/build/native-resilient-dataplane/native-artifacts.json \
NDP_PYTHON_BIN="$EMENDER_PYTHON" \
bash scripts/frontier/submit_native_dataplane_2n_gate.sh clean
```

It exited 69 with the expected fail-closed diagnostic:

```text
refusing to overlap another user allocation
```

The conflicting allocation observed immediately after the attempt was:

```text
5035539|validate-native-pool-32n-late-ready-fixed|RUNNING|32|3:24|30:00
```

No `sbatch` invocation was reached and no job of any size was submitted by
this task. In particular, no 4-node-or-larger job was submitted.

## Acceptance status

Because the approved launcher refused submission, no live two-node artifact
was created and none of the runtime-only claims are made. Five atomic K40
generations, overlap/cadence timing, foreground-idle percentage, bounded
latest-only behavior under loss/rejoin, rejection evidence, and atomic
checkpoint restart remain unverified by this attempt.

There is also an acceptance mismatch that must be resolved before a retry:
`submit_native_dataplane_2n_gate.sh clean` fixes `NDP_GENERATIONS=3` and its
batch payload requires one warm-up plus three timed synthetic native data-plane
generations. The present task requires at least five atomic **K40 trainer**
generations plus checkpoint/failure behavior. The real K40 split-role path is
`scripts/frontier/resilient_e97_true_2n.sbatch`; a retry must use a reviewed,
fail-closed two-node submit artifact for that path rather than treating three
synthetic data-plane generations as the requested live training proof.

The merged real-model batch artifact has a second hard mismatch.  Its
supervisor bounds each K40 stage at 420 seconds, while the batch script admits
only 20- or 30-minute non-startup gates.  Five worst-case K40 windows alone are
35 minutes, before foreground handoff/apply, integrity, redistribution, and
checkpoint publication.  Submitting that payload with five generations would
therefore not be a fail-closed way to establish the requested acceptance
artifact.  A reviewed launcher must provide a bounded walltime sufficient for
five K40 generations and must explicitly orchestrate and validate all requested
fault/restart phases; those controls are not present in the named submitter.

## Architecture conformance

Authority checked: *Resilient DiLoCo Compute Pool*, version 1 (2026-07-17),
and *Native resilient DiLoCo data plane v1*. Applicable gap-matrix requirement
IDs are R01-R16 and NDP01-NDP17.

The attempted command conformed to the admission safety portion of R01/R14 and
NDP17: authoritative identity was pinned, the current native bundle was built
and tested under the canonical environment, the launcher was exactly two-node
and CXI-only, and its overlap guard failed closed before scheduler mutation.
The remaining conformance checklist items require a live allocation and are not
claimed here.
