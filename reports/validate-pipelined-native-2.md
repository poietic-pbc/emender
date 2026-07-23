# Pipelined native DiLoCo two-node gate attempt

## Final-seed/debug-QoS replacement checkpoint (2026-07-23)

The evaluated final-seed/debug-QoS integration is authoritative at
`7ab92adabcd63ae4c5d0cf2d2c408b0fe182a944`; `git fetch`, the local
`origin/main` ref, and `git ls-remote origin refs/heads/main` agreed exactly.
The previous generation-gap replacement, job `5059293`, was cancelled before
allocation after its obsolete seed binding was discovered. Terminal accounting
records zero elapsed time on exactly two requested nodes, with
`Partition=batch` and `QOS=debug`.

A fresh clean clone and native stage were created at:

```text
/lustre/orion/bif148/scratch/erikgarrison/emender-exact2n-final-seed-20260723T123000Z
```

The canonical Frontier environment selected the approved Python 3.12 and
ROCm/Cray stack. The exact-source native build, install, artifact recording,
and CTest completed successfully (**10/10 passed**). After an adjacent empty
user-queue check, exact-source G2 job `5059531` ran on exactly two nodes and
completed `0:0` in 2:52. Both live and terminal scheduler evidence explicitly
record `Partition=batch` and `QOS=debug`. The full-layout correctness and
integrity gate is retained at:

```text
/lustre/orion/bif148/scratch/erikgarrison/emender-exact2n-final-seed-20260723T123000Z/g2-artifacts/5059531/full-layout-gate.json
sha256 df042cce1743ff8c4a0f5623aa9e8ba7d61a41ab6c6c6497b2873a2c12ed9004
```

The gate attests source `7ab92ada...` and native bundle
`9884a02d84bd9560a15314c26e386868350b865e688cc4c701c802f4f686227a`.
After G2 reached terminal state and the queue was empty again, the canonical
serial acceptance controller rebuilt and re-attested the exact native bundle
and submitted only clean-overlap job `5059548`. It requests exactly two nodes,
five K40 generations, a two-hour bound, `Partition=batch`, and `QOS=debug`.
The scheduler-exported immutable seed identity is step `2300930`, tokens
`150793748480`, size `7719680116`, and SHA256
`0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2`.
Job `5059548` subsequently ran on exactly two nodes and terminated `FAILED
1:0` after 33 seconds. Terminal accounting explicitly records
`Partition=batch`, `QOS=debug`, `NNodes=2`, and
`frontier[01918,09456]`. The user queue was empty after termination. No
duplicate, later serial phase, or job larger than two nodes was submitted.

The batch passed canonical Python/ROCm identity and exact native source,
bundle, and G2 attestation, then failed before model load on both nodes while
materializing the immutable step-2300930 seed:

```text
ValueError: destination must be scoped by the current SLURM_JOB_ID
```

The final-seed launcher rendered the destination with a submit-side job-id
expansion rather than preserving a live batch-side `${SLURM_JOB_ID}` binding.
Consequently the materializer's fail-closed node-local destination check
rejected both tasks. No K40 step, contribution, generation, handoff, apply, or
checkpoint occurred. The clean phase therefore cannot admit the serial
loss/rejoin, invalid-result, failed-publication, or fresh-restart phases. A
reviewed launcher fix with a regression proving batch-time job-id scoping is
required before a newly authorized replacement; this attempt does not
authorize an unreviewed duplicate.

The exact scheduler record is:

```text
5059548|FAILED|1:0|batch|debug|2|00:00:33|
2026-07-23T08:48:29|2026-07-23T08:49:02|frontier[01918,09456]
```

The terminal logs are retained at:

```text
/lustre/orion/bif148/scratch/erikgarrison/emender-exact2n-final-seed-20260723T123000Z/source/logs/frontier/trainpy_async_quorum/resilient-e97-true-2n-5059548.out
/lustre/orion/bif148/scratch/erikgarrison/emender-exact2n-final-seed-20260723T123000Z/source/logs/frontier/trainpy_async_quorum/resilient-e97-true-2n-5059548.err
```

Conformance was checked against the complete Compute Pool v1 checklist and gap
matrix. Exact source, G2 integrity, fail-closed seed admission, and scheduler
binding exercise R01, R10, R13-R14, R16 and NDP02-NDP03, NDP13-NDP14,
NDP16-NDP17. Runtime acceptance under R04, R06-R07, R11-R12, R14 and NDP10,
NDP15-NDP17 remains unproven because the job failed before training. Thus zero
of five required atomic K40 generations completed; overlap, separated timing,
foreground idle, cadence, bounded latest-only recovery, invalid-result
rejection, prior-checkpoint retention, and fresh restart are all unexercised.

Date: 2026-07-20  
Task: `validate-pipelined-native-2`  
Result: **exact two-node clean phase submitted; multi-phase chain blocked by the debug-QoS submit limit**

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
