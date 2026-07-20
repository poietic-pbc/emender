# Pipelined native DiLoCo two-node gate attempt

Date: 2026-07-20  
Task: `validate-pipelined-native-2`  
Result: **exact two-node clean phase submitted; multi-phase chain blocked by the debug-QoS submit limit**

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
4-node-or-larger job.  At `2026-07-20T13:16:06Z`, job `5035685` remained
RUNNING at 2:29 elapsed and was still in canonical environment/runtime
startup; its stdout/stderr contained module activation only, so no completed
K40 generation was yet available to claim.

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

Two additional lineage issues remain fail-closed validation points for the
live job: the installed native manifest records source `176ae0bc...` rather
than the submitted source `08a26a0f...`, and the renderer's phase-specific
`run_dir`/`restart_from` values are recorded in JSON but are not exported as
`RUN_DIR`/`RESILIENT_E97_RESUME_HANDOFF` to each `sbatch` command.  Runtime
attestation and restart evidence must not be claimed unless the job artifacts
prove those bindings.  These observations do not justify cancelling or
mutating the accepted job.

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
