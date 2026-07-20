# Pipelined native DiLoCo scale-ladder admission record

Date: 2026-07-20 UTC

WG task: `validate-pipelined-native`

Source inspected: `6a24618d` (`feat: implement-pipelined-native`)

## Result

**Stopped before the four-node submission.** No 4-, 8-, 32-, 64-, or
larger-node allocation was submitted by this task. This is the required
fail-closed outcome because the prerequisite two-node runtime acceptance does
not exist for the pipelined implementation and the pipeline is not connected
to the Frontier runtime.

The upstream implementation record explicitly states that its two-node job
was not submitted and that the five-generation performance/failure gate,
foreground-idle target, and cadence target remain outstanding. Source search
at `6a24618d` also finds `ndm/native_pipeline.py` used only by
`tests/test_native_pipeline.py` and the implementation report. None of
`scripts/frontier/resilient_e97_role.py`,
`scripts/frontier/resilient_e97_allocation_supervisor.py`, or
`scripts/frontier/resilient_e97_true_2n.sbatch` constructs a
`NativeGenerationPipeline`. Consequently, a launch through the existing
Frontier recipe would exercise the earlier serial native runtime, not the
accepted pipelined implementation.

Older four- and eight-node artifacts are intentionally not promoted as
evidence. They predate commit `6a24618d`, identify different final source
commits, and do not measure the pipeline's snapshot handoff, latest-only
mailbox, safe-boundary application, foreground wait, or overlap cadence.

At the admission checkpoint, `squeue -u "$USER"` contained no jobs. Scheduler
availability therefore was not the blocker. Submission was withheld solely to
obey the ordered scale rule: a four-node rung may follow only an accepted
two-node rung of the exact implementation.

## Concrete validation attempted

The canonical Frontier environment was activated before Python and native
build commands, as required by the project guide.

```text
source scripts/frontier/activate_emender_frontier.sh
bash scripts/frontier/build_native_resilient_dataplane.sh

Result: build passed; native ctest 10/10 passed; a clean local artifact
manifest was recorded at build/native-resilient-dataplane/native-artifacts.json.
```

The first combined pytest invocation correctly failed setup because it was
made before the local service artifact existed. After building and binding the
three explicit local artifacts, the complete accepted suite passed:

```text
export EMENDER_NDP_SERVICE="$PWD/build/native-resilient-dataplane/bin/ndp_cxi_service"
export EMENDER_NDP_LIBRARY="$PWD/build/native-resilient-dataplane/lib64/libemender_ndp.so"
export EMENDER_NDP_TRANSPORT_LIBRARY="$PWD/build/native-resilient-dataplane/lib64/libemender_ndp_transport.so"
$EMENDER_PYTHON -m pytest -q tests/test_native_pipeline.py \
  tests/test_native_dataplane_failure.py tests/test_native_pool_integration.py

Result: 33 passed in 24.60s.
```

These local results prove that the isolated pipeline policy and existing
compiled service remain internally correct. They do not substitute for live
K40 overlap measurements and therefore do not authorize a four-node launch.

## Outstanding admission evidence

Before this ladder can resume, the exact source must:

1. connect `NativeGenerationPipeline` to the persistent Frontier
   trainer/manager service path without moving dense bytes into Python;
2. run the two-node gate for at least five generations and retain stage timing
   for K40 compute, snapshot handoff, collection/ownership, redistribution,
   integrity, checkpoint publication, safe-boundary apply, GPU foreground
   idle, queue depth/staleness, useful/wire bytes, and cadence;
3. prove generation-g outer work overlaps generation-(g+1) local compute,
   foreground control-plane idle is below 10% when work fits inside K40, and
   steady cadence is at most 1.25 times measured K40 compute;
4. retain latest-only bounded queues and reject stale, partial, corrupt,
   non-finite, and obsolete-fence results; and
5. exercise the required delayed peer, node-peer loss/rejoin, bounded replay,
   and restart from a committed checkpoint.

Only a passing, committed, pushed two-node artifact may admit the four-node
rung; the same ordered five-generation/five-generation/three-generation gates
then apply to 4/8/32 nodes. No 64-node submission is part of this task.

## Architecture conformance checklist

This admission decision conforms to *Resilient DiLoCo Compute Pool*, version
1, and the companion gap-matrix requirements R01-R16 and NDP01-NDP17.

- R01-R08 and R11-R13: no READY membership, fenced five-generation commit
  chain, bounded wait, atomic checkpoint chain, or restart evidence exists for
  the integrated pipeline, so no higher rung was admitted.
- R09-R10 and NDP01-NDP14: the local test/build pass preserves trainer model
  ownership and the compiled bounded non-Lustre data plane, but source
  inspection shows the new metadata pipeline is not yet wired into it.
- R14-R16 and NDP16-NDP17: unit counters exist, but required live queue,
  foreground-idle, cadence, ownership, and wire-volume telemetry does not.
- The minimum progress floor cannot be claimed from a simulation or an older
  source revision. There is no launched-rank/all-rank inference and no
  central-broker evidence because no scale job was launched.

This record is an admission artifact, not a passing scale artifact.

## Retry audit

The task was automatically retried at 2026-07-20 11:41 UTC while the newly
created `prereq-integrate-and` task was still open.  The retry did not weaken
the admission rule or submit a scale rung.  At 11:44 UTC, the scheduler showed
job `5035341` (`native-ndp-g2-fault`) running on exactly two nodes.  A running
prerequisite job is not an accepted two-node artifact: its five committed
generations, overlap and cadence gates, fault/rejoin behavior, checkpoint
restart, telemetry, source identity, commit, and push must all be verified
before this task may submit four nodes.

The prerequisite is recorded as a graph dependency of this ladder.  An
accidental reciprocal edge visible at the start of the retry was reconciled
by WG before mutation, leaving `prereq-integrate-and` able to run ahead of this
task.  No 4-, 8-, 32-, or 64-node job was submitted during this retry audit.
