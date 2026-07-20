# Pipelined native DiLoCo two-node prerequisite

Date: 2026-07-20
Task: `prereq-integrate-and`
Source commit: `b66b4462`

## Result

The exact Frontier trainer now constructs one fenced
`NativeGenerationPipeline` per trainer incarnation. A native submission is
reserved and handed off only after the persistent service acknowledges its
sealed memfd retention, after which the producer slot is released. A native
result enters the latest-only mailbox only after the manager's committed result
marker has passed identity, completeness, root, weight, and finite validation;
the trainer consumes it at the K boundary. Pipeline counters are written as the
`native_generation_pipeline` stage telemetry record.

The known-good E97 model, data, ScheduleFree optimizer, K40 cadence, direct
memfd data plane, and streamed apply calls are unchanged. The integration test
was written first and failed before the runtime import/construction was added.

## Validation

Commands were run after sourcing
`scripts/frontier/activate_emender_frontier.sh` and using
`"$EMENDER_PYTHON"`:

```text
$EMENDER_PYTHON -m pytest -q tests/test_native_pipeline.py \
  tests/test_resilient_e97_true_2n_launcher.py \
  tests/test_resilient_e97_runtime.py
88 passed in 116.02s

PYTHON_BIN="$EMENDER_PYTHON" BUILD_JOBS=8 \
  bash scripts/frontier/build_native_resilient_dataplane.sh
10/10 native CTest tests passed
build/native-resilient-dataplane/native-artifacts.json recorded
```

The approved concrete two-node gate command was attempted:

```text
NDP_BUILD_MANIFEST="$PWD/build/native-resilient-dataplane/native-artifacts.json" \
  bash scripts/frontier/submit_native_dataplane_2n_gate.sh clean
exit 64: authoritative Frontier gate must be submitted from main
```

At the same time `squeue` showed job `5035347`
(`validate-native-pool-32n-third-atomic`) running on 32 nodes. The approved
submitter also forbids overlapping another user allocation. These are genuine
external launch gates: this WG branch must merge to `main`, the exact-source G2
artifact must then be generated, and the unrelated allocation must finish
before the five-generation real K40 job can be submitted. No 4+ node job was
submitted by this task.

Consequently, five live two-node generations, measured cross-generation
overlap/foreground-idle/cadence, live late-peer/loss/rejoin/replay, and
newer-fence checkpoint restart evidence are not claimed by this artifact.

## Conformance checklist

Authority: Resilient DiLoCo Compute Pool v1 and Native resilient DiLoCo data
plane v1. Applicable matrix requirements are R01-R16 and NDP01-NDP17.

- R01-R05, R09-R16: the integration preserves the existing fenced allocation,
  READY membership, bounded generation stages, strict current-generation
  admission, weighted commit, immutable checkpoint, and recovery paths.
- R06-R08 and NDP01-NDP16: the pipeline adds bounded two-slot ownership and a
  latest-only result mailbox without becoming a committer or dense transport;
  native memfd/XPMEM and bounded point-to-point owner transport remain the only
  production dense path.
- NDP17: the exact source builds and passes local native tests, but its required
  retained two-node CXI G2 artifact remains pending the authoritative-main and
  no-overlap launch gates above.
- Minimum progress remains `Q_min=2`, `T_min=3,934,080` accepted tokens for the
  approved two-node E97 gate. No launched-rank collective or central full-model
  broker was introduced.

This report is intentionally an integration/attempt record, not scale
admission. The downstream four-node rung must remain blocked until a subsequent
main-branch runner commits the missing five-generation live evidence.

## Retry on 2026-07-20

A subsequent worker resumed the committed branch at `4b9960a0` and repeated
the exact approved submission attempt with the retained manifest:

```text
NDP_BUILD_MANIFEST="$PWD/build/native-resilient-dataplane/native-artifacts.json" \
  bash scripts/frontier/submit_native_dataplane_2n_gate.sh clean
exit 64: authoritative Frontier gate must be submitted from main
```

The independent no-overlap precondition also remains false: Slurm job
`5035347` (`validate-native-pool-32n-third-atomic`) is still running on 32
nodes. The wrapper was allowed to fail closed; it was not modified or bypassed,
and no new Slurm job (including no job at four or more nodes) was submitted.

## Retry after allocation clearance on 2026-07-20

The exact approved command was attempted once more from committed source
`58bc6a30` after the unrelated allocation completed:

```text
NDP_BUILD_MANIFEST="$PWD/build/native-resilient-dataplane/native-artifacts.json" \
  bash scripts/frontier/submit_native_dataplane_2n_gate.sh clean
exit 64: authoritative Frontier gate must be submitted from main
```

The accompanying `squeue -u "$USER"` output contained no jobs, so the
no-overlap prerequisite is now satisfied. The remaining blocker is narrowly
the authoritative-source rule: this isolated worktree is on
`wg/agent-1337/prereq-integrate-and`, and the accepted launcher deliberately
permits submission only after the integration commits have merged to `main`.
The rule was not bypassed. The command exited before `sbatch`; therefore no
two-node job, and no job of four or more nodes, was submitted in this retry.

This change in scheduler state does not supply the missing live evidence. Five
committed K40 generations, pipeline overlap/idle/cadence measurements, live
failure/rejoin/replay, and newer-fence checkpoint restart remain pending a
main-branch runner using this exact source and retained native manifest.
