# Async v2.1 exact-two-node clean qualification: non-passing attempt 5079966

Verdict: `passed=false`. Job `5079966` was the sole changed-payload clean
attempt for source `7309551b552ff327619fcf2061a5af1d44a645cb`. It is retained
as failure evidence, does not authorize fault/restart or scale work, and its
payload digest
`bb3885709d773b4550a572e8ff9f9ff55dbe8cb4eb8f16390da4f26cece961d3`
must never be resubmitted.

## Terminal scheduler evidence

The queued/running commands were:

```text
squeue -h -j 5079966 -o '%i|%T|%P|%q|%D|%R|%M'
scontrol show job -dd 5079966
```

They retained `Nodes=2`, `Partition=batch`, `QOS=debug`, and
`frontier[09305-09306]`. Terminal accounting was:

```text
5079966|FAILED|1:0|2|batch|debug|2026-07-26T06:56:14|2026-07-26T07:03:43|00:07:29
```

The exact terminal command was:

```text
sacct -n -X -j 5079966 --format=JobIDRaw,State,ExitCode,NNodes,Partition,QOS,Start,End,Elapsed -P
```

## Exact launch bindings

The controller argv was:

```text
"$EMENDER_PYTHON" scripts/frontier/run_async_v21_qualification.py \
  --gate clean --nodes 2 --repo "$SNAPSHOT" \
  --seed-config configs/frontier/e97_async_256.yaml \
  --native-build-manifest "$BUILD_MANIFEST" \
  --full-layout-gate "$G2_GATE" \
  --run-root "$RUN_ROOT" --state "$STATE_JSON" \
  --output "$MANIFEST_JSON" --submit
```

The acceptance manifest SHA-256 is
`b33556a57b98d039d8530d5892462d364072fa1e727692f3e8f6626bbc2834cf`.
It binds:

- source commit `7309551b552ff327619fcf2061a5af1d44a645cb`;
- source digest
  `d765fcfa3d5bb3139f99b7a3ddde6f5355e78e6c696a02f75bdc474c6ac4704d`;
- native manifest SHA-256
  `95fa5eb50b2d5ea69c0bde77c76154491aa75da367e4d9977c9a34f4ae47d392`;
- passed G2 job `5079946` and gate SHA-256
  `17f81078f6a54e8974de2cd494be6d2f6f20225c59934da24107ef8bff169d32`;
- policy digest
  `fa9def95daf7bce25f1b962ca5437e7a76317b94ccfb9a710fbf126a344e7d98`;
- launcher digest
  `70b96385b5ec0795d2d1c6b6495846b20e94fe53e5256e9c53c824b65c223fb7`;
- exact canonical seed step `2300930`, accepted tokens `150793748480`,
  bytes `7719680116`, SHA-256
  `0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2`;
- offline node path
  `/tmp/emender-e97-seed-5079966/checkpoint-step-2300930.pt` and
  `network_fetches=0`.

Both nodes independently retained seed materialization attestations before
model load.

## Exact failure and preliminary metrics

All 16 real trainers reached active generation-0 K40 work. Each then raised:

```text
TimeoutError: persistent async lane admission deadline expired
```

Both atomic eight-trainer node cohorts exited under the reviewed zero-restart
policy. The attempt retained 481 supervision events, zero checkpoints, zero
atomic commits, and no completed measured K40 series. Consequently no cadence
or foreground-idle statistic is claimed for this attempt.

The phase evidence localizes the defect. Every coherent 2,753,385,248-byte
endpoint capture itself met the one-second policy: 16 events, minimum
0.413974589 s, median 0.446390327 s, p99/maximum 0.555780250 s. The runtime
then performed a full-state digest and causal-telemetry persistence before
local `OWNED` and the next lane start. That work consumed the remainder of the
same one-second admission deadline.

The smallest fix is covered by the failing-first regression
`test_snapshot_admission_deadline_excludes_telemetry_io`: capture and admission
completion are timestamped, the next mutable lane owns state first, telemetry
is persisted from the frozen timestamps afterward, and the already-verified
native generation base digest replaces the redundant full-state hash.

## Authority mapping

This attempt was evaluated against
`RESILIENT_DILOCO_COMPUTE_POOL.md` version 1, R01–R16; NDP01–NDP17;
ADR-002 v2.1, V21S01–V21S17; and ISP01–ISP07. Exact identities, scheduler
binding, native G2, offline seed verification, real trainer count, bounded
capacity policy, and failure containment were present. ISP02/ISP06 and thus
R14, NDP13/NDP16, V21S06/V21S13/V21S15 failed because `OWNED` was not reached
within one second. The remaining pass-only performance, commit, checkpoint,
atomic-apply, and fresh-process criteria are not satisfied by a generation-0
failure.

Machine-readable evidence is
[`reports/frontier/qualify-simple-async-v21-2n-clean-attempt-5079966.json`](../../reports/frontier/qualify-simple-async-v21-2n-clean-attempt-5079966.json).
