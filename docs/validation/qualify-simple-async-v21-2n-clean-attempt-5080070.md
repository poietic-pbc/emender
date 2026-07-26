# Async v2.1 exact-two-node clean qualification: non-passing attempt 5080070

Verdict: `passed=false`. Job `5080070` was the sole changed-payload clean
attempt for source `3603c85db25850a5f545b886045a76740af610b4`. Its payload
digest `29d87d9102b6a4cd523e62f7e28fdcb2459af5cfdf75c68a23df24c16e533a75`
is retired and must never be resubmitted. This attempt authorizes neither a
fault run nor scale work.

## Terminal scheduler and exact launch evidence

Queued/running monitoring used:

```text
squeue -h -j 5080070 -o '%i|%T|%P|%q'
scontrol show job -dd 5080070
```

Both commands retained exactly two allocated nodes,
`frontier[02244,02253]`, `Partition=batch`, and `QOS=debug`. Terminal
accounting used:

```text
sacct -n -X -j 5080070 \
  --format=JobIDRaw,State,ExitCode,NNodes,Partition,QOS,Start,End,Elapsed -P
```

and returned:

```text
5080070|FAILED|1:0|2|batch|debug|2026-07-26T07:25:44|2026-07-26T07:37:09|00:11:25
```

The serial controller argv was:

```text
"$EMENDER_PYTHON" scripts/frontier/run_async_v21_qualification.py \
  --gate clean --nodes 2 --repo "$SNAPSHOT" \
  --seed-config configs/frontier/e97_async_256.yaml \
  --native-build-manifest "$BUILD_MANIFEST" \
  --full-layout-gate "$G2_GATE" \
  --run-root "$RUN_ROOT" --state "$STATE_JSON" \
  --output "$MANIFEST_JSON" --submit
```

The acceptance-manifest SHA-256 is
`f55bc4729a401a1987a2d2220f42310edc43f27eafd4dd6fdceb14046b799a46`.
It binds source `3603c85d...`, native manifest
`78dd05561df181595d6c840b2e4be42a4e4a86f877bc1b62e39ed0129c636ef2`,
bundle `f19e10be...2441`, policy `fa9def95...7d98`, launcher
`70b96385...3fb7`, data `91321b2b...962`, tokenizer
`94b5ca7d...069`, and exact passed G2 job `5080048`, gate
`909084bb...337`.

G2 terminal accounting was
`5080048|COMPLETED|0:0|2|batch|debug|2026-07-26T07:21:09|2026-07-26T07:24:03|00:02:54`.
Its three timed full-layout samples had median `23.697565472` seconds,
maximum `23.871680488` seconds, and `4.176017434616889x` speedup, with zero
native retries, route errors, Python dense socket bytes, disk replay bytes, or
all-rank barriers.

## Exact seed and retained partial progress

The submit-side attestation and both compute nodes agreed on immutable step
`2300930`, tokens `150793748480`, size `7719680116`, and SHA-256
`0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2`.
Both nodes independently verified
`/tmp/emender-e97-seed-5080070/checkpoint-step-2300930.pt` offline before model
load and reported `network_fetches=0`.

All 16 trainers were alive. Generation zero reached result root
`7031b1ee58512dbc9c9bc29bd2b9444876d03a9f9953944fd69eaf0291d67207`,
exact accepted weight `5245440`, and one immutable checkpoint,
`generation-00000001-fence-05080070.pt`, `2753437091` bytes, SHA-256
`167e5ee105b5e64083bb90f8b7253b66f8c46fbbedb09d7c56040cca296838fe`.
That partial handoff is not an atomic commit and cannot satisfy the required
ten commits or twelve K40 windows per trainer.

## Exact terminal cause

Seven node-0 peers raised:

```text
TimeoutError: checkpoint leader apply deadline expired
```

The retained causal telemetry makes the ordering defect quantitative. Rank 0
spent `153.61372950021178` seconds in background result readiness, then
`42.85253917379305` seconds materializing the complete candidate result, then
`2.2889361325651407` seconds writing its immutable checkpoint. The composite
path through checkpoint write was therefore `198.75520480657097` seconds.
The peers had incorrectly started one `180`-second wait before all of those
independently bounded stages, so their deadline expired before the valid
leader release could exist.

The failing-first regression is
`test_checkpoint_leader_wait_uses_enclosing_result_preparation_deadline`.
The smallest repair gives this composite wait ADR-002's already reviewed
`420`-second freeze-to-reload-verified-result bound. It does not enlarge the
separate `180`-second result/checkpoint inner stages or the `60`-second
all-eight apply/swap bound.

## Authority mapping

The attempt was checked against the compute-pool conformance checklist,
R01–R16, NDP01–NDP17, ADR-002 V21S01–V21S17, and ISP01–ISP07. Exact source,
native bundle, scheduler queue, seed, data/tokenizer, policy/schema, capacity
declarations, result identity, and fail-closed cohort behavior were retained.
R14, NDP13/NDP15/NDP16, V21S11/V21S13/V21S15, and ISP05 remain non-passing
because seven trainers never crossed the generation-zero apply boundary.
All pass-only window, cadence, idle, commit, atomic-apply, checkpoint-reload,
and closure requirements remain unsatisfied by this failed attempt.

Machine-readable evidence is
[`reports/frontier/qualify-simple-async-v21-2n-clean-attempt-5080070.json`](../../reports/frontier/qualify-simple-async-v21-2n-clean-attempt-5080070.json).
