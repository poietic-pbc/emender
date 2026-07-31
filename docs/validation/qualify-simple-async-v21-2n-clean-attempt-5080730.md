# Async v2.1 exact-two-node clean qualification: non-passing attempt 5080730

Verdict: `passed=false`. Job `5080730` was the sole changed-payload clean
attempt for exact source `e2832f2b796b153b1177a3fc9ab6ad0b3ed3ec38`. Its
payload digest
`534c40b0cbace765ff06bca1b2ebab63dbafcd47ec2f450cad8be5db2a7d3331`
is retired and must never be resubmitted. This attempt authorizes neither a
fault run nor scale work.

## Terminal scheduler and exact launch evidence

Queued and running monitoring retained:

```text
squeue -h -j 5080730 -o '%i|%T|%P|%q'
scontrol show job -dd 5080730
```

The records prove exactly two allocated nodes,
`frontier[06179-06180]`, `Partition=batch`, `QOS=debug`, and
`Restarts=0`. Terminal accounting used:

```text
sacct -n -X -j 5080730 \
  --format=JobIDRaw,State,ExitCode,NNodes,Partition,QOS,Start,End,Elapsed -P
```

and returned:

```text
5080730|FAILED|143:0|2|batch|debug|2026-07-26T10:13:04|2026-07-26T10:29:51|00:16:47
```

The canonical serial-controller argv was:

```text
"$EMENDER_PYTHON" "$SNAPSHOT/scripts/frontier/run_async_v21_qualification.py" \
  --gate clean --nodes 2 --repo "$SNAPSHOT" \
  --seed-config "$SNAPSHOT/configs/frontier/e97_async_256.yaml" \
  --native-build-manifest "$BUILD_MANIFEST" \
  --full-layout-gate "$G2_GATE" \
  --run-root "$RUN_ROOT" --state "$STATE_JSON" \
  --output "$MANIFEST_JSON" --submit
```

The acceptance-manifest SHA-256 is
`35efd7e133d3fafa1288eda5c5aae693ba0b9bfd46f137924db97a7795b1af8e`.
It binds exact source `e2832f2b...ec38`, source digest
`c6d53885...fc0b`, native build-manifest SHA-256
`9a3d58ca...4aaa`, native bundle `f19e10be...2441`, policy
`fa9def95...7d98`, launcher `70b96385...3fb7`, data
`91321b2b...7962`, tokenizer `94b5ca7d...2069`, and passed G2 job
`5080660` with gate SHA-256 `44d7829c...f60`.

G2 terminal accounting was
`5080660|COMPLETED|0:0|2|batch|debug|2026-07-26T10:03:40|2026-07-26T10:06:35|00:02:55`.
Its median full-layout sample was `23.512473889` seconds, maximum
`23.840366785` seconds, and speedup `4.20889129048x`, with zero route
errors, CQ errors, Python dense socket bytes, disk replay bytes, or all-rank
barriers. The 111 bounded transport retries were retained rather than
excluded, and the exact-source native gate passed.

## Exact seed

The submit-side attestation and both compute nodes agreed on immutable step
`2300930`, accepted tokens `150793748480`, size `7719680116`, and SHA-256
`0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2`.
Both nodes independently verified
`/tmp/emender-e97-seed-5080730/checkpoint-step-2300930.pt` offline before
model load and reported `network_fetches=0`.

## Exact terminal cause and preliminary interleave metrics

The new all-eight ready/release barrier worked fail closed: no manager began a
foreground x/z correction without eight reload-verified candidates. However,
the prior repair had removed the capacity-one reader credit for the single
shared node aggregate. All eight trainers on each node consequently attempted
to materialize the same 5,506,770,496-byte result concurrently.

Node-zero rank zero alone completed in `43.095` seconds and retained
`native-apply-ready-00000000-00.json`. Its sibling ranks took
`265.881`, `266.850`, `266.746`, `266.342`, `261.952`, `267.428`, and
`266.264` seconds. Node-one ranks took `298.632`, `301.168`, `308.083`,
`306.712`, `305.326`, `300.188`, `306.910`, and `304.189` seconds. Fifteen
trainers therefore failed with
`TimeoutError: native_result_materialize exceeded 60.0s stage SLO`.
No `native-apply-release`, `native-applied`, or complete node-applied marker
was emitted. This was a real, inclusive failure; no unhealthy or
backpressured interval was selected out.

Before the failure, every one of the 16 real trainers completed exactly nine
K40 windows, for 144 total. Preliminary raw K40 median/max were
`68.138536907034`/`71.76527546904981` seconds; K-boundary cadence
median/max were `68.38901302893646`/`71.7786946343258` seconds
(`1.0036759832727875x`). Honest foreground idle was
`0.00007584119568409718`; median/max inter-window gaps were
`0.001856392715126276`/`0.04096293402835727` seconds. Endpoint snapshot
duration had count/median/max `16`/`0.0090716597624`/`0.0238479534164`
seconds, snapshot admission had `16`/`0.002698626835`/`0.035389152821`
seconds, and direct native memfd materialization had median/max
`34.9636`/`35.72215` seconds before the contested apply-result read. These
are not pass metrics: no trainer retained the required 12 windows and the run
retained only one atomic commit.

The immutable checkpoint
`generation-00000001-fence-05080730.pt` is `2753437091` bytes with SHA-256
`87cb9b359d80e2167d0137cab57aa25f0cd110683ea035376706c23c2a99d49b`.
Its ready receipt independently reload- and CAS-verified result root
`dab51ca5...35fc1`.

The failing-first regression is
`test_native_all_eight_apply_releases_only_after_bounded_serial_preparation`.
The smallest repair restores a capacity-one, authenticated, node-local reader
credit before each trainer materializes the one shared aggregate. Each reader
must finish within the unchanged 60-second materialization SLO; its durable
checkpoint, hash, and reload verification may overlap the following reader.
Only after all eight reload-verified ready receipts match does the manager
emit the all-eight release and begin the separate reviewed 60-second
foreground atomic-apply interval. The manager's already reviewed
`result_preparation` progress stage remains bounded at 420 seconds.

## Authority mapping

The attempt was checked against the compute-pool conformance checklist and
R01–R16, native NDP01–NDP17, accepted ADR-002 V21S01–V21S17, and
ISP01–ISP07. Exact source, native bundle, queue identity, seed,
data/tokenizer, policy/schema, launcher, capacity declarations, native owner
transport, bounded snapshot ownership, continuous training, immutable
checkpoint, and fail-closed supervision were retained. R14,
NDP13/NDP15/NDP16, V21S03/V21S11/V21S13/V21S15, and ISP05 remain
non-passing because only one immutable commit and nine windows per trainer
were reached and the all-16 atomic apply was incomplete.

Machine-readable evidence is
[`reports/frontier/qualify-simple-async-v21-2n-clean-attempt-5080730.json`](../../reports/frontier/qualify-simple-async-v21-2n-clean-attempt-5080730.json).
