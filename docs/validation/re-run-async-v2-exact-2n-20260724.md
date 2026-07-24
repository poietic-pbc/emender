# Async-v2 exact-two-node Frontier acceptance rerun

Date: 2026-07-24

Task: `re-run-async`

Status: **IN PROGRESS — no scale-out is authorized.**

## Authority

This rerun conforms to `docs/RESILIENT_DILOCO_COMPUTE_POOL.md` version 1,
`docs/RESILIENT_DILOCO_GAP_MATRIX.md`, and
`docs/ASYNC_DECOUPLED_DILOCO_V2.md` ADR-002. The applicable requirements are
the complete sets **R01–R16**, **NDP01–NDP17**, and **V2A01–V2A18**.

The reviewed order is strictly serialized:

1. clean overlap;
2. fault/rejoin;
3. invalid-result rejection;
4. failed checkpoint publication;
5. fresh-allocation restart.

Every phase requests exactly `Nodes=2`, `Partition=batch`, and `QOS=debug`.
No 4+ node job is permitted. A clean-phase failure stops the sequence without
submitting a later phase.

## Merged source and exact identities

The two terminal fixes were reconciled with and fast-forward pushed to
`origin/main`. The authoritative clean branch-`main` checkout is:

```text
path              /lustre/orion/bif148/proj-shared/emender/source-snapshots/emender-59101f42d23b-main
source commit     59101f42d23b13dc6b11357461adb79cfd6704c6
source tree       209cf43449b2f3077ece63643fcf91aec0b75871
source inventory  225556244a5b2fde82c0db902396fc2bcb2e4ffd25ae1f1118fe840a52c9f644
native bundle     9884a02d84bd9560a15314c26e386868350b865e688cc4c701c802f4f686227a
```

The authoritative run root is:

```text
/lustre/orion/bif148/proj-shared/emender/validation/re-run-async-59101f42d23b
```

The current-source native build passed all 10 CTests. Its manifest is
`native-g2-install/native-artifacts.json`, SHA-256
`ce09c15cef51ed1bed730f8db6b5cfdc0b7e3ec502be4fad55724a8b63db7b98`.

## Refreshed G2

Current-source full-layout G2 job `5066119` completed `0:0` in `00:03:02`.
Terminal accounting reports `AllocNodes=2`, `Partition=batch`, and
`QOS=debug`. The passing immutable gate is
`native-g2-evidence/5066119/full-layout-gate.json`, SHA-256
`6637cbc1fea5876161572d47b5d92e567f3ecd153bdf494005db29634e4f86ac`.
It binds provider `cxi`, source commit `59101f42...`, and native bundle
`9884a02d...`.

## Final E97 seed

The controller resolved the immutable S3 step manifest and latest discovery
pointer on the submit host, then verified the complete cached checkpoint
before submission:

```text
step        2300930
tokens      150793748480
bytes       7719680116
SHA-256     0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2
attestation 27e234891df02b64b9db77fc784c341e5a3ae6e87418b8f1af167776d1d710bb
```

Each allocation must receive this verified file via `sbcast`, re-hash it on
each node while offline, and report `network_fetches=0`.

## Serialized acceptance

The controller rebuilt the authoritative native stage (10/10 CTests), matched
the G2 bundle exactly, and attested the current source before submitting only
the clean phase:

```text
acceptance manifest  e53dc0d2f4211dabc0052adf278a91a7ab239c79c0f98c78927bfa16e861efd6
authoritative stage  a65b505768ab854a92c48633b5876031f8358a4e584d0a6e18a260ab730519c0
native manifest      ce09c15cef51ed1bed730f8db6b5cfdc0b7e3ec502be4fad55724a8b63db7b98
rendered sbatch      2a5a19144be83b7e207a80b1528a3b3d34d91c76c2208355cc504f5d59c1681e
```

Clean-overlap job `5066162` requests exactly two nodes, `batch`, and `debug`.
It is currently pending for scheduler priority. The state has
`next_phase=0`, `active.phase=clean-overlap`, `history=[]`; therefore no later
phase has been submitted.

## Commands

All Python and native commands followed:

```bash
source scripts/frontier/activate_emender_frontier.sh
```

The native prerequisite used
`scripts/frontier/build_native_resilient_dataplane.sh` followed by:

```bash
REPO="$SNAPSHOT" \
NDP_BUILD_MANIFEST="$VALIDATION_ROOT/native-g2-install/native-artifacts.json" \
NDP_ARTIFACT_ROOT="$VALIDATION_ROOT/native-g2-evidence" \
NDP_PYTHON_BIN="$EMENDER_PYTHON" \
scripts/frontier/submit_native_dataplane_2n_gate.sh clean
```

The real serialized controller uses:

```bash
"$EMENDER_PYTHON" scripts/frontier/render_resilient_e97_exact_2n_acceptance.py \
  --repo "$SNAPSHOT" \
  --native-build-manifest "$VALIDATION_ROOT/native-g2-install/native-artifacts.json" \
  --full-layout-gate "$VALIDATION_ROOT/native-g2-evidence/5066119/full-layout-gate.json" \
  --run-root "$VALIDATION_ROOT/acceptance-runs" \
  --output "$VALIDATION_ROOT/acceptance-manifest.json" \
  --state "$VALIDATION_ROOT/acceptance-state.json" \
  --native-stage-root "$VALIDATION_ROOT/acceptance-native-stage" \
  --submit
```

## Verdict

The terminal verdict is pending the clean gate. Until all criteria pass,
including honest base/commit/apply lag, at least five sequential local K40
windows, bounded queues/memory, cadence at most `1.25x` raw K, foreground idle
below 10%, local `OWNED` at most one second, leased/dynamic READY membership,
all rejection/recovery/publication/restart cases, and immutable artifacts,
**scale-out is explicitly denied**.
