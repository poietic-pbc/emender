# Async v2.1 two-node clean attempt 5080902

## Verdict

Job `5080902` is immutable, non-passing evidence. It ran exactly two nodes
(`frontier[03026,03032]`) on `Partition=batch`, `QOS=debug`, and terminated
`FAILED|1:0` after `00:16:33`. Payload
`d5665474f3b080be376ee2a05475e5ad52015bb11b1397d8c1b23ec4ea8845d7`
is retired and must never be resubmitted.

The terminal accounting command and record were:

```bash
sacct -n -X -j 5080902 \
  --format=JobIDRaw,State,ExitCode,NNodes,Partition,QOS,Start,End,Elapsed -P
5080902|FAILED|1:0|2|batch|debug|2026-07-26T10:55:49|2026-07-26T11:12:22|00:16:33
```

Queued, running, and terminal scheduler transcripts are under
`/lustre/orion/bif148/scratch/erikgarrison/emender-qualification/qualify-simple-async-v21-2n-clean/24ae4dd64f42213cf5dc7cf62038c4594cf3f5b8/scheduler-evidence/clean-5080902`.
They retain the required `squeue -o '%i|%T|%P|%q'`, `scontrol show job -dd`,
and terminal `sacct` results. The first queued transcript is SHA-256
`8de8ed18a73e950f1bdb29e7abcac36651d9c786d9dfe31f3f7ebca16b4cb716`;
the first running transcript is
`ab26a37f361619079a4c1c933c15d28b1786e5fa0ed29d0c6c4b3e42dac536b0`;
the terminal transcript is
`2016c46be69bbc0f5c34c514c041b014a6d6d467e2d76ee31aa4cbadcb268297`.

## Exact identity

The canonical controller submitted the exact argv retained in the plan:

```bash
"$EMENDER_PYTHON" scripts/frontier/run_async_v21_qualification.py \
  --gate clean \
  --nodes 2 \
  --repo "$SNAPSHOT" \
  --seed-config configs/frontier/e97_async_256.yaml \
  --native-build-manifest "$BUILD_MANIFEST" \
  --full-layout-gate "$G2_GATE" \
  --run-root "$ROOT/clean/clean-overlap" \
  --state "$STATE_JSON" \
  --output "$ROOT/clean-qualification-plan.json" \
  --submit
```

The plan SHA-256 is
`c32893b238efe12662ea9a24ab68717689863fe9e89d3606b32ce99ede79c675`.
It binds:

- source `24ae4dd64f42213cf5dc7cf62038c4594cf3f5b8`, source digest
  `b7a21e02224571b6b46b81a1e0b54040d749b33b398d10f3b82561aab0c2ecfa`;
- native build manifest
  `ed60e10b1fb1026e0b79fa81b98c421eeaacdbe572fc6d2a3afd3d15c2b802d9`
  and bundle
  `f19e10be9987cfdb551a8dd75c5c88145c3cf35b73c54d3898fe562ce4182441`;
- exact-source G2 gate
  `a696ffd22c4ac0af4dd76553770f5b2816ceba58dcb7022086e3e2b376f837cd`;
- policy
  `fa9def95daf7bce25f1b962ca5437e7a76317b94ccfb9a710fbf126a344e7d98`,
  launcher
  `70b96385b5ec0795d2d1c6b6495846b20e94fe53e5256e9c53c824b65c223fb7`,
  data
  `91321b2b90bb159f3aa73881455778f10e8df588edd526b1066281fa72997962`,
  tokenizer
  `94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069`,
  and train arguments
  `afc2a65fd8c73499e74e21cb9531c978206c3a9c898e42d18cc58bb93eb9fe9c`.

Both nodes independently verified, offline before model load, the exact seed
at `/tmp/emender-e97-seed-5080902/checkpoint-step-2300930.pt`: step `2300930`,
tokens `150793748480`, bytes `7719680116`, SHA-256
`0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2`,
and `network_fetches=0`. The seed authority attestation SHA-256 is
`27e234891df02b64b9db77fc784c341e5a3ae6e87418b8f1af167776d1d710bb`.

## Exact terminal cause

All 20 roles started without a process or scheduler restart. Both managers
failed closed with `reason=first_atomic_generation_deadline`:

- `node-1-manager` at `1785078738.1051574`;
- `node-0-manager` at `1785078739.1381934`.

This was not a missing commit. Generation 1 had already advanced through
immutable receipt
`e3cee1514dfd29ba728316016a6877b6313784c6467d59990989cef357757678`
at `1785078419221517000 ns`, with accepted-token clock `150798993920`,
result root
`67b0516f69df1e505d97f035efc1c4185ce3f7239bdefe4061a23584aedc6845`,
and checkpoint SHA-256
`90edf36b44b2963acc0ac3168aa72cc1ab13206307dad01618f731db18961b91`.

The defect was a semantic conflation in the live supervisor. The accepted
`720 s` authority is “allocation to first committed `latest`,” but the
supervisor recognized only the later `published_node_applied` marker. The
managers had verified the commit through native peer control and were still in
bounded `peer_apply`; because the intervening heartbeat did not retain the
commit receipt, the supervisor incorrectly declared that no first commit
existed.

## Preliminary interleave and idle metrics

Capacity-one serial result preparation itself succeeded:

- 16/16 `native_result_materialize` intervals completed in
  `42.053830–43.078616 s` (median `42.473577 s`) under the `60 s` bound;
- 16 authenticated result-materialized receipts, 16 apply-ready receipts, and
  both node release receipts exist;
- the unrelated supervisor deadline interrupted the two foreground
  transactions before any complete `native-applied` receipt, approximately
  `19–23 s` into their independent `60 s` apply allowance.

The strict performance validator cannot pass this truncated attempt because
every real trainer retained exactly 9 complete K40 windows rather than the
required 12. Applying the validator’s last-window timing semantics as
preliminary, explicitly non-promotion evidence gives:

- 144 raw K40 samples: median `66.350352 s`, maximum `70.739319 s`;
- 128 cadence samples: median `66.213755 s`, maximum `70.758810 s`;
- median cadence/raw ratio `0.997941`;
- aggregate foreground idle `0.000069021`, p99 `0.041617 s`, maximum
  `0.057848 s`.

The machine-readable companion report is
`reports/frontier/qualify-simple-async-v21-2n-clean-attempt-5080902.json`.

## Validation

- R01–R16: retained as a non-passing, exact-scheduler attempt; no promotion
  claim is made.
- NDP01–NDP17: current-source G2 and live native execution remain exact, but
  the clean semantic gate did not complete.
- V21S01–V21S17: the run ended before ten commits, the required K40 windows,
  all-eight apply receipts, and fresh-process checkpoint correctness could be
  validated.
- ISP01–ISP07: serial immutable preparation completed; the external
  first-commit diagnostic interrupted final atomic apply.

This report is failure evidence and a regression input, never authorization to
call the clean gate passed.
