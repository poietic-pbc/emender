# Async-v2 result-root token accounting correction

Date: 2026-07-24

Task: `fix-async-v2-2`

## Result

Job 5066162 exposed a control-plane identity error after generation 0 froze
two READY contributions. The frozen admission set correctly recorded
`accepted_tokens=5,245,440`, while the native result correctly carried the
distinct ADR-002 lag-zero aggregation weight
`5,245,440 * (7 - 0) = 36,718,080`. Result-root consensus incorrectly compared
the latter with the former and failed before publication.

`PoolControlServer._result_root` now derives the expected native weight from
the immutable frozen identities and their admitted `aggregation_weight`
values, exactly as the shard-owner result gate already does. It still requires
every frozen manager to report the same nonzero root, weight, and byte count.
It fails closed if the frozen sum is nonpositive or if the reported result
weight differs. The accepted-token clock remains independently bound by the
generation close result and is not inferred from native aggregation weight.

The regression uses the job 5066162 envelope:

| field | value |
|---|---:|
| READY/frozen contributions | 2 |
| exact accepted tokens | 5,245,440 |
| async-v2 commit lag | 0 |
| native aggregation weight | 36,718,080 |
| minimum token floor | 3,934,080 |

It proves that both managers validate the native result at weight 36,718,080,
that reporting 5,245,440 as the native weight still raises
`native result-root token accounting mismatch`, and that owner-result
validation rejects zero, the token total, and both adjacent off-by-one
aggregation weights.

## Authority and conformance

This correction conforms to
`docs/RESILIENT_DILOCO_COMPUTE_POOL.md` version 1 and ADR-002 in
`docs/ASYNC_DECOUPLED_DILOCO_V2.md`.

- **R04-R08:** frozen fenced identities remain the accounting authority;
  exact token weighting, floors, atomicity, deterministic owners, and bounded
  native transport are unchanged.
- **R14-R16:** the result gate remains bounded and fail-closed, preserves
  committed-token evidence, and authorizes no rung beyond exactly two nodes.
- **NDP05-NDP10:** native arithmetic continues to use the exact separately
  admitted aggregation weights; frame/root identity, credits, checksums,
  idempotence, and corruption rejection are not weakened.
- **NDP15-NDP17:** publication still follows validated consensus and current
  fence; this code correction does not itself constitute a current-source G2
  or a real-model Frontier acceptance result.
- **V2A03-V2A07:** canonical v2 identity and clocks remain separate; exact
  tokens advance the token clock, while `tokens * (7 - commit_lag)` controls
  aggregation and the half-step outer equation.
- **V2A13-V2A18:** conflicting replay and invalid weights still fail before
  publication; atomic bundle/reload, compiled P2P transport, honest
  decoupling telemetry, correctness latency, and the two-node-only promotion
  ladder are unchanged.

No 4+ node submission or scale authorization was made.

## Validation

All commands used the canonical Frontier environment:

```bash
source scripts/frontier/activate_emender_frontier.sh
```

Focused regression:

```text
$EMENDER_PYTHON -m pytest -q tests/test_resilient_pool_runtime.py \
  -k async_v2_owner_results
1 passed, 6 deselected
```

Focused Python suites:

```text
$EMENDER_PYTHON -m pytest -q \
  tests/test_resilient_pool_runtime.py \
  tests/test_async_diloco_v2.py \
  tests/test_resilient_e97_runtime.py
60 passed
```

Current-source native build and suite:

```text
scripts/frontier/build_native_resilient_dataplane.sh
10/10 CTests passed
```

The first Python-suite attempt found the native library absent; after the
required current-source native build, the complete focused suite passed.
`git diff --check` also passed.

The next scheduler action remains serialized and fail-closed: produce a
current-commit full-layout G2 artifact, explicitly verify `Nodes=2`,
`Partition=batch`, and `QOS=debug`, and only then rerun the exactly-two-node
clean admission. No clean admission may reuse the pre-fix G2 identity.
