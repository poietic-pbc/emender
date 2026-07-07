# Train.py-native async quorum DiLoCo design

Date: 2026-07-07  
Task: `design-train-py`  
Target: Emender E97 1.3B scaleout on Frontier

## Decision

Implement async quorum DiLoCo as a **train.py-native** distributed mode with
**one train.py training rank per GPU**. This is not DDP. In the main path, the
model must not be wrapped in `DistributedDataParallel`, and there must be no
per-step gradient all-reduce. Each GPU rank remains a single-GPU learner that
does local optimizer steps on its own token stream, then contributes an update
to an outer async/quorum merge protocol.

The first supported train.py-native path should be:

- one process per GPU, normally 8 ranks per Frontier node;
- local training for `K = --diloco_k` optimizer steps;
- delta-form update semantics against a named global generation;
- reject-stale quorum merge;
- stateless averaged outer optimizer first (`avg`, `eta_outer=1.0` default);
- durable global checkpoints and small manifests owned only by the global
  merger;
- dense per-generation update files avoided outside debug tests.

The design intentionally preserves the current single-GPU learner shape inside
`train.py`: model construction, optimizer stepping, data loading, loss logging,
held-out scoring, fused E97 guards, and schedule-free train/eval handling should
remain train.py code paths. The async layer replaces only the periodic
all-rank DiLoCo merge/checkpoint ownership path.

## Current systems compared

### Synchronous train.py DiLoCo

Current `train.py --diloco` already avoids per-step DDP gradient all-reduce:
`use_ddp = dist_enabled and not args.diloco`; `use_diloco = dist_enabled and
args.diloco`. In DiLoCo mode it broadcasts rank-0 initial weights, keeps each
rank independent for K optimizer steps, and calls `diloco_merge(...)` every K
steps. `diloco_merge` all-reduces flattened model or ScheduleFree tensor state
across all ranks and applies optional outer optimizer state.

Strengths to reuse:

- CLI and operational precedent for "distributed but not DDP".
- Correct ScheduleFree handling in `diloco_merge`: switch to eval for x,
  merge x and z, preserve scalar clocks locally, and translate z with x for
  nontrivial outer movement.
- Outer-state checkpoint/resume guard for synchronous `avg`, `momentum`, and
  `sfsgd`.
- Rank-0 checkpoint writing with atomic `latest.pt` symlink replacement.
- Existing final-checkpoint walltime/signal coordination concepts.

Limits that must change:

- The merge is a static all-rank collective. Slow/missing ranks hang or abort
  the round.
- Every rank receives the consensus by collective side effect; there is no
  explicit generation state, quorum, or late-update handling.
- Rank 0 owns train.py checkpoints, but async needs a global merger role that
  may not be a learner rank.
- Synchronous `momentum` and `sfsgd` outer state assumes a strict global round
  boundary with all ranks participating; this is not the safest initial async
  baseline.

### Existing async/quorum harness

`ndm/async_diloco.py` already provides pure quorum state math,
`AsyncDiLoCoUpdate`, `quorum_merge`, metrics objects, and
`AsyncDiLoCoCheckpointManager`. It correctly models accepted/stale/timed-out/
failed/invalid updates and defers advancement when quorum is missed. It also
has an authoritative global-merger-only latest publisher and ignores
non-authoritative cache manifests for resume.

`ndm/async_diloco_real.py` and Frontier scripts show a real-token prototype
direction, but prior quality-pass findings noted that parts of the real path
were serial/in-process or wrapper-dependent rather than a true one-rank-per-GPU
train.py mode.

Strengths to reuse:

- `AsyncDiLoCoUpdate` metadata shape.
- `quorum_merge` semantics for reject-stale avg merges.
- `AsyncDiLoCoGenerationMetrics` and JSON/JSONL summary helpers as the seed for
  the log schema.
- `AsyncDiLoCoCheckpointManager` role guard, finalized generation manifests,
  and latest selection logic.
- Unit tests around quorum deferral, stale rejection, and checkpoint latest
  safety.

Limits that must change:

- The production path must be reached from `train.py`, not by a side harness
  that simulates many ranks in one process.
- Tensor transport must become a rank-local streaming/sharded data plane. Dense
  Lustre `update.pt` files are acceptable for unit/debug tests only.
- Metrics need rank/GPU progress fields and train.py loss windows, not only
  synthetic worker summaries.
- The checkpoint publisher must eventually write loadable train.py checkpoint
  payloads, not only manifest placeholders.

### Proposed train.py-native async quorum path

Add a new explicit mode, for example `--async_quorum_diloco`, separate from the
existing synchronous `--diloco` all-reduce path. The mode should still use the
same local training loop and `--diloco_k`, but K-boundary behavior becomes:

1. Ensure the learner has a named global base generation `g` loaded.
2. Train locally for K optimizer steps or until a finalization/stop condition.
3. Extract a delta/update from local ScheduleFree/model state against `S_g`.
4. Submit update metadata and tensor shards to the node supervisor/global merge
   protocol.
5. Continue only after receiving either `S_{g+1}` or a rebase/resync command.

For the first implementation, the local learner can block at the K boundary
until the next global state is available. That is still async/quorum at the job
level because late or missing ranks do not block the global generation; it also
keeps the single-GPU learner simple and avoids unbounded local drift in v1.
Later work can allow speculative local training across open generations.

## State and math

Let the authoritative global generation be:

```text
S_g = (X_g, Z_g, C_g, O_g)
```

where:

- `X_g` is the exported model/eval tensor basis. For ScheduleFree this is the
  x/eval basis held in parameters after `optimizer.eval()`. For AdamW it is the
  model parameter tensor state.
- `Z_g` is the ScheduleFree z tensor state. It is absent for non-ScheduleFree
  optimizers.
- `C_g` is scalar metadata: generation id, global train step, token counters,
  quorum policy, loss windows, wall-clock fields, and schedule-free scalar
  clocks that are tracked but not globally reduced in the first path.
- `O_g` is outer optimizer state. In v1 this is empty for `avg`. Non-avg outer
  state is deferred.

A learner rank `r` pulls `S_g`, loads it into train.py model/optimizer state,
and runs K local optimizer steps on its rank-specific token stream:

```text
S_{r,g}^{local} = F_r(S_g, K)
```

The submitted update is logically:

```text
U_{r,g} = {
  rank_id: r,
  node_id,
  base_generation: g,
  local_step_start,
  local_step_end,
  tokens,
  loss_window,
  delta: D_{r,g}
}

D_{r,g} = S_{r,g}^{local}.tensors - S_g.tensors
```

For ScheduleFree:

```text
dX_{r,g} = X_{r,g}^{local} - X_g
dZ_{r,g} = Z_{r,g}^{local} - Z_g
```

For AdamW:

```text
dW_{r,g} = W_{r,g}^{local} - W_g
```

The accepted quorum set for generation `g` is:

```text
Q_g = { U_{r,g} | U accepted and U.base_generation == g }
```

The initial quorum rule is reject-stale:

```text
accept(U) iff U.base_generation == current_open_generation
reject stale iff U.base_generation < current_open_generation
reject future/invalid iff U.base_generation > current_open_generation
```

The v1 avg outer merge is:

```text
mean_delta_g = weighted_mean(D_{r,g}, r in Q_g)
S_{g+1}.tensors = S_g.tensors + eta_outer * mean_delta_g
```

Default `eta_outer = 1.0`, so if every rank participates from the same base:

```text
X_{g+1} = mean_r X_{r,g}^{local}
Z_{g+1} = mean_r Z_{r,g}^{local}
```

This is equivalent to synchronous train.py DiLoCo's all-rank periodic average
when `Q_g` is the full world and the same tensor basis is merged.

Weighting defaults:

- v1 production: weight by accepted token count, because ranks may produce
  unequal local work under timeout/recovery.
- debug/correctness: equal weights, to match synchronous all-rank tests.
- reject zero-token updates unless explicitly marked as health/failure records.

## Staleness and recovery

### Quorum rule

Use a configurable quorum threshold:

```text
quorum_threshold = max(min_quorum, ceil(quorum_fraction * requested_workers))
```

Recommended initial policy:

- 1n debug: `quorum_fraction=1.0` by default, with a test mode at `7/8`.
- 2n/8n debug: `quorum_fraction=0.75` or explicit threshold for failure tests.
- 64n+ validation: start at `0.75` to `0.875`, then tune from observed
  straggler behavior and loss impact.

A generation advances when either:

- accepted fresh updates reach threshold; or
- a timeout fires and accepted fresh updates still meet threshold.

If timeout fires below threshold, generation status is `deferred`; latest does
not advance. The system should either extend the generation, reduce only under
an explicit emergency policy, or stop cleanly if repeated deferrals exceed a
configured run-level limit.

### Staleness bound

The first mainline path uses `tau = 0` accepted staleness:

```text
current_generation - update.base_generation must equal 0
```

Stale updates are counted and discarded. They are not down-weighted into the
main avg merge. This keeps v1 correctness close to synchronous DiLoCo and makes
loss regressions easier to diagnose.

Bounded-stale weighting is a later experimental arm:

```text
0 < current_generation - update.base_generation <= tau
weight_stale = weight_fresh * gamma^(staleness)
```

That arm should be labeled research-only until it has separate correctness and
loss validation.

### Rank recovery and resync

If a rank misses generation `g`:

1. Its late update is marked `stale` when it arrives after `latest=g+1`.
2. The supervisor sends the newest global generation id and state location.
3. The rank discards the stale update payload and reloads or rebases to
   `S_{g+1}` before producing another accepted update.

Two recovery modes are useful:

- **Reload**: load full `S_latest` into model/optimizer. This is simplest and
  should be the default after rank restart, non-finite loss, shape mismatch, or
  failed update validation.
- **Rebase**: if the rank still has local state derived from `S_b` and has not
  advanced optimizer state beyond the stale K-window, shift local tensors:

```text
X_local <- X_local + (X_latest - X_b)
Z_local <- Z_local + (Z_latest - Z_b)
base_generation <- latest
```

This preserves local displacement relative to the base, including
ScheduleFree x/z geometry. It is for continued local training, not for
retroactively accepting an old stale update as fresh.

## Outer optimizer scope

### Supported initially: avg

The initial train.py-native async path should support only:

```text
--diloco_outer_optimizer avg
--diloco_outer_lr 1.0 by default, optionally eta_outer != 1 for tests
--diloco_outer_beta 0.0
```

Reasons:

- avg has no global momentum/anchor state beyond `S_g`.
- avg full-quorum equivalence to synchronous DiLoCo is straightforward.
- stale rejection plus avg is easier to validate at 1n/2n/8n.
- It avoids accidental reuse of synchronous all-rank outer-state assumptions.

### Deferred: momentum and ScheduleFree outer

Current train.py synchronous `momentum` and `sfsgd` outer optimizers can inform
the later design, but should not be enabled in the first async train.py mode.

Non-avg async support needs a separate design/test pass for:

- global ownership and checkpointing of `O_g` anchor/moment or outer x/z/y;
- how missed quorum/deferred generations affect outer clocks;
- whether stale accepted updates, if ever allowed, enter the momentum buffer;
- compatibility guards when resuming checkpoints without async outer state;
- exact ScheduleFree geometry preservation across async rebase and global
  outer displacement.

CLI should fail closed if `--async_quorum_diloco` is combined with
`--diloco_outer_optimizer momentum` or `sfsgd` until that work lands.

## Delta-form vs full-weight exchange

### Cost model for E97 1.3B bf16

Assume 1.3B trainable parameters, bf16 tensor payloads, 2 bytes/element.

```text
one tensor state (X or W): 1.3e9 * 2 bytes = 2.6 GB = 2.42 GiB
ScheduleFree X+Z state:   5.2 GB = 4.84 GiB
2048 GPU ranks, X only:   5.32 TB per generation
2048 GPU ranks, X+Z:      10.65 TB per generation
256 node updates, X+Z:    1.33 TB upstream per generation
global X+Z broadcast to 256 nodes: 1.33 TB inter-node per generation
node-local fanout to 2048 GPUs:    10.65 TB local aggregate per generation
```

These are payload estimates before framing, alignment, compression, checksums,
and retries. Frontier jobs at 256 nodes have 2048 learner ranks, so directly
shipping one dense X+Z update from every GPU to a single global receiver is not
acceptable.

### Full endpoint vs delta representation

Full endpoint payload:

```text
send X_local, Z_local
merger computes mean endpoints or endpoint-base delta
```

Delta payload:

```text
send dX = X_local - X_base, dZ = Z_local - Z_base
merger applies S_g + weighted_mean(delta)
```

For dense bf16 tensors, endpoint and delta payload bytes are the same if both
are materialized in bf16. Delta form is still preferable as the protocol
semantics because it names the base generation and makes stale rejection,
weighted updates, update norms, and avg equivalence explicit.

Implementation should not require materializing an extra full-size delta tensor
on every learner. Use streaming/bucketed extraction:

1. Keep or reload the base generation tensor shard needed for comparison.
2. For each bucket, switch ScheduleFree to the requested export basis.
3. Compute `local_bucket - base_bucket` into a reusable communication buffer.
4. Send the bucket and immediately reuse/free it.

Memory implications per GPU:

- live E97 bf16 model X/W: about 2.6 GB;
- ScheduleFree z tensor state: about 2.6 GB if using ScheduleFree;
- base X+Z cache for delta extraction: about 5.2 GB if resident in bf16;
- one streaming bucket: configurable, for example 256 MB to 1 GB;
- avoid an additional full 5.2 GB dense delta clone.

If memory pressure is too high, prefer endpoint streaming for `avg` and compute
delta at the merger, or keep base tensors on CPU/NVMe per node and stream them
bucket-by-bucket. The protocol should still record the update as delta-form
against `base_generation`.

### Network shape

The production data plane should be hierarchical:

```text
GPU learner rank
  -> node supervisor / node reducer
  -> optional group aggregator
  -> sharded global merger
```

Node aggregation should combine 8 GPU rank deltas into one node delta before
inter-node transfer. For 256 nodes this reduces global fan-in from 2048 updates
to 256 node updates. The global merger should be sharded by flat tensor ranges
or parameter buckets so no single Python process owns all payload ingress.

Avoid Lustre for dense per-generation update payloads. Use Lustre for manifests,
periodic recovery/export checkpoints, and debug artifacts. Dense `update.pt`
files are acceptable only for unit tests, local simulations, and explicitly
labeled debug runs.

## Global state, update.pt, and file-backed state

Authoritative global state is the newest finalized generation:

```text
G_g = {
  generation,
  train_step,
  tensor_state: X/Z or W shards,
  optimizer_metadata,
  outer_state: empty in v1 avg,
  metrics,
  checkpoint_manifest
}
```

An `update.pt`, if used in debug mode, means:

```text
{
  schema: train_py_async_quorum_update_v1,
  run_id,
  rank_id,
  node_id,
  base_generation,
  local_step_start,
  local_step_end,
  tokens,
  loss_window,
  tensor_basis: xz | weights,
  representation: delta | endpoint,
  tensor_payload,
  payload_dtype,
  checksum,
}
```

It is not a checkpoint, not a resume source, and not allowed to advance
`latest.pt` or `latest.json`. Production should record only small update
manifests unless a debug flag requests dense payload retention.

File-backed dense state is acceptable for:

- CPU unit tests;
- local 1-process correctness simulations;
- 1n/2n debug runs where the report explicitly states dense files were used;
- periodic recovery/export checkpoints.

File-backed dense state is not acceptable as the high-volume production update
transport at E97 256-node scale.

## Checkpoint ownership and latest semantics

The global merger owns authoritative async latest advancement. Learner ranks,
node supervisors, and group aggregators may write cache manifests and local
debug artifacts, but those are non-authoritative and ignored by resume
selection.

Recommended async run layout:

```text
<output_dir>/async_quorum/
  latest.json
  latest.pt -> checkpoints/global_gen_000123.pt
  generations/
    gen_000123/
      manifest.json
      metrics.json
  checkpoints/
    global_gen_000120.pt
    global_gen_000123.pt
  updates_debug/
    gen_000123/rank_000042_update.pt
  cache/
    node_000/...
```

`latest.json` advances only after:

1. quorum merge advanced the generation;
2. the global tensor state for `g+1` is complete and validated;
3. the generation manifest is written atomically;
4. any due recovery/export/finalization checkpoint is written atomically;
5. the latest pointer is atomically replaced.

`latest.pt` should point to the newest loadable train.py global checkpoint when
one was written. It may lag `latest.json` if manifests advance every generation
but recovery checkpoints are less frequent. Resume policy must make this
explicit:

- default resume: use `latest.json` to find the newest finalized generation and
  its loadable checkpoint/shards;
- if only manifests advanced since the last full checkpoint, either reconstruct
  from retained global shards or resume from the newest full recovery checkpoint
  and mark skipped generations in metadata;
- never resume from worker `update.pt` or cache manifests by default.

Checkpoint cadence:

- generation manifest every advanced generation;
- metrics JSONL append every attempted generation, including deferred ones;
- recovery checkpoint by generation interval or wall-clock interval, whichever
  fires first;
- export checkpoint less frequently for evaluation/S3/continuation;
- walltime finalization checkpoint once remaining time is below reserve plus
  estimated checkpoint duration.

At Frontier scale, use a measured cadence rather than a fixed hourly-only
checkpoint. The prior async docs estimate that K=40 at 256 nodes can process
hundreds of millions of tokens per generation; losing many generations on
failure wastes substantial node-hours. Start 256-node validation with a
5-to-10-minute recovery target if measured checkpoint overhead is acceptable.

## Metrics and log schema

Emit one JSONL record per attempted generation and a small per-rank heartbeat
record at local boundaries. Extend `AsyncDiLoCoGenerationMetrics` rather than
creating an unrelated schema.

Required generation fields:

```json
{
  "schema_version": 1,
  "run_id": "e97-async-...",
  "mode": "train_py_async_quorum_diloco",
  "generation": 123,
  "base_train_step": 492000,
  "diloco_k": 40,
  "requested_workers": 2048,
  "participating_workers": 1987,
  "quorum_threshold": 1792,
  "quorum_size": 1904,
  "quorum_status": "advanced",
  "accepted_updates": 1904,
  "stale_updates": 51,
  "timed_out_updates": 93,
  "failed_updates": 0,
  "invalid_updates": 0,
  "staleness": {
    "max": 1,
    "p50": 0,
    "p95": 1,
    "histogram": {"0": 1904, "1": 51}
  },
  "generation_duration_s": 74.2,
  "local_train_duration_s": {"p50": 61.0, "p95": 69.8, "max": 73.1},
  "merge_duration_s": 8.4,
  "rebase_duration_s": 1.2,
  "checkpoint_duration_s": 3.7,
  "tokens_per_generation": 638000000,
  "tokens_per_accepted_update": {"mean": 335000, "min": 327680, "max": 327680},
  "tokens_per_sec": 8600000,
  "per_rank_progress": {
    "min_step": 492040,
    "max_step": 492040,
    "missing_ranks": [17, 812]
  },
  "loss_window": {
    "accepted_token_weighted_mean": 2.41,
    "p50": 2.40,
    "p95": 2.49,
    "max": 2.61
  },
  "update_norms": {
    "mean_delta_l2": 123.4,
    "mean_delta_l2_over_weight_l2": 0.00021
  },
  "update_bytes": {
    "accepted_payload": 990000000000,
    "broadcast_payload": 1330000000000
  },
  "latest_advanced": true,
  "checkpoint_paths": [".../generations/gen_000123/manifest.json"]
}
```

Required rank heartbeat/update fields:

- `run_id`, `rank`, `local_rank`, `node_id`, `generation`, `base_generation`;
- `local_step_start`, `local_step_end`, `tokens`, `batch_size`, `chunk_size`;
- local train duration, update extraction duration, send duration, wait duration;
- loss moving average/window min/max;
- non-finite/skip flags;
- update status: `accepted`, `stale`, `timed_out`, `failed`, `invalid`,
  `resynced`;
- resume/rebase source generation when recovery occurs.

These metrics are required to answer whether quorum is making progress, whether
late ranks are repeatedly excluded, whether merge/checkpoint overhead dominates,
and whether accepted-token loss differs from stale/missing ranks.

## Train.py integration points

Add the async mode around these existing regions:

- Argument parser near the current DiLoCo flags:
  `--async_quorum_diloco`, quorum threshold/fraction, generation timeout,
  staleness policy, update representation, metrics path, transport/debug
  options, and checkpoint cadence.
- Distributed setup in `train(args)`: keep one rank per GPU, but route
  `use_async_quorum_diloco` separately from `use_ddp` and synchronous
  `use_diloco`. Do not call DDP wrapping in async mode.
- Initial state: reuse the current rank-0 broadcast idea for debug, but
  production should load/pull `S_g` from global merger/checkpoint state rather
  than relying on all-rank collectives.
- State extraction helpers: factor `diloco_merge`'s ScheduleFree x/z basis logic
  into reusable functions that can flatten/stream X and Z without performing an
  all-reduce.
- K-boundary branch currently calling `diloco_merge`: in async mode call
  `async_quorum_boundary(...)`, submit/update wait for generation result, then
  load or rebase to the returned global state.
- Checkpointing: regular learner ranks do not call `save_checkpoint` as the
  authoritative writer in async mode. The global merger publishes manifests and
  due train.py-compatible checkpoints.
- Finalization: replace all-rank final merge with global-merger finalization.
  Learners stop at a safe K boundary, flush any accepted update, then the merger
  writes the final authoritative checkpoint if quorum advanced or records the
  final deferred state.

Concrete helper APIs to add or extract:

```python
extract_train_py_async_state(model, optimizer, args, basis="x") -> TensorState
load_train_py_async_state(model, optimizer, state, args) -> None
stream_train_py_async_delta(model, optimizer, base_state, args, bucket_numel) -> Iterator[BucketDelta]
apply_train_py_async_global_state(model, optimizer, global_state, args) -> None
rebase_train_py_async_state(model, optimizer, old_base, new_base, args) -> None
async_quorum_boundary(core_model, optimizer, args, local_metrics, generation) -> BoundaryResult
```

## Implementation plan

1. Add async quorum mode flags and fail-closed compatibility checks.
2. Extract train.py state helpers from the synchronous DiLoCo merge path.
3. Add unit/correctness tests proving full-quorum avg equivalence to
   synchronous DiLoCo and ScheduleFree x/z geometry preservation.
4. Wire a debug/local coordinator path using existing `ndm.async_diloco`
   quorum math and checkpoint manager.
5. Replace dense debug file transport with streaming/sharded transport hooks
   before 64n+ validation.
6. Add train.py-compatible global checkpoint publication and latest semantics.
7. Add 1n/2n Slurm smokes, then 8n and 64n validation reports.

## Test ladder

Unit/correctness:

- delta extraction round-trip for AdamW and ScheduleFree;
- full quorum avg equivalence against synchronous `diloco_merge`;
- reject stale, reject invalid, defer missed quorum;
- token-weighted vs equal-weighted merge behavior;
- rebase preserves local displacement and ScheduleFree x/z gap;
- worker/cache manifests cannot advance authoritative latest;
- resume selects finalized global generations only.

Debug smokes:

- 1n, 8 ranks: one train.py rank per GPU, async mode enabled, no DDP wrapper,
  full quorum, real token stream if available.
- 1n fault injection: one delayed/missing rank, quorum still advances when
  threshold permits.
- 2n, 16 ranks: stale/rejected update path appears in metrics without killing
  the job.

Scale ladder:

- 8n debug: verify train.py-native path, checkpoint cadence, loss windows, and
  quorum behavior.
- 64n validation: compare throughput/loss against synchronous train.py DiLoCo
  and the existing async harness at matched K/batch where possible.
- Larger Frontier candidate: only after an explicit report-level go decision,
  with job id, logs, node-hours, token estimate, latest policy, and cancel
  criteria.

## Follow-up WG graph

The implementation/testing graph created from this design is:

| Task ID | Purpose | Depends on |
| --- | --- | --- |
| `review-train-py` | Review this design before implementation. | `design-train-py` |
| `implement-train-py` | Add train.py async quorum mode flags and compatibility checks. | `review-train-py` |
| `implement-train-py-2` | Extract/apply async tensor state and delta helpers. | `review-train-py` |
| `implement-train-py-3` | Implement train.py async quorum coordinator using the reviewed helpers. | `implement-train-py`, `implement-train-py-2` |
| `integrate-async-checkpoint` | Wire authoritative checkpoint/latest semantics. | `implement-train-py-3` |
| `add-train-py` | Add 1n/2n train.py-native async quorum smokes. | `integrate-async-checkpoint` |
| `validate-train-py` | Run/report 8n, 64n, and larger-scale validation ladder. | `add-train-py` |

## Open risks

- Holding a full base X+Z cache on every learner may be too expensive at the
  exact E97 optimizer/memory configuration. The implementation must support
  bucketed/streaming delta extraction or endpoint streaming for avg.
- Python-level tensor transport may be sufficient for 1n/2n debug but is not
  enough evidence for 64n+. The transport abstraction must be ready for
  GPU-aware MPI/RCCL-adjacent point-to-point or another measured Frontier data
  plane.
- Async avg with quorum changes the effective data weighting when ranks are
  missing. Metrics must expose accepted-token coverage and stale rank patterns
  before any production go decision.
- Non-avg outer optimizers are intentionally deferred. Enabling them without
  new tests would invalidate the v1 correctness argument.
