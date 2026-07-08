# Resilient quorum DiLoCo catchup design

Task: `design-resilient-quorum-diloco-catchup`  
Date: 2026-07-08  
Target: E97 `train.py` scaleout on Frontier

## Decision

Keep two separate DiLoCo modes:

1. **Strict compiled-MPICH fast path**: preserve the validated compiled Cray
   MPICH `MPI_Reduce` path as the fastest strict collective mode. It assumes the
   launched world is healthy and all required ranks enter the same generation
   reduction. Missing or stuck ranks are fatal to the generation. This mode is
   used for performance validation and as the strict equivalence baseline.
2. **Resilient quorum mode**: add an explicit second mode that never waits in an
   all-rank blocking collective for a missing rank. It accepts fresh updates for
   an open generation until quorum or timeout, advances only when quorum is met,
   rejects or records stale/invalid/failed updates, and catches stale or
   restarted ranks up to the newest finalized global generation before they
   produce another accepted delta.

The resilient mode is not a replacement for the strict fast path. It is a
correctness and liveness mode for cases where some launched ranks never join,
stall before merge, or fall behind. It must be selected by a distinct CLI/env
switch and must emit mode-specific metrics so scale decisions do not mix strict
collective throughput with quorum-resilience behavior.

No Slurm jobs were submitted for this design task.

## State model

The authoritative global state for generation `g` is:

```text
G_g = {
  run_id,
  global_generation: g,
  state_id,
  checkpoint_id,
  tensor_basis,
  tensors,
  outer_optimizer_state,
  scalar_metadata,
  finalized_at_s,
  parent_state_id,
}
```

Fields:

- `global_generation`: monotonically increasing integer owned by the global
  merger. Generation `g` is the latest finalized state. Generation `g + 1` is
  not visible to workers until its manifest is finalized.
- `state_id`: content-addressed or run-local unique id for the complete global
  tensor state. It should include `run_id`, generation, tensor basis, shape
  schema hash, and payload checksum.
- `checkpoint_id`: id of the loadable train.py checkpoint or shard manifest
  from which `G_g` can be reconstructed. It may lag `state_id` if manifests
  advance more frequently than full recovery checkpoints, but resume must be
  explicit about that lag.
- `tensor_basis`: `schedulefree_xz` for ScheduleFree mainline, or `weights`
  for non-ScheduleFree/debug controls.
- `tensors`: `X_g` and `Z_g` for ScheduleFree, or `W_g` for plain weights.
  `X_g` is the eval/model basis after the ScheduleFree eval swap; `Z_g` is the
  ScheduleFree base iterate state. Shape and dtype are fixed by the run schema.
- `outer_optimizer_state`: empty for v1 `avg`; for later non-avg modes it owns
  global momentum/anchor/ScheduleFree-outer tensors and clocks. This state is
  merger-owned and checkpointed with `G_g`.
- `scalar_metadata`: global train step range, accepted token counters,
  accepted loss windows, quorum policy, mode, transport, wall-clock timings,
  and ScheduleFree scalar clocks treated as metadata rather than reduced scalar
  collectives.
- `parent_state_id`: `state_id` for `G_{g-1}`.

A worker update is:

```text
U_{r,u} = {
  update_id,
  worker_id,
  rank_id,
  node_id,
  local_generation_epoch,
  base_generation: b,
  base_state_id,
  base_checkpoint_id,
  target_open_generation,
  tensor_basis,
  representation: delta | endpoint,
  payload_manifest,
  payload_checksum,
  local_step_start,
  local_step_end,
  local_steps,
  tokens,
  loss_window,
  update_norms,
  nonfinite_flags,
  started_at_s,
  submitted_at_s,
}
```

`update_id` must be unique and idempotent:

```text
update_id = hash(run_id, worker_id, local_generation_epoch,
                 base_generation, base_state_id, payload_checksum)
```

Duplicate `update_id` values are counted once. A duplicate with different
payload bytes is `invalid`.

The logical delta for ScheduleFree is:

```text
dX_{r,b} = X_{r,b}^{local} - X_b
dZ_{r,b} = Z_{r,b}^{local} - Z_b
D_{r,b}  = (dX_{r,b}, dZ_{r,b})
```

For an `avg` outer update with accepted set `Q_g`:

```text
mean_delta_g = weighted_mean(D_{r,g}, r in Q_g)
G_{g+1}.tensors = G_g.tensors + eta_outer * mean_delta_g
```

Default production weighting is accepted token count:

```text
weight_r = U_r.tokens
```

Debug/equivalence tests may use equal weights. Zero-token updates are invalid
unless explicitly marked as health-only records that do not enter `Q_g`.

Full-cohort equivalence invariant:

```text
If Q_g contains every strict-mode worker, all updates have base_generation = g,
eta_outer = 1, and equal local work, then:

X_{g+1} = mean_r X_{r,g}^{local}
Z_{g+1} = mean_r Z_{r,g}^{local}
```

This must match the existing synchronous train.py DiLoCo average in the same
tensor basis.

## Quorum semantics

The resilient mode opens exactly one merge generation at a time:

```text
open_generation = latest_finalized.global_generation
deadline = open_started_at_s + quorum_timeout_s
quorum_threshold = max(min_quorum,
                       ceil(quorum_fraction * expected_participants))
```

`expected_participants` must be explicit in metrics. For GPU-rank quorum it is
the expected GPU ranks. For node-aggregated quorum it is the expected node
updates. The protocol must not silently change denominator after some ranks
fail to launch; nonjoining ranks are `timed_out` or `failed`, not removed from
the denominator, unless a separate human-approved reduced-world run starts with
a new `run_id`.

Update classification for generation `g`:

| Class | Condition | Action |
| --- | --- | --- |
| `accepted` | schema valid, finite tensors, expected tensor basis, `base_generation == g`, `base_state_id == G_g.state_id`, token/local-step policy valid, received before close | Enters `Q_g` once. |
| `stale` | `base_generation < g` or `base_state_id` names an older finalized state | Never merged into `G_{g+1}` in v1; worker is told to catch up. |
| `future` / `invalid` | `base_generation > g`, unknown `base_state_id`, shape/dtype mismatch, checksum mismatch, duplicate id with different payload, non-finite payload, forbidden zero-token payload | Counted invalid; worker must reload. |
| `failed` | Worker reports local train failure, NaN loss, helper error, local IPC failure, or transport failure before payload validity | Counted failed; may still allow quorum without it. |
| `timed_out` | No valid update by `deadline` for an expected participant | Counted timed out. |
| `late` | Payload arrives after `G_{g+1}` finalized | Reclassified against current state: normally stale, sometimes duplicate of an already accepted update. |

Generation advancement:

```text
advance iff count(accepted fresh updates) >= quorum_threshold
```

The merger can advance immediately when quorum is reached or at timeout when
quorum is already satisfied. The default should advance as soon as quorum is
met for liveness, while still waiting for optional `grace_s` in debug modes if
we want full-cohort comparison data.

Timeout below quorum:

```text
if now >= deadline and accepted < quorum_threshold:
  generation_status = deferred
  global_generation remains g
  latest.json/latest.pt do not advance
```

After a deferred generation, policy is explicit:

- `extend`: reopen the same `g` with a new deadline, preserving already
  accepted update ids.
- `retry-generation`: discard accepted payloads for `g`, keep metrics, and
  ask workers to resubmit from `G_g`.
- `stop-fail-closed`: terminate cleanly without latest advancement.

The first implementation should use `stop-fail-closed` after a small configured
number of deferrals for scale gates, because repeated below-quorum advancement
without human review risks silently training a different system.

Late updates are never used to revise a finalized generation. A finalized
`G_{g+1}` is immutable. Late payloads are retained only as metrics/debug
artifacts subject to retention policy.

## Catchup semantics

A worker carries:

```text
worker_state = {
  loaded_generation,
  base_generation,
  base_state_id,
  base_checkpoint_id,
  local_generation_epoch,
  local_optimizer_state,
  current_tensor_state,
  last_seen_global_generation,
}
```

A worker detects it is behind when any of these is true:

- coordinator/global merger response says `latest_generation >
  base_generation`;
- submitted update receives `stale` or `invalid_base_state`;
- heartbeat/latest metadata shows a newer finalized `latest.json`;
- restart finds durable `latest.json` generation greater than its persisted
  `base_generation`;
- local helper result names a newer decision generation than the worker loaded.

Catchup rule:

```text
staleness = latest_generation - base_generation
```

For v1 accepted updates, required staleness is zero:

```text
accept only if staleness == 0
```

If `staleness > 0`, the worker must not submit or resubmit its old delta as
fresh. It must discard that update payload and choose one of two catchup paths.

### Reload catchup

Reload is the default and mandatory after restart, non-finite local state,
shape/schema mismatch, optimizer mismatch, missing local base cache, or unknown
local progress:

1. Stop local training at the boundary or abort the current partial K-window.
2. Delete or mark the pending stale `update_id` as discarded.
3. Load `G_latest` from its finalized state/checkpoint manifest.
4. Load global tensor state into train.py in the correct tensor basis.
5. Load outer optimizer state from `G_latest.outer_optimizer_state`.
6. Reset worker `base_generation = latest_generation`,
   `base_state_id = latest.state_id`, and increment `local_generation_epoch`.
7. Resume local K-step training from the new base.

For v1 `avg`, `outer_optimizer_state` is empty, but the reload path still
records that it loaded empty avg state. This prevents later non-avg modes from
silently dropping momentum/outer ScheduleFree state.

### Rebase catchup

Rebase is allowed only when the process is still alive, has exact `G_b` base
tensors cached, has not restarted, has no non-finite values, has no optimizer
schema change, and has not continued speculative training beyond the stale
K-window.

For ScheduleFree:

```text
shift_X = X_latest - X_b
shift_Z = Z_latest - Z_b

X_local <- X_local + shift_X
Z_local <- Z_local + shift_Z
base_generation <- latest_generation
base_state_id <- latest.state_id
```

This preserves local displacement:

```text
X_local' - X_latest = X_local - X_b
Z_local' - Z_latest = Z_local - Z_b
```

The old stale update remains stale and is not accepted. Rebase is only a way to
continue local training without a full reload. If non-avg outer state is ever
enabled, rebase must also apply a formally defined transform to the local view
of global outer state or fail closed to reload. Until then, resilient quorum
mode should reject `momentum` and `sfsgd` outer optimizers.

### Restarted ranks

A restarted rank must assume its in-memory local delta and optimizer state are
untrusted. It reads only authoritative global latest:

```text
latest = async_quorum/latest.json
load latest.manifest/checkpoint
base_generation = latest.global_generation
local_generation_epoch += 1
```

It must not resume from `updates_debug/`, node cache manifests, helper IPC
files, or partially written local checkpoint fragments. If a rank's prior
`update_id` later arrives from transport retry, the merger classifies it as
duplicate or stale and does not merge it a second time.

## Outer optimizer state

Initial resilient quorum mode supports only:

```text
diloco_outer_optimizer = avg
eta_outer = diloco_outer_lr
diloco_outer_beta = 0
outer_optimizer_state = {}
```

The implementation should fail closed if resilient quorum mode is combined with
`momentum` or `sfsgd` until a separate state design lands. Reasons:

- non-avg outer state is global, not rank-local;
- missed/deferred generations must not advance momentum clocks;
- stale update admission would need a defined momentum/staleness rule;
- catchup/rebase must not corrupt global anchor or ScheduleFree-outer state;
- checkpoints must contain enough state to reproduce the exact global outer
  optimizer after restart.

Strict compiled-MPICH mode may continue to use the existing synchronous
train.py outer-state logic, because it has a strict all-rank round boundary.

## Transport

### Strict compiled-MPICH fast path

Strict mode remains the compiled Cray MPICH `MPI_Reduce`/collective helper path
that has already been validated. Its contract:

- static Slurm MPI world;
- all required ranks enter every generation collective;
- dense tensor buckets reduced in the compiled helper;
- no per-step DDP or RCCL gradient all-reduce;
- rank/global failure at collective time fails the generation/job;
- fastest path for healthy worlds and strict equivalence baselines.

This path is allowed to use blocking collective semantics because resilience to
missing ranks is explicitly out of scope for it.

### Resilient quorum mode

Resilient mode should use nonblocking, rank-scoped transport with bounded
deadlines. The chosen implementation direction is:

1. **Debug/control plane**: rank-0 TCP or existing bounded metadata path for
   smoke tests and failure injection. Payloads are small metadata summaries,
   not dense E97 tensors.
2. **Dense production direction**: compiled Cray MPICH helper point-to-point or
   one-sided/RMA style movement with root/sharded aggregators, not all-rank
   blocking collectives. Helpers send per-rank or per-node payloads with
   deadlines. The receiver stops waiting when quorum is met or timeout fires.
3. **Hierarchy**: GPU ranks reduce to a node update first; optional group
   aggregators reduce node updates; global merger shards by tensor bucket. This
   limits 256-node fan-in to node/group updates rather than 2048 direct GPU
   payloads into one Python process.

The resilient transport must avoid hot Lustre sync:

- no remote polling of dense `update.pt` files for quorum decisions;
- no dense per-generation payloads written to Lustre as the live data plane;
- Lustre is for run artifacts, immutable manifests, metrics, recovery/export
  checkpoints, logs, and optional debug-retained payloads;
- same-rank Python-to-helper IPC may use node-local shared memory or run-local
  staged files, but those files are not remote quorum files.

The resilient transport must avoid blocking collectives for missing ranks:

- use per-source receives/sends or sharded request queues with deadlines;
- classify non-arrivals as timed out rather than entering a collective that
  requires them;
- broadcast only compact decisions to surviving ranks after generation close;
- never require a newly restarted rank to join an old generation collective.

## Checkpoint and latest semantics

Only the global merger publishes authoritative async global state. Workers,
node supervisors, group aggregators, and helpers may write local caches,
heartbeats, and debug artifacts, but those are non-authoritative.

Recommended layout:

```text
<run_root>/async_quorum/
  latest.json
  latest.pt -> checkpoints/global_gen_000123.pt
  metrics.jsonl
  generations/
    gen_000122/
      manifest.json
      metrics.json
    gen_000123/
      manifest.json
      metrics.json
  checkpoints/
    global_gen_000120.pt
    global_gen_000123.pt
  recovery_checkpoints/
  export_checkpoints/
  updates_debug/
  cache/
  ipc/
```

`latest.json` advances only after:

1. quorum status is `advanced`;
2. merged tensors for `G_{g+1}` are complete, finite, shape-valid, and checksum
   validated;
3. outer optimizer state for `G_{g+1}` is present and valid;
4. generation manifest is written by temp-file plus atomic rename;
5. any due recovery/export/finalization checkpoint is written atomically;
6. `latest.json` is atomically replaced.

`latest.pt` points to the newest loadable train.py checkpoint if one was
written. If full checkpoints are less frequent than generation manifests, the
resume manifest must identify whether `latest.pt` is exact for `latest.json` or
which generation gap must be reconstructed from retained global shards. The
implementation should start with recovery checkpoints every advanced
generation for debug gates, then relax cadence only after recovery is proven.

Run-local latest advancement:

- debug and validation jobs write only under their isolated run root;
- run-local latest may advance after each finalized generation;
- no production chain `latest.pt` or `last` pointer changes during 1n/8n/64n
  debug ladders or optional 256n debug smoke.

Production latest guard:

- record production `latest`/`last` identity before and after every debug or
  approval-package run;
- fail validation if production identity changes without an explicit
  production task/human approval;
- 256n x 1h or 12h production promotion remains outside this workstream and
  requires later explicit approval.

Finalization:

- on walltime reserve or signal, close the current generation only if quorum
  is already satisfied or can be satisfied before the deadline;
- otherwise publish no new global latest and record a terminal
  `deferred_finalization` metrics event;
- always write a final metrics summary and production latest guard result.

## Required metrics

Per generation:

- `mode`: `strict_compiled_mpich_reduce` or `resilient_quorum`;
- transport name and version;
- `global_generation`, `base_generation`, `state_id`, `checkpoint_id`;
- `expected_participants`, `participating_workers`, `quorum_fraction`,
  `min_quorum`, `quorum_threshold`, `accepted_count`;
- rank/node id lists for accepted, stale, failed, timed-out, invalid, late, and
  duplicate updates;
- deferred/advanced/fail-closed status and reason;
- update id list and duplicate suppression count;
- token count per accepted update and aggregate accepted tokens;
- local step count per accepted update;
- loss window min/median/max/p95 and non-finite loss count;
- update norm summaries for X/Z or W, including min/median/max/p95 and
  clipping/rejection reason if any;
- bytes sent/received by hierarchy level and by transport;
- submit latency, queue latency, receive latency, merge duration, catchup
  duration, checkpoint duration, total generation duration;
- helper/root memory high-water mark where available;
- state/checkpoint sizes and checksums;
- latest advancement boolean and latest path;
- resume/catchup counts by type: reload, rebase, restart reload,
  stale-discard, invalid-reload;
- production latest guard before/after identity and changed flag.

Run summary:

- quorum distribution across generations;
- sustained accepted fraction by rank/node;
- consecutive timeout/failure streaks;
- number of deferred generations and deferral policy actions;
- catchup success rate and time-to-catchup distribution;
- tokens/sec based on accepted tokens and wall clock;
- loss trend by accepted tokens, separate from launched-token estimates;
- checkpoint overhead percentage;
- strict-vs-quorum comparison metadata when a paired baseline exists.

Scale decisions should require separate evidence for:

- nonjoining-rank tolerance;
- stuck-before-merge timeout;
- late update rejection;
- stale/restarted-rank catchup;
- run-local checkpoint/latest finalization;
- unchanged production latest/last guard.

## Test plan

### Unit tests

1. `test_quorum_advances_on_threshold_not_unanimity`: expected participants 8,
   quorum 6, six valid updates from `base_generation=g`; assert `G_{g+1}`
   advances and timed-out list contains ranks 6 and 7.
2. `test_quorum_deferred_below_threshold`: expected participants 8, quorum 6,
   five valid updates, timeout fires; assert generation remains `g`,
   latest does not advance, and metrics status is `deferred`.
3. `test_stale_update_rejected_after_generation_advance`: finalize `G_{g+1}`,
   then submit update with `base_generation=g`; assert stale count increments
   and tensors/checkpoint ids are unchanged.
4. `test_late_duplicate_update_id_idempotent`: accept an update, finalize, then
   deliver the same `update_id`; assert duplicate count increments and no second
   merge occurs.
5. `test_invalid_future_or_wrong_state_id_fails_closed`: submit
   `base_generation=g+1` or mismatched `base_state_id`; assert invalid and
   worker reload command.
6. `test_rebase_preserves_schedulefree_displacement`: verify
   `X_local' - X_latest == X_local - X_base` and the same for `Z`.
7. `test_restart_ignores_update_debug_and_cache`: create worker update/cache
   files plus authoritative latest; restart selector must load only
   authoritative latest.
8. `test_non_avg_outer_optimizer_rejected_in_resilient_mode`: resilient quorum
   with `momentum` or `sfsgd` must fail closed.
9. `test_run_local_latest_advances_only_after_checkpoint_manifest`: simulate
   manifest/checkpoint write failure; assert no latest advancement.
10. `test_production_latest_guard_detects_mutation`: debug run records
    production latest before/after and fails validation if identity changes.

### Local failure-injection tests

1. **Nonjoining rank**: expected 8 workers, launch 7 or make rank 7 never
   connect. Quorum 6. Assert generation advances, rank 7 is timed out, and the
   job does not enter a blocking all-rank collective.
2. **Stuck rank before merge**: rank starts and writes heartbeat, then sleeps
   past `quorum_timeout_s` before payload submission. Assert timeout accounting,
   accepted quorum advancement, and catchup command when it wakes.
3. **Late rank**: rank submits valid payload after `G_{g+1}` finalized. Assert
   `late`/`stale` classification, no mutation of `G_{g+1}`, and reload/rebase
   instruction.
4. **Stale rank continuing from old base**: worker misses two generations,
   tries to submit `base_generation=g`. Assert discard, latest detection,
   reload to `G_{g+2}`, and next update accepted only from base `g+2`.
5. **Restarted rank**: kill rank after local training but before submit,
   restart it with local cache present. Assert it reads authoritative
   `latest.json`, discards partial local delta, and emits restart catchup
   metrics.
6. **Corrupt payload**: valid metadata but bad checksum or non-finite tensor.
   Assert invalid count, no merge use, and reload command.
7. **Below-quorum 64n-style coordinator lag**: all workers produce local
   artifacts but coordinator receives fewer than threshold before timeout.
   Assert status `deferred`, latest/checkpoint paths missing or unchanged, and
   terminal fail-closed if configured.

### Strict fast-path regression tests

1. Existing compiled helper strict `MPI_Reduce` smoke stays green.
2. Strict mode full-cohort result matches synchronous train.py DiLoCo average
   for a tiny ScheduleFree model.
3. Strict mode config does not emit resilient quorum metadata that would imply
   missed-rank tolerance.
4. Resilient mode config does not call strict all-rank `MPI_Reduce` for the
   live quorum decision.

### Frontier debug ladder

The implementation/validation workstream should use the approved boundary:

```text
1n -> 8n -> 64n -> optional single bounded 256n debug smoke
```

The 256n x 1h run is approval-package-only and must not be submitted by this
design or by the early implementation tasks.

Per rung:

- prove no DDP/per-step all-reduce;
- record mode and transport;
- record accepted/stale/failed/timed-out/invalid/late lists;
- record catchup reload/rebase/restart evidence;
- record run-local latest/checkpoint behavior;
- record production latest/last guard unchanged;
- keep strict compiled-MPICH fast-path and resilient quorum evidence separate.

## Implementation guardrails

- Add an explicit mode switch, for example `--diloco_mode strict_mpich_reduce`
  vs `--diloco_mode resilient_quorum`, or a similarly unambiguous pair of
  flags. Ambiguous combinations fail closed.
- Keep `global_generation` and `base_generation` in every update envelope and
  every checkpoint manifest.
- Make finalized generation manifests immutable.
- Never merge stale updates in v1 resilient quorum mode.
- Never treat worker caches, debug updates, helper IPC files, or node manifests
  as authoritative resume state.
- Record all failure accounting even when quorum advances.
- Do not mutate production `latest` or `last` in debug/validation ladders.
