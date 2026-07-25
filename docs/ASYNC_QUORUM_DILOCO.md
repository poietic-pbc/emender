# Async Quorum DiLoCo

Date: 2026-07-05

> **Historical v1 scaffolding, not current bounded-lag authority.** This note
> defines the stale-reject quorum baseline implemented by the original local
> simulation and core tests. Strict `tau = 0` remains a compatibility mode
> under [Resilient DiLoCo Compute Pool](RESILIENT_DILOCO_COMPUTE_POOL.md).
> The only reviewed policy for continuous decoupled local windows and bounded
> contribution admission is
> [ADR-002: simple asynchronous DiLoCo v2.1](ASYNC_DECOUPLED_DILOCO_V2.md),
> policy `async-decoupled-v2.1-simple`, with its V21S01–V21S17 namespace in the
> [gap matrix](RESILIENT_DILOCO_GAP_MATRIX.md). Historical
> `async-decoupled-v2.0-exp` artifacts and V2A requirements are incompatible
> evidence, not implementation or promotion authority.
> Where this note calls bounded staleness a future experiment or suggests a
> transport that conflicts with the native authority, ADR-002 and
> [Native resilient DiLoCo data plane v1](NATIVE_RESILIENT_DILOCO_DATAPLANE.md)
> take precedence.

## Goal

Train E97 at Frontier scale without requiring a static all-rank collective at
every DiLoCo boundary. The target research system is asynchronous DiLoCo with
bounded liveness risk: slow, missing, or failed workers should reduce the number
of updates in a round, not kill the job.

The first implementation should be quorum DiLoCo with stale rejection. This is
the A/C hybrid discussed in the Frontier chat:

- workers produce local DiLoCo deltas from a named global generation;
- the merger advances the global generation after a quorum or timeout;
- late updates from older generations are rejected at first;
- lagging workers pull or rebase to the newest generation before producing the
  next accepted update.

This keeps the zero-staleness/full-cohort case equivalent to synchronous DiLoCo
while removing all-rank startup and merge failure modes.

## Non-Goals

- Do not use Lustre as the high-volume update data plane.
- Do not require a `torch.distributed` all-rank process group for the global
  optimizer step.
- Do not make stale update weighting the first mainline path. Bounded-stale
  Hogwild is an experimental extension after quorum DiLoCo is correct.
- Do not merge only visible model weights in the main ScheduleFree arm. Model
  only merge is a control.

## State And Math

Let a global ScheduleFree state at generation `g` be:

```text
S_g = (x_g, z_g, c_g)
```

`x` is the ScheduleFree eval/running weight tensor state, `z` is the
ScheduleFree base iterate tensor state, and `c` denotes scalar clocks and
metadata. The main async DiLoCo arm communicates `x` and `z` deltas. Scalar
clock policy is tracked in metadata and should not require global scalar
collectives.

A worker pulls `S_g`, trains locally for `K` optimizer steps or a bounded local
time window, and obtains:

```text
S_i = F_i(S_g) = (x_i, z_i, c_i)
```

It submits:

```text
dx_i = x_i - x_g
dz_i = z_i - z_g
base_generation = g
tokens_i
local_steps_i
loss/window metrics
```

For a quorum set `Q_g` of accepted updates from the same base generation, the
stateless average outer update is:

```text
x_{g+1} = x_g + eta_outer * weighted_mean_i(dx_i, i in Q_g)
z_{g+1} = z_g + eta_outer * weighted_mean_i(dz_i, i in Q_g)
```

Weights should default to accepted token count or local optimizer steps. Equal
weights are a valid debug/control mode.

When `Q_g` contains all synchronous workers, `eta_outer=1`, and all workers
started from `S_g`, this is exactly the synchronous DiLoCo average:

```text
x_{g+1} = mean_i(x_i)
z_{g+1} = mean_i(z_i)
```

This equivalence is the first correctness invariant.

## Staleness Policy

The first mainline policy rejects stale deltas:

```text
accept update iff update.base_generation == current_open_generation
```

Late updates are marked stale and ignored for the generation update. Workers
whose base is stale must pull or rebase to the current generation before
submitting another accepted update.

This is intentionally conservative. A later bounded-stale Hogwild arm can allow:

```text
0 < current_generation - base_generation <= tau
```

with a clearly labeled server learning-rate or staleness weighting policy. That
arm is research, not the first correctness baseline, because stale deltas are
not mathematically identical to fresh deltas.

## Rebase

For a worker that has local state `S_i` based on `S_b`, and a newer global state
`S_g`, a geometry-preserving rebase shifts `x` and `z` together:

```text
shift_x = x_g - x_b
shift_z = z_g - z_b

x_i <- x_i + shift_x
z_i <- z_i + shift_z
base_generation <- g
```

This preserves the worker's local displacement relative to its base:

```text
(x_i + shift_x) - x_g = x_i - x_b
(z_i + shift_z) - z_g = z_i - z_b
```

The already-computed stale delta is still stale with respect to the global
optimizer. Rebase is primarily a recovery mechanism for continued local
training, not a proof that an old update is fresh.

## System Shape

The performance-oriented design is hierarchical:

```text
GPU worker(s)
    -> node supervisor
        -> group aggregator(s)
            -> sharded global merger
```

### GPU Worker

- Runs one E97 training process on one GPU.
- Does not join a global distributed process group.
- Pulls a generation state from local supervisor or durable cache.
- Trains local chunks.
- Extracts `dx,dz` against the current base generation.
- Sends deltas to the node supervisor.
- Rebases or reloads when instructed.

### Node Supervisor

- Runs one per Frontier node.
- Receives deltas from 8 local GPU workers.
- Averages local GPU deltas into one node delta when local quorum or timeout is
  reached.
- Tracks per-GPU health, tokens, loss windows, and update norms.
- Sends one node delta upstream.

Node aggregation is required for scale. At 256 nodes, the global system should
see at most 256 node updates per generation, not 2048 GPU updates.

### Group Aggregator

Optional but likely useful for 256-node jobs. It aggregates node deltas for a
fixed node group, for example 8 or 16 nodes, and forwards group deltas to the
global merger. This reduces fan-in and allows hierarchical monitoring.

### Sharded Global Merger

- Owns global generation state.
- Shards tensors by parameter or flat ranges so no single process receives all
  bytes.
- Applies accepted deltas by quorum or timeout.
- Publishes a new generation.
- Sends new generation shards or rebase instructions downstream.
- Writes durable checkpoints periodically, not every update.

## Transport

Best-performance target:

- GPU-aware Cray MPICH point-to-point or one-sided/RMA for tensor data.
- Python is acceptable for orchestration and metadata, but not as the final
  high-volume tensor transport.

Rationale:

- OLCF documents Frontier's Cray MPICH as GPU-aware when configured with
  `craype-accel-amd-gfx90a`, `rocm`, and `MPICH_GPU_SUPPORT_ENABLED=1`.
- Avoiding Lustre for dense deltas is essential. Lustre should be used for
  durability, manifests, and periodic checkpoints, not per-round dense tensor
  synchronization.
- Avoiding `torch.distributed` all-rank collectives is essential for robustness.
  Point-to-point scoped communication has a smaller failure domain.

Initial implementation may use CPU tensors and local files for unit tests, then
single-node IPC, then GPU-aware MPI for Frontier validation.

## Durable State

Durable generation directories should be immutable:

```text
async_diloco_runs/<run_id>/
  generations/
    gen_000000/
      state.pt
      manifest.json
    gen_000001/
      state.pt
      manifest.json
  latest -> generations/gen_000001
  updates/
    gen_000000/
      node_000/
        manifest.json
```

For production runs, dense update payloads should not be stored here unless in
debug mode. Store small manifests, accepted/rejected update summaries, and
periodic full checkpoints.

The global merger owns the authoritative `latest` pointer. GPU workers, node
supervisors, and group aggregators may cache states and write update manifests,
but they must not advance the production checkpoint pointer. This keeps
continuation semantics simple: a new job starts from the latest finalized global
generation, not from a worker-local or partially merged state.

Checkpointing should have three layers:

1. Generation manifests: small metadata written every generation. These record
   accepted nodes, rejected stale nodes, missing nodes, quorum size, token
   weights, merge latency, update norms, loss windows, and timeout cause.
2. Recovery checkpoints: full global `x,z` state written every fixed wall-clock
   interval or generation interval, and always during walltime finalization.
3. Export checkpoints: user-facing model checkpoints written less frequently
   for evaluation, S3 upload, and continuation by other training paths.

Initial production defaults must be scale-adaptive rather than a fixed
20-to-30-minute recovery interval. At 256 nodes with batch size 4 and `K=40`,
each local step covers 16,777,216 tokens and each DiLoCo generation covers
671,088,640 tokens. If 64-node batch-4 scans extrapolate to roughly one to two
minutes per `K=40` generation at 256 nodes, a 20-minute recovery interval can
lose many accepted global generations and about 85 node-hours on a failure.

Use this cadence policy until measurements justify changing it:

- write a generation manifest every DiLoCo generation;
- write recovery checkpoints by whichever fires first: a configured generation
  interval or a configured wall-clock interval;
- for the first 256-node `K=40` launch package, start with a recovery target of
  about 5 to 10 minutes if checkpoint write time and training overhead are
  acceptable;
- write export checkpoints about hourly unless evaluation/transfer needs a
  different lower-frequency cadence;
- trigger a final recovery checkpoint with enough walltime remaining to finish
  cleanly.

These numbers are policy defaults, not mathematical constants. The 32/64-node
configuration tests and 256-node launch preparation must measure checkpoint
write duration, checkpoint size, generation duration, and percent overhead
before finalizing any production cadence.

## Metrics Required From Tests

Every practical test should emit a machine-readable summary, not only logs. At
minimum:

- requested nodes and GPUs;
- participating nodes and GPUs;
- average, minimum, maximum, and percentile quorum size;
- accepted, stale rejected, timed-out, and failed updates per generation;
- generation duration, merge duration, rebase duration, and checkpoint duration;
- tokens/sec per GPU, per node, and aggregate;
- tokens per accepted generation;
- update byte volume by level;
- loss moving averages over a large enough window to compare noisy runs;
- update norm and clipped/invalid update counts;
- checkpoint paths, checkpoint sizes, and whether `latest` advanced;
- restart/resume source generation when applicable.

The key quorum metric is effective quorum, not only configured quorum. For a
run with 256 nodes and quorum 192, a healthy system should show whether it
typically advances with 250-plus nodes, 220 nodes, or barely at threshold. That
distinction tells us if the system is robust or merely surviving.

## Payload Cost

For a 1.3B parameter model:

```text
bf16 tensor state:      ~2.6 GB
x + z dense delta:     ~5.2 GB per worker or node update
int8 x + z delta:      ~2.6 GB
int4 x + z delta:      ~1.3 GB
observed full ckpt:    ~7.7 GB
```

Dense delta form is not smaller than dense state form by itself. Delta form is
still the right semantic representation because it is tied to a base generation
and supports async application, staleness handling, quantization, and error
feedback.

The first correctness prototype should use dense bf16/fp32 deltas. The scaling
track should add quantized deltas and error feedback only after equivalence tests
pass.

## Correctness Invariants

1. Zero-staleness/full-cohort quorum DiLoCo with `eta_outer=1` equals synchronous
   DiLoCo averaging of `x` and `z`.
2. Partial quorum with equal weights equals the mean over accepted worker states.
3. Token-weighted quorum equals the token-weighted accepted delta mean.
4. Stale updates are rejected in the first mainline policy.
5. Rebase shifts `x` and `z` together and preserves local `x-z` geometry.
6. A failed or missing worker cannot block generation advancement once quorum or
   timeout is satisfied.
7. Durable metadata is sufficient to reproduce which updates contributed to each
   global generation.

## Implementation Phases

### Phase 1: Pure Math And State Utilities

- Implement extraction of ScheduleFree `x,z` tensor state from an E97 model and
  optimizer.
- Implement dense delta compute/apply.
- Implement quorum merge over synthetic states.
- Implement rebase over synthetic and real ScheduleFree states.
- Add unit tests for the invariants above.

### Phase 2: Local Async Simulation

- Simulate workers with small torch models and ScheduleFree AdamW.
- Randomly delay or drop worker updates.
- Verify quorum generation advancement and stale rejection.
- Verify equivalence to synchronous DiLoCo in the zero-staleness/full-cohort
  case.

### Phase 3: Single-Node Prototype

- Run 8 single-GPU workers and one node supervisor.
- Use the real E97 training target with a short debug budget.
- Exercise worker pull, local train, delta extraction, node aggregation, and
  generation publish.
- Keep dense payloads local for correctness; measure extraction and aggregation
  time.

### Phase 4: Frontier Multi-Node Debug

- Run 2, 4, and 8 nodes.
- Validate no global startup rendezvous is required.
- Validate a missing or intentionally delayed worker does not block generation
  advancement.
- Measure end-to-end tokens/sec, merge latency, update bytes, and checkpoint
  overhead.

### Phase 5: Transport And Scale

- Implement GPU-aware Cray MPICH data transport for node and group deltas.
- Add optional group aggregators.
- Add sharded global merger.
- Test 32 and 64 nodes.
- Prepare 256-node 12-hour launch package after debug evidence is clean.

## Experiment Arms

Main arm:

- quorum DiLoCo, dense `dx,dz`, stale rejection, ScheduleFree `x,z` merge.

Controls:

- synchronous hierarchical DiLoCo in current `train.py`;
- async model-only delta merge;
- async equal-weight vs token-weighted delta merge;
- later bounded-stale Hogwild with explicit `tau` and server learning-rate
  policy.

## Initial WG Graph

The implementation graph should be:

1. design review and acceptance of this document;
2. core state/delta/quorum math implementation with unit tests;
3. local async simulation;
4. single-node E97 worker/supervisor prototype;
5. 2-node Frontier debug;
6. 8-node Frontier debug with induced worker lag/failure;
7. transport benchmark and GPU-aware MPI prototype;
8. 32/64-node configuration tests;
9. 256-node 12-hour launch package, gated on prior evidence.

Large compute submissions must remain gated by explicit validation artifacts and
should record job IDs, elapsed time, node-hours, logs, update/generation
manifests, and pass/no-go conclusions.
