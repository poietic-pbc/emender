# Real Async DiLoCo E97 Training Path Design

Date: 2026-07-06
Task: `design-real-async`

## Decision

Implement the first real async E97 path as a new Frontier entrypoint that imports
small, import-safe helpers extracted from `train.py`; do not extend `train.py`
into the async orchestrator.

The recommended implementation target is:

- new script: `scripts/frontier/async_diloco_e97_real.py`
- shared helpers: `ndm/training/e97_runtime.py` or `ndm/train_runtime.py`
- existing pure async math extension: `ndm/async_diloco.py`
- optional local simulation helpers: `ndm/async_diloco_local.py`
- new Slurm wrapper only after local/short validation: `scripts/frontier/async_diloco_e97_real.sbatch`

Rationale:

- `train.py` already owns correct E97 model, ScheduleFree, checkpoint,
  data-stream, walltime, finalization, and DiLoCo semantics, but its `train(args)`
  function is a monolithic CLI routine with distributed setup, process-group
  state, output directory creation, training loop, checkpointing, evaluation, and
  shutdown coordination interleaved.
- The real async runner needs one independent GPU worker per GPU, one
  node-local supervisor, and a global supervisor. Embedding that orchestration
  directly in `train.py` would make the existing synchronous/DDP/local-DiLoCo
  path harder to protect.
- A new script can keep the current stable training path unchanged while
  reusing the same model/optimizer/checkpoint/data helpers so the async path is
  real token training, not a separate prototype.

## Prototype Versus Real Path

The current Frontier async entrypoint is a prototype harness:

- `scripts/frontier/async_diloco_e97_multinode.py` imports `main` from
  `scripts/frontier/async_diloco_e97_2n8n_debug.py`.
- `async_diloco_e97_2n8n_debug.py` loads checkpoint tensors through
  `load_async_diloco_readonly_state`, creates synthetic worker specs, generates
  synthetic dense deltas, optionally drops/lag-simulates workers, and writes
  metrics/checkpoint manifests.
- It validates quorum math, metrics, checkpoint publication, resume selection,
  and production-latest guards.
- It does not instantiate the E97 model from `train.py`, does not build the
  real optimizer, does not read the real token stream, and does not run real
  forward/backward optimizer steps.

The real path must instead:

- build the same E97 model configuration as `train.py`;
- load the same checkpoint model and optimizer state semantics as `train.py`;
- run real `DocumentStreamDataset`, `TokenizedStreamDataset`, or
  `BatchedStreamDataset` batches;
- execute real forward/backward/optimizer steps on each GPU island;
- extract ScheduleFree `x` and `z` deltas after local training;
- merge accepted node/global deltas through the async quorum protocol;
- publish durable global checkpoints without mutating any production latest
  pointer from the old synchronous/paused run tree.

`docs/ASYNC_QUORUM_DILOCO.md` remains the math/system design reference. This
document is the implementation bridge from the prototype harness to the real
`train.py`-derived path.

## Import-Safe Helpers To Extract From `train.py`

The downstream `refactor-train-py` task should extract the minimal helpers below
without changing existing `train.py` behavior. `train.py` should become a thin
CLI consumer of these helpers where practical, but the first refactor may leave
`train(args)` intact and duplicate only calls into the new helper module.

### Argument And Configuration

Expose a parser builder instead of forcing CLI parsing:

- `build_train_arg_parser() -> argparse.ArgumentParser`
- `parse_train_args(argv=None) -> argparse.Namespace`
- `normalize_train_args(args) -> argparse.Namespace`

`normalize_train_args` should cover the non-I/O setup currently buried near the
top of `train(args)`: `parse_level`, E97 Triton auto-selection, model variant
labels, default distributed flags for single-process workers, and seed handling.

### Model Build

Extract:

- `resolve_vocab_size(args) -> int`
- `resolve_r_h_mode(args) -> str`
- `build_model(args, vocab_size, device=None) -> tuple[nn.Module, dict]`
- `move_model_for_training(model, args, device) -> nn.Module`

This must preserve all current E97 behavior:

- `--level E97`/`97`/`E97-M2` plus `--bf16` auto-enables `--use_triton 1`;
- fused guard remains loud on GPU workers;
- all explicit geometry knobs used by the E97 1.3B recipe are passed through;
- `attach_model_run_metadata` metadata remains identical for checkpoints.

### Optimizer And ScheduleFree

Extract:

- `build_param_groups(core_model, args) -> list | Iterable`
- `build_optimizer(core_model, args) -> torch.optim.Optimizer`
- `set_schedulefree_train(optimizer, args)`
- `set_schedulefree_eval(optimizer, args, mode=None)`
- `extract_schedulefree_state(core_model, optimizer, basis="xz")`
- `load_schedulefree_state(core_model, optimizer, state, basis="xz")`

The async merge basis should default to ScheduleFree `x` plus `z`, not visible
model weights alone. For AdamW controls, the state extractor can use model
weights as `x` and omit `z`, but E97 production uses ScheduleFree.

### Checkpoint Load/Save

Keep existing `save_checkpoint` and `load_checkpoint` semantics available, but
add async-safe wrappers:

- `load_training_checkpoint(path, model, optimizer=None) -> TrainingCheckpoint`
- `save_training_checkpoint_atomic(model, optimizer, step, loss, output_dir, *, metadata, outer_state=None)`
- `checkpoint_payload_from_state(schedulefree_state, metadata) -> dict`
- `restore_state_from_payload(payload, model, optimizer)`

The async global supervisor must be able to publish a checkpoint from the merged
global ScheduleFree state even when no worker process is rank 0 of a
`torch.distributed` group. The save path must atomically write checkpoint files
and atomically advance run-local latest manifests/symlinks.

### Data Iterator

Extract:

- `build_training_dataset(args, *, data_seed) -> DatasetLike`
- `build_validation_loader(args, device)`
- `make_training_batch_fn(dataset, args, device) -> Callable`
- optional `PrefetchingBatchIterator`

Each GPU worker must read a distinct real token stream. Seed policy should be:

```text
data_seed = args.seed + global_worker_id
global_worker_id = node_index * gpus_per_node + local_gpu_id
```

This mirrors the existing distributed rank offset in `train.py` without
requiring a global process group.

### One/Few Step Function

Extract a reusable local training step object:

- `TrainingStepState(step, accumulated_steps, hidden_state, running_loss, tokens_processed, last_losses, stopped_nonfinite)`
- `run_one_microstep(model, optimizer, batch_fn, args, state, device) -> StepResult`
- `run_optimizer_step_if_ready(model, optimizer, args, state, device) -> StepResult`
- `run_local_training_window(runtime, *, max_optimizer_steps, max_seconds, stop_event) -> LocalWindowResult`

This function must include:

- BF16 autocast;
- TBPTT hidden-state reset;
- grad accumulation;
- gradient clipping;
- nonfinite loss/grad handling;
- AdamW LR schedule when selected;
- ScheduleFree train/eval mode preservation;
- token accounting using `actual_lengths.sum()`;
- loss moving windows compatible with current printed metrics.

The async worker should call `run_local_training_window` for `K` local optimizer
steps or a short time window, then produce an update.

### Metrics And Finalization

Extract:

- `LossWindowTracker`
- `compute_tokens_per_sec`
- `FinalCheckpointController` or a smaller walltime helper usable without
  distributed collectives
- `build_checkpoint_metadata(args, kind, step, loss, extra)`

The real async runner should not reuse `train.py`'s default distributed
collective finalization. Each worker reacts to supervisor stop/rebase commands;
the global supervisor owns final checkpoint publication.

## Worker/Supervisor Protocol

The first real implementation should use conservative process boundaries:

```text
one GPU worker process per GPU
    -> one node supervisor process per node
        -> one global supervisor process for the debug/smoke path
```

For 256-node production, insert group aggregators only after the single global
supervisor path passes 8/32/64-node smokes:

```text
GPU worker -> node supervisor -> group aggregator -> global supervisor
```

### Worker Lifecycle

Each GPU worker:

1. Starts with a real E97 runtime built from extracted `train.py` helpers.
2. Receives or loads global generation `g` with ScheduleFree state `S_g`.
3. Restores model/optimizer state to `S_g`.
4. Runs a local window of `K` optimizer steps or until `max_window_seconds`.
5. Extracts `S_i` and computes a dense delta against `S_g`:
   `delta_i = (x_i - x_g, z_i - z_g)`.
6. Sends metadata plus the update to the node supervisor:
   worker id, local GPU id, base generation, local steps, accepted tokens,
   loss windows, grad/loss health, elapsed time, update norms, and tensor payload.
7. Waits for `advance`, `rebase`, `reload`, `retry`, or `stop`.

Workers must not write authoritative global checkpoints or production latest
pointers. They may write non-authoritative cache manifests for debugging.

### Node Supervisor

Each node supervisor:

- launches or attaches to 8 GPU workers by default;
- tracks the current open local generation;
- accepts only worker updates with `base_generation == current_generation`;
- treats failed, invalid, timed-out, and stale worker reports as generation
  metadata, not process-fatal conditions;
- merges accepted GPU updates into one node update when local quorum is reached
  or a local timeout/health policy fires;
- sends one token-weighted node delta to the global supervisor;
- broadcasts the next generation state, rebase instruction, or deferred
  generation command to workers.

Default Frontier local quorum:

```text
local_quorum = 6 of 8 GPU workers per node
```

The CLI should also accept `--local-quorum-fraction`, default `0.75`, and
resolve it with `ceil(fraction * gpus_per_node)` so non-8-GPU unit tests are
natural.

### Global Supervisor

The global supervisor:

- owns authoritative global generation state and checkpoint publication;
- accepts one update per node per generation;
- rejects stale node updates by default;
- advances a generation after global quorum is reached;
- defers a generation on quorum miss instead of crashing;
- records accepted/stale/timed-out/failed/invalid nodes every generation;
- sends new generation state or rebase/reload instructions downstream.

Default Frontier global quorum:

```text
global_quorum = ceil(2/3 * requested_node_count)
```

The CLI should accept `--global-quorum-fraction`, default `0.667`, and resolve
with `ceil(fraction * node_count)`.

### Update Representation

The default correctness path uses dense deltas:

```text
AsyncTrainingUpdate(
    worker_id: str,
    scope: "gpu" | "node" | "group",
    base_generation: int,
    x_delta: state_dict-like tensors,
    z_delta: state_dict-like tensors,
    tokens: int,
    local_steps: int,
    loss_windows: dict[str, float],
    update_norms: dict[str, float],
    elapsed_s: float,
    status: "ok" | "stale" | "timed_out" | "failed" | "invalid",
)
```

Dense deltas are expensive but simplest to validate against synchronous DiLoCo
and current `ndm.async_diloco.quorum_merge` math. Compression, chunk streaming,
and sharded tensor transport should be follow-up performance work after the
real-token smoke ladder passes.

Merge policy:

```text
merged_delta = weighted_mean(delta_i, weight=tokens_i)
S_{g+1} = S_g + eta_outer * merged_delta
```

Accepted token count is the default weight. `local_steps` and equal weighting
remain debug/control modes.

### Rebase And Recovery

Default staleness policy is strict:

```text
accept iff update.base_generation == current_open_generation
```

Late updates are rejected for that generation. The worker then either reloads
`S_current` or rebases its local state:

```text
local_x <- local_x + (global_x_current - global_x_old_base)
local_z <- local_z + (global_z_current - global_z_old_base)
base_generation <- current_generation
```

Rebase preserves local displacement but does not make an already submitted stale
delta fresh. The first production path should use rebase/reload only to resume
training, not to accept stale updates. Bounded-stale Hogwild acceptance can be a
later explicit experiment with a separate `--max-staleness > 0` gate and
staleness weighting.

## Nonfatal Quorum Semantics

Quorum misses must not raise or crash the job in the normal async path.

Local generation outcomes:

- `advance`: accepted worker updates meet `local_quorum`; emit node update.
- `defer`: accepted updates below quorum before timeout; keep generation open
  and continue collecting unless walltime/global stop intervenes.
- `skip`: timeout or health policy says this node should not emit a node update
  for this global generation; report `timed_out`/`missing` to global supervisor.
- `cancel_node`: sustained local health failure exceeds configured threshold;
  stop this node's workers and report failed node health upstream.

Global generation outcomes:

- `advance`: accepted node updates meet `global_quorum`; publish generation
  `g+1`.
- `defer`: quorum miss before timeout; do not publish `g+1`; keep the same
  global generation open and wait for more node updates or a bounded retry.
- `skip`: global timeout/health policy closes the generation without advancing;
  instruct healthy nodes to retry/reload from the same current generation.
- `cancel_run`: sustained health criteria indicate the run is wasting allocation
  or cannot make progress; enter graceful finalization and publish the latest
  finalized global generation only.

Suggested sustained-health cancel defaults for first implementation:

- cancel if no global generation advances for `max(3, --global-stall-generations)`
  consecutive attempted windows;
- cancel if fewer than `global_quorum` nodes are healthy for more than
  `--global-health-grace-seconds` after startup;
- cancel if any required global supervisor/checkpoint invariant fails;
- cancel if walltime finalization reserve is reached.

All quorum misses must be visible in manifests and metrics:

- configured quorum threshold;
- effective accepted count;
- missing worker/node ids;
- stale/timed-out/failed/invalid counts;
- whether latest advanced;
- action taken: `advance`, `defer`, `skip`, `cancel_node`, or `cancel_run`.

## One-Node Behavior

A one-node run must reduce to local DiLoCo/model averaging semantics:

- `node_count=1` makes `global_quorum=1`;
- the node supervisor merges GPU worker updates using the same token-weighted
  local merge as multi-node;
- the global supervisor accepts the single node update and publishes the next
  global generation;
- with all 8 GPUs participating, `eta_outer=1`, equal local steps, and fresh
  bases, the result equals the synchronous local DiLoCo average over those GPU
  workers' ScheduleFree `x,z` states;
- with `gpus_per_node=1`, the path is one independent real training worker whose
  update is applied by the global supervisor and checkpointed.

Required tests:

- CPU toy ScheduleFree-state test: 1 node, N workers, full quorum equals direct
  average.
- Real short training test: 1 node / 1 GPU or CPU fallback, tiny E97-like config,
  one local window, checkpoint advances run-local latest.
- One-node quorum miss test: local quorum miss returns `defer`/`skip` metrics,
  not an exception.

## Checkpoint Policy

The global supervisor is the only authoritative checkpoint publisher.

Run-local layout:

```text
async_diloco_runs/<run_id>/
  generations/
    gen_000000/
      manifest.json
      state.pt              # optional/full recovery state
    gen_000001/
      manifest.json
      state.pt
  latest.json               # atomic pointer to latest finalized generation
  checkpoints/
    recovery/
    export/
  cache/
    workers/
    nodes/
```

Rules:

- Never mutate any production `latest.pt` or previous E97 continuation latest
  path from the async runner.
- The CLI must require an explicit `--run-dir` under the async run tree.
- `--production-latest-path` may be accepted only as a read-only guard path for
  before/after identity checks, matching the prototype.
- All manifests are written with temp-file plus atomic rename.
- The run-local latest pointer is advanced only after the generation manifest
  and any required checkpoint payload are durably written.
- Worker/node caches are non-authoritative and ignored by resume selection.
- Resume selection reads only finalized global generation manifests owned by the
  global supervisor.

Recovery cadence:

- write generation manifests every attempted global generation, including
  deferred/skipped attempts;
- write recovery checkpoints by generation interval or wall-clock interval,
  whichever fires first;
- write finalization checkpoint before walltime reserve expires;
- write export checkpoints less frequently for evaluation/S3 workflows.

Initial smoke defaults can keep recovery frequent, for example every generation
or every 5-10 minutes, then relax after checkpoint overhead is measured.

## Production Submit Gates

No production-scale Slurm jobs are authorized by this design task. The old
`retry-refreshed-e97` / 256-node retry remains paused and no-go until the real
training path below passes.

Exact gate ladder:

1. Unit tests for extracted train helpers, async merge outcomes, nonfatal quorum
   actions, one-node reduction, checkpoint latest atomicity, and stale rejection.
2. Local/CI tiny real-token training test with a small config and fake/CPU async
   supervisors.
3. Frontier `1n20m` real E97 async smoke: one node, real data, real checkpoint
   load, real worker training, local quorum behavior, run-local latest only.
4. Frontier `2n20m` real E97 async smoke: two nodes, global quorum
   `ceil(2/3 * 2)=2` unless explicitly testing quorum miss; verify no crash on
   induced local/global miss.
5. Frontier `8n20m` exact production wrapper dry/smoke: same wrapper shape and
   environment as the eventual larger launch, but short walltime and debug run
   directory.
6. Frontier `32n` smoke only after 8-node pass and artifact review.
7. Frontier `64n` smoke only after 32-node pass and artifact review.
8. Frontier `256n12h` only after the 64-node smoke passes, the WG artifacts show
   real token training and checkpoint safety, and a human/coordinator explicitly
   reopens the production gate.

Any wrapper named like the old synthetic path must be treated as no-go for
production unless it calls `async_diloco_e97_real.py` and its report proves real
`train.py` token training. The current synthetic/prototype harness may continue
as a metrics/checkpoint/quorum regression test, but it is not sufficient for
production submission.

## Concrete Implementation Tasks

Downstream tasks should implement in this order:

1. `refactor-train-py`: extract the import-safe helpers listed above, preserving
   `train.py` CLI behavior. File scope: `train.py`, new `ndm/training/*.py`,
   focused tests.
2. `implement-nonfatal-async`: extend `ndm.async_diloco` with explicit
   `advance`/`defer`/`skip`/`cancel` outcomes instead of raising on quorum miss.
   File scope: `ndm/async_diloco.py`, `ndm/async_diloco_local.py`, tests.
3. Add real worker runtime: `ndm/async_diloco_real_worker.py` or
   `ndm/async_training_worker.py`, using the extracted train helpers for
   checkpoint restore, batch iteration, and local windows.
4. Add node/global supervisor runtime:
   `ndm/async_diloco_supervisor.py`, with in-process/IPC transport first and a
   transport abstraction for Frontier GPU-aware MPI later.
5. Add CLI entrypoint: `scripts/frontier/async_diloco_e97_real.py`, with flags
   for run id, checkpoint, run dir, data, local/global quorum fractions,
   timeouts, local steps, walltime finalization, and production-latest guard.
6. Add smoke wrapper: `scripts/frontier/async_diloco_e97_real.sbatch` only for
   1n/2n/8n short gates initially. Larger-node wrappers should remain absent or
   guarded until the validation ladder reaches them.
7. Add reports for each validation run under `reports/frontier/` and register
   them as WG artifacts.

## Validation Checklist For This Design

- Written under `reports/frontier/`.
- Separates the synthetic prototype harness from the real token-training path.
- Defines nonfatal quorum semantics and default quorum fractions:
  local `ceil(0.75 * gpus_per_node)` / Frontier default 6 of 8, global
  `ceil(2/3 * node_count)`.
- States exact production-submit gates and keeps old `retry-refreshed-e97` /
  256-node retry paused and no-go.
- Identifies concrete files/modules/scripts to implement next.
- Submits no Slurm production job.
