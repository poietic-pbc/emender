# Frontier E97 non-avg outer resume bootstrap design

Date: 2026-06-25
Task: `design-e97-non`

## Scope and decision

This is a design/audit artifact only. I did not implement code and did not
submit any Slurm job.

The required code change is an explicit bootstrap path for missing DiLoCo
outer optimizer state when resuming from a checkpoint that lacks compatible
`diloco_outer_state`, such as an `avg`-outer E97 checkpoint or a user-provided
pretrained checkpoint. The default behavior must remain fail-closed: a
non-`avg` resume without compatible outer state should raise before training
unless the operator explicitly requests bootstrap.

Recommended CLI/config guard:

```text
--diloco_bootstrap_outer_state {none,from-loaded-model}
```

Default: `none`.

`from-loaded-model` means: after model and available optimizer state have been
loaded, create only the missing DiLoCo outer optimizer state from the current
loaded model/optimizer tensors. The helper must restore the live model tensors
byte-for-byte to their pre-bootstrap values before returning. It must not
average ranks, reinitialize parameters, apply an outer step, or silently change
the loaded model basis.

For Frontier wrappers, mirror the guard as:

```text
DILOCO_BOOTSTRAP_OUTER_STATE=from-loaded-model
```

and pass it through only when set. The absence of that environment variable
must preserve today's hard failure.

## Current code audit

Relevant command surface:

- `train.py:337-356` exposes `--diloco`,
  `--diloco_outer_optimizer {avg,momentum,sfsgd}`, `--diloco_outer_lr`,
  `--diloco_outer_beta`, and `--diloco_export_basis {x,y}`.
- `train.py:873-889` saves optional `checkpoint_metadata` and optional
  `diloco_outer_state`.
- `train.py:924-933` loads `model_state_dict`, optional optimizer state, and can
  return the raw checkpoint payload.
- `train.py:1010-1063` restores or initializes explicit outer state.
- `train.py:1070-1347` consumes that state in the real DiLoCo merge.
- `train.py:2121-2125` currently rejects `--resume` plus non-`avg`
  `--diloco_outer_optimizer` when `diloco_outer_state` is missing.
- `scripts/frontier/diloco_scaleout_readiness.sbatch:31-35` exposes the DiLoCo
  outer config through environment variables, and
  `scripts/frontier/diloco_scaleout_readiness.sbatch:127-130` passes
  `RESUME_CHECKPOINT` to `--resume`.

That existing failure is correct and should remain the default. The only change
should be to admit a clearly named operator opt-in that records what happened.

## State required by each outer path

### `avg`

`avg` is stateless. `initialize_diloco_outer_state()` returns `None`, and
`diloco_merge()` performs only the rank average of model weights plus the
ScheduleFree inner `x/z` consensus when the inner optimizer is schedule-free.

Resume requirement: no `diloco_outer_state` required. A checkpoint from `avg`
should remain directly resumable with `--diloco_outer_optimizer avg`.

### `momentum`

`momentum` requires:

```text
{
  "mode": "momentum",
  "anchor": [tensor_like_each_param],
  "moment": [tensor_like_each_param],
  optional diagnostics such as "last_metrics"
}
```

`diloco_merge()` interprets `anchor` as `W_r`, the outer round anchor in the
same basis that is merged at a DiLoCo boundary. For inner ScheduleFree, that is
the eval/averaged `x` basis because `diloco_merge()` calls `optimizer.eval()`
before all-reducing model weights. `moment` is the accumulated outer momentum.

Partial-average is not a separate code path today. It is represented by
`momentum` with `outer_beta=0.0` and `0 < outer_lr < 1`, giving:

```text
W_{r+1} = W_r + outer_lr * (mean_i(W_{r,i}) - W_r)
```

Resume requirement: either compatible checkpoint state with `mode="momentum"`,
or explicit bootstrap from loaded weights.

### `sfsgd`

`sfsgd` requires inner `--optimizer schedulefree` and outer state:

```text
{
  "mode": "sfsgd",
  "x": [tensor_like_each_param],
  "z": [tensor_like_each_param],
  "y": [tensor_like_each_param],
  "k": int,
  "weight_sum": float,
  "lr_max": float
}
```

For the desired `sfsgd_y` arm, the CLI shape is:

```text
--optimizer schedulefree
--diloco_outer_optimizer sfsgd
--diloco_export_basis y
--diloco_outer_lr 1.0
--diloco_outer_beta 0.1
```

At each merge, `diloco_merge()` exports the inner train point `y` when
`--diloco_export_basis y` is set, advances the separate outer ScheduleFree-SGD
state, then translates the inner ScheduleFree `x/z/y` geometry by the resulting
outer displacement. Resume requirement: either compatible checkpoint state with
`mode="sfsgd"`, or explicit bootstrap from loaded weights.

## Bootstrap semantics

The bootstrap source is the already-loaded model state, not random
initialization and not a cross-rank average. It is a bookkeeping operation that
defines outer optimizer anchors at the resume boundary.

Before creating any bootstrap state, take a byte-for-byte snapshot of the live
model parameters:

```text
pre_bootstrap_params = [p.detach().clone() for p in core_model.parameters()]
```

The bootstrap helper may switch ScheduleFree modes internally to derive the
needed basis, but it must restore every `p.data` from `pre_bootstrap_params`
before returning and tests must assert exact equality. This protects against
the existing `optimizer.eval()` / `optimizer.train()` basis swaps becoming a
hidden model mutation.

### Momentum and partial-average initialization

For `--diloco_outer_optimizer momentum`:

1. If inner optimizer is ScheduleFree and optimizer state exists, derive the
   anchor in eval/`x` basis:

   ```text
   optimizer.eval()
   anchor = clone(current p.data for each parameter)
   restore pre_bootstrap_params exactly
   ```

2. If inner optimizer is not ScheduleFree, or if this is a pretrained
   schedule-free continuation with no loaded ScheduleFree optimizer state and
   therefore `x == y == loaded weights` by construction, use the loaded
   parameter tensors as the anchor.

3. Initialize momentum buffers to exact zeros:

   ```text
   moment = [zeros_like(p.data) for p in parameters]
   ```

4. Return:

   ```text
   {"mode": "momentum", "anchor": anchor, "moment": moment, "bootstrap_metadata": ...}
   ```

Mathematical coherence: the first post-resume merge compares each rank's
post-local-training boundary weights to the resume boundary anchor. With
`moment=0`, the first outer step is exactly the same as starting a fresh
momentum or partial-average outer optimizer at the loaded checkpoint. For
partial-average (`outer_beta=0`, `outer_lr<1`), the first merge lands a fraction
of the K-step drift away from the loaded checkpoint rather than pretending the
checkpoint had prior outer history.

### `sfsgd_y` initialization

For `--diloco_outer_optimizer sfsgd --diloco_export_basis y`:

1. Require inner `--optimizer schedulefree`.

2. Determine the loaded train point `y0`.

   If a compatible ScheduleFree optimizer state was loaded, the helper may call
   `optimizer.train()` to derive `y0`, then must restore the exact
   `pre_bootstrap_params`. If no optimizer state exists, treat the loaded model
   weights as the fresh train point: `x0 == z0 == y0 == loaded weights`.

3. Initialize:

   ```text
   outer_y = clone(y0)
   outer_z = clone(y0)
   outer_x = clone(y0)
   k = 0
   weight_sum = 0.0
   lr_max = float(args.diloco_outer_lr)
   ```

4. Return:

   ```text
   {
     "mode": "sfsgd",
     "x": outer_x,
     "z": outer_z,
     "y": outer_y,
     "k": 0,
     "weight_sum": 0.0,
     "lr_max": outer_lr,
     "bootstrap_metadata": ...
   }
   ```

Mathematical coherence: the outer ScheduleFree-SGD system starts at the resume
boundary train point. Its first merge computes `delta = exported_y_after_K -
outer_y`, with no fake outer averaging mass. This is equivalent to starting the
outer `sfsgd_y` optimizer fresh at the checkpoint and continuing from there.

For `sfsgd` export `x`, the same structure applies except `outer_x/z/y` should
be initialized from the export basis that will be used for the first boundary.
Because current Frontier evidence prefers `sfsgd_y`, tests should cover `y`
first and may include `x` as a lower-priority regression.

## Fail-closed behavior

The resume decision should be centralized before calling
`initialize_diloco_outer_state()`:

```text
if use_diloco and args.resume and args.diloco_outer_optimizer != "avg":
    if loaded_outer_state is None:
        if args.diloco_bootstrap_outer_state != "from-loaded-model":
            raise ValueError(
                "checkpoint lacks diloco_outer_state for non-avg DiLoCo outer "
                "optimizer; rerun with --diloco_bootstrap_outer_state "
                "from-loaded-model only if you intend to start fresh outer state "
                "from the loaded model weights"
            )
        bootstrap_reason = "missing_diloco_outer_state"
    elif loaded_outer_state.get("mode") != args.diloco_outer_optimizer:
        if args.diloco_bootstrap_outer_state != "from-loaded-model":
            raise ValueError(...)
        bootstrap_reason = (
            "incompatible_diloco_outer_state_mode:"
            f"{loaded_outer_state.get('mode')}->{args.diloco_outer_optimizer}"
        )
        loaded_outer_state = None
```

If `loaded_outer_state` exists and has the requested mode, ignore
`--diloco_bootstrap_outer_state from-loaded-model` and restore the real state,
or fail with a clear "bootstrap requested but compatible state exists" error.
The stricter choice is preferable for auditability: bootstrap should only be
used when state is absent or incompatible.

Never silently downgrade `momentum` or `sfsgd` to `avg`. Never create fresh
outer state on resume under the existing default configuration.

## Checkpoint metadata and reporting

Every checkpoint saved after a bootstrap must carry metadata that makes the run
honest in future evals and reports. Extend the existing `checkpoint_metadata`
payload passed to `save_checkpoint()` with:

```text
"diloco_outer_state_bootstrap": {
  "performed": true,
  "guard": "from-loaded-model",
  "requested_cli": "--diloco_bootstrap_outer_state=from-loaded-model",
  "source_checkpoint": args.resume,
  "source_checkpoint_step": ckpt.get("step"),
  "source_checkpoint_has_diloco_outer_state": false,
  "missing_or_incompatible_reason": "missing_diloco_outer_state",
  "target_outer_optimizer": args.diloco_outer_optimizer,
  "target_export_basis": args.diloco_export_basis,
  "target_outer_lr": args.diloco_outer_lr,
  "target_outer_beta": args.diloco_outer_beta,
  "bootstrap_source": "loaded_model_weights",
  "model_weight_mutation": "none; restored byte-identical after bootstrap",
  "inner_optimizer_state_loaded": true_or_false,
  "inner_schedulefree_basis_used_for_anchor": "x" | "y" | "loaded",
  "created_at_utc": "...",
  "code_commit": git_commit_if_available
}
```

Also print a rank-0 startup line such as:

```text
[DiLoCo] bootstrapped missing outer state from loaded model weights:
mode=sfsgd export_basis=y reason=missing_diloco_outer_state guard=from-loaded-model
source=<path> step=<step>; model tensors restored byte-identical after bootstrap
```

The Frontier launch manifest in
`scripts/frontier/diloco_scaleout_readiness.sbatch` should record
`diloco_bootstrap_outer_state` alongside `resume_checkpoint`,
`diloco_outer_optimizer`, `diloco_outer_lr`, `diloco_outer_beta`, and
`diloco_export_basis`.

## Files and functions likely needing changes

- `train.py:327-356`: add parser argument
  `--diloco_bootstrap_outer_state {none,from-loaded-model}`.
- `train.py:873-889`: no structural change required, but saved metadata should
  include the bootstrap block.
- `train.py:924-933`: optionally expose whether optimizer state was present, or
  derive it from the returned checkpoint.
- `train.py:997-1007`: reuse cloning helpers, but add exact snapshot/restore
  helpers for bootstrap.
- `train.py:1010-1063`: extend `initialize_diloco_outer_state()` with explicit
  `bootstrap=False`, `bootstrap_reason=None`, and metadata return, or add a
  separate `bootstrap_diloco_outer_state_from_loaded_model()` wrapper to avoid
  blurring restore versus bootstrap.
- `train.py:2111-2139`: replace the current missing-state hard error with the
  fail-closed guard above; print restore/bootstrap status.
- `scripts/frontier/diloco_scaleout_readiness.sbatch:31-35` and
  `scripts/frontier/diloco_scaleout_readiness.sbatch:97-130`: add optional
  `DILOCO_BOOTSTRAP_OUTER_STATE` passthrough and manifest/env logging.
- `tests/test_diloco_merge.py`: add focused CPU tests for bootstrap behavior.

## Unit tests

Add tests that run on CPU and do not submit jobs:

1. `test_nonavg_resume_missing_outer_state_fails_closed`

   Build a tiny real ScheduleFree model, save a checkpoint without
   `diloco_outer_state`, load it, and assert non-`avg` resume decision raises
   unless `--diloco_bootstrap_outer_state from-loaded-model` is set.

2. `test_bootstrap_momentum_preserves_loaded_model_tensors`

   Load a checkpoint without outer state, snapshot every `p.data`, bootstrap
   `momentum`, then assert every parameter is `torch.equal()` to its snapshot.
   Assert `outer_state["mode"] == "momentum"`, `moment` tensors are exact zeros,
   and `anchor` tensors equal the expected eval/`x` basis for ScheduleFree.

3. `test_bootstrap_partial_average_first_merge_math`

   Two-rank gloo CPU test using the existing `tests/test_diloco_merge.py`
   harness. Bootstrap `momentum` with `outer_beta=0.0`, `outer_lr=0.5`, run one
   real divergent local window, call `diloco_merge()`, and assert the result is
   `anchor + 0.5 * (mean_x_after_local - anchor)` with exact consensus.

4. `test_bootstrap_sfsgd_y_preserves_loaded_model_tensors`

   With inner ScheduleFree optimizer state present, snapshot live tensors,
   bootstrap `sfsgd` export `y`, assert parameters are exactly unchanged, and
   assert `outer_x`, `outer_z`, and `outer_y` all equal the derived train point
   used for `sfsgd_y`. Assert scalars `k=0`, `weight_sum=0.0`, and
   `lr_max=args.diloco_outer_lr`.

5. `test_bootstrap_sfsgd_y_pretrained_no_optimizer_state`

   Simulate a user-provided pretrained checkpoint with model weights but no
   optimizer state. Bootstrap `sfsgd_y` and assert `outer_x == outer_z ==
   outer_y == loaded model weights`, all parameters remain unchanged, and the
   metadata reports `inner_optimizer_state_loaded=false`.

6. `test_checkpoint_metadata_records_outer_bootstrap`

   Save a post-bootstrap checkpoint and reload it. Assert
   `checkpoint_metadata["diloco_outer_state_bootstrap"]` records the guard,
   source checkpoint, missing-state reason, target outer config, source basis,
   and "no model mutation" statement.

7. `test_compatible_outer_state_not_overwritten_by_bootstrap_flag`

   Save a checkpoint with `diloco_outer_state.mode="sfsgd"`, then request
   `sfsgd` with bootstrap flag set. Assert the implementation either restores
   compatible state and records no bootstrap, or raises a clear error. Do not
   allow silent overwrite.

These tests should explicitly compare tensor values with `torch.equal()`, not
only tolerances, for the "no model mutation" invariant.

## Smallest live smoke

No 32-node allocation is needed. The smallest live smoke should be local CPU or
single-node only:

1. Create a tiny checkpoint without `diloco_outer_state` using the existing unit
   model in `tests/test_diloco_merge.py`.
2. Run a two-process gloo CPU DiLoCo smoke through the real `diloco_merge()`:
   one arm for `momentum` partial-average (`outer_lr=0.5`, `outer_beta=0.0`),
   one arm for `sfsgd_y` (`outer_lr=1.0`, `outer_beta=0.1`).
3. Verify startup logs show the explicit bootstrap guard and reason.
4. Verify saved checkpoint contains `diloco_outer_state` and bootstrap metadata.
5. Verify no parameter tensor changed during bootstrap by recording hashes before
   the first training step and after bootstrap.

A later optional Frontier smoke can be a one-node or two-node short job only if
a separate human-authorized task permits it. This design task does not
authorize `sbatch`.

## User-provided pretrained E97 continuation

This design enables pretrained continuation by treating the external checkpoint
as a legitimate resume boundary:

1. A separate intake task verifies model architecture, tensor names/shapes,
   tokenizer/vocab, context length, dtype, and eval compatibility.
2. `train.py` loads the model weights and any optimizer state present.
3. The user explicitly requests:

   ```text
   --diloco --diloco_outer_optimizer sfsgd --diloco_export_basis y \
   --diloco_bootstrap_outer_state from-loaded-model
   ```

   or:

   ```text
   --diloco --diloco_outer_optimizer momentum \
   --diloco_outer_lr 0.5 --diloco_outer_beta 0.0 \
   --diloco_bootstrap_outer_state from-loaded-model
   ```

4. If the checkpoint has no compatible outer state, the code creates fresh
   outer bookkeeping at that loaded model point, records the bootstrap, and
   keeps model weights unchanged.

This is honest continuation from a pretrained model with fresh outer optimizer
state. It is not equivalent to restoring a historical `sfsgd` or momentum
outer trajectory, and reports must say so.

## Implementation recommendation

Create a follow-up implementation task. Keep it test/smoke-only:

- implement the parser guard and bootstrap helper;
- add the unit tests above;
- add wrapper passthrough/manifest fields;
- run `python tests/test_diloco_merge.py` or the narrow pytest selection;
- do not submit Slurm jobs;
- do not modify or resume `run-64-node-e97`.

Any 4/8/16/32/64-node training use of the new guard should require a separate
human-authorized launch task with checkpoint path, target config, validation
gate, and allocation budget.

## Scope confirmation

- No Slurm training jobs were submitted from this task.
- No `sbatch` command was run from this task.
- `run-64-node-e97` remains paused/open per the current graph and upstream
  quality-pass constraints.
