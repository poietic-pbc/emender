# E97 / E88 Kernel Naming Clarification

Date: 2026-08-02

Status: provenance finding and implemented compatibility layer

## Decision in one paragraph

The completed 150B-token Emender run is an **E97 split-edit model**, not an E88
run. Its nonlinear sequential recurrence was executed by the fused Triton engine
whose Python modules and public functions still carry the older `e88_*` names.
E97 was added by extending that engine with `SPLIT_EDIT`, erase/read-gate, and
value-write-gate inputs instead of copying the recurrence into a new module. Later,
a genuinely different, linear-state chunked implementation was added under
`e97_chunked*`. The compatibility change adds an E97-named **sequential façade**
that delegates to the proven shared engine, an E97-named model wrapper,
unambiguous runtime metadata, and a strict checkpoint/generation API. It does not
copy or rename the underlying kernels. User instructions are in
[E97_CHECKPOINT_LOADING_AND_GENERATION.md](E97_CHECKPOINT_LOADING_AND_GENERATION.md).

## What the 150B artifact actually is

The completed run is:

| Fact | Recorded value |
|---|---|
| Run | `emender_E97_1.3B_20260709_084606` |
| Level | `E97` |
| Parameters | `1,286,589,072` |
| Final step | `2,300,930` |
| Tokens per step | `65,536` |
| Final tokens | `150,793,748,480` |
| Triton | `use_triton=1` |
| E97 split edit | `use_split_edit=True` at runtime |
| State map | nonlinear, `linear_state=0` |
| Recurrence | delta-correcting, `e88_raw_write=0` |
| Chunked E97 | disabled, `use_chunked_e97=0` |
| Output gate | enabled, `use_gate=1`, `gate_activation=silu` |

Primary evidence:

- The frozen arguments are at
  `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/emender_E97_1.3B_20260709_084606/args.json`.
- The run log says, on every rank, `level=E97 ... fused split-edit Triton
  kernel, NO eager fallback`. It also reports
  `path=e88-sequential-split-edit-triton`, `use_split_edit=True`,
  `use_chunked_e97=False`, and `linear_state=False`.
- The final token count, checkpoint, checksum, and upload location are recorded in
  [EMENDER_DILOCO_OPS_HANDOFF_20260722.md](EMENDER_DILOCO_OPS_HANDOFF_20260722.md).
- The launch recipe explicitly selects `--level E97`, `--use_triton 1`, and the
  SiLU output gate in
  [`scripts/launch_emender_8gpu_diloco.sh`](../scripts/launch_emender_8gpu_diloco.sh).

The correct compact description is therefore:

> **Emender/nonlin (historical level E97), using the fused sequential Triton
> split-edit recurrence backed by the shared E88-derived engine.**

Calling the artifact simply “E88” is incorrect. Calling the shared implementation
core “E88-derived” is correct.

## The E97 recurrence

For each batch element and head, E97 has matrix state `S` with key dimension `N`
and value dimension `V`. Omitting projection preprocessing for clarity:

```text
erase_t       = sigmoid(W_e x_t)                 # [N]
write_t       = sigmoid(W_w x_t)                 # [V]
read_key_t    = erase_t * k_t                    # [N]
write_value_t = write_t * v_t                    # [V]
delta_t       = write_value_t - S_(t-1)^T read_key_t
S_t           = tanh(decay_t * S_(t-1) + k_t delta_t^T)
out_t         = silu(g_t) * (S_t^T q_t)
```

This is the GDN-2-inspired part: erase/read and value-write control are separated
along the two axes of the matrix state. It is **inspired by GDN-2**, not the same
model as the repository's separate `gdn2-mlp` control.

Three similarly named gates must not be conflated:

1. `erase_gate_proj` and `value_write_gate_proj` are the E97 split-edit gates.
2. `use_gate=1` enables the SiLU output gate after the state readout.
3. `use_write_gate=0` disables an older optional FLA-GDN beta-style scalar write
   gate. That zero does not disable the E97 value-axis write gate.

## How E97 ended up inside E88-named code

The repository history shows the evolution clearly:

1. E88 already supplied the projections, Mamba-style decay, nonlinear matrix
   state, delta correction, output gating, and fused sequential Triton machinery.
2. Commit `21f03a3d` (`feat: add E97 split-edit Triton path`) extended
   `E88FLAHybrid` and the existing E88 Triton forward/backward kernels in place.
   It added `use_split_edit`, `erase_gate`, `value_write_gate`, `RAW_WRITE`, and
   `SPLIT_EDIT`, then registered level E97 as:

   ```python
   E88FLAHybrid(..., use_split_edit=True)
   ```

3. Commit `23a6c487` added the E97 linear-state ablation to the same shared
   sequential kernel.
4. The E97 production route consequently became:

   ```text
   --level E97
     -> LadderLM level registry
     -> E88FLAHybrid(use_split_edit=True)
     -> e88_triton_optimized_apply(..., erase_gate=..., value_write_gate=...)
     -> e88_triton / E88TritonFunction
     -> e88_triton_forward + e88_triton_backward with SPLIT_EDIT=True
   ```

5. Later work added `e97_chunked.py`, `e97_chunked_autograd.py`, and related
   modules. Those names describe a **new chunked-parallel algorithm for the
   linear-state E97 recurrence**, not an alias for the nonlinear sequential
   production path.

The implementation reuse was reasonable: the two recurrences share most of their
math, and keeping one fused engine avoided duplicated forward/backward code. The
problem is that the public symbols, class name, file names, tests, and runtime path
continued to expose only the ancestor name. Tools or people inspecting the class or
kernel path without also checking `level` and `SPLIT_EDIT` can therefore report the
wrong model identity.

## What existed before the compatibility change

### Already present at the start

- An E97 model-level selector: `--level E97` maps to
  `E88FLAHybrid(use_split_edit=True)`.
- E97 behavior in the sequential fused forward and backward kernels, selected by
  the compile-time `SPLIT_EDIT` flag and the two split-edit tensors.
- A loud production guard proving the E97 bf16 run uses fused Triton without an
  eager fallback.
- An E97 runtime diagnostic, although its path value currently begins with `e88-`.
- E97-named **chunked** APIs for the linear-state path.

### Missing at the start

- No E97-named public façade for the **sequential** fused path.
- No E97-named model class; an instantiated E97 layer is still an
  `E88FLAHybrid`.
- No structured runtime identity separating model, API, implementation core, and
  execution algorithm.

So the forwarding layer was **not already present** for the path used by the
150B run.

## Implemented cleanup: façade, not fork

### Runtime and documentation identity

The runtime record now separates identity from implementation lineage:

```text
[e97-runtime] model=emender/nonlin historical_level=E97 backend=cuda
path=e97-sequential-split-edit-triton
kernel_api=e97-sequential-split-edit-triton kernel_core=e88-shared-triton
recurrence=sequential state=tanh eager_fallback=False
```

Keep the old run logs immutable. Document their translation:

```text
e88-sequential-split-edit-triton
  == e97-sequential-split-edit-triton backed by the shared E88-derived core
```

### E97 sequential API façade

`ndm/triton/e97_sequential.py` now provides the clearly named entry point:

```python
def e97_split_edit_triton_apply(..., erase_gate, value_write_gate, ...):
    """Run the sequential fused E97 split-edit recurrence."""
    # Validate that both split-edit gates are supplied.
    return e88_triton_optimized_apply(
        ...,
        erase_gate=erase_gate,
        value_write_gate=value_write_gate,
        ...,
    )
```

E97 calls route through this façade while E88 calls stay on the existing API.
The façade is exported from `ndm.triton`, and every current `e88_*` symbol remains
intact for backward compatibility.

This is preferable to copying a kernel because it:

- gives users, profilers, stack traces, and documentation an E97-native entry;
- preserves the exact proven forward/backward implementation;
- prevents E88 and E97 copies from drifting numerically;
- does not alter parameter names or checkpoint compatibility; and
- leaves room to rename the common core later, if that is ever worthwhile.

### E97 model wrapper

The thin `E97SplitEditLayer(E88FLAHybrid)` wrapper forces
`use_split_edit=True`, and the level registry constructs that type. This gives
introspection an E97 identity without changing the module tree or state-dict keys.
Reject attempts to pass `use_split_edit=False` through the wrapper.

The long-term dynamics name can be `EmenderNonlinearLayer`; the wrapper can retain
`E97SplitEditLayer` as the historical compatibility name. Run and checkpoint labels
must continue to preserve `E97` because that is the artifact's recorded identity.

### Identity, loading, and generation tests

Tests assert that:

- level E97 constructs the E97 wrapper and always enables split edit;
- the E97 sequential façade rejects a missing erase or write gate;
- the façade delegates its unchanged arguments to the shared core;
- E97 runtime metadata says E97 even though the implementation core is shared;
- E88 without split edit continues to use the E88 API and identity; and
- E97 state-dict keys remain identical to the historical split-edit layer, and
  E97 checkpoints strict-load through the public loader.

The public `ndm.e97` loader reconstructs the complete training geometry, recovers
schedule-free y/train weights when requested, and exposes safe fused full-context
and exact eager stateful generation modes.

### P2: describe the common core honestly

If further cleanup is useful, introduce neutral internal names such as
`matrix_delta_triton_forward` and keep `e88_*` and `e97_*` as compatibility
wrappers. This is optional and higher risk because many tests and imports use the
existing internal names. Do not block the P0/P1 clarity fixes on it.

## What not to do

- **Do not duplicate the sequential kernel into an E97 copy.** The backward is
  substantial and duplication creates an immediate parity and maintenance risk.
- **Do not route the nonlinear 150B configuration through `e97_chunked`.** Chunked
  E97 requires `linear_state=True`; the completed model has per-step `tanh` and
  must scan sequentially.
- **Do not rename checkpoint parameters.** The trained artifact's state dict is the
  compatibility boundary.
- **Do not rewrite historical logs or call the 150B artifact E88.** Preserve the
  evidence and provide the translation above.
- **Do not call the E97 model GDN-2.** “GDN-2-inspired split-edit gating” is precise;
  the repository has a separate GDN-2 control model.

## Acceptance criterion for the cleanup

After the change, a person who inspects any one of the model type, runtime banner,
public Triton entry point, frozen arguments, or documentation should reach the same
conclusion:

> This is E97/Emender nonlinear split edit, executing the sequential fused Triton
> algorithm through a shared E88-derived implementation core.
