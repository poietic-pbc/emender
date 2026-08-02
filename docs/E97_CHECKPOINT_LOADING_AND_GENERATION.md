# Loading and Generating with E97 Checkpoints

This is the supported path for native `train.py` E97/Emender checkpoints,
including the completed 150B-token run. It reconstructs the model as E97,
strictly loads the historical state dict, and makes the shared E88-derived
implementation core an internal detail.

For the naming and implementation history, see
[E97_E88_KERNEL_NAMING_CLARIFICATION_20260802.md](E97_E88_KERNEL_NAMING_CLARIFICATION_20260802.md).

## Required files

Keep these together in one directory:

```text
args.json
checkpoint_step_<step>_loss_<loss>.pt
```

Alternatively, pass the arguments separately with `--args-json`. The `.pt`
checkpoint contains weights and optimizer state but does not embed the complete
training arguments, so `args.json` is part of the loadable artifact.

The completed local reference is:

```text
/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/
  emender_E97_1.3B_20260709_084606/
    args.json
    checkpoint_step_2300930_loss_2.4365.pt
```

Its recorded identity is E97/Emender nonlinear split edit, 1,286,589,072
parameters, step 2,300,930, and 150,793,748,480 training tokens.

## Generate from the command line

From the repository root:

```bash
python scripts/generate_e97.py \
  --checkpoint /path/to/emender_E97_run/checkpoint_step_2300930_loss_2.4365.pt \
  --prompt $'\x1eThe theorem states' \
  --max-new-tokens 64
```

The checkpoint argument may instead name its run directory or `latest.pt`. If
`args.json` is elsewhere:

```bash
python scripts/generate_e97.py \
  --checkpoint /path/to/checkpoint.pt \
  --args-json /path/to/args.json \
  --prompt "Once upon a time"
```

Defaults for the 150B checkpoint are intentional:

- `--weight-mode train` recovers the schedule-free y/train weights using the
  checkpoint's optimizer state. The stored x/averaged weights are available via
  `--weight-mode saved`, but are not the established generation weights for this
  run.
- `--dtype bfloat16 --device cuda` matches the trained numerical path.
- `--mode auto` selects fused full-context generation when Triton is enabled.

The model is a raw base language model, not an instruction- or chat-tuned model.

## Load and generate from Python

```python
from ndm.e97 import generate_e97, load_e97_checkpoint

loaded = load_e97_checkpoint(
    "/path/to/emender_E97_run",  # run directory, latest.pt, or checkpoint .pt
    device="cuda",
    dtype="bfloat16",
    weight_mode="train",
)

print(type(next(m for m in loaded.model.modules()
                if m.__class__.__name__ == "E97SplitEditLayer")))

sample = generate_e97(
    loaded,
    "\x1eThe theorem states",
    max_new_tokens=64,
    temperature=0.8,
    top_k=40,
)
print(sample["text"])
print(sample["kernel_api"])
```

The reconstructed recurrent layer type is `E97SplitEditLayer`. Existing
checkpoint keys remain unchanged, so strict loading is compatible with the
historical checkpoint written when the type was reported as `E88FLAHybrid`.

## Generation modes

### Fused full-context mode

`--mode full-context`, and `auto` for a Triton-enabled checkpoint, recomputes a
bounded context for every new token. It executes through the public
`e97-sequential-split-edit-triton` façade, backed by the shared E88-derived
forward/backward implementation.

This is the safe fused path today. The shared sequential kernel pads lengths to
its sparse-checkpoint interval. Outputs at real token positions are causal and
unaffected by the padded tail, so using the last real output is correct.

### Exact stateful mode

```bash
python scripts/generate_e97.py \
  --checkpoint /path/to/emender_E97_run \
  --prompt "Hello" \
  --mode stateful
```

Stateful mode processes the prompt once and carries the matrix state one token
at a time. The CLI deliberately disables Triton for this mode and uses the exact
E97 reference recurrence. The current fused wrapper's padded final state must not
be carried across token calls. A future parity-cleared E97 single-token kernel can
remove this restriction without changing the public generation API.

## What the compatibility layer guarantees

- `--level E97` constructs `E97SplitEditLayer`, which forces
  `use_split_edit=True`.
- `ndm.triton.e97_split_edit_triton_apply` is the E97-named sequential fused API.
- The implementation still delegates to the proven E88-derived Triton core.
- State-dict keys and tensor shapes are unchanged.
- Loading is strict: missing or unexpected checkpoint tensors raise an error.
- The full E97 training geometry is reconstructed, including `mlp_ratio`, gate
  configuration, state geometry, decay, raw/delta mode, and chunked/sequential
  selection.
- Runtime metadata separates `model=emender/nonlin`, `historical_level=E97`,
  `path=e97-sequential-split-edit-triton`, and `kernel_core=e88-shared-triton`.

## Packaging future checkpoints

Every native E97 checkpoint release should include:

1. the `.pt` checkpoint;
2. its exact `args.json`;
3. a checksum manifest;
4. the training step, token count, loss, and tokenizer;
5. whether schedule-free weights are stored as x/saved or exported as y/train;
6. a command using `scripts/generate_e97.py`; and
7. the source revision containing the E97 compatibility layer.

If a future public artifact exports inference-ready y/train weights without the
optimizer, record that explicitly in its manifest and load it with
`weight_mode="saved"`.
