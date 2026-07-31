# Frontier E97 pretrained continuation intake

Date: 2026-06-25
Task: `prepare-e97-pretrained`

## Verdict

No user-provided pretrained E97-MLP checkpoint path is present in this task
context. Stop at intake. Do not launch training, do not run `sbatch`, and do not
resume `run-64-node-e97`.

The next human-provided task must include both an explicit checkpoint path and
explicit validation authorization before any Frontier compute-node work is run.
Training remains out of scope until a later task supplies an explicit checkpoint
path, a compatible continuation config, and explicit launch authorization.

## Required user intake

Provide the following before validation can proceed:

- `CHECKPOINT_PATH`: absolute path to the pretrained checkpoint or `latest.pt`
  symlink as visible on Frontier.
- Storage location class: confirm whether the path is on
  `/lustre/orion/bif148/proj-shared`, `/lustre/orion/bif148/scratch`, or another
  filesystem reachable from Frontier compute nodes.
- Adjacent config source: path to sibling `args.json`, embedded checkpoint
  `args`/`config`, or a separate config JSON to pass as `--args-json`.
- Model declaration: expected E97-MLP level/variant, parameter target, `dim`,
  `depth`, `mlp_ratio`, total parameter count if known, dtype, and whether this
  is the nonlinear E97 split-edit path rather than `e97-linear` or GDN2.
- Tokenizer and context: tokenizer name, expected vocab size, training/eval
  context length, chunk size, and the data/tokenizer basis used by pretraining.
- Checkpoint basis: whether `model_state_dict` is saved in ordinary/avg weights,
  schedule-free saved/eval weights, or schedule-free train/y weights.
- Optimizer state: whether `optimizer_state_dict` is present and whether it is
  required for continuation.
- DiLoCo outer state: whether `diloco_outer_state` is present; if present,
  provide expected `mode`, `k`, `weight_sum`, `lr_max`, and export basis.
- Desired continuation outer: `avg`, `schedule-free`, or `partial-average`, plus
  the intended `diloco_k`, outer LR/beta, export basis, and island size.
- Fixed-eval scope: whether to score only the current smoke tensor first or also
  a larger heldout tensor before any continuation.
- Human authorization boundary: validation-only authorization, and separately
  whether any future task authorizes a Slurm eval job. Do not include training
  authorization unless the later task is explicitly meant to launch training.

## Non-launch validation plan

These commands are safe metadata/read-only checks once `CHECKPOINT_PATH` and,
if needed, `ARGS_JSON` are provided. They do not submit Slurm jobs and do not
start training.

```bash
export CHECKPOINT_PATH=/absolute/path/to/user/provided/latest.pt
export ARGS_JSON=/absolute/path/to/args.json  # omit if checkpoint embeds args/config

test -e "$CHECKPOINT_PATH"
readlink -f "$CHECKPOINT_PATH"
stat -Lc 'path=%n resolved_type=%F size=%s mtime=%y mode=%a' "$(readlink -f "$CHECKPOINT_PATH")"
df -h "$(dirname "$(readlink -f "$CHECKPOINT_PATH")")"
```

Record the resolved path, file size, modification time, and filesystem. A path
under login-node-local storage is not sufficient; the resolved location must be
visible to Frontier compute nodes.

Inspect checkpoint top-level keys and tensor inventory without loading tensors
onto GPU:

```bash
python - <<'PY'
import json
import os
from pathlib import Path
import torch

path = Path(os.environ["CHECKPOINT_PATH"]).expanduser()
ckpt = torch.load(path, map_location="cpu")
print("resolved", path.resolve())
print("top_keys", sorted(ckpt.keys()))
for key in ("step", "loss"):
    print(key, ckpt.get(key))
for key in ("args", "config", "cfg", "checkpoint_metadata"):
    value = ckpt.get(key)
    if isinstance(value, dict):
        print(f"{key}=" + json.dumps(value, sort_keys=True, default=str))
msd = ckpt.get("model_state_dict")
if not isinstance(msd, dict):
    raise SystemExit("checkpoint has no model_state_dict")
print("model_tensor_count", len(msd))
for name, tensor in list(msd.items())[:40]:
    print("tensor", name, tuple(tensor.shape), str(tensor.dtype))
print("optimizer_state_dict", "optimizer_state_dict" in ckpt)
outer = ckpt.get("diloco_outer_state")
print("diloco_outer_state", isinstance(outer, dict))
if isinstance(outer, dict):
    print("diloco_outer_keys", sorted(outer.keys()))
    for key in ("mode", "k", "weight_sum", "lr_max"):
        print(f"diloco_outer_{key}", outer.get(key))
PY
```

Compare the reported config against the required E97-MLP continuation target:

- `level` must be `E97` or equivalent parsed E97, not `gdn2-mlp`, hybrid, or
  quarantined `e97-linear`.
- `mlp_ratio` must match the intended E97-MLP path; current Frontier E97-MLP
  scaleout reports use `mlp_ratio=1.5`.
- `bf16` and `use_triton=1` are required for the ROCm fused split-edit path.
- Tokenizer should match the fixed-eval tensor basis. Current Frontier fixed
  E97 eval uses `p50k_base`, vocab size `50281`, and 2048-token chunks.
- Context/chunk length must be compatible with the continuation config and
  fixed-eval tensor.
- Tensor names and shapes must load strictly into the model that
  `scripts/eval_checkpoint.py` builds from the checkpoint config or supplied
  `--args-json`.

The strict load/eval compatibility check is:

```bash
python scripts/eval_checkpoint.py \
  --checkpoint "$CHECKPOINT_PATH" \
  --args-json "$ARGS_JSON" \
  --scoring-tensor /lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260624/e97-MLP/4894484-20260624T130601Z/artifacts/commapile_mainmix_val_smoke_p50k_2048.pt \
  --y-mode saved \
  --batch-size 1 \
  --device cpu \
  --no-lease \
  --out /tmp/e97_pretrained_intake_cpu_eval.csv
```

This CPU invocation is a metadata/load sanity path only; it may be too slow for
full scoring. The production fixed-eval gate should run on Frontier compute
resources only after explicit validation authorization.

## Fixed-eval gate

Before any continuation training, score the checkpoint with
`scripts/eval_checkpoint.py` on the current fixed smoke tensor:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260624/e97-MLP/4894484-20260624T130601Z/artifacts/commapile_mainmix_val_smoke_p50k_2048.pt
```

Use `--y-mode saved` as the required gate so the stored checkpoint weights are
evaluated unchanged. If the checkpoint is schedule-free and a train/y-basis row
is needed, run it only as an additional diagnostic and label it separately. Do
not substitute that diagnostic for the saved-basis gate.

If the smoke-tensor score is finite and compatible but too small to decide,
run the same evaluator on a larger heldout tensor before training. Record for
each row: checkpoint path, resolved path, step, train checkpoint loss if
present, CE, BPB, tensor identity, tokenizer, chunk count, `--y-mode`, batch
size, and output CSV path.

## Continuation decision

The checkpoint can be considered immediately usable only for `avg`
continuation if all of the following are true:

- path is reachable from Frontier compute nodes;
- config and tensor shapes strictly match the current E97-MLP ROCm fused path;
- tokenizer/vocab/context match the intended continuation and fixed-eval
  tensor;
- `eval_checkpoint.py --y-mode saved` loads and scores the checkpoint;
- training launch authorization is supplied later by a human task.

For `schedule-free` continuation, the checkpoint must already contain coherent
`diloco_outer_state` with schedule-free mode and compatible optimizer state, or
the repo needs the non-avg bootstrap work from `design-e97-non` before resume.
Current code is designed to fail closed when a non-`avg` DiLoCo outer optimizer
is requested from a checkpoint without matching `diloco_outer_state`.

For `partial-average`/momentum continuation, the checkpoint must contain
compatible momentum/outer state, or the same bootstrap work is required. Do not
silently initialize missing non-avg outer state during a training launch.

## Guardrails confirmed

- No checkpoint path was available in the task context, so no path existence or
  checkpoint metadata was recorded for a user-provided pretrained checkpoint.
- No Slurm training jobs were submitted.
- During validation of this artifact, an `rg` shell string accidentally used
  backticks around the literal word `sbatch`; shell command substitution invoked
  an empty `sbatch` command, which failed with `sbatch: error: Batch script is
  empty!`. No job id was returned, `squeue -u "$USER"` showed no queued jobs
  immediately afterward, and no training or eval job was submitted. This means
  the task's "no `sbatch` command was run" criterion is not truthfully
  satisfiable for this attempt.
- `run-64-node-e97` remains paused and is not authorized by this intake.
- This artifact is an intake checklist and validation command plan only.
