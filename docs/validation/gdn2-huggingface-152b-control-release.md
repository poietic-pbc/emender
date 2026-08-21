# GDN2-MLP 152B-token Hugging Face control release

Date: 2026-08-21

## Published repository

- Repository: <https://huggingface.co/spinozans/gdn2-mlp-1.3b>
- Default `main`: final 152,280,498,176-token authority
- Main commit: `e6087fa6984265404d99c419c095692d840da472`
- Immutable tag: `tokens-152280498176`
- Tag commit: `e6087fa6984265404d99c419c095692d840da472`

## Source checkpoint

```text
/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/runs/
gdn2_gdn2-mlp_1.3B_20260722_083444/
checkpoint_step_2323616_loss_2.4034.pt
```

- Step: `2,323,616`
- Aggregate tokens: `152,280,498,176`
- Parameters: `1,286,713,448`
- Training-log last-100 loss: `2.4033774927` nats/token
- File size: `7,720,577,595` bytes
- SHA-256: `ffae61cfeeeff820f469f7d66dd31c60e48c455872ff9102e57b15ad75bad59b`
- Shutdown: manual target-reached request, eight-rank finalization, final DiLoCo consensus merge, then Schedule-Free averaged checkpoint save

The run used eight independent local-DiLoCo islands, batch size 4 per island,
2,048-token chunks, `K=250`, outer averaging with learning rate 1 and beta 0,
BF16 Schedule-Free AdamW, `p50k_base`, and the Pile stream. The final checkpoint
is 2.280B tokens beyond the nominal 150B target and 1.487B beyond E97's
150.794B historical authority.

## Export

The curated release contains:

- BF16 Schedule-Free train/y weights in `model.safetensors`;
- sanitized architecture arguments;
- `p50k_base` tokenizer files;
- a standalone custom Transformers loader;
- provenance, validation, and SHA-256 manifests;
- a bounded model card using conservative `license: other` metadata.

Raw pickle checkpoint and optimizer state are excluded. NVIDIA's GatedDeltaNet-2
source is also excluded. The portable loader implements the published recurrence
in clean PyTorch; the original fused source remains separately available at
NVIDIA commit `95709fc250357c2dd109361c353192f2aa5913f9` under NVIDIA's Source
Code License-NC.

Exported `model.safetensors`:

- Size: `2,792,280,168` bytes
- SHA-256: `06da94e6fc666927547858b23207cacac41e966c6a85e7bb53bc1f1cba388444`
- Tensor count: `267`
- Weight semantics: Schedule-Free train/y (`schedulefree_train_weight_swap=true`)

## Validation

Local validation established:

1. all 267 tensor keys and values exactly match the recovered source-checkpoint train/y state;
2. the custom Transformers wrapper loads with zero missing, unexpected, mismatched, or error keys;
3. `p50k_base` encodes `The theorem states` as `[464, 44728, 2585]`;
4. CPU BF16 logits have shape `[1, 3, 50281]` and are finite;
5. the independently reconstructed portable source-checkpoint logits are bit-identical to exported Transformers logits (SHA-256 `0b302c84563fad1cf51572a0130949fd498164ab7f0af0a77addfb051dbf4865`);
6. greedy generation emits token `326` after the three-token prompt;
7. the original fused CUDA path and portable CUDA path preserve the same argmax; BF16 evaluation-order differences measured mean absolute `0.0124367` and maximum absolute `0.125` logits.

The release-authored `SHA256SUMS` verifies every supplied file. A fresh public
Hub readback from `tokens-152280498176` at commit
`e6087fa6984265404d99c419c095692d840da472` verified the manifest, strictly
loaded with zero key discrepancies, produced finite `[1, 3, 50281]` logits, and
greedily generated token `326` without relying on the upload source directory.

## Matched E97 comparison

At the last common regular log point, step `2,300,925` / `150,793,420,800`
tokens:

- E97 80-point moving average: `2.437045`
- GDN2 80-point moving average: `2.426705`
- GDN2 minus E97: `-0.010340` nats/token
- E97 last-1000 mean: `2.435820`
- GDN2 last-1000 mean: `2.427318`
- last-1000 delta: `-0.008503`

These are aligned training-log summaries, not fixed held-out evaluation.

## Reproducibility and repository reconciliation

Release construction is implemented by `scripts/prepare_gdn2_hf_release.py`;
validation is implemented by `scripts/validate_gdn2_hf_release.py`; focused
tests live in `tests/test_prepare_gdn2_hf_release.py`.

The canonical staging checkout is `/home/erikg/emender`, fast-forwarded from
`spinozans/emender` main. Historical GDN2/E97 ops documents and scripts that
existed only in `/home/erikg/ndm` were copied into this canonical tree. Newer
canonical source implementations were retained rather than overwritten by
stale local variants. WG control-plane directories, worktrees, private scratch
notes, raw logs, generated PNGs, and checkpoints were intentionally excluded.
