# E97 dense base and CLI-agent Hugging Face release

Date: 2026-08-21

## Published repositories

### Dense base lineage

- Repository: <https://huggingface.co/spinozans/emender-e97-1.3b>
- Default `main`: 513,013,841,920-token authority
- Main commit: `611b6e452114c3a3cd78d2fba445ce403e3e1332`
- Immutable tag: `tokens-513013841920`
- Historical branch/tag: `tokens-150793748480`
- Historical revision commit: `119c6623f414aa7fc6e101667d3ad89b7ed6af73`

The 513B source checkpoint is:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/
final-seed-production-256n/milestones/
step-2322520-tokens-513013841920/
checkpoint_step_2322520_loss_2.2798.pt
```

Its SHA-256 is
`e559df3e8c540aef59ce8c9d73338f255cbe2fb9c7301ab45c7ef36a5b0fb857`.
The published BF16 `model.safetensors` is 2,753,401,472 bytes with SHA-256
`b86a874c06537254047542c9ea16e4f8272748d0f098a8a2ef7213d9d5922640`.
It contains the recovered ScheduleFree train/y weights, not the checkpoint's
stored averaged x weights.

The 150B source was downloaded anonymously from the already-public prefix:

```text
s3://spinozans/emender/e97-diloco/
emender_E97_1.3B_20260709_084606/step_2300930/
checkpoint_step_2300930_loss_2.4365.pt
```

The downloaded source matched the recorded 7,719,680,116-byte size and SHA-256
`0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2`.
The corresponding published BF16 safetensors has SHA-256
`b10e1fa728030e86352c924ecefe62f44ec68ec4b6f6d0ad0b09c58447a5c2f9`.

### Direct CLI agent

- Repository: <https://huggingface.co/spinozans/emender-e97-1.3b-cli-agent>
- Main commit: `cb8569a1c3cced997119c93bc703313f3824cc56`
- Immutable tag: `direct-cli-40of40`

Source checkpoint:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-dense-agent/
cli-json-replay-v1-from-direct128-64u-lr3e6/checkpoints/
checkpoint_agent_sft_u000064.pt
```

Its SHA-256 is
`fca9b06c521fb9407cd7bed2d7049f36e9a93948d6cd98899b6ad9b9e4fc6b01`.
The published BF16 safetensors is 2,753,401,488 bytes with SHA-256
`b494c01a17098dc00a79d2358c3e1dcf5466e51d3cddb26c13bef6639b7645a0`.
The repository includes the 40/40 real-Pi evaluation summary.

## Artifact policy

All three revisions are curated weights-only releases. They include:

- BF16 `model.safetensors`;
- sanitized architecture arguments;
- `p50k_base` tokenizer files;
- Transformers custom loader files;
- provenance, validation, and SHA-256 manifests;
- bounded model cards.

Raw pickle checkpoints and optimizer states were not uploaded. Model-card
metadata intentionally uses `license: other`; no permissive license was inferred
or invented while the project has no standalone model license.

## Validation

For the 513B base, 150B base, and CLI agent independently:

1. the custom Transformers wrapper strictly loaded all 146 tensors with zero
   missing, unexpected, or mismatched keys;
2. `p50k_base` encoded `The theorem states` as `[464, 44728, 2585]`;
3. CPU BF16 logits had shape `[1, 3, 50281]` and were finite;
4. exported-checkpoint logits were bit-exact with the corresponding source
   checkpoint under the same native CPU execution path.

The 513B model also completed cache-free Transformers generation from the live
immutable Hugging Face tag and generated token ID `326` for the one-token greedy
smoke. The live CLI-agent tag strictly loaded and produced finite logits.

`hf cache verify` matched every locally supplied file against each live tagged
revision. Its only warning was the expected Hub-generated `.gitattributes`,
which is remote-only. Browser QA confirmed that both public model cards render
and contain their expected identity and evaluation text.

Release construction is implemented by
`scripts/prepare_e97_hf_release.py`; focused tests are in
`tests/test_prepare_e97_hf_release.py`.
