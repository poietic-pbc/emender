# Verify: corrected E97 513B checkpoint metadata

Task: `verify-e97-513b-correction-20260805`
Date: 2026-08-05

## RESULT

Live S3 state now corrects the prior finding in
`reports/audits/audit-e97-513b-args-20260805.md`: the current prefix
`s3://spinozans/emender/e97-diloco/emender_E97_1.3B_20260804_513B/`
does include decisive architecture sidecars. The previously missing
`step_2322520/args.json` is now present and was read directly from S3. It
records `"level": "E97"` and `"mlp_ratio": 2.2623`; `model_config.json` also
records `"mlp_ratio": 2.2623`; `manifest.json` records
`"mlp_branch_enabled": true` and
`"frontier_launcher_explicit_mlp_argument": "--mlp_ratio 2.2623"`. Therefore
`mlp_ratio` is nonzero, exactly `2.2623`, and the E97 loader should construct
the FFN/MLP branch rather than a mixer-only model.

The checkpoint object itself does not appear to have been replaced relative to
the prior audit evidence: the current checkpoint size
`7719680180` and SHA256 `e559df3e8c540aef59ce8c9d73338f255cbe2fb9c7301ab45c7ef36a5b0fb857`
match the values already recorded in the prior audit. Its current S3 HEAD is
`LastModified=2026-08-05T12:42:36+00:00`,
`ETag="80b6c2d97c7d88510b5f3f76e269713b-921"`,
`ContentLength=7719680180`, S3 `ChecksumSHA256=r6+fKAjV57997HXEvyCNoFI1tfPTdn/Bg2J+1XsGHvM=-921`,
with object metadata `sha256=e559df3e8c540aef59ce8c9d73338f255cbe2fb9c7301ab45c7ef36a5b0fb857`,
`source_job=5157856`, `checkpoint_kind=final`, `step=2322520`,
`total_tokens=513013841920`. The change is sidecar/provenance correction:
`args.json` and `model_config.json` were added, `manifest.json`/`latest.json`
were updated, and `SHA256SUMS` was updated; the checkpoint payload identity is
unchanged by the evidence available.

## Live 513B S3 inventory

Fresh command:

```text
aws s3api list-objects-v2 --bucket spinozans --prefix emender/e97-diloco/emender_E97_1.3B_20260804_513B/ --no-sign-request --output json
```

Current objects:

```text
emender/e97-diloco/emender_E97_1.3B_20260804_513B/latest.json
  LastModified=2026-08-05T14:47:35+00:00
  Size=1843
  ETag="405fab05f1b1907b2a1bbf16a649e645"
  ChecksumAlgorithm=SHA256
  head ChecksumSHA256=pSF6pHnZmxdSqaGSeoe+bbKfOEjNgkbD9SpmBNF1IlM=
  content: byte-identical to step_2322520/manifest.json by ETag/checksum/size

emender/e97-diloco/emender_E97_1.3B_20260804_513B/step_2322520/SHA256SUMS
  LastModified=2026-08-05T14:47:33+00:00
  Size=424
  ETag="1f0783c9e9962db176abf14eb29668c7"
  ChecksumAlgorithm=SHA256
  head ChecksumSHA256=wGN8kmyVnaUM4qY0NUT9x1RkmU+Gz1+gCMDqRAZjkOQ=

emender/e97-diloco/emender_E97_1.3B_20260804_513B/step_2322520/args.json
  LastModified=2026-08-05T14:47:29+00:00
  Size=4799
  ETag="9568e74950927f439c52bf051a72e60e"
  ChecksumAlgorithm=SHA256
  head ChecksumSHA256=F0qqkfAsPmzVN+ypc3UGsLsIinsIwwwSE055+KlMabY=
  SHA256SUMS hex=174aaa91f02c3e6cd537eca9737506b0bb088a7b08c30c12134e79f8a94c69b6

emender/e97-diloco/emender_E97_1.3B_20260804_513B/step_2322520/checkpoint_step_2322520_loss_2.2798.pt
  LastModified=2026-08-05T12:42:36+00:00
  Size=7719680180
  ETag="80b6c2d97c7d88510b5f3f76e269713b-921"
  ChecksumAlgorithm=SHA256
  head ChecksumSHA256=r6+fKAjV57997HXEvyCNoFI1tfPTdn/Bg2J+1XsGHvM=-921
  Metadata.sha256=e559df3e8c540aef59ce8c9d73338f255cbe2fb9c7301ab45c7ef36a5b0fb857
  Metadata.source_job=5157856
  Metadata.checkpoint_kind=final
  Metadata.step=2322520
  Metadata.total_tokens=513013841920
  SHA256SUMS hex=e559df3e8c540aef59ce8c9d73338f255cbe2fb9c7301ab45c7ef36a5b0fb857

emender/e97-diloco/emender_E97_1.3B_20260804_513B/step_2322520/manifest.json
  LastModified=2026-08-05T14:47:32+00:00
  Size=1843
  ETag="405fab05f1b1907b2a1bbf16a649e645"
  ChecksumAlgorithm=SHA256
  head ChecksumSHA256=pSF6pHnZmxdSqaGSeoe+bbKfOEjNgkbD9SpmBNF1IlM=
  SHA256SUMS hex=a5217aa479d99b1752a9a1927a87be6db29f3848cd8246c3f52a6604d1752253

emender/e97-diloco/emender_E97_1.3B_20260804_513B/step_2322520/metadata.env
  LastModified=2026-08-05T12:43:09+00:00
  Size=488
  ETag="74986383dd52fb74609eede0a2dca89e"
  ChecksumAlgorithm=SHA256
  head ChecksumSHA256=l7p8hBPgq9hXDY4BVAtrEM1kObI8fckJOuCuMtTVPJY=
  SHA256SUMS hex=97ba7c8413e0abd8570d8e01540b6b10cd6439b23c7dc9093ae0ae32d4d53c96

emender/e97-diloco/emender_E97_1.3B_20260804_513B/step_2322520/model_config.json
  LastModified=2026-08-05T14:47:30+00:00
  Size=1163
  ETag="a007cd57b6941936f2575675447eae7a"
  ChecksumAlgorithm=SHA256
  head ChecksumSHA256=A88/qrG7n3g/7G3qVTIlpvQjJ4SbJqIpkjARq+cfYB0=
  SHA256SUMS hex=03cf3faab1bb9f783fec6dea553225a6f42327849b26a229923011abe71f601d
```

## Corrected 513B architecture evidence

`step_2322520/args.json` decisive fields read from S3:

```text
"_execution_source_commit": "33bd7a2dc8f7705daf1685cdb825da0cbeed64f0"
"_final_step": 2322520
"_final_total_tokens": 513013841920
"_model_variant": "level=E97,params_arg=100m,derived_params=1.3B,total_params=1286589072,mlp_ratio=2.2623"
"_resume_step_at_launch": 2315840
"_resume_target_at_launch": "checkpoint_step_2315840_loss_2.2928.pt"
"_resume_total_tokens_at_launch": 400942039040
"_slurm_job_id": "5157856"
"_world_size": 2048
"batch_size": 4
"chunk_size": 2048
"depth": 11
"diloco": true
"diloco_k": 40
"diloco_outer_optimizer": "avg"
"diloco_outer_lr": 1.0
"diloco_outer_beta": 0.0
"dim": 1792
"level": "E97"
"mlp_multiple": 64
"mlp_ratio": 2.2623
"n_groups": 32
"n_heads": 216
"n_slots": 64
"n_state": 32
"params": "100m"
"source_commit": "33bd7a2dc8f7705daf1685cdb825da0cbeed64f0"
"use_gate": 1
"use_permutation": 1
"use_triton": 1
```

`step_2322520/model_config.json` independently records:

```text
"schema": "emender-model-config-v1"
"level": "E97"
"total_params": 1286589072
"trainable_params": 1286589072
"dim": 1792
"depth": 11
"n_heads": 216
"n_state": 32
"n_groups": 32
"n_slots": 64
"mlp_ratio": 2.2623
"mlp_multiple": 64
"source_launcher": "scripts/frontier/e97_same_allocation_restart.sbatch"
"source_launcher_argument": "--mlp_ratio 2.2623"
"checkpoint_step": 2322520
"checkpoint_total_tokens": 513013841920
```

`step_2322520/manifest.json` independently records:

```text
"schema": "emender-e97-checkpoint-export-v1"
"source_job": "5157856"
"source_job_state": "FAILED_AFTER_FINAL_CHECKPOINT"
"source_job_exit": "143:0"
"post_checkpoint_shutdown_fault": true
"checkpoint_atomic_publication_complete": true
"all_ranks_finalization_ready": 2048
"final_consensus_merge": 167
"total_params": 1286589072
"trainable_params": 1286589072
"mlp_ratio": 2.2623
"mlp_branch_enabled": true
"frontier_launcher_explicit_mlp_argument": "--mlp_ratio 2.2623"
"source_commit": "33bd7a2dc8f7705daf1685cdb825da0cbeed64f0"
```

Conclusion: this is an E97 1.3B FFN/MLP checkpoint. Loading it with
`mlp_ratio=0` or as mixer-only would be incorrect.

## Checkpoint payload metadata

To avoid a 7.7 GB download, I read only the first 1 MiB of the checkpoint object
with an S3 range request and unpickled the first ZIP member
`.checkpoint_step_2322520_loss_2.2798.pt.9wqfk83n/data.pkl` using placeholder
tensor rebuild hooks. That reads pickle metadata and key names without tensor
payloads.

Top-level checkpoint dict keys read from the live checkpoint object:

```text
step
total_tokens
model_state_dict
optimizer_state_dict
loss
checkpoint_metadata
```

Additional values:

```text
step=2322520
total_tokens=513013841920
loss=2.279764811873436
model_state_dict key count=146
optimizer_state_dict keys=["state", "param_groups"]
diloco_outer_state present=false
```

Embedded `checkpoint_metadata` keys and values:

```text
kind="final"
reason="walltime:SLURM_JOB_END_TIME"
model_variant="level=E97,params_arg=100m,derived_params=1.3B,total_params=1286589072,mlp_ratio=2.2623"
model={"model_family":"emender","level":"E97","params_arg":"100m","derived_param_slug":"1.3B","run_label_prefix":"emender_E97_1.3B","total_params":1286589072,"trainable_params":1286589072}
run_label_prefix="emender_E97_1.3B"
total_params=1286589072
trainable_params=1286589072
rank=0
world_size=2048
is_head=true
walltime_deadline_source="SLURM_JOB_END_TIME"
walltime_remaining_s=848.0948951244354
walltime_margin_s=900.0
shutdown_signal=null
total_tokens=513013841920
```

## Change versus prior audit

Prior audit conclusion:

```text
The exact 513B public export does not include a public args.json,
launch_manifest.json, run_manifest.json, captured command, or log tail at the
documented S3 prefix.
```

Correction from live S3:

```text
step_2322520/args.json exists and was read.
step_2322520/model_config.json exists and was read.
step_2322520/manifest.json/latest.json now include mlp_ratio and MLP branch fields.
step_2322520/SHA256SUMS now includes args.json, model_config.json, manifest.json, and metadata.env hashes.
```

Object timing supports sidecar/provenance-only correction:

```text
checkpoint LastModified=2026-08-05T12:42:36+00:00
metadata.env LastModified=2026-08-05T12:43:09+00:00
args.json LastModified=2026-08-05T14:47:29+00:00
model_config.json LastModified=2026-08-05T14:47:30+00:00
manifest.json LastModified=2026-08-05T14:47:32+00:00
SHA256SUMS LastModified=2026-08-05T14:47:33+00:00
latest.json LastModified=2026-08-05T14:47:35+00:00
```

The checkpoint hash/size in the corrected sidecars and checkpoint object
metadata match the prior audit values, so the checkpoint object was not replaced
by the evidence available; the corrected architecture was published through
sidecars/provenance.

## Cross-checks

Completed 150B E97 public prefix:

```text
s3://spinozans/emender/e97-diloco/emender_E97_1.3B_20260709_084606/step_2300930/
```

Current 150B S3 inventory includes `args.json`, `manifest.json`,
`run_manifest.json`, `run_command_process_snapshot.txt`, `run_log_tail_1000.txt`,
and `supervisor.log`. The 150B `args.json` records:

```text
"level": "E97"
"dim": 1792
"depth": 11
"n_heads": 216
"n_state": 32
"n_groups": 32
"n_slots": 64
"mlp_ratio": 2.2623
"mlp_multiple": 64
"_model_variant": "level=E97,params_arg=100m,derived_params=1.3B,total_params=1286589072,mlp_ratio=2.2623"
```

The 150B `manifest.json` records:

```text
"run": "emender_E97_1.3B_20260709_084606"
"step": 2300930
"tokens": 150793748480
"checkpoint_sha256": "0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2"
"checkpoint_size_bytes": 7719680116
"world_size": 8
"diloco_k": 250
```

Accessible Frontier launch provenance is consistent with the corrected 513B
sidecars. `scripts/frontier/e97_1p3b_pretrained_8n_k320.sbatch` has model args
`--level E97 --params 100m --dim 1792 --depth 11 --n_heads 216 --n_state 32
--n_groups 32 --n_slots 64 --mlp_ratio 2.2623 --mlp_multiple 64`.
`reports/frontier/monitor-e97-1p3b-k160-32n-rccl-4906910.md` records a
delegated training command using `--dim 1792 --depth 11 --n_heads 216
--n_state 32 --n_groups 32 --n_slots 64 --mlp_ratio 2.2623`.
`docs/FRONTIER_E97_1P3B_PRETRAINED_VALIDATION_20260625.md` records
`mlp_ratio=2.2623`, representative `layers.0.mlp.*` tensor shapes, and runtime
strict-load `model_variant=...mlp_ratio=2.2623`. The exact source launcher named
by the corrected 513B `model_config.json`,
`scripts/frontier/e97_same_allocation_restart.sbatch`, is not present in this
working tree, but the corrected S3 sidecars now provide the exact 513B job
architecture directly.

## Validation

- Live S3 prefix freshly enumerated: yes.
- Decisive updated artifacts read: `args.json`, `model_config.json`,
  `manifest.json`, `SHA256SUMS`, `metadata.env`, and checkpoint `data.pkl` range.
- Exact `mlp_ratio`: `2.2623`.
- MLP conclusion: FFN/MLP branch enabled; not mixer-only.
- Checkpoint replacement versus sidecar update: checkpoint hash/size match prior
  audit values; current correction is sidecar/provenance publication.
- No S3, checkpoint, source, or training artifacts mutated.
