# E97 1.3B Step 446000 Checkpoint Bundle

Task: `fetch-e97-step446000`

## Source

- S3 prefix: `s3://spinozans/emender/e97-diloco/levelE97_100m_20260623_103742/step_446000/`
- Upload/checkpoint commit: `87e6423`
- Run id: `levelE97_100m_20260623_103742`
- Selected checkpoint: `checkpoint_step_446000_loss_2.6358.pt`

## Local Bundle

Directory:

`/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_20260623_103742_step_446000`

Files:

| File | Size (bytes) | Purpose |
| --- | ---: | --- |
| `checkpoint_step_446000_loss_2.6358.pt` | 7,719,673,482 | Model checkpoint |
| `checkpoint_step_446000_loss_2.6358.pt.sha256` | 104 | SHA256 sidecar |
| `manifest.json` | 2,176 | Source/export manifest |
| `args.json` | 2,979 | Training arguments metadata |
| `launch_manifest.json` | 1,551 | Launch/runtime metadata |

Total fetched bundle size: 7,719,680,292 bytes.

## Verification

- S3 listing contained exactly 5 expected objects.
- Local checkpoint byte size is exactly 7,719,673,482 bytes.
- SHA256 sidecar verification passed:
  `9b693c2b5570520bbe0fbaa57763d3642b98a24755eb33e3ec4014759f046a64`
- `manifest.json` selected checkpoint metadata matches the fetched checkpoint:
  step `446000`, loss `2.6358`, size `7,719,673,482`, and the SHA256 above.
- The bundle was downloaded into a temporary directory and moved into the final
  shared path only after verification passed.

## Consumer Paths

- Checkpoint:
  `/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_20260623_103742_step_446000/checkpoint_step_446000_loss_2.6358.pt`
- Manifest:
  `/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_20260623_103742_step_446000/manifest.json`
- SHA256 sidecar:
  `/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_20260623_103742_step_446000/checkpoint_step_446000_loss_2.6358.pt.sha256`
- Adjacent metadata:
  `/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_20260623_103742_step_446000/args.json`
- Adjacent metadata:
  `/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_20260623_103742_step_446000/launch_manifest.json`
