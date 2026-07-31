# E97 1.3B Step 483000 Checkpoint Fetch

Task: `fetch-e97-step483000`

Fetched and verified the stable E97 1.3B foundation-generation checkpoint
bundle from:

`s3://spinozans/emender/e97-diloco/levelE97_100m_20260623_103742/step_483000/`

Local bundle path:

`/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_20260623_103742_step_483000`

Selected checkpoint:

`/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_20260623_103742_step_483000/checkpoint_step_483000_loss_2.5431.pt`

## Bundle Objects

| Object | Bytes | Local path |
| --- | ---: | --- |
| `args.json` | 2,979 | `/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_20260623_103742_step_483000/args.json` |
| `checkpoint_step_483000_loss_2.5431.pt` | 7,719,673,482 | `/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_20260623_103742_step_483000/checkpoint_step_483000_loss_2.5431.pt` |
| `checkpoint_step_483000_loss_2.5431.pt.sha256` | 104 | `/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_20260623_103742_step_483000/checkpoint_step_483000_loss_2.5431.pt.sha256` |
| `launch_manifest.json` | 1,551 | `/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_20260623_103742_step_483000/launch_manifest.json` |
| `manifest.json` | 2,786 | `/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_20260623_103742_step_483000/manifest.json` |

Object count: 5

Total bundle bytes: 7,719,680,902

## Verification

- S3 prefix listing returned exactly 5 objects and was not truncated.
- Local object sizes match the S3 listing.
- Checkpoint size matches the expected and S3 size: 7,719,673,482 bytes.
- SHA256 sidecar is present at `checkpoint_step_483000_loss_2.5431.pt.sha256`.
- Checkpoint SHA256 matches the sidecar and expected digest:
  `c8806d51a7f3611299e0491856075f6df8b5e98e860eb6d75defa0aad59955b5`.
- `manifest.json` is present and records the selected checkpoint step/loss:
  step `483000`, loss `2.5431`.

Verification result: PASS.
