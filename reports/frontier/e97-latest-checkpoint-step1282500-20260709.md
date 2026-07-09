# E97 Latest Checkpoint Refresh: Step 1282500

Date: 2026-07-09

## Summary

The local E97 seed checkpoint used by Frontier launch recipes was refreshed to the uploaded S3 checkpoint at step `1282500`. No Slurm training job was submitted for this task.

## S3 Inputs

- Checkpoint: `s3://spinozans/emender/e97-diloco/emender_E97_1.3B_20260709_084606/step_1282500/checkpoint_step_1282500_loss_2.5175.pt`
- Pointer JSON: `s3://spinozans/emender/e97-diloco/latest_emender_E97_1.3B.json`
- Prefix: `s3://spinozans/emender/e97-diloco/emender_E97_1.3B_20260709_084606/step_1282500/`

The S3 prefix was listed over HTTPS and contained 11 objects:

- `args.json`
- `checkpoint_step_1282500_loss_2.5175.pt`
- `checkpoint_step_1282500_loss_2.5175.pt.sha256`
- `latest_symlink_target.txt`
- `launch_manifest.json`
- `manifest.json`
- `metadata_files.sha256`
- `run_command_process_snapshot.txt`
- `run_log_tail_1000.txt`
- `run_manifest.json`
- `supervisor.log`

All adjacent metadata files above, plus the project-level latest pointer JSON, were downloaded into the local checkpoint directory.

## Local Install

- Local directory: `/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260709_084606_step_1282500/`
- Local checkpoint: `/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260709_084606_step_1282500/checkpoint_step_1282500_loss_2.5175.pt`
- File size: `7719679924` bytes
- Local latest symlink: `/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260709_084606_step_1282500/latest.pt`
- Local latest symlink target: `checkpoint_step_1282500_loss_2.5175.pt`
- Resolved latest path: `/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260709_084606_step_1282500/checkpoint_step_1282500_loss_2.5175.pt`

The `latest.pt` symlink was installed with an atomic local symlink swap inside the new dated directory:

```bash
ln -sfn checkpoint_step_1282500_loss_2.5175.pt .latest.pt.tmp
mv -Tf .latest.pt.tmp latest.pt
```

Older checkpoint directories under `/lustre/orion/bif148/proj-shared/emender/checkpoints/` were not deleted or overwritten.

## Verification

Required SHA256:

```text
0ddf1279e80756bcd195d971b02175cfd4faf1e4f753e6f5c6c47789e81dc5c4
```

Verification result:

```text
actual=0ddf1279e80756bcd195d971b02175cfd4faf1e4f753e6f5c6c47789e81dc5c4
expected=0ddf1279e80756bcd195d971b02175cfd4faf1e4f753e6f5c6c47789e81dc5c4
sidecar=0ddf1279e80756bcd195d971b02175cfd4faf1e4f753e6f5c6c47789e81dc5c4
manifest=0ddf1279e80756bcd195d971b02175cfd4faf1e4f753e6f5c6c47789e81dc5c4
pointer=0ddf1279e80756bcd195d971b02175cfd4faf1e4f753e6f5c6c47789e81dc5c4
```

The local checkpoint hash exactly matched the task-provided SHA256, the downloaded `.sha256` sidecar, the downloaded `manifest.json`, and the downloaded latest pointer JSON. `sha256sum -c metadata_files.sha256` also passed for the downloaded metadata and pointer files:

```text
args.json: OK
run_manifest.json: OK
launch_manifest.json: OK
supervisor.log: OK
latest_symlink_target.txt: OK
run_log_tail_1000.txt: OK
run_command_process_snapshot.txt: OK
manifest.json: OK
latest_emender_E97_1.3B.json: OK
```

## Checkpoint Metadata

From `manifest.json` and `latest_emender_E97_1.3B.json`:

- Model: `emender_E97_1.3B`
- Run name: `emender_E97_1.3B_20260709_084606`
- Checkpoint step: `1282500`
- Checkpoint filename: `checkpoint_step_1282500_loss_2.5175.pt`
- Checkpoint loss: `2.5175`
- Tokens at checkpoint: `84049920000` (`84.050B`, rounded from `84.04992B`)
- Estimated BPB: `0.9601` (manifest value `0.960081`)
- Tokens per step: `65536`
- Created UTC: `2026-07-09T15:29:01Z`
- Size bytes: `7719679924`
- SHA256: `0ddf1279e80756bcd195d971b02175cfd4faf1e4f753e6f5c6c47789e81dc5c4`

Key pointer JSON fields:

```json
{
  "checkpoint_filename": "checkpoint_step_1282500_loss_2.5175.pt",
  "checkpoint_loss": 2.5175,
  "checkpoint_step": 1282500,
  "created_utc": "2026-07-09T15:29:01Z",
  "estimated_bpb_at_checkpoint": 0.960081,
  "model": "emender_E97_1.3B",
  "run_name": "emender_E97_1.3B_20260709_084606",
  "s3_checkpoint_uri": "s3://spinozans/emender/e97-diloco/emender_E97_1.3B_20260709_084606/step_1282500/checkpoint_step_1282500_loss_2.5175.pt",
  "sha256": "0ddf1279e80756bcd195d971b02175cfd4faf1e4f753e6f5c6c47789e81dc5c4",
  "size_bytes": 7719679924,
  "total_tokens_at_checkpoint": 84049920000,
  "total_tokens_at_checkpoint_b": 84.04992
}
```

## Launch Recipe Updates

No project-wide filesystem symlink outside the new dated checkpoint directory was found or changed.

The Frontier launch recipe defaults that select the local refreshed seed were updated to point at:

```text
/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260709_084606_step_1282500/latest.pt
```

Changed recipe files:

- `scripts/frontier/async_diloco_e97_256n12h_launch.sbatch`
- `scripts/frontier/trainpy_async_quorum_smoke_common.sh`
- `scripts/frontier/trainpy_async_quorum_1n_smoke.sbatch`
- `scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch`
- `scripts/frontier/e97_1p3b_step1065000_b4_trainpy_smoke.sbatch`

The launch labels in those recipes were also advanced from step `1065000` to step `1282500` where they describe the refreshed seed. The compatibility filename `e97_1p3b_step1065000_b4_trainpy_smoke.sbatch` was not renamed.

## Validation Commands

Commands run:

```bash
sha256sum checkpoint_step_1282500_loss_2.5175.pt
sha256sum -c metadata_files.sha256
readlink -f latest.pt
stat -c '%n %s bytes' checkpoint_step_1282500_loss_2.5175.pt
.envs/olcf-rocm711-torch210-py312/bin/pytest tests/test_async_diloco_e97_2n8n_debug_runner.py tests/test_frontier_runtime_plumbing.py
git diff --check -- reports/frontier/e97-latest-checkpoint-step1282500-20260709.md scripts/frontier/async_diloco_e97_256n12h_launch.sbatch scripts/frontier/trainpy_async_quorum_smoke_common.sh scripts/frontier/trainpy_async_quorum_1n_smoke.sbatch scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch scripts/frontier/e97_1p3b_step1065000_b4_trainpy_smoke.sbatch tests/test_async_diloco_e97_2n8n_debug_runner.py
```
