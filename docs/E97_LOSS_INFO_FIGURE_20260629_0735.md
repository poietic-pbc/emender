# E97 Loss Info And Hosted Figure Refresh - 2026-06-29 07:35 UTC

Scope: refreshed the hosted E97/Emender 8-GPU DiLoCo loss plot from the current live logs using the existing repository workflow. I inspected the live run read-only and did not stop, restart, modify, rename, delete, or launch training/eval.

## Inputs

- Plot script: `scripts/plot_e97_diloco_loss.py`
- Command: `python scripts/plot_e97_diloco_loss.py --run-root /mnt/nvme1n1/erikg/diloco_8gpu/emender --output docs/experiments/figures/e97_diloco_loss_curve_20260623.png`
- Run root: `/mnt/nvme1n1/erikg/diloco_8gpu/emender`
- Log sources:
  - `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run_phase1.log`
  - `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run_pre_supervisor_20260622T101450Z.log`
  - `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run_20260623T103727Z.log`
  - `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run.log`

## Plotted State

- Raw parsed points: `25,475`
- Effective plotted points after resume deduplication: `22,560`
- Superseded pre-resume points: `2,915`
- Resume steps: `500`, `72000`, `72500`
- Latest plotted step: `564000`
- Latest plotted timestamp: `2026-06-29T07:36:24+00:00`
- Latest raw loss: `2.6261`
- Plot smoothing: moving average over `80` points
- Plot-smoothed loss: `2.582807`
- Global tokens seen estimate: `36,962,304,000` (`564000 * 4 batch_size * 2048 chunk_size * 8 world_size`; `65,536` global tokens/step)
- Latest checkpoint at inspection time: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_564000_loss_2.6261.pt`

## BPB Conversion

Measured tokenizer conversion:

```text
bytes_per_token=3.783
```

Source: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/args.json:heldout_bytes_per_token`

Formula:

```text
BPB = loss * log2(e) / bytes_per_token
```

For `bytes_per_token=3.783`, `log2(e) / bytes_per_token = 0.381362685934`.

- Latest raw estimated BPB: `1.001497`
- Plot-smoothed estimated BPB: `0.984986`

## Output And Verification

- Local PNG: `docs/experiments/figures/e97_diloco_loss_curve_20260623.png`
- Public URL: `http://hypervolu.me/~erik/emender/e97_diloco_loss_curve_20260623.png`
- SSH target: `erik@hypervolu.me:www/emender/e97_diloco_loss_curve_20260623.png`
- Local file: `PNG image data, 2240 x 1200, 8-bit/color RGBA, non-interlaced`
- Local SHA-256: `3f3d98f7d9f9a58890d2c7a9c04bf0657027d8e6217acb61ab0409512cae7792`
- Local size: `292491`
- SSH remote SHA-256: `3f3d98f7d9f9a58890d2c7a9c04bf0657027d8e6217acb61ab0409512cae7792`
- SSH remote size: `292491`
- SSH remote MIME type: `image/png`
- HTTP status: `200 OK`
- HTTP `Content-Type`: `image/png`
- HTTP `Content-Length`: `292491`
- HTTP SHA-256: `3f3d98f7d9f9a58890d2c7a9c04bf0657027d8e6217acb61ab0409512cae7792`
- Local, SSH remote, and HTTP download hashes matched: yes
