# E97 Loss Curve BPB Refresh - 2026-06-30 07:55 UTC

Scope: refreshed the hosted E97/Emender 8-GPU DiLoCo loss curve from current
live logs using the existing repository plotting workflow. I inspected the live
run read-only and did not stop, restart, modify, rename, delete, or launch
training/eval.

## Inputs

- Plot script: `scripts/plot_e97_diloco_loss.py`
- Run root: `/mnt/nvme1n1/erikg/diloco_8gpu/emender`
- Log sources:
  - `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run_phase1.log`
  - `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run_pre_supervisor_20260622T101450Z.log`
  - `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run_20260623T103727Z.log`
  - `/mnt/nvme1n1/erikg/diloco_8gpu/emender/run.log`
- Output PNG: `docs/experiments/figures/e97_diloco_loss_curve_20260623.png`
- Hosted URL: `http://hypervolu.me/~erik/emender/e97_diloco_loss_curve_20260623.png`

## Current Training Snapshot

- Raw parsed points: `28,865`
- Effective plotted points after resume deduplication: `25,950`
- Superseded pre-resume points: `2,915`
- Resume steps: `500`, `72000`, `72500`
- Latest plotted step: `648750`
- Latest plotted timestamp: `2026-06-30T07:54:50+00:00`
- Latest raw training loss: `2.6016`
- Plot smoothing: moving average over `80` points
- Plot-smoothed loss: `2.578550`

## Token Accounting

This is an 8-GPU run with `batch_size=4`, `chunk_size=2048`,
`grad_accum=1`, and `world_size=8`.

```text
tokens_per_step = 4 * 2048 * 8 = 65,536
global_tokens_seen = 648750 * 65,536 = 42,516,480,000
```

## BPB

Using `bytes_per_token=3.783` from
`/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/args.json:heldout_bytes_per_token`.

Formula:

```text
BPB = loss * log2(e) / bytes_per_token
```

For `bytes_per_token=3.783`, `log2(e) / bytes_per_token = 0.381362685934`.

- Latest raw estimated BPB: `0.992153`
- Plot-smoothed estimated BPB: `0.983363`

## Checkpoint Context

- Latest checkpoint: `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/levelE97_100m_20260623_103742/checkpoint_step_648500_loss_2.5999.pt`
- Latest checkpoint step: `648500`
- Latest checkpoint loss: `2.5999`
- Latest checkpoint estimated BPB: `0.991505`

## Verification

- Local PNG SHA-256:
  `a2976aff5019a9edca0310708b117b8b043ac48794e998ee6ce542029c8cd24a`
- SSH remote target:
  `erik@hypervolu.me:www/emender/e97_diloco_loss_curve_20260623.png`
- SSH remote SHA-256:
  `a2976aff5019a9edca0310708b117b8b043ac48794e998ee6ce542029c8cd24a`
- Public HTTP status: `HTTP/1.1 200 OK`
- Public HTTP content type: `image/png`
- Public HTTP SHA-256:
  `a2976aff5019a9edca0310708b117b8b043ac48794e998ee6ce542029c8cd24a`

Local, SSH remote, and public HTTP artifact hashes matched.
