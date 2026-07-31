# E97 1.3B Pretrained K160 32-Node Scale Probe - 2026-06-26

WG task: `run-e97-1p3b-k160-32n`

## Verdict

**Systems result: fail. Quality/loss result before failure: stable-to-indeterminate, no observed blowup.**

The single 32-node K160 singleton-island continuation was submitted under the
relaxed blowup/systems gate from
`docs/FRONTIER_E97_1P3B_K160_SCALE_GATE_OVERRIDE_20260626.md`. It trained for
2,941 local steps after the 16-node checkpoint and produced periodic
checkpoints through step `269920`, but the Slurm job failed before clean
finalization with NCCL process-group watchdog timeouts on a 1-element
`ALLREDUCE` (`SeqNum=12003`) across many ranks.

This is an actual collective/runtime failure under the override policy. Do not
advance the K160 ladder to the 64-node successor from this run. The last
periodic `latest.pt` exists, but there was no final consensus merge, no final
checkpoint, and no final-ready marker set.

## Submission

- Commit/launcher: `0939342cc513a023c6c22f9373e2ab5942cb6641`;
  launcher at `scripts/frontier/e97_1p3b_pretrained_k160_scale_ladder.sbatch`.
- Slurm job id: `4903889`
- Submitted command:

```bash
sbatch -N 32 -J e97-1p3b-k160-32n --export=ALL,WG_TASK_ID=run-e97-1p3b-k160-32n,SCALEOUT_VARIANT=E97_1.3B_pretrained_k160_32n,RESUME_CHECKPOINT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_k160_16n/4902278-20260625T213328Z/train/emender_E97_1.3B_20260625_173454/latest.pt,REQUESTED_NODE_HOURS=32.0 scripts/frontier/e97_1p3b_pretrained_k160_scale_ladder.sbatch
```

- Submission count: exactly one allocated 32-node training job; no
  resubmission and no scheduler-only adjustment.
- Slurm state: `FAILED`, exit `137:0`
- Slurm elapsed: `00:59:04`
- Requested node-hours: `32.0`
- Actual node-hours: about `31.502` node-hours (`32 * 3544 / 3600`)
- AllocTRES: `billing=3584,cpu=3584,energy=157756994,mem=16000G,node=32`
- Step state: `4903889.0|bash|CANCELLED|0:9|00:58:56|32`

## Artifacts

- Run root: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260626/E97_1.3B_pretrained_k160_32n/4903889-20260626T035916Z`
- Train log: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260626/E97_1.3B_pretrained_k160_32n/4903889-20260626T035916Z/logs/train.log`
- Manifest: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260626/E97_1.3B_pretrained_k160_32n/4903889-20260626T035916Z/artifacts/manifest.json`
- Run manifest: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260626/E97_1.3B_pretrained_k160_32n/4903889-20260626T035916Z/train/emender_E97_1.3B_20260626_000120/run_manifest.json`
- Summary: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260626/E97_1.3B_pretrained_k160_32n/4903889-20260626T035916Z/summaries/summary.md`
- Last periodic checkpoint: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260626/E97_1.3B_pretrained_k160_32n/4903889-20260626T035916Z/train/emender_E97_1.3B_20260626_000120/checkpoint_step_269920_loss_2.9823.pt`
- Periodic `latest.pt`: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260626/E97_1.3B_pretrained_k160_32n/4903889-20260626T035916Z/train/emender_E97_1.3B_20260626_000120/latest.pt`
- `latest.pt` resolves to: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260626/E97_1.3B_pretrained_k160_32n/4903889-20260626T035916Z/train/emender_E97_1.3B_20260626_000120/checkpoint_step_269920_loss_2.9823.pt`

## Topology And Configuration

Confirmed from `env.txt`, `args.json`, `manifest.json`, and the training log:

- Nodes: `32`
- Ranks/world size: `256`
- Tasks per node: `8`
- `DILOCO_ISLAND_SIZE=1`
- `_ddp_enabled=false`
- `_use_diloco=true`
- `DILOCO_K=160`
- `DILOCO_OUTER_OPTIMIZER=avg`
- `DILOCO_EXPORT_BASIS=x`
- `BATCH_SIZE=1`
- `CHUNK_SIZE=2048`
- `SAVE_EVERY=160`
- Resume checkpoint: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_k160_16n/4902278-20260625T213328Z/train/emender_E97_1.3B_20260625_173454/latest.pt`

No GDN2, CMAES, schedule-free outer optimizer, K sweep, LR sweep, DDP, or
island-size change was submitted. The inner optimizer remained `schedulefree`
as in the launcher, but the DiLoCo outer optimizer was `avg`; no schedule-free
outer was used.

## Failure And Finalization

Finalization was not clean:

- Slurm job failed with exit `137:0`.
- `srun` cancelled step `4903889.0` after task failure.
- The training log contains repeated NCCL watchdog failures of this form:
  `Watchdog caught collective operation timeout: WorkNCCL(SeqNum=12003, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000)`.
- The timeout was reported on many ranks, including ranks `80-95`, and the
  step was force terminated.
- No `FINAL merge` line was logged.
- No final checkpoint was logged.
- `.final_checkpoint_ready` marker count was `0`.
- Periodic `latest.pt` points to `checkpoint_step_269920_loss_2.9823.pt`, not
  to a final checkpoint.
- No final-ready mismatch was observed because final checkpoint coordination
  did not complete or produce marker files.

## Training Loss Windows

Rank-0 training metrics were logged every 5 local steps. The windows below use
all logged samples available in each local-step span of the partial run.

| Window | First-window steps | First avg loss | Last-window steps | Last avg loss | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 500 local steps | `267060-267555` (`100` samples) | `2.585022` | `269505-270000` (`100` samples) | `2.668400` | `+0.083378` |
| 1000 local steps | `267060-268055` (`200` samples) | `2.640959` | `269005-270000` (`200` samples) | `2.695651` | `+0.054692` |
| 2000 local steps | `267060-269055` (`400` samples) | `2.651523` | `268005-270000` (`400` samples) | `2.677335` | `+0.025812` |

Additional loss context:

- Resume checkpoint step: `267059`
- First logged training step: `267060`
- Last logged training step: `270000`
- Last logged point loss: `2.4144` at step `270000`
- Last periodic checkpoint loss: `2.9823` at step `269920`

Loss classification: **stable-to-indeterminate, no observed blowup**. The
trailing windows are mildly worse, but losses remained finite and in-family
through the observed training samples. The rung stops on the collective/runtime
failure, not on the short-window loss deltas.

## Throughput

Parsed from `589` rank-0 training metric lines:

- Mean global tokens/s: `606,719`
- Median global tokens/s: `633,670`
- Min/max global tokens/s: `132,936` / `652,656`

Filtered throughput excludes merge/save/startup dips by requiring
`global_tok/s >= 500,000`:

- Filtered samples: `548`; excluded samples: `41`
- Filtered mean global tokens/s: `631,145`
- Filtered median global tokens/s: `634,468`
- Filtered min/max global tokens/s: `519,914` / `652,656`

Approximate ensemble tokens processed:

- By last logged step minus resume step:
  `(270000 - 267059) * 2048 * 256 = 1,541,931,008` tokens
- By logged step-line coverage: `589 * 5 * 2048 * 256 = 1,544,028,160` tokens

## DiLoCo Sync

- Non-final merges completed: `18`
- Final merge: none
- `DILOCO_MERGES`: `18` completed before failure
- `DILOCO_SYNC_TOTAL_S`: `77.564`
- `DILOCO_SYNC_AVG_MS`: `4309.1`
- Median merge ms: `4226.0`
- Min/max merge ms: `4164` / `4783`
- Sync fraction estimate using Slurm elapsed `00:59:04`: `2.189%`
- Sync fraction estimate using rank-0 logged training elapsed `0.753 h`:
  `2.861%`

## Validation Checklist

- [x] Read `docs/FRONTIER_E97_1P3B_K160_SCALE_GATE_OVERRIDE_20260626.md` and
  applied the relaxed blowup/systems gate.
- [x] Submitted no more than one allocated 32-node training job for this task.
- [x] Confirmed topology: 32 nodes, 256 ranks, singleton DiLoCo islands, no
  DDP, avg outer, K160, export basis `x`.
- [ ] Clean finalization: failed. No final consensus merge, no final
  checkpoint, zero final-ready marker files, and `latest.pt` points to the
  last periodic checkpoint only.
- [x] Reported Slurm job id, elapsed time, requested and actual node-hours,
  run root, train log, manifest, summary, and final/last available
  `latest.pt`.
- [x] Reported trailing rank-0 training loss averages over last 500, 1000, and
  2000 local steps where available.
- [x] Classified loss behavior as stable-to-indeterminate with no observed
  blowup; the rung stops on systems failure.
- [x] Reported throughput, approximate ensemble tokens processed,
  `DILOCO_MERGES`, sync total/avg, and sync fraction.
- [x] Confirmed no GDN2, CMAES, schedule-free outer, K sweep, LR sweep, or
  DDP/island-size change was submitted.
