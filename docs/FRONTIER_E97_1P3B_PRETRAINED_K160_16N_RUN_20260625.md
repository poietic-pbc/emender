# E97 1.3B Pretrained K160 16-Node Scale Probe - 2026-06-25

WG task: `run-e97-1p3b-k160-16n`

## Verdict

**Systems result: pass. Quality gate: not passed / blocker.**

The 16-node K160 singleton-island continuation completed cleanly and produced a final consensus checkpoint, but the primary quality signal does not show a clean, strongly improving large trailing training-loss-window result. The last-500 window is only slightly better than the first-500 window, while the last-1000 and last-2000 windows are slightly worse. The larger deterministic eval from `evaluate-e97-1-3b` also favors K40 over K160 and is included as secondary context.

Do **not** unblock `run-e97-1p3b-k160-32n` from this result without a synthesis/review decision.

## Submission

- Commit/launcher: `6b67efb26108ad8a5fc648f9eb241571174136e2`; launcher existed at `scripts/frontier/e97_1p3b_pretrained_k160_scale_ladder.sbatch`.
- Slurm job id: `4902278`
- Submitted command:

```bash
sbatch -N 16 -J e97-1p3b-k160-16n --export=ALL,WG_TASK_ID=run-e97-1p3b-k160-16n,SCALEOUT_VARIANT=E97_1.3B_pretrained_k160_16n,REQUESTED_NODE_HOURS=16.0 scripts/frontier/e97_1p3b_pretrained_k160_scale_ladder.sbatch
```

- Submission count: one allocated 16-node training job; no resubmission.
- Slurm state: `COMPLETED`, exit `0:0`
- Submit/start/end: `2026-06-25T17:18:28` / `2026-06-25T17:33:27` / `2026-06-25T18:23:45` America/New_York
- Elapsed: `00:50:18`
- Requested node-hours: `16.0`
- Actual node-hours: about `13.413` node-hours (`16 * 50.3 / 60`)
- AllocTRES: `billing=1792,cpu=1792,energy=76978803,mem=8000G,node=16`

## Artifacts

- Run root: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_k160_16n/4902278-20260625T213328Z`
- Train log: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_k160_16n/4902278-20260625T213328Z/logs/train.log`
- Manifest: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_k160_16n/4902278-20260625T213328Z/artifacts/manifest.json`
- Run manifest: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_k160_16n/4902278-20260625T213328Z/train/emender_E97_1.3B_20260625_173454/run_manifest.json`
- Summary: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_k160_16n/4902278-20260625T213328Z/summaries/summary.md`
- Final checkpoint: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_k160_16n/4902278-20260625T213328Z/train/emender_E97_1.3B_20260625_173454/checkpoint_step_267059_loss_2.6179.pt`
- Final `latest.pt`: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_k160_16n/4902278-20260625T213328Z/train/emender_E97_1.3B_20260625_173454/latest.pt`
- `latest.pt` resolves to: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_k160_16n/4902278-20260625T213328Z/train/emender_E97_1.3B_20260625_173454/checkpoint_step_267059_loss_2.6179.pt`

## Topology And Configuration

Confirmed from `env.txt`, `args.json`, `manifest.json`, and training logs:

- Nodes: `16`
- Ranks/world size: `128`
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
- Resume checkpoint: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_k160_8node/4900869-20260625T183437Z/train/emender_E97_1.3B_20260625_143545/latest.pt`

No GDN2, CMAES, schedule-free outer optimizer, K sweep, LR sweep, DDP, or island-size change was submitted. The inner optimizer remained `schedulefree` as in the launcher, but the DiLoCo outer optimizer was `avg`; no schedule-free outer was used.

## Finalization

Finalization was clean:

- Finalization reason: `walltime:SLURM_JOB_END_TIME`
- Final step: `267059`
- Final consensus merge: `FINAL merge #21 at step 267059`, consensus averaged across `128` ranks in `4125 ms`
- Final checkpoint START/END logged by rank 0.
- Final checkpoint path: `checkpoint_step_267059_loss_2.6179.pt`
- `latest.pt` points at the final checkpoint.
- Final-ready markers: `128` files under `.final_checkpoint_ready`.
- No final-ready mismatch observed.

## Training Loss Windows

Rank-0 training metrics were logged every 5 local steps. The windows below use all logged samples in each local-step span.

| Window | First-window steps | First avg loss | Last-window steps | Last avg loss | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 500 local steps | `263845-264340` (`100` samples) | `2.628524` | `266560-267055` (`100` samples) | `2.617852` | `-0.010672` |
| 1000 local steps | `263845-264840` (`200` samples) | `2.664922` | `266060-267055` (`200` samples) | `2.672554` | `+0.007633` |
| 2000 local steps | `263845-265840` (`400` samples) | `2.661771` | `265060-267055` (`400` samples) | `2.671046` | `+0.009275` |

Additional loss context:

- Start step after resume: `263840`
- First logged training step: `263845`
- Last logged training step: `267055`
- Final step: `267059`
- Final `FINAL_LOSS_LAST100`: `2.6179`
- Last logged point loss: `2.1348` at step `267055`

Interpretation: the 500-step trailing average is marginally improved, but the larger 1000- and 2000-step trailing averages are slightly worse. This is not the clean, strongly improving large-window signal requested for unblocking the 32-node rung.

## Throughput

Parsed from `643` rank-0 training metric lines:

- Mean global tokens/s: `311,499`
- Median global tokens/s: `324,840`
- Min/max global tokens/s: `105,803` / `331,637`

Filtered throughput excludes merge-step lines, the first post-save line after each merge, and any `global_tok/s < 250,000` to remove save/merge dips:

- Filtered samples: `599`; excluded samples: `44`
- Filtered mean global tokens/s: `323,997`
- Filtered median global tokens/s: `325,155`
- Filtered min/max global tokens/s: `260,069` / `331,637`

Approximate ensemble tokens processed:

- By final step minus resume step: `(267059 - 263840) * 2048 * 128 = 843,841,536` tokens
- By logged step-line coverage: `643 * 5 * 2048 * 128 = 842,792,960` tokens

## DiLoCo Sync

- Non-final merges: `20`
- Final merge: `1`
- `DILOCO_MERGES: 21`
- `DILOCO_SYNC_TOTAL_S: 90.091`
- `DILOCO_SYNC_AVG_MS: 4290.1`
- Mean merge ms including final: `4290.048`
- Median merge ms including final: `4296`
- Sync fraction estimate using Slurm elapsed `00:50:18`: `90.091 / 3018 = 2.985%`
- Sync fraction estimate using the 50-minute training timer: `90.091 / 3000 = 3.003%`

## Eval Context

The fixed smoke eval remains secondary. A larger deterministic eval from `evaluate-e97-1-3b` completed during this run and reported:

- Scored tokens: `69,632` (`4.25x` smoke)
- Source CE/BPB: `26.83747050` / `10.31650033`
- K40 CE/BPB: `26.85514759` / `10.32329551`, delta `+0.01767709` / `+0.00679518`
- K160 CE/BPB: `27.08281208` / `10.41081125`, delta `+0.24534158` / `+0.09431092`

This larger eval favors K40 over K160. Per the task correction, it should not replace the training-window gate by itself, but it reinforces the decision not to advance K160 to 32 nodes from this result.

## Validation Checklist

- [x] Confirmed main is at commit `6b67efb` or later and the scale ladder launcher exists.
- [x] Submitted no more than one allocated 16-node training job for this task.
- [x] Confirmed topology: 16 nodes, 128 ranks, singleton DiLoCo islands, no DDP, avg outer, K160, export basis `x`.
- [x] Confirmed clean finalization: final consensus merge, final checkpoint, `latest.pt` pointing at final checkpoint, no final-ready mismatch.
- [x] Reported Slurm job id, elapsed time, requested and actual node-hours, run root, train log, manifest, summary, and final `latest.pt`.
- [x] Reported large trailing rank-0 training loss averages over last 500, 1000, and 2000 local steps, first-window comparison, and final loss.
- [x] Reported throughput mean/median, filtered mean/median excluding save/merge dips, and approximate ensemble tokens processed.
- [x] Reported `DILOCO_MERGES`, `DILOCO_SYNC_TOTAL_S`, `DILOCO_SYNC_AVG_MS`, and sync fraction estimate.
- [x] Treated eval as secondary context only.
- [x] Confirmed no GDN2, CMAES, schedule-free outer, K sweep, LR sweep, DDP, or island-size change was submitted.
