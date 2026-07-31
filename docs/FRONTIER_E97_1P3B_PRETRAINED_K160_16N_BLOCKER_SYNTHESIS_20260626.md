# E97 1.3B Pretrained K160 16-Node Blocker Synthesis - 2026-06-26

WG task: `synthesize-e97-k160-16`

## Decision

Do **not** continue the K160 singleton-GPU-island ladder to 32 nodes from the
completed 16-node result.

The 16-node systems run passed and produced a valid final consensus checkpoint,
but the primary quality gate for this ladder is the large trailing averaged
training-loss-window signal. That signal is not cleanly improving:

| Window | First avg loss | Last avg loss | Delta |
| --- | ---: | ---: | ---: |
| 500 local steps | `2.628524` | `2.617852` | `-0.010672` |
| 1000 local steps | `2.664922` | `2.672554` | `+0.007633` |
| 2000 local steps | `2.661771` | `2.671046` | `+0.009275` |

The last-500 window improves only marginally, while the larger 1000- and
2000-step windows regress. This is not sufficient quality evidence for any
32-node K160 continuation.

## Source Evidence Read

Primary report:

- `docs/FRONTIER_E97_1P3B_PRETRAINED_K160_16N_RUN_20260625.md`

Source run artifacts inspected:

- Run root:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_k160_16n/4902278-20260625T213328Z`
- Train log:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_k160_16n/4902278-20260625T213328Z/logs/train.log`
- Artifact manifest:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_k160_16n/4902278-20260625T213328Z/artifacts/manifest.json`
- Run manifest:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_k160_16n/4902278-20260625T213328Z/train/emender_E97_1.3B_20260625_173454/run_manifest.json`
- Summary:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_k160_16n/4902278-20260625T213328Z/summaries/summary.md`

The artifacts confirm the systems result:

- Slurm job `4902278` completed with exit `0`.
- Topology was 16 nodes / 128 ranks, singleton islands, no DDP.
- Recipe was `DILOCO_K=160`, `DILOCO_OUTER_OPTIMIZER=avg`,
  `DILOCO_EXPORT_BASIS=x`, `BATCH_SIZE=1`, `CHUNK_SIZE=2048`,
  `SAVE_EVERY=160`.
- Resume checkpoint was the 8-node K160 final `latest.pt`.
- Finalization was clean: `FINAL merge #21 at step 267059`, final checkpoint
  written, and `latest.pt` resolves to the final checkpoint.

Final K160 16-node checkpoint, for audit only:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_k160_16n/4902278-20260625T213328Z/train/emender_E97_1.3B_20260625_173454/checkpoint_step_267059_loss_2.6179.pt
```

Final `latest.pt`, for audit only:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_k160_16n/4902278-20260625T213328Z/train/emender_E97_1.3B_20260625_173454/latest.pt
```

These paths are not approved as the source for a 32-node K160 continuation.

## Eval Context

The larger deterministic eval from `evaluate-e97-1-3b` is treated as secondary
context, not as a replacement for the training-window gate. It nevertheless
reinforces the no-go decision:

| Row | CE | BPB | Delta CE vs source | Delta BPB vs source |
| --- | ---: | ---: | ---: | ---: |
| Source | `26.83747050` | `10.31650033` | `+0.00000000` | `+0.00000000` |
| K40 | `26.85514759` | `10.32329551` | `+0.01767709` | `+0.00679518` |
| K160 | `27.08281208` | `10.41081125` | `+0.24534158` | `+0.09431092` |

Policy interpretation: because the primary K160 16-node training windows already
fail to show a clean strong improvement, no policy change is needed to block
the 32-node K160 rung. The deterministic eval is supporting evidence only.

## Downstream Action

Keep `run-e97-1p3b-k160-32n` blocked/paused. Do not submit a 32-node K160 job
from the 16-node final checkpoint above.

If the project wants a further pretrained 1.3B continuation, redirect away from
this K160 ladder. The better-supported branch is K40, based on the larger
deterministic eval and the earlier K-sweep synthesis. That would require a
fresh, explicitly approved K40 follow-up task because the previous 16-node K40
task (`run-e97-1-3b-4`) was abandoned before the larger deterministic eval was
available.

No Slurm training or eval job was submitted from this synthesis task.

## Validation Checklist

- [x] Read `docs/FRONTIER_E97_1P3B_PRETRAINED_K160_16N_RUN_20260625.md`.
- [x] Inspected source run artifacts: train log, manifests, summary, and final
  checkpoint/latest path.
- [x] Decided explicitly that K160 16-node quality is insufficient for any
  32-node continuation.
- [x] Did not approve the final checkpoint as a 32-node resume source.
- [x] Treated deterministic eval as secondary context and made no policy change
  that would let eval alone override the training-window gate.
