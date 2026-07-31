# Frontier E97 32-Node Fixed Validation Eval

Date: 2026-06-24
WG task: `fixed-eval-e97-32`

## Verdict

The apparent 32-node E97-MLP train-loss regression is reflected on a fixed
validation slice. Both 32-node checkpoints score worse than the 16-node source
checkpoint under the same forward-only checkpoint evaluator, same fixed tensor,
same seed-free deterministic slice construction, same batch size, same saved
checkpoint-weight basis, and same E97-MLP eval code.

Decision: treat the 32-node regression as real on fixed validation, not merely
a rank-0 train-loss comparability artifact. This fixed eval completed before
any further 32-node diagnostic training was authorized.

Important caveat: the absolute CE/BPB values are near-random on this very small
smoke validation slice, so they should not be used as publication-quality model
quality numbers. The gate decision is based on the row-matched relative result:
both 32-node checkpoints regress versus the 16-node source on the identical
slice and invocation.

## Eval Invocation

Committed wrapper:

```bash
scripts/frontier/e97_fixed_eval_32node.sbatch
```

The wrapper uses:

- `scripts/eval_checkpoint.py`
- forward-only checkpoint scoring
- `--y-mode saved`, so the checkpoint model weights are evaluated as stored
- `HELDOUT_EVAL_BS=1`
- `MAX_CHUNKS=8`
- one Frontier node per eval attempt
- no training command, no `train.py` launch, no DiLoCo launch

Fixed validation tensor:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260624/e97-MLP/4894484-20260624T130601Z/artifacts/commapile_mainmix_val_smoke_p50k_2048.pt
```

Tensor construction:

- source file: `/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_val_smoke.txt`
- tokenizer: `p50k_base`
- construction: first 8 contiguous, non-overlapping full 2048-token chunks
- scored tokens: 16,384
- bytes/token: 3.467529296875

Result CSV:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260624/e97-MLP/4894484-20260624T130601Z/artifacts/e97_fixed_eval_32node.csv
```

## Results

| Checkpoint | Step | Train checkpoint loss | Fixed CE | Fixed BPB | CE delta vs 16-node | BPB delta vs 16-node |
|---|---:|---:|---:|---:|---:|---:|
| 16-node source | 1328 | 5.2531 | 10.49609756 | 4.36699062 | 0.00000000 | 0.00000000 |
| 32-node original | 1780 | 5.8701 | 10.68286133 | 4.44469527 | +0.18676377 | +0.07770465 |
| 32-node retry | 1740 | 5.8164 | 10.67199516 | 4.44017431 | +0.17589760 | +0.07318369 |

Perplexity equivalents from the same CE values:

- 16-node source: 36,174.06
- 32-node original: 43,602.13
- 32-node retry: 43,130.91

The retry did improve slightly relative to the original 32-node checkpoint, but
it remains worse than the 16-node source on the fixed slice.

## Slurm Accounting

All Slurm jobs were one-node E97-MLP forward-only eval attempts. No Slurm job
from this task requested 32 or more nodes, and none launched training.

| Job | Purpose | Nodes | Requested walltime | Requested node-hours | State | `sacct` elapsed | Actual node-hours by `NNodes*Elapsed` |
|---|---:|---:|---:|---:|---|---:|---:|
| 4894476 | initial 34-chunk eval attempt, canceled as impractical | 1 | 01:00:00 | 1.0000 | CANCELLED by user | 00:01:06 | 0.0183 |
| 4894484 | 8-chunk eval, completed 16-node and original-32 rows, canceled before retry row timeout risk | 1 | 02:00:00 | 2.0000 | CANCELLED by user | 00:02:05 | 0.0347 |
| 4894488 | retry-only row against the same tensor/CSV | 1 | 01:00:00 | 1.0000 | COMPLETED | 00:00:58 | 0.0161 |

Total requested node-hours: 4.0000.

Total actual node-hours from `sacct` elapsed times: approximately 0.0692.

## Scope And Gate Confirmations

- This fixed eval completed before any further 32-node diagnostic training was
  authorized.
- `run-64-node-e97` remains open and paused. It was not resumed by this task.
- No 64+ node job was authorized or submitted from this task.
- No 32+ node training job was submitted from this task.
- The only evaluated research arm was E97-MLP.
- GDN2, schedule-free variants, CMAES variants, and other unrelated arms stayed
  out of scope.

## Recommendation

Downstream `diagnose-e97-32-node` should proceed on the assumption that the
32-node E97-MLP avg path has a real fixed-validation regression. The next
diagnostic should look for a bounded E97-MLP-only scale dynamics cause rather
than treating the loss gap as only a rank-0 train-loss logging artifact.
