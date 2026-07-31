# Frontier E97 schedule-free 4/8 synthesis

Date: 2026-06-25
Task: `synthesize-e97-schedule`

## Decision

Proceed to one bounded 16-node E97-MLP schedule-free `sfsgd_y` smoke.

The 4-node and 8-node schedule-free outer probes are operationally clean enough
and promising enough for a bounded 16-node probe. This is an authorization for
the downstream `run-e97-schedule-3` scope only. It does not authorize any
32-node or 64-node schedule-free run.

The evidence supports an operational scale probe, not a claim that
schedule-free has beaten the prior 16-node avg trajectory. Both schedule-free
probes started from scratch because the audited avg checkpoint cannot initialize
a coherent `sfsgd` outer optimizer state. Fixed eval is therefore useful here as
a health and comparability check on the same saved tensor, not as a
same-trajectory quality comparison.

## Scope guard

- No Slurm job was submitted from this synthesis task.
- I queried existing accounting and task state only; I did not run `sbatch` or
  `srun`.
- No 32-node or 64-node schedule-free job is authorized by this synthesis.
- `run-64-node-e97` remains open and paused. `wg show run-64-node-e97` reports
  `Status: open (PAUSED)`, with the prior pause reason tied to the 32-node avg
  loss regression.

## Comparison

| Dimension | 4-node schedule-free | 8-node schedule-free | Synthesis |
| --- | ---: | ---: | --- |
| Training job | `4899141` | `4899142` | Both completed cleanly. |
| Fixed eval job | `4899197` | `4899198` | Both completed cleanly. |
| Nodes / islands | 4 nodes / 4 islands | 8 nodes / 8 islands | 8-node topology scaled the same path to 64 ranks. |
| Slurm state / exit | `COMPLETED` / `0:0` | `COMPLETED` / `0:0` | No operational launch failure. |
| Training elapsed | `00:20:26` | `00:20:27` | Same bounded runtime envelope. |
| Training actual node-hours | `1.362222` | `2.726667` | 8-node cost is near exactly 2x 4-node cost. |
| Fixed-eval elapsed | `00:00:50` | `00:00:51` | Eval overhead is small. |
| Fixed-eval actual node-hours | `0.013889` | `0.014167` | One-node eval cost is negligible relative to training. |
| Final step | `1204` | `1190` | Comparable token budget under the same walltime. |
| Final train loss summary | `FINAL_LOSS_LAST100: 4.7773` | `FINAL_LOSS_LAST100: 4.8115` | Both finite and productive; neither shows instability. |
| Fixed eval CE / BPB | `4.85631931` / `2.02051293` | `4.85920626` / `2.02171407` | Essentially matched on the saved tensor. |
| DiLoCo merges | 5 total, final at step 1204 | 5 total, final at step 1190 | Periodic and final consensus merge paths both worked. |
| DiLoCo sync avg | `6599.5 ms` | `6504.5 ms` | No scale-up sync regression at 8 nodes. |
| Checkpoint retention | retained 500/750/1000/final | retained 500/750/1000/final | `KEEP_CHECKPOINTS=4` behaved correctly. |
| `latest.pt` | final checkpoint symlink | final checkpoint symlink | Finalization/latest behavior is clean. |
| Schedule-free state | `diloco_outer_state` present; `mode=sfsgd`, `k=5`, `weight_sum=5.0`, `lr_max=1.0`, `x/y/z` present | same | Saved optimizer state looks complete for this probe depth. |

## Operational health

Both jobs launched with the audited schedule-free outer configuration:
`--optimizer schedulefree`, DiLoCo enabled, `--diloco_k 250`,
`--diloco_outer_optimizer sfsgd`, `--diloco_outer_lr 1.0`,
`--diloco_outer_beta 0.1`, `--diloco_export_basis y`, no resume checkpoint, and
bounded checkpoint retention. Both runs exited `COMPLETED` with exit code `0:0`.

The 4-node run did not show a traceback, non-finite loss, OOM, RCCL/NCCL
watchdog timeout, collective mismatch, checkpoint write failure, or eval load
failure. The 8-node run had post-exit PyTorch TCPStore heartbeat warnings after
the successful final checkpoint, but these did not affect Slurm exit status,
checkpoint finalization, or fixed eval. That warning should be watched at 16
nodes, but it is not a blocker for one bounded probe.

## Loss behavior

The 4-node loss improved from about 10.5 near the first logged steps into the
4-5 range, with final summary `FINAL_LOSS_LAST100: 4.7773`. The 8-node run
stayed finite through the same short window and ended at
`FINAL_LOSS_LAST100: 4.8115`. The short-run loss streams are not monotone, but
they are productive and do not show divergence or a scale-specific instability
at 8 nodes.

The late-window 8-node throughput was roughly `150k-157k` global tokens/s,
compared with roughly `30k` global tokens/s for the 4-node representative
merge/checkpoint windows and a higher final sample. Throughput evidence is
noisy over a short smoke, but the 8-node run did not reveal an operational
throughput collapse.

## Fixed eval comparability

Both probes used saved-basis fixed eval on the audited tensor:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260624/e97-MLP/4894484-20260624T130601Z/artifacts/commapile_mainmix_val_smoke_p50k_2048.pt
```

Rows:

```csv
step,tokens,ce,bpb,split,checkpoint
1204,2465792,4.85631931,2.02051293,primary,/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/e97-MLP/4899141-20260625T091236Z/train/emender_E97_1.3B_20260625_051345/checkpoint_step_001204_loss_4.7773.pt
1190,2437120,4.85920626,2.02171407,primary,/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/e97-MLP/4899142-20260625T091312Z/train/emender_E97_1.3B_20260625_051423/checkpoint_step_001190_loss_4.8115.pt
```

The 4-node and 8-node fixed eval scores differ by only about `0.0029` CE and
`0.0012` BPB. That close agreement is a good saved-basis/y-handling signal:
the checkpointed schedule-free basis can be loaded and scored consistently at
both sizes.

The prior 16-node avg source fixed eval row was much worse on this tensor
(`CE=10.49609756`, `BPB=4.36699062`), but the comparison is not a controlled
continuation because these probes were from scratch and the source checkpoint is
an avg-outer checkpoint. The correct interpretation is: fixed eval machinery and
saved-basis handling are healthy enough to compare the 4-node and 8-node
schedule-free probes and to support one bounded 16-node probe.

## Checkpoint and latest behavior

Both runs performed four periodic DiLoCo merges plus one final consensus merge.
Both final checkpoints were written after final consensus, and both `latest.pt`
links advanced to the final checkpoint:

- 4-node: `latest.pt -> checkpoint_step_001204_loss_4.7773.pt`
- 8-node: `latest.pt -> checkpoint_step_001190_loss_4.8115.pt`

Retention was bounded at four checkpoint files in both runs. The 4-node report
also checked for final `.tmp`, `.partial`, or incomplete checkpoint files and
found none. The available evidence indicates schedule-free finalization,
retention, and `latest.pt` behavior are ready for a bounded 16-node probe.

## Node-hour cost

Training cost:

- 4-node requested: `2.000000` node-hours; actual: `1.362222`.
- 8-node requested: `4.000000` node-hours; actual: `2.726667`.

Fixed eval cost:

- 4-node eval actual: about `0.013889` node-hours.
- 8-node eval actual: about `0.014167` node-hours.

Combined actual cost for the two training probes plus fixed eval was about
`4.116945` node-hours. A bounded 16-node probe with the same about-20.5 minute
runtime would cost roughly `5.45` actual node-hours for training, plus about
`0.014` node-hours for one-node fixed eval if repeated. Requested cost at a
30-minute walltime would be `8.0` node-hours.

This is a reasonable next probe cost given the clean 4-node and 8-node evidence.
It is not enough evidence to authorize 32-node or 64-node schedule-free work.

## Schedule-free-specific issues

The main schedule-free risk called out by the audit was coherent `y`/saved-basis
handling. Both probes used export basis `y`, retained `diloco_outer_state` in
the final checkpoint, and passed fixed eval with `--y-mode saved`.

The final checkpoint state in both runs included `mode=sfsgd`, `k=5`,
`weight_sum=5.0`, `lr_max=1.0`, and `x/y/z` state. That is the expected shape
for this short probe and is sufficient for operational continuation of the
schedule-free path.

Residual issues to watch at 16 nodes:

- Treat the 8-node post-final TCPStore heartbeat warning as a teardown warning
  to monitor, not a current blocker.
- Keep using saved-basis fixed eval; do not switch to export basis `x` in the
  bounded 16-node probe.
- Preserve the bounded scope: one 16-node smoke, no 32/64-node escalation from
  this decision.

## Recommendation for `run-e97-schedule-3`

Run exactly one bounded 16-node E97-MLP schedule-free `sfsgd_y` smoke using the
same audited configuration family and the same guardrails:

- no resume from the avg checkpoint;
- `diloco_export_basis=y`;
- bounded walltime and checkpoint retention;
- inspect final consensus merge, final checkpoint, `latest.pt`, retained
  `diloco_outer_state`, and saved-basis fixed eval;
- stop after the 16-node result for a separate synthesis before any larger
  schedule-free scale.

Do not submit or authorize any 32-node or 64-node schedule-free run from this
synthesis.
