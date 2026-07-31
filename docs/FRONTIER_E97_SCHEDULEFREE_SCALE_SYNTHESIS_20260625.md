# Frontier E97 schedule-free scale synthesis

Date: 2026-06-25
Task: `synthesize-e97-schedule-2`

## Decision

Create one bounded 32-node E97-MLP schedule-free diagnostic, but keep it as a
separate from-scratch schedule-free scale probe. Do not treat it as a
same-source fix for the existing 32-node avg K80/partial-average ladder, and do
not use it to unblock `run-64-node-e97`.

The 4-node, 8-node, and 16-node `sfsgd_y` probes are operationally clean and
consistent enough to justify one 32-node continuation of that schedule-free
probe track. They are not controlled continuation evidence against the 16-node
avg source because all schedule-free probes start from scratch. Current
non-`avg` resume semantics still prevent coherent `sfsgd` continuation from the
avg source checkpoint that anchors the 32-node avg ladder.

Operational classification:

- Proceed to 32-node schedule-free as a bounded from-scratch diagnostic.
- Fix or explicitly define non-`avg` outer-state initialization before using
  schedule-free as a same-source 32-node avg-ladder candidate.
- Keep the avg K80 result as the best same-source 32-node evidence so far, even
  though it is not green on the fixed gate.

## Schedule-Free Evidence

All schedule-free probes used E97-MLP, inner `schedulefree`, DiLoCo outer
`sfsgd`, export basis `y`, `DILOCO_K=250`, `DILOCO_ISLAND_SIZE=8`,
`DILOCO_OUTER_LR=1.0`, `DILOCO_OUTER_BETA=0.1`, no resume checkpoint,
`SAVE_EVERY=250`, and `KEEP_CHECKPOINTS=4`. Fixed eval used saved-basis scoring
on:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260624/e97-MLP/4894484-20260624T130601Z/artifacts/commapile_mainmix_val_smoke_p50k_2048.pt
```

| Run | Job ids | Nodes / islands | State | Final step | Final train loss | Fixed CE / BPB | Merge/checkpoint state |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| 4-node `sfsgd_y` | train `4899141`, eval `4899197` | 4 / 4 | `COMPLETED` `0:0` | 1204 | 4.7773 | 4.85631931 / 2.02051293 | 5 merges; final consensus; `latest.pt` final; `diloco_outer_state` present |
| 8-node `sfsgd_y` | train `4899142`, eval `4899198` | 8 / 8 | `COMPLETED` `0:0` | 1190 | 4.8115 | 4.85920626 / 2.02171407 | 5 merges; final consensus; `latest.pt` final; `diloco_outer_state` present |
| 16-node `sfsgd_y` | train `4899221`, eval `4899229` | 16 / 16 | `COMPLETED` `0:0` | 1119 | 4.8905 | 4.84385067 / 2.01532525 | 5 merges; final consensus; `latest.pt` final; `diloco_outer_state` present |

The 16-node result is the strongest schedule-free gate so far: it scaled the
same audited path to 128 ranks, had finite productive loss, retained complete
outer state (`mode=sfsgd`, `k=5`, `weight_sum=5.0`, `lr_max=1.0`, `x/y/z`),
and slightly improved fixed CE/BPB versus the 4-node and 8-node schedule-free
rows.

The schedule-free fixed-eval rows are much lower than the older 16-node avg
source row on the same tensor, but that is not a same-trajectory comparison.
The correct interpretation is that saved-basis checkpoint/eval handling is
healthy within the schedule-free from-scratch track.

## Comparison To Avg Evidence

Known avg evidence has a different source policy: it resumes from a retained
16-node avg checkpoint and uses source-vs-candidate fixed validation.

| Evidence | Config | Result | Operational meaning |
| --- | --- | --- | --- |
| 8-node avg smoke | avg outer, `K=10`, export `x`, from scratch | clean launch/finalization; final step 451; `FINAL_LOSS_LAST100=5.7391` | Avg path was a clean low-node systems smoke. |
| 8-node avg continuation | avg outer, `K=10`, resumed from 8-node smoke | reached step 970; `FINAL_LOSS_LAST100=5.2538`, delta `-0.4853` | Avg continuation was productive enough to gate 16 nodes. |
| 16-node avg smoke | avg outer, `K=10`, resumed from 8-node continuation | clean; step 1328; `FINAL_LOSS_LAST100=5.2531`, delta `-0.0007` vs 8-node continuation | Clean systems gate, flat quality signal; became the common 32-node source. |
| 32-node avg K10 original | avg outer, `K=10`, same 16-node source | clean systems path but `FINAL_LOSS_LAST100=5.8701`; fixed BPB delta `+0.07770465` | Regressed; not a 64-node gate. |
| 32-node avg K10 retry | same as K10 original | clean systems path but `FINAL_LOSS_LAST100=5.8164`; fixed BPB delta `+0.07318369` | Reproduced K10 failure class. |
| 32-node avg K40 | avg outer, `K=40`, same 16-node source | train loss mostly repaired; `FINAL_LOSS_LAST100=5.2822`; fixed BPB delta `+0.05462847` | Cadence is a real scale-control knob, but K40 still failed fixed gate. |
| 32-node avg K80 | avg outer, `K=80`, same 16-node source | best train loss; `FINAL_LOSS_LAST100=5.1084`; fixed BPB delta `+0.03081585` | Best current same-source 32-node recipe, but still not green. |
| Partial-average | intended K80 half-average via `momentum`, same source | no job submitted | Existing semantics require non-`avg` outer state absent from the avg source. |

The schedule-free evidence complements this ladder but does not replace it.
The avg ladder asks: "from the 16-node avg checkpoint, which 32-node recipe
improves or at least does not regress fixed validation?" Schedule-free currently
answers a different question: "does the audited `sfsgd_y` path scale cleanly
when started with coherent schedule-free outer state from scratch?"

That difference is caused by a real semantic guard, not merely a conservative
choice. Current training rejects `--resume` with `--diloco_outer_optimizer
sfsgd` when the checkpoint lacks `diloco_outer_state`; the common avg source
checkpoint lacks that state.

## Proposed 32-Node Schedule-Free Diagnostic

Recommended exact training shape:

```text
SCALEOUT_VARIANT=e97-MLP
SCALEOUT_NODES=32
SCALEOUT_WALLTIME=00:30:00
TRAIN_MINUTES=20
OUTPUT_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout
DATA=/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt
VAL_DATA=/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_val_smoke.txt
TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache
BATCH_SIZE=1
CHUNK_SIZE=2048
LOG_EVERY=5
VAL_EVERY=10000
COMPILE_WARMUP_STEPS=1
```

Schedule-free/DiLoCo settings:

```text
--optimizer schedulefree
--diloco
DILOCO_K=250
DILOCO_ISLAND_SIZE=8
DILOCO_OUTER_OPTIMIZER=sfsgd
DILOCO_OUTER_LR=1.0
DILOCO_OUTER_BETA=0.1
DILOCO_EXPORT_BASIS=y
SAVE_EVERY=250
KEEP_CHECKPOINTS=4
RESUME_CHECKPOINT unset
```

Expected topology: 32 Frontier nodes, 256 ranks, 32 islands x 8 GPUs, per-step
DDP within each island, DiLoCo consensus every 250 steps across islands.

Recommended post-run checks:

- Slurm state/exit, node-hours, run root, runtime git commit, and run label.
- Finite loss trend, final step, throughput, `DILOCO_MERGES`, `DILOCO_K`,
  sync timing, and absence of watchdog/collective/OOM/non-finite signatures.
- Final consensus merge behavior, retained checkpoint set, no `.tmp`/partial
  checkpoint leftovers, and `latest.pt` pointing to the final checkpoint.
- `diloco_outer_state` present with `mode=sfsgd`, `k`, `weight_sum`, `lr_max`,
  and `x/y/z`.
- One-node saved-basis fixed eval on the same tensor, outputting a row-matched
  CSV for 4/8/16/32 schedule-free comparison.

Do not add LR sweeps, beta sweeps, `sfsgd` export basis `x`, fixed momentum,
GDN2, CMAES, 64-node training, changed island size, or a resume checkpoint to
this diagnostic. Those would change the question being tested.

## Relationship To 32-Node Avg Ladder

The 32-node schedule-free diagnostic should be created as a sibling/parallel
diagnostic track, not as a rung inside the same-source avg ladder:

- It may answer whether the `sfsgd_y` schedule-free path itself has a 32-node
  operational cliff after clean 4/8/16 evidence.
- It should not be scored against the avg ladder's 16-node-source fixed gate.
- It should not supersede K80 as the best available same-source 32-node avg
  recipe unless a later semantics task defines a coherent non-`avg` resume
  policy and reruns schedule-free from the same source.
- It should not unblock `run-64-node-e97`; that task is avg-outer scope and
  remains blocked by the not-green 32-node same-source ladder.

If the 32-node from-scratch schedule-free diagnostic is clean, the next
schedule-free-specific decision should be either a longer same-track
schedule-free run or an implementation/design task for non-`avg` outer-state
initialization. If it fails operationally, deprioritize schedule-free and keep
cadence/LR work on the avg ladder as the main path.

## Scope Confirmations

- No Slurm jobs were submitted from this synthesis task.
- `run-64-node-e97` remains `open (PAUSED)`.
- This report is E97-MLP-only. GDN2 remains a separate control track.
- No CMAES, GDN2, 64-node, or uncontrolled-source recommendation is introduced.
