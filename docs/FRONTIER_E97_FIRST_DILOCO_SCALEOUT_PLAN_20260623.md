# E97-First DiLoCo Scaleout Plan

Date: 2026-06-23
Task: `plan-e97-first-diloco-scaleout`

## Recommendation

2026-06-25 policy update: future E97-MLP scaleout tasks should treat
`DILOCO_ISLAND_SIZE=1` GPU islands, with no per-step DDP gradient averaging, as
the primary hypothesis when the model fits on one GPU. The older
`DILOCO_ISLAND_SIZE=8` node-island shape in this plan is retained as historical
baseline/control context, not as the preferred launch recipe. See
`docs/FRONTIER_E97_SCALEOUT_NO_DDP_POLICY_20260625.md`.

Decision: **PLAN ONLY; do not submit 4-node or 8-node jobs yet**.

Run a cheap E97-MLP 2-node canary first, then compare E97-MLP regular
model-averaging DiLoCo against the schedule-free outer arm at 2, 4, and 8
Frontier nodes only after the checkpoint/resume gate is clean. GDN2-MLP should
remain a high-quality control arm because its one-node evidence is cleaner, but
it must not become the optimization target before E97-MLP has a fair checkpoint
and 2-node canary path.

The scaleout ladder is intentionally small:

1. E97-MLP 2-node canary for regular averaging and schedule-free outer.
2. E97-MLP 4-node pilots only after the 2-node checkpoint/resume evidence
   passes.
3. E97-MLP 8-node pilots only after 4-node metrics are interpretable and no
   checkpoint, resume, or merge defect is open.
4. GDN2-MLP controls are run after the matching E97 stage has produced usable
   evidence, not before it.

No expensive multi-node job was launched by this planning task.

## Evidence Separation

Implementation status:

- `train.py` exposes the required DiLoCo command surface:
  `--diloco`, `--diloco_k`, `--diloco_outer_optimizer`,
  `--diloco_outer_lr`, `--diloco_outer_beta`, `--diloco_export_basis`, and
  `--diloco_island_size`.
- `train.py` supports `--resume`, writes rank-0 checkpoints named
  `checkpoint_step_*.pt`, writes `latest.pt`, records DiLoCo outer state when
  the selected outer optimizer has state, and prints final DiLoCo sync metrics.
- `scripts/frontier/diloco_scaleout_readiness.sbatch` is the reusable Frontier
  template. This task updated it to expose `DILOCO_OUTER_OPTIMIZER`,
  `DILOCO_EXPORT_BASIS`, and `RESUME_CHECKPOINT` so the planned comparison can
  be expressed without editing training code.
- The template already captures environment, git status, command line,
  manifest, train log, final summary, requested node-hours, and failure lines.

Live validation and canary evidence:

- E97-MLP has Frontier one-node post-cache evidence for first finite loss,
  throughput, memory, fused/no-eager guard, and a checkpoint, but the cited
  job ended failed because it predated the rank-mapping/checkpoint-race script
  fix. That is not clean enough for 4/8-node approval.
- GDN2-MLP has cleaner one-node post-cache and resume evidence. Use it as a
  control and sanity benchmark, not as the first optimization target.
- The required live gate for this plan is
  `validate-e97-two-node-checkpoint-canary`: it must record successful E97-MLP
  2-node checkpoint, final-checkpoint, `latest.pt`, and resume behavior before
  any 4-node or 8-node task is created or submitted.

Future scaleout execution:

- This document is the plan. Workers should create separate execution tasks for
  each approved stage.
- Do not combine checkpoint fixes, canary execution, ledger accounting, and
  4/8-node launches in the same task.
- If E97 checkpoint/final-checkpoint/resume behavior fails, route the defect to
  `fix-e97-mlp-checkpoint-finalization`, `implement-walltime-final-checkpoint`,
  or a new checkpoint-specific task. Do not hide it with wrapper logic in the
  scaleout launch script.

## Audited Command Surface

Primary template:

```text
scripts/frontier/diloco_scaleout_readiness.sbatch
```

Common Frontier launch shape:

```bash
cd /lustre/orion/bif148/scratch/erikgarrison/emender
export WG_TASK_ID="<execution-task-id>"
export HUMAN_APPROVAL_RECORD="<WG message or committed approval artifact>"
export SCALEOUT_VARIANT="e97-MLP"
export SCALEOUT_NODES="<2|4|8>"
export SCALEOUT_WALLTIME="<walltime>"
export TRAIN_MINUTES="<walltime minutes minus checkpoint drain>"
export DATA="/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt"
export VAL_DATA="/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_val_smoke.txt"
export TIKTOKEN_CACHE_DIR="/lustre/orion/bif148/proj-shared/tiktoken_cache"
export EMENDER_CONDA_ENV="${MEMBERWORK}/emender/conda/emender-rocm711"
export DILOCO_K=250
export DILOCO_ISLAND_SIZE=1
export BATCH_SIZE=1
export CHUNK_SIZE=2048
export SAVE_EVERY=250
export KEEP_CHECKPOINTS=4
sbatch -N "$SCALEOUT_NODES" -t "$SCALEOUT_WALLTIME" \
  scripts/frontier/diloco_scaleout_readiness.sbatch
```

E97 model geometry used by the template for `SCALEOUT_VARIANT=e97-MLP`:

```text
--level E97
--linear_state 0
--use_triton 1
--n_state 32
--n_heads 323
--dim 1536
--depth 10
--expansion 1.0
--use_gate 1
--gate_activation silu
--mlp_ratio 1.5
--mlp_multiple 64
```

GDN2 control geometry used by the template for `SCALEOUT_VARIANT=gdn2-MLP`:

```text
--level gdn2-mlp
--dim 2176
--depth 12
--n_heads 30
--expansion 1.0
--use_conv 1
--d_conv 4
--gdn2_mlp_ratio 3.258732449079677
```

Regular model averaging arm:

```bash
export DILOCO_OUTER_OPTIMIZER=avg
export DILOCO_OUTER_LR=1.0
export DILOCO_OUTER_BETA=0.0
export DILOCO_EXPORT_BASIS=x
```

Schedule-free outer arm:

```bash
export DILOCO_OUTER_OPTIMIZER=sfsgd
export DILOCO_EXPORT_BASIS=y
export DILOCO_OUTER_LR=1.0
export DILOCO_OUTER_BETA=0.1
```

Rationale for the schedule-free outer choice: local P4/P5 planning retained
only `sfsgd` export-y as the plausible schedule-free outer comparator. Plain
averaging was still the best local arm, so the Frontier question is whether
the schedule-free outer becomes more useful as island count grows, not whether
it is already the favorite.

Resume command surface:

```bash
export RESUME_CHECKPOINT="/path/to/prior/run/train/<run-name>/latest.pt"
sbatch -N "$SCALEOUT_NODES" -t "$SCALEOUT_WALLTIME" \
  scripts/frontier/diloco_scaleout_readiness.sbatch
```

The template now checks that `RESUME_CHECKPOINT` is readable and passes
`--resume` through to `train.py`. It does not change checkpoint semantics.

## E97 Scaleout Matrix

Policy note: for new primary-path E97-MLP scaleout rows, "islands" means
singleton GPU islands (`DILOCO_ISLAND_SIZE=1`) unless the task explicitly says
it is running a node-island DDP control. The historical one-node-island column in
this plan described the old `DILOCO_ISLAND_SIZE=8` topology and should not be
copied into new primary launch tasks.

All rows below are planned, not approved. Requested node-hours are conservative
scheduler requests, not consumed accounting. Before submission, the execution
task must add a row to `docs/FRONTIER_ALLOCATION_LEDGER_20260621.md` and record
the approval artifact.

| Stage | Arm | Nodes | GCDs | GPU islands | Walltime | Train minutes | Requested node-hours | Status | Required gate |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |
| `E97-DILOCO-2N-AVG-CANARY` | regular average | 2 | 16 | 16 | 02:00:00 | 90 | 4 | first canary candidate | `validate-e97-two-node-checkpoint-canary` prerequisite tasks have produced clean E97 checkpoint/final/resume implementation evidence |
| `E97-DILOCO-2N-SFOUTER-CANARY` | schedule-free outer | 2 | 16 | 16 | 02:00:00 | 90 | 4 | second canary candidate | 2-node avg canary reaches finite loss, at least one merge, checkpoint, final checkpoint, and resume |
| `E97-DILOCO-4N-AVG-PILOT` | regular average | 4 | 32 | 32 | 04:00:00 | 210 | 16 | blocked | both 2-node E97 arms pass and checkpoint/resume evidence is clean |
| `E97-DILOCO-4N-SFOUTER-PILOT` | schedule-free outer | 4 | 32 | 32 | 04:00:00 | 210 | 16 | blocked | 4-node avg reaches first merge, checkpoint, and stable loss/throughput |
| `E97-DILOCO-8N-AVG-PILOT` | regular average | 8 | 64 | 64 | 08:00:00 | 450 | 64 | blocked | 4-node comparison reviewed, no open checkpoint/resume/merge defect |
| `E97-DILOCO-8N-SFOUTER-PILOT` | schedule-free outer | 8 | 64 | 64 | 08:00:00 | 450 | 64 | blocked | 8-node avg is interpretable and approval is refreshed |

E97 requested node-hour envelope for the six planned rows:

```text
2-node comparison:   8 node-hours total
4-node comparison:  32 node-hours total
8-node comparison: 128 node-hours total
E97 total:         168 requested node-hours
```

The 2-node rows are allowed to be cheap canaries. The 4-node and 8-node rows
must remain separate future execution tasks and are not launchable from this
planning task.

## GDN2 Control Schedule

GDN2-MLP should benchmark the effective RNN-training state of the art under the
same data, tokenizer, context, batch, DiLoCo K, island size, logging, and
checkpoint cadence. It is lower priority than E97: run each GDN2 control only
after the matching E97 stage has produced usable evidence or after a human
explicitly asks for a control-first sanity check.

| Stage | Arm | Nodes | Walltime | Requested node-hours | Priority | Purpose |
| --- | --- | ---: | --- | ---: | --- | --- |
| `GDN2-DILOCO-2N-AVG-CONTROL` | regular average | 2 | 02:00:00 | 4 | lower than E97 2-node | Control for launch, loss, throughput, checkpoint, and resume behavior at the E97 canary scale. |
| `GDN2-DILOCO-4N-AVG-CONTROL` | regular average | 4 | 04:00:00 | 16 | lower than E97 4-node | Control for scaling efficiency and merge overhead at the 4-node point. |
| `GDN2-DILOCO-8N-AVG-CONTROL` | regular average | 8 | 08:00:00 | 64 | lower than E97 8-node | Control for whether E97 scaleout is competitive with the cleaner GDN2 arm. |

Optional GDN2 schedule-free outer controls should not be part of the first
control pass. Add them only if the E97 schedule-free outer shows a plausible
advantage or if GDN2 regular averaging exposes a scale-specific anomaly that
requires outer-optimizer separation.

GDN2 requested node-hour envelope for the first control pass:

```text
GDN2 controls: 84 requested node-hours
Combined E97 + first GDN2 controls: 252 requested node-hours
```

## Gates

Hard gate before any 4-node or 8-node launch:

- `validate-e97-two-node-checkpoint-canary` is done and records live E97-MLP
  evidence for checkpoint write, final checkpoint, `latest.pt`, resume from
  `latest.pt`, later finite loss after resume, and interpretable DiLoCo merge
  logs.

Hard gate before any scaleout submission:

- Ledger row is added before submission with requested node-hours, reserve,
  allocation before/after, model variant, arm, queue, walltime, output root, and
  approval artifact.
- Human approval is recorded in a WG message or committed approval artifact.
- Shared tokenizer cache is readable from compute nodes.
- Data and validation paths are readable from compute nodes.
- E97 fused guard prints `NO eager fallback` on all ranks.
- Primary-path E97-MLP runs use `--diloco_island_size 1`; any
  `--diloco_island_size 8` run is labeled as a node-island DDP control/baseline.
- `SAVE_EVERY` is aligned with `DILOCO_K=250` for consensus checkpoints.
- `RESUME_CHECKPOINT`, if set, is readable before launch.

Stop ladder and route to an implementation task if any of these occur:

- no checkpoint or broken `latest.pt` after a save boundary;
- final checkpoint missing after normal walltime drain;
- resume does not load the expected step or fails before a later finite loss;
- non-main ranks attempt to write checkpoints;
- DiLoCo state cannot be restored for `sfsgd` outer resumes;
- repeated nonfinite loss/gradient, OOM, RCCL/NCCL error, or no merge by
  `DILOCO_K`;
- fused E97 path falls back to eager;
- template wrapper changes would be required to mask a training/checkpoint bug.

## Metrics

Loss sanity:

- first finite loss and step;
- loss every `LOG_EVERY`;
- last-100 training loss;
- validation CE/BPB when `VAL_DATA` is enabled;
- post-merge loss jump and recovery within the next K-window;
- nonfinite loss/gradient count and rank-local outlier loss.

Throughput:

- per-rank `tok/s`;
- global `tok/s`;
- startup and compile time separated from timed training by
  `--timer_after_compile_warmup`;
- scaling efficiency versus the best clean one-node E97 baseline:
  `global_tok_s(N) / (N * one_node_global_tok_s)`;
- cost per billion tokens:
  `requested_node_hours / processed_tokens_billion`, later replaced by
  consumed node-hours when scheduler accounting is available.

Communication overhead:

- DiLoCo merge count;
- `DILOCO_SYNC_TOTAL_S`;
- `DILOCO_SYNC_AVG_MS`;
- sync fraction of walltime and sync fraction per K-window;
- RCCL/NCCL backend environment, rank mapping, node list, and failure messages;
- schedule-free outer overhead versus regular averaging at matched nodes.

Checkpoint and resume:

- periodic checkpoint filenames and step/loss values;
- `latest.pt` target and readability;
- final checkpoint written after the final merge or explicit skip when the last
  step already merged;
- rank/head behavior: only main rank writes checkpoint/final artifacts;
- resume command, source checkpoint, loaded step, later finite loss, and new
  checkpoint written after resume;
- `diloco_outer_state` presence for `sfsgd` resumes.

Failure modes:

- scheduler/QOS rejection;
- environment/module/import failure;
- tokenizer/data readability failure;
- fused-kernel guard failure or eager fallback;
- RCCL/NCCL/Slingshot failure;
- OOM or allocator fragmentation;
- nonfinite loss/gradient;
- checkpoint I/O/symlink failure;
- resume incompatibility;
- timeout before final checkpoint;
- missing summary/manifest/log artifact.

## Execution Task Boundaries

Create separate tasks in this order when the gates are satisfied:

1. `run-e97-diloco-2n-avg-canary`: execute only the regular-average E97
   2-node row and report metrics/checkpoint/resume evidence.
2. `run-e97-diloco-2n-sfouter-canary`: execute only the schedule-free outer
   E97 2-node row if the avg row is clean.
3. `review-e97-diloco-2n-scaleout-evidence`: compare the two 2-node rows and
   decide whether 4-node work is justified.
4. `run-e97-diloco-4n-comparison`: execute the two 4-node E97 rows if the
   review approves.
5. `review-e97-diloco-4n-scaleout-evidence`: decide whether 8-node work is
   justified.
6. `run-e97-diloco-8n-comparison`: execute the two 8-node E97 rows if approved.
7. `run-gdn2-diloco-controls`: run the lower-priority GDN2 controls at the
   scales already cleared by E97 evidence.

Each execution task should own its ledger update, launch command, and artifact
collection. No two tasks should write the same run root or update the same
ledger row concurrently.

## Validation Checklist

- Current origin/main DiLoCo and schedule-free outer command surface is audited
  above.
- E97-MLP regular averaging and E97-MLP schedule-free outer are planned at 2,
  4, and 8 nodes with requested node-hours.
- GDN2-MLP control schedule is fair but lower priority than E97.
- Metrics cover loss sanity, throughput, communication overhead,
  checkpoint/final-checkpoint behavior, resume, and failure modes.
- All 4-node and 8-node launches are gated on successful E97 2-node
  checkpoint/resume evidence from `validate-e97-two-node-checkpoint-canary`.
- Implementation prerequisites, live validation/canary evidence, and future
  scaleout execution tasks are separated.
