# Frontier E97 continuation and scaleout quality pass (2026-06-25)

Task: `quality-pass-e97-2`

## Scope reviewed

This quality pass reviewed the next E97 analysis/engineering batch before
dispatch:

- `research-e97-diloco`
- `design-e97-non`
- `prepare-e97-pretrained`

The user intent is to understand how serious the 32-node behavior is, what
convergence behavior to expect for large E97 models, and how to use a
pretrained or continuation checkpoint correctly.

## Verdict

Pass after spec tightening. The downstream tasks now have concrete validation
criteria and explicit launch guardrails. This quality pass does not authorize
Slurm training, does not authorize `run-64-node-e97`, and keeps E97-MLP as the
primary target. GDN2 remains only a control/comparator where mentioned.

## Validation checklist

- [x] Downstream tasks have concrete validation criteria.
- [x] The research task separates literature expectations, current Frontier
      evidence, and unvalidated hypotheses.
- [x] The resume-bootstrap task focuses on enabling schedule-free or
      partial-average continuation from avg/pretrained checkpoints without
      silently changing model weights.
- [x] The pretrained-intake task asks for and checks checkpoint path plus
      model/tokenizer compatibility before any training launch.
- [x] No Slurm training job or `run-64-node-e97` resume is authorized by this
      quality pass.

## Task-by-task review

### `research-e97-diloco`

Result: pass after clarification.

The task is explicitly research/report-only and now requires a report that
separates literature-backed DiLoCo/local-SGD expectations, repo/Frontier
observations, and hypotheses. Its validation requires primary-source citations
for external literature where used, explicit from-scratch versus pretrained
continuation analysis, fixed-eval/source comparability discussion, E97-MLP as
the primary target, and confirmation that no `sbatch` command was run.

### `design-e97-non`

Result: pass after clarification.

The task now requires an explicit fail-closed design for bootstrapping missing
non-avg outer optimizer state from already-loaded model weights. Its validation
requires semantics for schedule-free and momentum/partial-average state,
checkpoint metadata/reporting requirements, a CLI/config guard, and tests that
prove bootstrap creates only optimizer/outer state without changing loaded
model tensors. It remains design/audit-first and does not authorize Frontier
training.

### `prepare-e97-pretrained`

Result: pass after clarification.

The task now stops at an intake checklist when no checkpoint path is present.
If a path is available, it must verify path existence and record checkpoint
metadata without launching training. Validation now explicitly requires model
and tokenizer compatibility checks before any proposed continuation: E97-MLP
variant, params/dim/depth/mlp_ratio, vocab/tokenizer identity, context length,
dtype, checkpoint tensor names/shapes, and compatibility with `eval_checkpoint.py`
and the ROCm fused E97 path.

## Launch and pause audit

- `run-64-node-e97` remains `open (PAUSED)`.
- No downstream task in this batch authorizes `sbatch`.
- Any future training launch must come from a separate human-authorized task
  that provides the checkpoint path/config and explicit launch authorization.
- E97-MLP remains primary. GDN2 is not part of the requested scaleout diagnosis
  except as a control/comparator in analysis.

## Changes made

- Tightened `research-e97-diloco` validation around claim categories,
  source citation, E97-MLP primacy, and no-`sbatch` confirmation.
- Tightened `design-e97-non` validation around explicit bootstrap guards,
  fail-closed behavior, unchanged model weights, and metadata requirements.
- Tightened `prepare-e97-pretrained` validation around missing-path intake,
  model/tokenizer compatibility, no launch, and no-`sbatch` confirmation.

## Dimension scores

- Concrete validation criteria: 1.00
- Literature versus Frontier evidence separation: 1.00
- Resume bootstrap semantics and model-weight preservation: 1.00
- Pretrained intake path and compatibility gates: 1.00
- Slurm/run-64 authorization discipline: 1.00

Overall grade: 1.00.

Rubric underspecification flag: false. The quality-pass criteria were explicit
and directly checkable against the WG task descriptions and graph state.
