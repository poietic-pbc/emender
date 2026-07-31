# Frontier E97 32-node recipe ladder quality pass (2026-06-25)

Task: `quality-pass-e97-32`

## Scope reviewed

The quality-pass prompt names the intended downstream ladder as:

- `plan-e97-32node-recipe-ladder`
- `run-e97-32node-k80-diagnostic`
- `run-e97-32node-partialavg-diagnostic`
- `run-e97-32node-schedulefree-diagnostic`
- `synthesize-e97-32node-recipe-ladder`

The live WG graph uses these normalized task IDs:

- `plan-e97-32-node`
- `run-e97-32-node`
- `run-e97-32-node-2`
- `run-e97-32-node-3`
- `synthesize-e97-32-node`

This pass reviewed the live graph tasks.

## Verdict

Pass. The ladder is sequential, E97-MLP-only, allocation-disciplined, and has concrete validation criteria on each downstream task. The current `run-64-node-e97` task remains paused and no downstream ladder task authorizes a 64+ node job.

## Validation checklist

- [x] Each downstream task has concrete validation criteria.
- [x] Fixed validation is required after every submitted training diagnostic.
- [x] `run-64-node-e97` remains paused and no 64+ node job is authorized.
- [x] E97-MLP remains the only training/eval arm in this ladder.
- [x] The ladder runs one diagnostic at a time and each later rung can no-op if an earlier recipe is already green.

## Task-by-task review

### `plan-e97-32-node`

Result: pass.

The planning task is explicitly non-executing: it says not to submit Slurm jobs. Its validation requires a concise ordered ladder, exact first diagnostic and acceptance gate, confirmation that no jobs were submitted, confirmation that `run-64-node-e97` remains paused, and confirmation that GDN2/CMAES are out of scope while schedule-free is not the default first fix.

### `run-e97-32-node`

Result: pass.

This K80 diagnostic depends on `plan-e97-32-node`, keeping the first training rung sequential. It allows at most one bounded `<=32` node E97-MLP training job, only if K80 remains the selected next rung. It requires the same fixed eval source-vs-K80 checkpoint after any run, and explicitly requires confirmation that no 64+ jobs, GDN2/CMAES, or schedule-free run were submitted. It also requires leaving `run-64-node-e97` paused.

### `run-e97-32-node-2`

Result: pass.

This partial-average diagnostic depends on the K80 diagnostic. It can no-op if K80 cleared the gate, and otherwise requires a semantics audit before launch. It allows at most one bounded `<=32` node E97-MLP training diagnostic and requires same fixed-tensor source-vs-candidate eval after any job. It explicitly excludes 64+ jobs, GDN2/CMAES, and schedule-free runs, and leaves `run-64-node-e97` paused.

### `run-e97-32-node-3`

Result: pass.

This schedule-free diagnostic depends on the partial-average diagnostic, so schedule-free is a later controlled arm rather than an uncontrolled default. It can no-op if an earlier rung clears the gate. If run, it permits at most one bounded `<=32` node E97-MLP schedule-free diagnostic, requires exact config recording before launch, prohibits extra uncontrolled variables, and requires the same fixed source-vs-candidate eval. It also requires no 64+ jobs, no GDN2/CMAES, and leaving `run-64-node-e97` paused.

### `synthesize-e97-32-node`

Result: pass.

This synthesis task depends on the schedule-free conditional rung and is explicitly non-executing. It requires comparing all available 32-node E97-MLP recipes against the same 16-node source and fixed validation gate, recommends next action without automatically resuming `run-64-node-e97`, and confirms no Slurm jobs were submitted. It keeps GDN2 as a control track outside this scale-fix ladder.

## Dependency/order audit

The live task dependencies enforce the intended sequential ladder:

`quality-pass-e97-32` -> `plan-e97-32-node` -> `run-e97-32-node` -> `run-e97-32-node-2` -> `run-e97-32-node-3` -> `synthesize-e97-32-node`

The ladder therefore authorizes only one training diagnostic rung at a time. The diagnostic task descriptions additionally require later rungs to no-op when earlier evidence has already cleared the fixed gate.

## Allocation and scope audit

- Maximum training allocation per rung: at most one bounded `<=32` node diagnostic.
- Fixed eval: required after every submitted K80, partial-average, or schedule-free run.
- 64-node gate: `run-64-node-e97` is still `open (PAUSED)` and its own logs record that it was paused after the 32-node retry repeated the loss regression.
- Scope: all ladder training/eval tasks are E97-MLP-only. GDN2/CMAES are excluded from the ladder.
- Schedule-free: present only as rung 3 after cadence and merge-strength checks, with explicit no-op/defer behavior if earlier rungs are green.

## Dimension scores

- Concrete validation criteria: 1.00
- Fixed-validation discipline: 1.00
- Allocation discipline and 64-node pause: 1.00
- E97-MLP-only scope: 1.00
- Sequential/no-op ladder behavior: 1.00

Overall grade: 1.00.

Rubric underspecification flag: false. The quality-pass criteria were explicit and directly checkable in the WG task descriptions and dependency graph.
