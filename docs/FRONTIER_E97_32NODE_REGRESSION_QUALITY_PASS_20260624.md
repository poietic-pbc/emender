# Frontier E97 32-Node Regression Resolution Quality Pass

Date: 2026-06-24
Task: `quality-pass-32-node`

## Reviewed Tasks

- `merge-32-node-e97`
- `fixed-eval-e97-32`
- `diagnose-e97-32-node`

## Scope Checks

- E97-MLP remains the primary research arm for this resolution batch.
- GDN2, CMAES, and broad schedule-free exploration remain out of scope unless a separate task explicitly requests them.
- `run-64-node-e97` remains paused and this quality pass did not authorize any 64+ node jobs.
- This quality pass did not submit Slurm jobs.

## Task Spec Updates

- `merge-32-node-e97`: validation now explicitly requires confirmation that no 64+ node jobs were authorized and that no GDN2/schedule-free/CMAES scope was added while merging documentation/report evidence.
- `fixed-eval-e97-32`: description now states that fixed evaluation is the required gate before any further 32-node diagnostic training. Validation now requires recording that the gate completed first, that `run-64-node-e97` remains paused, and that E97-MLP is the only evaluated arm.
- `diagnose-e97-32-node`: description now requires the fixed-eval result before any diagnostic training decision. Validation now requires confirming that fixed eval completed first, that any diagnostic is at most one bounded <=32-node E97-MLP job, and that no 64+ node work was authorized.

## Validation

- Each downstream task has a concrete `## Validation` section.
- `diagnose-e97-32-node` depends on `fixed-eval-e97-32`, so fixed validation is prioritized before further 32-node scale-out diagnostics.
- `run-64-node-e97` is still `open (PAUSED)`.
- The revised task descriptions preserve E97-first scope and keep GDN2/schedule-free work out of this batch unless separately requested.
