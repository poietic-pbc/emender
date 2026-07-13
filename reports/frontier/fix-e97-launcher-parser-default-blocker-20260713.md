# E97 launcher minimal-change blocker

Task: `fix-e97-launcher`

Date: 2026-07-13

## Result

The requested minimal launcher edit is unsafe and was not made. Omitting
`--generations 1` and `--steps 40` does **not** mean unbounded execution in the
existing production entrypoint. The user explicitly required work to stop and
be reported if that precondition was false, so no launcher, renderer, manifest,
parity policy, training control flow, or test was changed. No Slurm job was
submitted.

The existing parser in `scripts/frontier/e97_async_diloco_train.py` defines:

```python
parser.add_argument("--generations", type=int, default=1)
parser.add_argument("--local-steps", type=int, default=1)
parser.add_argument("--steps", type=int, default=8)
```

Therefore the proposed omissions would resolve to `generations=1` and
`steps=8`, not an unbounded sentinel such as `None`. More importantly, the
canonical compiled-MPICH multinode branch does not use `args.generations` at
all. It creates one `RealAsyncFileRankConfig` and calls
`run_real_async_diloco_file_rank` once. That function in
`ndm/async_diloco_real.py` sets `generation = 0`, creates one worker spec with
`local_steps=config.local_steps`, invokes `_run_real_node_supervisor` once,
publishes once, and returns. There is no outer production generation loop for
an omitted argument to unbound.

## Exact argv before and after the requested omission

The canonical common launcher currently contributes this training argv
fragment to the production entrypoint:

```text
--generations 1 --local-steps 40 --steps 40 --timeout-s 1200
```

The narrowly requested textual edit would produce:

```text
--local-steps 40 --timeout-s 1200
```

After parsing, however, the effective values would be:

```text
generations=1 local_steps=40 steps=8 timeout_s=1200
```

The selected actual-multinode branch would still execute exactly one
`generation = 0` with 40 local steps because `--generations` is ignored by that
branch and `--local-steps 40` is intentionally retained. The edit would only
change the otherwise-unused/default training namespace value `steps` from 40
to 8; it would not create repeated K=40 merge rounds or scheduler-controlled
duration.

Both smoke and production renders currently share that same flawed argv. Their
Slurm submission argv differ only at the intended scheduler fields:

```diff
- -t 00:20:00 -p batch -q debug
+ -t 12:00:00 -p batch -q normal
```

Structural parity is therefore not the blocker. The blocker is that the shared
production entrypoint is intrinsically one-shot.

## Job 4975950 stop mechanism

The detailed retained diagnosis is
`reports/frontier/diagnose-e97-job-4975950-premature-completion-20260713.md`.
The exact mechanism was normal exhaustion of the one-shot compiled-MPICH
file-rank path:

1. The wrapper passed `--generations 1 --local-steps 40 --steps 40`.
2. The selected actual-multinode branch discarded `args.generations`, created
   one `RealAsyncFileRankConfig`, and called the one-shot runner once.
3. The runner hard-coded `generation = 0` and trained exactly
   `local_steps=40` before its single merge/publication and normal return.
4. The checkpoint consequently advanced from step 1,525,000 to 1,525,040.
5. Slurm recorded `COMPLETED 0:0` after `00:09:11`, far short of the requested
   `12:00:00`; there was no scheduler signal, requeue, exception, or timeout.

`--timeout-s 1200` was a quorum/transport timeout, not a runtime horizon.
`--walltime-remaining-s 1200` was a static finalization hint, not a scheduler
deadline. The retained canonical metadata records `signal: null` and
`requeue: null`, and the rendered batch body has no Slurm pre-timeout signal
directive. Although `train.py` already installs a real `SIGUSR1` shutdown
handler, the canonical async wrapper does not establish a scheduler signal or
a repeated outer loop that can reach that handler across merge rounds.

## Evidence and validation performed

- `build/e97-256/production/launch-inputs.json` records the exact production
  stop budget and Slurm argv.
- `build/e97-256/production/evidence/sacct-terminal.txt` records the allocation,
  batch, extern, and 2,048-rank step as `COMPLETED 0:0`; allocation elapsed time
  is `00:09:11` against a `12:00:00` limit.
- Static inspection followed the exact canonical wrapper, parser, selected
  actual-multinode branch, one-shot runner, checkpoint cadence inputs, and the
  existing `train.py` signal handler. No mocked signal or alternate run path
  was introduced.
- The existing focused promotion suite remains the appropriate baseline check,
  but its passing assertion encodes the defective one-generation stop budget;
  it cannot validate the proposed omission as unbounded.

## Required next decision

Completing the objective requires a scope expansion beyond deleting two argv
items: at minimum, the actual compiled-MPICH multinode path needs a repeated
K=40 generation loop driven to a scheduler-derived termination condition, and
the canonical Slurm render must request/deliver the existing production
shutdown signal. That control-flow change is expressly outside the user's final
minimal-change instruction. It should not be inferred or implemented without
new authority.
