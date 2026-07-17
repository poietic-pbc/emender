# Resilient E97 two-node live gate: pre-submit blocker report

Task: `run-resilient-e97-2`  
UTC preflight: `2026-07-17T08:20:08Z`  
Fetched authoritative payload: `f88498852ff1426fc21677f222f173cb7412c220`

## Outcome

No live Slurm allocation was submitted. The required two-node failure/restart
gate cannot be truthfully executed by payload `f884988`: the retained launch
surface has no failure-injection interface, and a supervisor restart starts a
trainer again from generation zero without its current model or ScheduleFree
inner-optimizer state. Submitting despite these preflight findings would spend
the sole allowed unchanged payload attempt on a run that cannot satisfy the
mandatory restart/rejoin proof.

No 4+ node, normal-QoS, or production job was submitted or mutated. Slurm job
number `5016933` printed below is the scheduler's `--test-only` estimate, not a
submitted job; `squeue -j 5016933` returned `Invalid job id specified`.

## Authoritative-source and local validation gate

`wire-split-role` was `Done`. After `git fetch origin`, both fetched
`origin/main` and this worktree were exactly
`f88498852ff1426fc21677f222f173cb7412c220`, with a clean tracked worktree.

The pinned ROCm/Torch environment ran the focused runtime, split-role,
transport, quorum, launcher, topology, Frontier plumbing, checkpoint, and
walltime suites:

```text
/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python -m pytest -q \
  tests/test_resilient_e97_runtime.py \
  tests/test_resilient_e97_split_roles.py \
  tests/test_resilient_node_transport.py \
  tests/test_resilient_node_quorum.py \
  tests/test_resilient_e97_true_2n_launcher.py \
  tests/test_resilient_e97_topology.py \
  tests/test_frontier_runtime_plumbing.py \
  tests/test_checkpoint_finalization.py \
  tests/test_walltime_final_checkpoint.py
```

Result: `56 passed in 74.36s`.

The 7,719,679,924-byte generation-zero seed was independently hashed:

```text
1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9  /lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260709_084606_step_1525000/checkpoint_step_1525000_loss_2.4378.pt
```

`bash -n scripts/frontier/resilient_e97_true_2n.sbatch`, Python compileall,
and `git diff --check` also passed.

## Concrete scheduler attempt and exact rendered command

The concrete scheduler preflight used exactly debug QoS, two nodes, and two
hours. It deliberately used a unique run/payload identity and `--test-only`,
because the runtime blockers below were already reproduced:

```text
sbatch --test-only -A bif148 -q debug -N 2 -t 02:00:00 \
  --export=ALL,RUN_DIR=/lustre/orion/bif148/proj-shared/emender/runs/run-resilient-e97-2-preflight-20260717T082008Z-f884988,RESILIENT_E97_RUN_ID=run-resilient-e97-2-preflight-20260717T082008Z-f884988,RESILIENT_E97_SOURCE_ID=step-1525000-sha256-1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9,RESILIENT_E97_PAYLOAD_ID=f884988-20260717T082008Z,RESILIENT_E97_CODE_ID=f88498852ff1426fc21677f222f173cb7412c220,RESILIENT_E97_SEED=/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260709_084606_step_1525000/checkpoint_step_1525000_loss_2.4378.pt,RESILIENT_E97_TRAIN_ARGS_JSON=/lustre/orion/bif148/scratch/erikgarrison/emender/configs/frontier/e97_async_256_job4962400_golden.json,RESILIENT_E97_DATA=/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt,RESILIENT_E97_COORDINATOR_HOST=SLURM_ASSIGNED_NODE0,RESILIENT_E97_GENERATIONS=8,RESILIENT_E97_LAUNCH_MODE=node-local \
  scripts/frontier/resilient_e97_true_2n.sbatch
```

Scheduler result:

```text
sbatch: Job 5016933 to start at 2026-07-18T03:40:10 a using 112 processors on nodes frontier[01919,02045] in partition batch
```

This estimate also shows that a real submission would queue for roughly 19
hours; queue latency itself is not treated as failure.

## Reproduced blockers

1. `resilient_e97_allocation_supervisor.py` restarts a failed child by invoking
   the unchanged role command. That command has `--initial-generation 0`; the
   trainer reloads only the pinned generation-zero seed. It does not persist or
   restore the failed trainer's current model and inner optimizer at each
   committed generation, nor replay a complete fenced aggregate chain.
2. Managers call `prune_aggregates(keep_generations=2)`. Consequently even a
   hypothetical seed replay cannot reliably reconstruct all generations by the
   required post-injection point, and optimizer state cannot be reconstructed
   from outer aggregates in any case.
3. The launcher/supervisor exposes no documented trainer-failure or distinct
   manager/node-step injection controls. Existing
   `resilient_e97_node_step_supervisor.py` has a timer injection option, but the
   retained true two-node sbatch does not invoke that program.
4. `RESILIENT_E97_COORDINATOR_HOST` is mandatory before submission, although
   Slurm assigns the physical node names only after allocation. The literal
   `SLURM_ASSIGNED_NODE0` in the test-only rendering marks this unresolved
   value; the launcher does not derive node zero inside the allocation.
5. The proposed `RESILIENT_E97_TRAIN_ARGS_JSON` is the retained golden launch
   record, not a validated flat override object for
   `default_tiny_e97_train_args(**overrides)`. No approved split-role arguments
   artifact is present, so the exact model/data/optimizer gate cannot be
   established by this command.
6. The sbatch has no way to pass `--resume-handoff` or
   `--initial-generation` for the mandatory fresh-allocation continuation.

These are payload defects, not queue conditions. A changed payload must add a
focused regression proving generation-gated injection, exact trainer and
manager recovery/catch-up including inner and outer state, allocation-time
coordinator discovery, an approved flat training-arguments artifact, and an
immutable-handoff restart launch. Only after that payload is committed,
pushed, fetched, and locally validated should the one live attempt be made.

## Production-path identity disclosure requested before submission

At the final preflight checkpoint, fetched authoritative `origin/main`, local
`HEAD`, and the clean tracked worktree were all `f884988` (the evidence-only
report was subsequently committed as `88e1b6e`). The proposed live gate is
**not** identical to the retained 256-node production path with only QoS,
walltime, and node count changed:

- It intends to use the same pinned step-1525000 checkpoint and same CommaPile
  data path, with no synthetic-data, synthetic-model, or `--control` flag.
- Exact model/optimizer equivalence is not yet established because the split
  launcher needs a flat `default_tiny_e97_train_args` override JSON, whereas
  the available golden artifact is a nested production launch record.
- The split-role gate uses two model-free CPU managers and sixteen independent
  GPU trainers with dynamic local 6/8 quorum. The retained production job used
  2,048 worker ranks across 256 nodes and a global quorum of 2,048.
- Its hot path is the new bounded node-local mailbox plus persistent non-MPI
  manager network stream. The retained production path used the compiled Cray
  MPICH helper collective-reduce transport.
- Its default launcher mode is `node-local`: one durable Slurm step per node
  supervises one manager and eight trainer children. The alternative
  `independent-step` mode creates eighteen separate manager/trainer steps. This
  differs from the retained production launcher.
- No live fault-injection controls are exposed by the true two-node launcher.
  The unused legacy node-step supervisor has a timer-based kill option, but it
  is not on this launch path.
- The only scheduler-only flag actually exercised was `sbatch --test-only`.
  The rendered command also used literal placeholder
  `RESILIENT_E97_COORDINATOR_HOST=SLURM_ASSIGNED_NODE0`, because allocation-time
  node-zero discovery is missing. Neither item is suitable for the live job.

Therefore no claim of identical production-path execution was made and no live
submission followed this disclosure.
