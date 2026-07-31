# Quality pass: real async E97 training path

Task: `quality-pass-real`  
Date: 2026-07-06  
Reviewed checkout: `main` at `9bd95a7`  
Dependency implementation reviewed for intent: `ed36645` (`wire-production-async`)

## Verdict

**No-go for 1n/2n debug validation from the current checkout.** The checked-out
production wrapper still defaults to the synthetic/prototype path and the real
trainer files advertised by the dependency are not present in `HEAD`.

Readiness score: **0.35 / 1.00**.

Dimension scores:

- Wrapper/entrypoint correctness: **0.10**. Current `HEAD` points production to
  `scripts/frontier/async_diloco_e97_multinode.py`, not the real trainer
  (`scripts/frontier/async_diloco_e97_256n12h_launch.sbatch:43`), and that
  entrypoint imports the 2n8n debug harness (`scripts/frontier/async_diloco_e97_multinode.py:6`).
- Real token training vs synthetic deltas: **0.40**. The dependency branch's
  intended trainer uses `train.build_training_model`, `train.train_one_optimizer_step`,
  and `train.build_training_dataset` (`ed36645:ndm/async_diloco_real.py:276`,
  `ed36645:ndm/async_diloco_real.py:524`, `ed36645:ndm/async_diloco_real.py:581`),
  but the current production wrapper does not reach that path.
- Quorum/deferred semantics: **0.85**. The shared quorum primitive defers missed
  quorum without throwing (`ndm/async_diloco.py:1057`-`1078`), and the dependency
  tests assert local/global quorum deferral is nonfatal
  (`ed36645:tests/test_async_diloco_real_trainer.py:83`-`107`,
  `ed36645:tests/test_async_diloco_real_trainer.py:110`-`133`).
- Checkpoint/finalization safety: **0.70**. The checkpoint manager publishes only
  from the global merger role (`ndm/async_diloco.py:673`-`683`), writes run-local
  `latest.json` (`ndm/async_diloco.py:637`, `ndm/async_diloco.py:733`-`742`),
  and ignores non-authoritative cache manifests for resume
  (`ndm/async_diloco.py:640`-`671`, `ndm/async_diloco.py:781`-`809`). However,
  the dependency wrapper records recovery/export/finalization knobs but does not
  pass a checkpoint cadence or walltime budget into the real trainer
  (`ed36645:scripts/frontier/async_diloco_e97_256n12h_launch.sbatch:82`-`88`,
  `ed36645:scripts/frontier/async_diloco_e97_256n12h_launch.sbatch:133`-`153`);
  long-job finalization is therefore not actually exercised by that command.
- Ops safety/no production mutation: **0.80**. Current and dependency wrappers
  use run-local `async_run/latest.json` policy rather than mutating the external
  chain latest (`scripts/frontier/async_diloco_e97_256n12h_launch.sbatch:79`-`83`,
  `ed36645:scripts/frontier/async_diloco_e97_256n12h_launch.sbatch:79`-`90`).
  The dependency wrapper also rejects runtime probes and synthetic/protocol
  entrypoints for production (`ed36645:scripts/frontier/async_diloco_e97_256n12h_launch.sbatch:118`-`131`).
- Hidden DDP/barrier risk: **0.55**. The dependency trainer does not wrap models
  in `DistributedDataParallel`; it calls train helper APIs directly. But it also
  serializes all workers inside one Python process by iterating worker specs and
  node ids (`ed36645:ndm/async_diloco_real.py:291`-`326`,
  `ed36645:ndm/async_diloco_real.py:373`-`386`), so a 2-node Slurm debug run
  would not validate actual inter-node worker execution or transport.

## Findings

### F1 - Current checkout cannot launch the advertised real trainer

`HEAD` does not contain `scripts/frontier/e97_async_diloco_train.py`,
`ndm/async_diloco_real.py`, or `tests/test_async_diloco_real_trainer.py`.
The production wrapper in `HEAD` still has:

- `ASYNC_ENTRYPOINT=${ASYNC_ENTRYPOINT:-scripts/frontier/async_diloco_e97_multinode.py}`
  at `scripts/frontier/async_diloco_e97_256n12h_launch.sbatch:43`.
- `scripts/frontier/async_diloco_e97_multinode.py:6` imports
  `async_diloco_e97_2n8n_debug.main`.

This rejects the dependency log's claim as applied to the current checkout.
The path available from `HEAD` is the synthetic/prototype harness, not the real
token trainer.

### F2 - Dependency branch intent is closer, but it is not production-distributed

In `ed36645`, `scripts/frontier/e97_async_diloco_train.py` is a thin CLI wrapper
around `run_real_async_diloco` (`ed36645:scripts/frontier/e97_async_diloco_train.py:19`-`25`,
`ed36645:scripts/frontier/e97_async_diloco_train.py:95`-`109`).
That helper does real train.py-backed optimizer work:

- Builds a model via `train.build_training_model`
  (`ed36645:ndm/async_diloco_real.py:276`, `ed36645:ndm/async_diloco_real.py:510`).
- Loads the seed checkpoint into the global model
  (`ed36645:ndm/async_diloco_real.py:277`-`279`).
- Runs `train.train_one_optimizer_step`
  (`ed36645:ndm/async_diloco_real.py:523`-`535`).
- Uses real dataset batches unless `--synthetic-token-stream` is passed
  (`ed36645:ndm/async_diloco_real.py:571`-`587`).

However, the implementation loops over node ids and workers in-process
(`ed36645:ndm/async_diloco_real.py:291`-`326`,
`ed36645:ndm/async_diloco_real.py:373`-`386`). The dependency wrapper computes
`ASYNC_WORKER_COUNT=ASYNC_NODE_COUNT * ASYNC_WORKER_COUNT_PER_NODE`
(`ed36645:scripts/frontier/async_diloco_e97_256n12h_launch.sbatch:51`-`55`) and
passes that as a single CLI value
(`ed36645:scripts/frontier/async_diloco_e97_256n12h_launch.sbatch:140`-`143`),
but no rank-local partitioning, multi-process launch, or inter-node transport is
connected. This can exercise real token math, but not a real async distributed
training path.

### F3 - Missed quorum is nonfatal/deferred

The core quorum merge returns the base state and `quorum_status="deferred"` when
accepted updates are below threshold (`ndm/async_diloco.py:1057`-`1078`) instead
of raising or forcing a hard barrier. The dependency real trainer propagates this
at both levels: local node misses yield `node_update is None`
(`ed36645:ndm/async_diloco_real.py:417`-`433`), and global misses only publish
when `merge_result.advanced` is true
(`ed36645:ndm/async_diloco_real.py:476`-`480`). Tests cover local and global
defer cases without process failure
(`ed36645:tests/test_async_diloco_real_trainer.py:83`-`107`,
`ed36645:tests/test_async_diloco_real_trainer.py:110`-`133`).

### F4 - Checkpoint latest mutation is run-local, but long-job cadence is not wired

The shared checkpoint manager only allows the `GLOBAL_MERGER_ROLE` to publish
authoritative generations (`ndm/async_diloco.py:673`-`683`), writes the latest
pointer at `run_dir/latest.json` (`ndm/async_diloco.py:637`,
`ndm/async_diloco.py:733`-`742`), and selects resume sources from finalized
global manifests (`ndm/async_diloco.py:781`-`809`). The wrappers describe the
external production latest as a guard path, not an update target
(`scripts/frontier/async_diloco_e97_256n12h_launch.sbatch:79`-`83`,
`ed36645:scripts/frontier/async_diloco_e97_256n12h_launch.sbatch:79`-`90`).

The remaining hazard is that the dependency command does not pass the recorded
cadence/finalization variables into `RealAsyncDiLoCoConfig`; the config default
sets recovery/export cadence fields to `None`
(`ed36645:ndm/async_diloco_real.py:74`-`81`), and
`run_real_async_diloco` constructs the manager from that default
(`ed36645:ndm/async_diloco_real.py:281`-`286`). `publish_global_generation`
supports walltime finalization inputs (`ndm/async_diloco.py:673`-`679`,
`ndm/async_diloco.py:713`-`724`), but the real trainer call site does not pass
them. That is not safe enough to rely on for 12-hour production jobs.

### F5 - No hidden DDP use found in the reviewed real helper, but no distributed execution either

The dependency real helper calls `train.build_training_dataset(args, rank=..., dist_enabled=True)`
(`ed36645:ndm/async_diloco_real.py:581`), which changes data stream seeding
(`train.py:1355`-`1357`). It does not initialize `torch.distributed` or wrap the
model in DDP. DDP wrapping in `train.py` is in the full training main path
(`train.py:2234`-`2239`, `train.py:2741`-`2748`, `train.py:2838`-`2845`), not in
the reviewed helper calls. The operational issue is therefore not hidden DDP;
it is that the "multi-node" execution is simulated by serial helper calls.

## Validation checklist

- Findings documented with file/line references: **yes**.
- Real token training vs synthetic deltas: **rejected for current checkout**;
  **partially confirmed for dependency branch intent** when
  `--synthetic-token-stream` is not used.
- Missed quorum nonfatal/deferred: **confirmed**.
- Checkpoint/finalization safe for long jobs: **partially rejected** because
  run-local latest semantics are safe, but long-job cadence/finalization inputs
  are not wired into the dependency real trainer command.
- Wrapper command matches intended real trainer: **rejected for current checkout**.
  The dependency branch wrapper matches the real CLI, but that state is not in
  `HEAD`.
- Recommendation for 1n/2n debug validation: **no-go** until the real trainer
  files and wrapper default are present in the validation checkout, and until the
  debug ladder is scoped as a serial real-token smoke rather than evidence of
  multi-node async transport.
- Slurm production job submitted: **no**.

