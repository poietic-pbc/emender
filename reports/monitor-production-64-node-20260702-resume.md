# Monitor production 64-node E97 B4/K80 chain: resumed pending-state pass

Task: `monitor-production-64-node`
Timestamp: 2026-07-02T22:36:15-04:00 / 2026-07-03T02:36:15+00:00

## Summary

This resumed monitor pass found that the current active scope jobs are still
queued, not running or terminal:

- `4931004`: original 64-node x 2h production job, still `PENDING` on
  `Priority`.
- `4932059`: independent 96-node x 6h continuation job, still `PENDING` on
  `Priority`.

The earlier jobs `4931005`, `4931947`, and `4932047` are intentionally
cancelled scope changes and are not training failures. Current user scope is to
monitor `4931004` and independent `4932059`. Serialization for `4932059` is via
the required `RESUME_CHECKPOINT` pointing at the production chain `latest.pt`
symlink, not via a Slurm dependency.

No training stdout/stderr files exist yet for `4931004` or `4932059`. The
production chain directory and `latest.pt` symlink also do not exist yet, while
the original seed checkpoint still exists. Therefore loss trend, token
throughput, DiLoCo merge timing, final checkpoint save, and chain advancement
cannot be evaluated in this pass.

There is no unattended hang: both active jobs are pending with `Elapsed=00:00:00`.

## Scheduler state

Observed commands:

```bash
sacct -j 4931004,4932059,4931005,4931947,4932047 --format=JobID,JobName%40,State,ExitCode,Elapsed,NNodes,Submit,Start,End -P
squeue -j 4931004,4932059,4931005,4931947,4932047 -o '%i|%T|%M|%D|%R|%S|%j'
scontrol show job 4931004
scontrol show job 4932059
```

Results:

| Job | Role | State | ExitCode | Elapsed | Nodes | Submit | Start estimate | End estimate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `4931004` | original production chain job | `PENDING` / `Priority` | `0:0` | `00:00:00` | 64 | `2026-07-02T08:30:50` | `2026-07-03T20:36:00` | `2026-07-03T22:36:00` |
| `4932059` | independent 96n continuation | `PENDING` / `Priority` | `0:0` | `00:00:00` | 96 | `2026-07-02T12:56:18` | `2026-07-04T08:34:00` | `2026-07-04T14:34:00` |
| `4931005` | cancelled earlier chain job | `CANCELLED by 19032` | `0:0` | `00:00:00` | 64 | `2026-07-02T08:30:50` | `None` | `2026-07-02T09:28:55` |
| `4931947` | cancelled replacement | `CANCELLED by 19032` | `0:0` | `00:00:00` | 64 | `2026-07-02T12:13:10` | `None` | `2026-07-02T12:49:27` |
| `4932047` | cancelled dependent 96n replacement | `CANCELLED by 19032` | `0:0` | `00:00:00` | 96 | `2026-07-02T12:49:29` | `None` | `2026-07-02T12:56:16` |

Current `squeue` rows:

```text
4931004|PENDING|0:00|64|(Priority)|2026-07-03T20:36:00|e97-b4-k80-64n2h
4932059|PENDING|0:00|96|(Priority)|2026-07-04T08:34:00|e97-b4-k80-96n6h
```

## Job details to preserve

`4931004`:

- Command:
  `/lustre/orion/bif148/scratch/erikgarrison/emender-diloco-scalar-free/scripts/frontier/e97_1p3b_b4_k80_64n_chain2h.sbatch`
- Workdir:
  `/lustre/orion/bif148/scratch/erikgarrison/emender-diloco-scalar-free`
- Stdout:
  `/lustre/orion/bif148/scratch/erikgarrison/emender-diloco-scalar-free/logs/frontier/scaleout/e97-b4-k80-64n2h-4931004.out`
- Stderr:
  `/lustre/orion/bif148/scratch/erikgarrison/emender-diloco-scalar-free/logs/frontier/scaleout/e97-b4-k80-64n2h-4931004.err`
- Submit line includes:
  `CHAIN_INDEX=1,WG_TASK_ID=chain-64n-b4-k80-2h-1`
- Time limit: `02:00:00`
- Network: `disable_rdzv_get`

`4932059`:

- Command:
  `/lustre/orion/bif148/scratch/erikgarrison/emender-diloco-scalar-free/scripts/frontier/e97_1p3b_b4_k80_64n_chain2h.sbatch`
- Workdir:
  `/lustre/orion/bif148/scratch/erikgarrison/emender-diloco-scalar-free`
- Stdout:
  `/lustre/orion/bif148/scratch/erikgarrison/emender-diloco-scalar-free/logs/frontier/scaleout/e97-b4-k80-96n6h-4932059.out`
- Stderr:
  `/lustre/orion/bif148/scratch/erikgarrison/emender-diloco-scalar-free/logs/frontier/scaleout/e97-b4-k80-96n6h-4932059.err`
- Submit line includes:
  `CHAIN_INDEX=2,WG_TASK_ID=chain-96n-b4-k80-6h-eligible`
- Submit line sets:
  `SCALEOUT_VARIANT=E97_1.3B_step489920_b4_k80_96n_hier_g4_bucket64m_avg_6h`
- Submit line requires:
  `RESUME_CHECKPOINT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt`
- Submit line records:
  `HUMAN_APPROVAL_RECORD=96-node 6h E97 B4/K80 continuation submitted eligible immediately; serialization via production latest.pt symlink, no Slurm dependency. Requires latest.pt at startup; updates same chain symlink atomically on checkpoint.`
- Time limit: `06:00:00`
- Network: `disable_rdzv_get`

## Log files

The expected logs do not exist yet:

```text
ls: cannot access '/lustre/orion/bif148/scratch/erikgarrison/emender-diloco-scalar-free/logs/frontier/scaleout/e97-b4-k80-64n2h-4931004.out': No such file or directory
ls: cannot access '/lustre/orion/bif148/scratch/erikgarrison/emender-diloco-scalar-free/logs/frontier/scaleout/e97-b4-k80-64n2h-4931004.err': No such file or directory
ls: cannot access '/lustre/orion/bif148/scratch/erikgarrison/emender-diloco-scalar-free/logs/frontier/scaleout/e97-b4-k80-96n6h-4932059.out': No such file or directory
ls: cannot access '/lustre/orion/bif148/scratch/erikgarrison/emender-diloco-scalar-free/logs/frontier/scaleout/e97-b4-k80-96n6h-4932059.err': No such file or directory
```

## Checkpoint pointer evidence

Production chain latest path:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt
```

Current state: absent. `readlink -f` returned non-zero with no target, and
listing the chain directory returned:

```text
ls: cannot access '/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg': No such file or directory
```

Seed checkpoint remains present:

```text
-rw------- 1 erikgarrison bif148 7719679569 Jun 30 16:04 /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260630/E97_1.3B_step483000_b4_k80_64n_hier_g4_bucket64m_avg_24h/4911454-20260630T164035Z/train/emender_E97_1.3B_20260630_124257/checkpoint_step_489920_loss_2.4894.pt
```

## Validation status

- `4931004` terminal state: not yet satisfied; job is still pending.
- `4931004` logs inspected: no log files exist yet because the job has not
  started.
- `4931004` final/latest checkpoint advancement: not yet applicable; no
  production chain `latest.pt` exists yet.
- `4932059` startup checkpoint: not yet satisfied; job has not started. Its
  submitted environment requires the production `latest.pt` symlink, which is
  currently absent.
- `4932059` terminal state/log inspection: not yet satisfied; job is still
  pending.
- Token throughput, final/last logged loss trend, DiLoCo merge timing, and
  checkpoint paths: not yet available because neither active job has started.
- Hang handling: satisfied for this pass; no running or wedged job was found.

## Next monitor action

Resume after `4931004` has likely reached terminal state, currently estimated
after `2026-07-03T22:36:00-04:00`. At that point:

1. Inspect `sacct` and `scontrol` for `4931004` terminal state and exit code.
2. Tail `4931004` stdout/stderr and extract token throughput, loss trend,
   DiLoCo merge timing, final checkpoint path, and any NCCL/RCCL errors.
3. Verify whether production `latest.pt` and its manifest were created and
   atomically advanced.
4. Before `4932059` starts, confirm that its required `RESUME_CHECKPOINT`
   target exists and points to the latest valid `4931004` checkpoint if
   `4931004` advanced the chain.
5. Continue monitoring `4932059` to terminal state, with the same log,
   throughput, loss, merge, checkpoint, and pointer checks.
