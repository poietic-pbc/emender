# Fix Refreshed E97 Async 256n Wrapper Python Env

Task: `fix-refreshed-async`

Decision: **fixed and validated**. This task did not submit the 256-node
production retry.

## Root Cause

Production job `4946500` started the refreshed E97 async 256n12h launch on
2026-07-06 and failed before the async entrypoint ran.

Stderr contained only the launcher failure:

```text
/var/spool/slurmd/job4946500/slurm_script: line 187: exec: python: not found
```

Stdout showed the wrapper had already recorded the intended refreshed seed and
production parameters:

- `e97_checkpoint=/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt`
- `async_node_count=256`
- `async_local_quorum=8`
- `async_global_quorum=240`
- `batch_size=4`
- `chunk_size=2048`
- `diloco_k=40`
- `requested_walltime=12:00:00`
- `requested_node_hours=3072.0`

The failed wrapper built its final command as `python -u
scripts/frontier/async_diloco_e97_multinode.py` and then `exec`ed that command
without first activating the OLCF runtime conda environment. On the compute
shell for job `4946500`, bare `python` was not on `PATH`, matching the earlier
4n debug failure mode documented in `validate-refreshed-seed-4n20m`.

## Patch

Patched `scripts/frontier/async_diloco_e97_256n12h_launch.sbatch` to follow the
validated debug wrapper pattern before any Python calls:

- Default `ENV_PREFIX` / `EMENDER_CONDA_ENV` to
  `/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312`,
  with the same persistent-prefix fallback used by the 2n/8n debug wrapper.
- Load Frontier modules via `frontier_runtime_env.sh`.
- Run `frontier_activate_emender_conda_env` before command recording or Python
  execution.
- Resolve `PYTHON_BIN=$(command -v python)` only after activation.
- Use `"$PYTHON_BIN" -u "$ASYNC_ENTRYPOINT"` in the recorded launch command and
  in the final `CMD`/`exec` path.
- Add `ASYNC_DILOCO_RUNTIME_PROBE_ONLY=1`, a non-training probe mode that uses
  the same production wrapper on compute nodes to confirm activated Python and
  entrypoint resolution without requiring `ASYNC_DILOCO_HUMAN_APPROVED=1`.

The production seed and scale parameters for the downstream retry remain
unchanged:

- `E97_CHECKPOINT=/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt`
- Global quorum `240`, local quorum `8`
- `DILOCO_K=40`, `batch_size=4`, `chunk_size=2048`
- Target shape `256` nodes x `12:00:00`

## Validation

Static wrapper validation:

```bash
bash -n scripts/frontier/async_diloco_e97_256n12h_launch.sbatch scripts/frontier/async_diloco_e97_2n8n_debug.sbatch
```

Result: passed.

Focused regression test:

```bash
/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python -m pytest tests/test_async_diloco_e97_2n8n_debug_runner.py -q
```

Result: `3 passed`.

The focused test now asserts that the 256n launch wrapper activates the runtime,
resolves `PYTHON_BIN`, uses `"$PYTHON_BIN" -u "$ASYNC_ENTRYPOINT"`, and no
longer contains the indented final-command form `python -u "$ASYNC_ENTRYPOINT"`.

Compute-node runtime probe:

```bash
sbatch -N 1 -p batch -q debug -t 00:05:00 -J async-e97-pyprobe \
  --export=ALL,WG_TASK_ID=fix-refreshed-async,TASK_ID=fix-refreshed-async,ASYNC_DILOCO_RUNTIME_PROBE_ONLY=1,OUTPUT_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/fix_refreshed_async_runtime_probe,SCALEOUT_VARIANT=fix_refreshed_async_runtime_probe,ASYNC_NODE_COUNT=1,ASYNC_GLOBAL_QUORUM=1,ASYNC_LOCAL_QUORUM=1,REQUESTED_WALLTIME=00:05:00,REQUESTED_NODE_HOURS=0.083333 \
  scripts/frontier/async_diloco_e97_256n12h_launch.sbatch
```

Result: Slurm job `4946963` completed successfully.

Accounting:

```text
4946963|async-e97-pyprobe|COMPLETED|0:0|00:00:17|1|2026-07-06T09:54:30|2026-07-06T09:54:32|2026-07-06T09:54:49
```

Probe artifacts:

- Stdout: `logs/frontier/async_diloco_e97/async-e97-pyprobe-4946963.out`
- Stderr: `logs/frontier/async_diloco_e97/async-e97-pyprobe-4946963.err`
- Env: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/fix_refreshed_async_runtime_probe/fix_refreshed_async_runtime_probe/20260706/4946963-20260706T135436Z/artifacts/env.txt`
- Command: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/fix_refreshed_async_runtime_probe/fix_refreshed_async_runtime_probe/20260706/4946963-20260706T135436Z/artifacts/command.txt`
- Metrics: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/fix_refreshed_async_runtime_probe/fix_refreshed_async_runtime_probe/20260706/4946963-20260706T135436Z/artifacts/async_diloco_e97_256n_metrics.json`

The compute-node env file recorded:

```text
python_bin=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python
python_executable=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python
python_version=3.12.13
torch.__version__=2.10.0+rocm7.1
torch.version.hip=7.1.25424
triton.__version__=3.6.0
runtime_probe_entrypoint=scripts/frontier/async_diloco_e97_multinode.py
```

The command artifact recorded the fixed command prefix:

```text
"/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python" -u "scripts/frontier/async_diloco_e97_multinode.py"
```

The probe metrics recorded:

```json
{
  "conclusion": "runtime-probe-pass",
  "entrypoint": "scripts/frontier/async_diloco_e97_multinode.py",
  "python_executable": "/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python",
  "python_prefix": "/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312",
  "python_version": "3.12.13",
  "slurm_job_id": "4946963",
  "slurm_job_num_nodes": "1"
}
```

No `sbatch -N 256` production retry was submitted by this task. The only Slurm
submission was the one-node debug-QOS runtime probe above.
