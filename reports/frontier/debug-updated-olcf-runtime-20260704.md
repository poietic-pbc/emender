# Updated OLCF Runtime Debug Validation

Task: `debug-updated-olcf-runtime`  
Date: 2026-07-04  
Candidate prefix: `/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312`

## Decision

The OLCF-aligned runtime candidate is suitable for the next larger scaleout
debug tests for E97 import/compile/rendezvous/kernel correctness, with one
launcher caveat: the task-specific E97 debug wrapper allowed the distributed
training loop to run until the debug walltime. That produced a Slurm `TIMEOUT`
terminal state even though the intended resume, compile, distributed init, and
finite-loss training smoke had already succeeded.

Use the candidate runtime for the next scaleout decision path, but do not reuse
the exact `TRAIN_MINUTES=6` distributed-debug wrapper unchanged. Either add a
hard step cap / shorter external timeout, or use the production launcher's
normal stop/checkpoint behavior in an isolated debug output root.

## Jobs

| Job | Purpose | Nodes | Slurm state | Elapsed | Actual node-hours | Notes |
| --- | --- | ---: | --- | ---: | ---: | --- |
| `4939804` | E97 2-node active-`latest.pt` resume smoke | 2 | `TIMEOUT` | `00:30:29` | 1.0161 | Passed resume, compile, distributed init, and finite-loss steps before walltime |
| `4939880` | Initial GDN2 delegated smoke attempt | 1 | `CANCELLED by 19032` | `00:02:26` | 0.0406 | Cancelled after stdout showed delegated wrapper reverted training to older Python/Triton/HIP stack |
| `4939888` | Direct GDN2/FLA candidate preflight | 1 | `COMPLETED` | `00:00:35` | 0.0097 | Candidate prefix GDN2 bf16 fwd/bwd passed |

Total actual node-hours spent: approximately `1.0664`.

Accounting command:

```text
sacct -j 4939804,4939880,4939888 -X --format=JobID,JobName%24,State,ExitCode,Elapsed,AllocNodes,AllocTRES%80 -P
```

## Submitted Wrappers

- `scripts/frontier/e97_updated_olcf_runtime_debug.sbatch`
- `scripts/frontier/gdn2_updated_olcf_runtime_debug.sbatch`
- `scripts/frontier/gdn2_updated_olcf_runtime_preflight.sbatch`

The first GDN2 wrapper is retained as a documented negative/invalid attempt:
it delegates to `debug_smoke_one_node.slurm`, whose conda handling reset the
actual training command to the older runtime. The direct preflight wrapper is
the valid GDN2 evidence.

## E97 Evidence

Job `4939804` ran on `frontier[03466,03521]` with `SLURM_NTASKS=16` and
`SLURM_JOB_QOS=debug`.

Primary artifacts:

- stdout: `logs/frontier/debug/e97-olcf-debug-4939804.out`
- stderr: `logs/frontier/debug/e97-olcf-debug-4939804.err`
- run root: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260704/E97_1.3B_active_latest_olcf_runtime_debug/4939804-20260704T151442Z`
- env manifest: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260704/E97_1.3B_active_latest_olcf_runtime_debug/4939804-20260704T151442Z/artifacts/env.txt`
- train log: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260704/E97_1.3B_active_latest_olcf_runtime_debug/4939804-20260704T151442Z/logs/train.log`

The job resolved the active production chain symlink at startup:

```text
production_latest=/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt
resolved_production_latest=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260630/E97_1.3B_step483000_b4_k80_64n_hier_g4_bucket64m_avg_24h/4911454-20260630T164035Z/train/emender_E97_1.3B_20260630_124257/checkpoint_step_489920_loss_2.4894.pt
production_latest_before='/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt'|7719679569|1782849877
```

It then deliberately cleared production chain-update variables:

```text
chain_latest_path_after_guard=''
CHAIN_LATEST_PATH=
CHAIN_MANIFEST_PATH=
CHAIN_UPDATE_ON_FAILURE=0
```

Candidate runtime and RCCL plugin evidence were logged in stdout and env:

```text
rccl_net_plugin_status=/sw/frontier/rccl-plugins/aws-ofi-nccl/1.19.2/rocm/7.1.1/lib/librccl-net.so
loaded_modules=...:rocm/7.1.1:craype-accel-amd-gfx90a:rccl-net-plugin/1.0
"prefix": "/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312"
"torch_version": "2.10.0+rocm7.1"
"torch_hip": "7.1.25424"
```

The E97 model path compiled and ran through the fused Triton path:

```text
[fused-guard] rank 0/16: level=E97 bf16 use_triton=1 -> fused split-edit Triton kernel, NO eager fallback
[e97-runtime] backend=hip path=e88-sequential-split-edit-triton use_triton=True use_chunked_e97=False e97_chunk_size=32 linear_state=False raw_write=False use_split_edit=True log_decay=True
```

Checkpoint loading succeeded on all ranks:

```text
Resuming from /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260630/E97_1.3B_step483000_b4_k80_64n_hier_g4_bucket64m_avg_24h/4911454-20260630T164035Z/train/emender_E97_1.3B_20260630_124257/checkpoint_step_489920_loss_2.4894.pt
Resumed at step 489920
```

Representative finite loss lines:

```text
step 489921 | loss 2.3118 | lr 1.01e-03 | grad 1.41 | tok/s 2938 | global_tok/s 47008 | elapsed_h 0.000 | time 2026-07-04T15:15:54+00:00
step 491062 | loss 2.7661 | lr 1.01e-03 | grad 1.86 | tok/s 3086 | global_tok/s 49380 | elapsed_h 0.485 | time 2026-07-04T15:44:59+00:00
```

The terminal state was timeout at the debug allocation boundary:

```text
[2026-07-04T11:44:59.919] error: *** STEP 4939804.0 ON frontier03466 CANCELLED AT 2026-07-04T11:44:59 DUE TO TIME LIMIT ***
```

No debug-run checkpoint or `latest.pt` was written under the E97 debug run root:

```text
find .../4939804-20260704T151442Z -name '*.pt' -o -name 'latest.pt'
# no output
```

## Production Symlink Guard

The production `latest.pt` metadata is unchanged after the E97 job:

```text
'/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt'|7719679569|1782849877
```

The resolved target after the job remains:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260630/E97_1.3B_step483000_b4_k80_64n_hier_g4_bucket64m_avg_24h/4911454-20260630T164035Z/train/emender_E97_1.3B_20260630_124257/checkpoint_step_489920_loss_2.4894.pt
```

No production chain pointer was modified.

## GDN2 Evidence

The valid GDN2 result is job `4939888`, not the cancelled delegated attempt
`4939880`.

Primary artifacts:

- stdout: `logs/frontier/debug/gdn2-olcf-pre-4939888.out`
- stderr: `logs/frontier/debug/gdn2-olcf-pre-4939888.err`
- run root: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260704/gdn2_olcf_runtime_preflight/4939888-20260704T155011Z`
- manifest: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260704/gdn2_olcf_runtime_preflight/4939888-20260704T155011Z/artifacts/manifest.json`
- env file: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260704/gdn2_olcf_runtime_preflight/4939888-20260704T155011Z/artifacts/env.txt`
- preflight log: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260704/gdn2_olcf_runtime_preflight/4939888-20260704T155011Z/logs/gdn2_preflight.log`

Manifest highlights:

```text
"job_id": "4939888"
"exit_status": 0
"env_prefix": "/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312"
"rccl_net_plugin_status": "/sw/frontier/rccl-plugins/aws-ofi-nccl/1.19.2/rocm/7.1.1/lib/librccl-net.so"
```

Candidate runtime and finite GDN2/FLA fwd/bwd evidence:

```text
"device_name": "AMD Instinct MI250X"
"dtype": "torch.bfloat16"
"finite_input_grad": true
"finite_loss": true
"finite_output": true
"finite_param_grads": true
"loss": 0.009861934930086136
"ok": true
"fla_version": "0.5.1"
"missing_required_symbols": []
"torch_version": "2.10.0+rocm7.1"
"torch_version_hip": "7.1.25424"
```

## Validation Checklist

- 1-2 node debug jobs submitted and monitored to terminal state: yes.
  `4939804` terminal `TIMEOUT`, `4939880` terminal `CANCELLED`, `4939888`
  terminal `COMPLETED`.
- E97 smoke loaded from active `latest.pt` and logged resolved target: yes.
- GDN2 compatibility tested: yes, direct candidate-prefix GDN2/FLA preflight
  job `4939888` completed.
- `librccl-net.so` resolves under `rccl-net-plugin/1.0`: yes, both valid
  E97 and GDN2 logs resolve
  `/sw/frontier/rccl-plugins/aws-ofi-nccl/1.19.2/rocm/7.1.1/lib/librccl-net.so`.
- Loss finite for completed training/preflight steps: yes. E97 finite losses
  from `489921` through `491062`; GDN2 bf16 fwd/bwd loss finite.
- No production chain symlink modified: yes. Metadata and target unchanged.
- Isolated debug output directories only: yes. Outputs went under
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260704/...`.

## Recommendation

Proceed with the OLCF-aligned runtime candidate for the next scaleout decision
task. The candidate stack passed:

- PyTorch `2.10.0+rocm7.1` / HIP `7.1.25424` import inside Slurm jobs.
- `rccl-net-plugin/1.0` plugin resolution to `librccl-net.so`.
- 2-node E97 distributed init and active-production-checkpoint resume.
- E97 fused split-edit Triton compile/run with finite loss.
- GDN2/FLA import and bf16 fwd/bwd with finite output, loss, input grad, and
  parameter grads.

Do not interpret `4939804` as an extended-training pass: it is a debug smoke
that exceeded its walltime because the wrapper stop condition did not end the
distributed loop early. For the next larger run, use a launcher with a proven
bounded stop condition and the same explicit candidate env / plugin loading.
