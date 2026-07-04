# Frontier OLCF PyTorch/ROCm/Triton/RCCL stack alignment

Date: 2026-07-04

Task: `study-olcf-pytorch`

## Executive recommendation

Load `rccl-net-plugin/1.0` for the next Frontier RCCL/rendezvous diagnostics and for future large E97/GDN2 training jobs. The recent launch wrappers set `NCCL_NET_PLUGIN=librccl-net.so`, but both the successful 256-node job `4908935` and failed 256-node job `4936017` recorded `rccl_net_plugin_status=not-found`; the plugin library directory was not in `LD_LIBRARY_PATH`. This means the jobs were hand-exporting only part of the OLCF RCCL guidance and were not actually loading the official AWS-OFI RCCL net plugin.

Do not make an unvalidated PyTorch/Triton/ROCm upgrade immediately before another production-scale run. The current runtime is old and misaligned with OLCF's current recommendation, but E97/GDN2 depends on custom Triton kernels, FLA kernels, checkpoint resume behavior, and large cold compile caches. Treat a full stack upgrade as a separate validation track. The low-risk next step is to align RCCL networking first by loading `rccl-net-plugin/1.0`, then run a bounded RCCL/rendezvous diagnostic and small-scale training smoke before 256-node work.

Most likely cause of job `4936017`: a c10d/TCPStore startup/rendezvous failure at 256 nodes, with 2016 of 2048 clients joining before the 601-second timeout. Triton 3.2 is unlikely to be the direct cause because the failure happened during `dist.init_process_group`, before model Triton kernels matter, and the same warning appears in completed jobs. Missing `rccl-net-plugin/1.0` is a real stack defect and should be fixed, but it is more directly relevant to RCCL/NCCL collectives and scale stability after process-group setup than to TCPStore's initial client count. The plausible model is launch/startup jitter or rank/node failure during process-group formation, with the incomplete RCCL network setup as a risk amplifier rather than a sole explanation.

No production Slurm jobs were submitted for this study.

## Current job environment

Recent jobs used the CPE 26.03 + ROCm 7.1.1 module stack, but the Python wheel reports ROCm 6.4:

| Item | Observed value | Evidence |
| --- | --- | --- |
| Python | `/autofs/nccs-svm1_sw/frontier/miniforge3/23.11.0-0/bin/python`, Python 3.10 | `logs/frontier/scaleout/emender-e97-resume-canary-4891784.out:61` and FLA Python warning |
| torch | `2.8.0.dev20250422+rocm6.4` | `logs/frontier/scaleout/emender-e97-resume-canary-4891784.out:62` |
| `torch.version.hip` | `6.4.43482-0f2d60242` | `logs/frontier/scaleout/emender-e97-resume-canary-4891784.out` Python block |
| Triton | `3.2.0` | `logs/frontier/scaleout/emender-e97-resume-canary-4891784.out:63` |
| Loaded modules | `PrgEnv-gnu/8.7.0`, `cpe/26.03`, `miniforge3/23.11.0-0`, `rocm/7.1.1`, `craype-accel-amd-gfx90a` | `reports/frontier/hierarchical-diloco-integrate-test-20260627.md:76`, `scripts/frontier/*.sbatch`, and module list in `logs/frontier/scaleout/emender-e97-resume-canary-4891784.out` |
| Rank layout | 8 tasks per node, 1 GPU per task, closest GPU binding | `srun -N ... -n ... -c7 --gpus-per-task=1 --gpu-bind=closest` in `logs/frontier/scaleout/emender-e97-resume-canary-4891784.out` and 4936017 `env.txt` |
| Large-scale Slurm network | `--network=disable_rdzv_get`, `FI_CXI_RDZV_PROTO=alt_read` | `SLURM_NETWORK=disable_rdzv_get` and `FI_CXI_RDZV_PROTO=alt_read` in 4936017 `env.txt` |
| RCCL plugin status | `rccl_net_plugin_status=not-found` | `reports/frontier/conditional-256n-e97-hierarchical-smoke-20260627.md:141`; 4936017 `artifacts/env.txt:29` |

The failed job `4936017` used:

- Run root: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260704/E97_1.3B_step489920_b4_k40_256n_hier_g4_bucket64m_avg_12h/4936017-20260704T131533Z`
- 256 nodes, 2048 tasks, 8 tasks per node, `SLURM_NETWORK=disable_rdzv_get`.
- `NCCL_NET_PLUGIN=librccl-net.so`, `NCCL_CROSS_NIC=1`, `NCCL_NET_GDR_LEVEL=3`, `NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3`.
- `FI_CXI_DEFAULT_CQ_SIZE=131072`, `FI_CXI_DEFAULT_TX_SIZE=2048`, `FI_CXI_RDZV_PROTO=alt_read`, `FI_CXI_RX_MATCH_MODE=hybrid`, `FI_MR_CACHE_MONITOR=kdreg2`.
- `HSA_ENABLE_IPC_MODE_LEGACY=1`.
- `FRONTIER_PER_RANK_TRITON_CACHE=1` with a per-run cache root under the job artifacts.
- Failure line: `/lustre/orion/bif148/scratch/erikgarrison/emender-diloco-scalar-free/logs/frontier/scaleout/e97-b4-k40-256n12h-4936017.out:20803`: `torch.distributed.DistStoreError: Timed out after 601 seconds waiting for clients. 2016/2048 clients joined.`

The successful 256-node smoke `4908935` is important negative evidence. It also reported `rccl_net_plugin_status=not-found` and emitted repeated FLA Triton/Python warnings, yet completed. That makes "missing plugin" and "Triton 3.2 warning" insufficient as single-factor explanations for `4936017`, while still leaving both as stack-alignment problems.

## Current OLCF recommendation

Official OLCF Frontier PyTorch documentation currently recommends a stable PyTorch stack built around ROCm 7.1.1 for the broad PyTorch/PyG/Flash Attention environment:

- OLCF says newer PyTorch versions generally have better ROCm integration and support.
- The page's concrete recommended example is `torch==2.10.0`, `torchvision==0.25.0`, `torchaudio==2.10.0`, installed from the `rocm7.1` PyTorch wheel index.
- The module stack in that example is `PrgEnv-gnu/8.7.0`, `cpe/26.03`, `miniforge3/23.11.0-0`, `rocm/7.1.1`, `craype-accel-amd-gfx90a`, plus `LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH`.
- OLCF recommends creating that environment with Python 3.12.

Source: <https://docs.olcf.ornl.gov/software/analytics/pytorch_frontier.html>, lines 763-786.

For PyTorch-only users, OLCF also says versions compatible with ROCm 7.0.2 and 7.2.0 are recommended. PyTorch's official wheel matrix currently lists:

- `torch==2.11.0` for ROCm 7.2.
- `torch==2.10.0` for ROCm 7.1.
- `torch==2.9.1` and `torch==2.8.0` for ROCm 6.4.

Source: <https://pytorch.org/get-started/previous-versions/>, lines 81-162.

OLCF's July 1, 2026 software news and Frontier User Guide add a stronger RCCL recommendation:

- `rccl-net-plugin/1.0` was added on July 1, 2026.
- It provides `aws-ofi-nccl` and best-practice variables for AMD RCCL on HPE Slingshot.
- OLCF states this module is recommended for all PyTorch users.
- Recommended module families after the July 2026 cleanup are ROCm `6.4.2 or 7.x`, CPE `25.09 or 26.03`, and Cray MPICH `9.0.1 or 9.1.0`.

Sources:

- <https://docs.olcf.ornl.gov/software/software-news.html>, lines 738-786.
- <https://docs.olcf.ornl.gov/systems/frontier_user_guide.html>, lines 4447-4454 and 4468-4482.

## Local module availability

Local `module avail` evidence:

- `module avail pytorch`: no PyTorch module was found.
- `module avail rocm`: `rocm/6.4.2`, `rocm/7.0.2`, `rocm/7.1.1`, `rocm/7.2.0`, and `rocm/7.13.0` are available, among others. The current default in this environment is `rocm/6.2.4`, so scripts should continue to load ROCm explicitly.
- `module show rccl-net-plugin/1.0`: the module exists and is installed for ROCm `6.2.4`, `6.3.1`, `6.4.1`, `6.4.2`, `7.0.2`, `7.1.1`, and `7.2.0`.

`rccl-net-plugin/1.0` module metadata:

- Provides AWS NCCL/RCCL OFI interfaces, version `1.19.2`.
- Prepends `/sw/frontier/rccl-plugins/aws-ofi-nccl/1.19.2/lib` to `LD_LIBRARY_PATH`.
- Sets `OLCF_OFI_NCCL_ROOT=/sw/frontier/rccl-plugins/aws-ofi-nccl/1.19.2`.
- Sets `NCCL_CROSS_NIC=1`, `NCCL_NET_GDR_LEVEL=PHB`, `NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3`.
- Sets `HSA_FORCE_FINE_GRAIN_PCIE=1`.
- Sets `FI_MR_CACHE_MONITOR=kdreg2`, `FI_CXI_DISABLE_HOST_REGISTER=1`, `FI_CXI_DEFAULT_CQ_SIZE=131072`, `FI_CXI_RDZV_PROTO=alt_read`, `FI_CXI_RDZV_EAGER_SIZE=0`, `FI_CXI_RDZV_THRESHOLD=0`, `FI_CXI_RDZV_GET_MIN=0`, `FI_CXI_DEFAULT_TX_SIZE=2048`, and `FI_CXI_RX_MATCH_MODE=hybrid`.

The module does not appear to set `NCCL_NET_PLUGIN` itself. OLCF's PyTorch page says the plugin path must be in `LD_LIBRARY_PATH` and `NCCL_NET_PLUGIN=librccl-net.so` must be set so RCCL actually loads it; otherwise the AWS-OFI RCCL plugin is not used and performance is worse than expected. Source: <https://docs.olcf.ornl.gov/software/analytics/pytorch_frontier.html>, lines 1241-1246.

Recommended wrapper change, after review:

```bash
module load rocm/7.1.1
module load rccl-net-plugin/1.0
export NCCL_NET_PLUGIN=librccl-net.so
```

If testing ROCm 7.2.0 instead, load `rocm/7.2.0` before `rccl-net-plugin/1.0`. The module has a ROCm prerequisite and selects a precompiled plugin compatible with the loaded ROCm family.

## Comparison against OLCF guidance

| Area | Current E97/GDN2 jobs | OLCF/current evidence | Assessment |
| --- | --- | --- | --- |
| PyTorch | `2.8.0.dev20250422+rocm6.4` | OLCF example: stable `2.10.0` on ROCm 7.1. PyTorch official: 2.10 ROCm 7.1, 2.11 ROCm 7.2 | Current wheel is older, pre-release, and not aligned with loaded ROCm 7.1.1. |
| ROCm module | `rocm/7.1.1` | OLCF PyTorch example uses `rocm/7.1.1`; broader Frontier recommendation includes CPE 26.03 + ROCm 7.x or 7.2.0 | Module is acceptable, but the wheel reports ROCm 6.4. |
| Python | 3.10 | OLCF example creates Python 3.12; FLA warns Python 3.11+ recommended | Current Python works in completed jobs but is below current recommendation. |
| Triton | 3.2.0 | FLA recommends 3.3.0+; newer PyTorch wheels will carry newer compatible Triton dependencies | Warning is real for FLA/custom kernels, but not a direct c10d rendezvous cause. |
| RCCL plugin | `NCCL_NET_PLUGIN=librccl-net.so`, plugin status `not-found` | OLCF says put plugin lib in `LD_LIBRARY_PATH` and set `NCCL_NET_PLUGIN`; July 2026 module is recommended for all PyTorch users | Current wrappers are incomplete. Load `rccl-net-plugin/1.0`. |
| FI_CXI/NCCL env | Several recommended variables set by hand | `rccl-net-plugin/1.0` sets these plus additional HPE/OLCF values | Hand exports miss several official settings and the library path. |
| Slurm network | `--network=disable_rdzv_get` plus `FI_CXI_RDZV_PROTO=alt_read` | OLCF recommends this pair for alternative RCCL rendezvous protocol | Aligned. |
| srun mapping | 8 ranks/node, `-c7`, one GPU per task, `--gpu-bind=closest` | OLCF recommends `srun` and shows this layout as NUMA-appropriate | Aligned. |
| Env location | Python packages appear under `/ccs/home/.../.local/23.11.0-0`; run data on Lustre | OLCF says environment location matters; `NVMe > Orion >> NFS` and strongly recommends `sbcast` to NVMe at scale | Potential startup-jitter contributor at 2048 ranks. |

## Triton 3.2 warning

The warning text is from the FLA stack:

```text
Current Triton version 3.2.0 is below the recommended 3.3.0 version. Errors may occur and these issues will not be fixed. Please consider upgrading Triton.
```

It appears once per rank or many times per job because the imports run on each rank. It was present in completed jobs, including earlier canaries and the completed 256-node smoke, so it should not be treated as proof of the `4936017` failure.

What the warning does mean:

- FLA upstream does not intend to support all problems seen on Triton 3.2.
- E97/GDN2 custom Triton kernels and FLA kernels should be validated on a newer stack before production use.
- Kernel code generation, autotune choices, compile cache keys, and numerical behavior can change across Triton/ROCm/PyTorch versions.

What it probably does not mean:

- It is unlikely to explain a TCPStore timeout while waiting for clients in `dist.init_process_group`.
- It is unlikely to explain exactly 2016/2048 clients joining unless Triton imports/compiles delayed or killed a subset of ranks before process-group init. In the available logs, the failure signature is the c10d rendezvous timeout, not a Triton compilation exception.

Do not upgrade only Triton in place as a quick fix. PyTorch ROCm wheels and Triton are coupled through wheel dependencies, compiler/runtime expectations, and kernel API compatibility. Move torch+Triton+Python as a tested environment, not as a one-package mutation.

## E97/GDN2 upgrade risk

A full stack upgrade is feasible locally but not safe to deploy directly at 256 nodes.

Risk areas:

- Custom `ndm` Triton kernels: split edit kernels, chunked E97 kernels, multiquery kernels, GDN2 nonlinear kernels, MLP memory kernels, refit kernels, and pinned autotune settings.
- FLA/GDN2 kernels: FLA explicitly warns that Triton 3.2 is below its recommendation, so the upgrade may fix latent issues but can also change generated kernels and autotune behavior.
- Checkpoint compatibility: PyTorch `state_dict` tensor storage should usually load across these versions, but optimizer state, schedulefree state, dtype handling, serialization defaults, and checkpoint wrapper metadata must be tested.
- Compile cache behavior: changing torch/Triton/ROCm invalidates caches. At 2048 ranks, a cold compile/autotune wave can create startup jitter and filesystem pressure. Use a separate cache root per candidate environment and keep `NDM_PIN_TRITON_AUTOTUNE=1` for controlled comparisons.
- Numerical behavior: upgrade risk is not only "runtime." Kernel codegen, fused reductions, atomics, bf16 math, and FLA implementations can change bitwise behavior and possibly training trajectory. Expect small drift; reject obvious loss spikes, NaNs, divergent validation loss, or changed checkpoint contents without explanation.

Candidate environments:

1. Low-disruption OLCF example: Python 3.12, `rocm/7.1.1`, `torch==2.10.0` from `rocm7.1`, with OLCF module stack and `rccl-net-plugin/1.0`.
2. PyTorch-only newer candidate: Python 3.12, `rocm/7.2.0`, `torch==2.11.0` from `rocm7.2`, with `rccl-net-plugin/1.0`.

Because no local PyTorch module exists, these should be user-managed conda/pip environments or source builds if a dependency blocks wheels. Prefer wheels first for reproducibility and speed; build from source only if E97/GDN2 dependencies require it.

## Minimal validation ladder

Do not submit production jobs as part of the stack change. Use this ladder before another 256-node production run:

1. Import/version check.
   - Scope: login node if imports are CPU-safe, otherwise one compute node.
   - Command shape: import `torch`, `triton`, FLA, project modules; print `torch.__version__`, `torch.version.hip`, `triton.__version__`, Python version, and `LD_LIBRARY_PATH` plugin status.
   - Cost: no allocation to less than 0.1 node-hour.
   - Pass: versions match the intended stack; no import exceptions; `librccl-net.so` resolves when the module is loaded.

2. One-node E97/GDN2 forward/backward or short training smoke.
   - Scope: 1 node, 8 ranks if possible.
   - Cost: roughly 5-20 minutes, under 0.5 node-hour.
   - Pass: checkpoint loads, kernels compile, no NaNs, loss finite, no unexpected eager fallback, cache directory behavior sane.

3. RCCL allreduce/rendezvous diagnostic with `rccl-net-plugin/1.0` loaded.
   - Scope: existing `scripts/frontier/rccl_allreduce_diag.sbatch` style, first 2 nodes, then 8 or 32 nodes if needed.
   - Cost: roughly 0.1-2 node-hours depending on scale.
   - Pass: plugin status resolves, all ranks enter process group, allreduce completes, no TCPStore timeout, no severe RCCL errors.

4. Small-scale E97/GDN2 training smoke with the plugin.
   - Scope: 8 nodes, then 32 nodes if the 8-node smoke is clean.
   - Cost: roughly 1-3 node-hours for 8 nodes and 8-16 node-hours for 32 nodes depending on requested duration.
   - Pass: all ranks initialize, DiLoCo merge path works, checkpoint save/resume works, throughput is not obviously worse, and loss trajectory matches the current stack within expected noise.

5. Only after the above: 64-node or 128-node scale probe, then 256-node run.
   - Use the same checkpoint seed and short walltime first.
   - Keep the old environment available for rollback.

Run the ladder twice if changing both RCCL plugin and PyTorch/Triton. First validate `rccl-net-plugin/1.0` with the current torch stack. Then validate the candidate PyTorch/Triton/ROCm environment with the plugin already loaded. This isolates the effect of the networking fix from the runtime upgrade.

## Answer on `4936017` failure attribution

`4936017` failed during distributed initialization:

```text
torch.distributed.DistStoreError: Timed out after 601 seconds waiting for clients. 2016/2048 clients joined.
```

Attribution:

- Triton 3.2 warning: low likelihood as direct cause. It is an import/kernel-support warning and appears in successful jobs. It should drive validation of a newer stack, not be blamed for the rendezvous timeout.
- Missing `rccl-net-plugin/1.0`: medium likelihood as a stack risk, low-to-medium as the direct cause of this exact TCPStore client-count timeout. The plugin affects RCCL/NCCL network transport and scale stability; however, TCPStore client rendezvous timing can fail before meaningful RCCL allreduce traffic. Still, the current "set `NCCL_NET_PLUGIN` but library not found" state is incorrect and should be fixed before more large jobs.
- c10d timeout/startup jitter/rank failure: highest likelihood. The timeout says 32 clients never joined. That can come from rank startup skew, node-local process failure, Python environment startup latency at scale, filesystem pressure, or ranks killed while other ranks are waiting. The subsequent TCPStore broken-pipe and `srun` killed-task messages are consistent with a cascading abort after the rendezvous failure.
- Combination: plausible. At 2048 ranks, partial RCCL/Slingshot env, Python packages under home/local paths, per-rank imports, per-rank compile-cache setup, and cold startup variance can combine into a c10d timeout even if no single setting is deterministically broken.

Operational conclusion: before retrying a large run, load `rccl-net-plugin/1.0`, verify the plugin is found, keep `--network=disable_rdzv_get` with `FI_CXI_RDZV_PROTO=alt_read`, and run a small RCCL/c10d diagnostic. Defer full PyTorch/Triton/ROCm upgrade until it clears the E97/GDN2 validation ladder.

## Validation checklist status

- [x] Exact current runtime versions and module stack identified.
- [x] Official/current OLCF recommendation identified with source links and local module evidence.
- [x] Triton 3.2 warning explained and assessed against the rendezvous failure.
- [x] `rccl-net-plugin/1.0` recommendation and changed variables/library paths documented.
- [x] Go/no-go recommendation provided: go for `rccl-net-plugin/1.0` diagnostics; no-go for unvalidated torch/Triton/ROCm upgrade before production-scale jobs.
- [x] Minimal smoke/diagnostic ladder proposed with risk and rough resource cost.
- [x] No production Slurm jobs submitted for this task.
