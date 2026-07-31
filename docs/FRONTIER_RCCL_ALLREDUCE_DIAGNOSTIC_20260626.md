# Frontier RCCL/Slingshot Allreduce Diagnostic - 2026-06-26

WG task: `diagnose-frontier-rccl`

## Verdict

The E97 1.3B K160 32-node r4 failure remains a systems blocker, not a
model-quality failure, but a minimal standalone `torch.distributed` allreduce
does **not** reproduce the hang. The diagnostic exercised 1-element, medium
`1048576`-element, and model-sized `1286589072`-element float32 allreduces at
2/4/8/16/32 Frontier nodes under both the current launcher-style env and an
OLCF-guidance env with alternative rendezvous. All ten jobs completed with
Slurm `0:0`, and all rank-0 tensor checks passed.

This means:

- Keep the r4 scalar-collective removal. It moved the observed failure from the
  prior 1-element scalar `ALLREDUCE` to the intended model-sized
  `ALLREDUCE NumelIn=1286589072`, and the scalar diagnostic now passes through
  32 nodes.
- Do **not** submit 64/128/256-node training. The next allowed training action
  is a single 32-node K160 retry using the patched launcher comm-env controls,
  because the failed r4 training merge path has not passed.
- Treat `aws-ofi-rccl` / `librccl-net.so` as still unverified. No module was
  visible, no readable `librccl-net.so` was found in the searched project/shared
  paths or runtime `LD_LIBRARY_PATH`, and the diagnostic artifacts record
  `RCCL_DIAG_PLUGIN_STATUS=not-found`.

OLCF's current Frontier PyTorch guidance recommends `srun` rank mapping,
`MASTER_ADDR`, high-speed `NCCL_SOCKET_IFNAME`, the AWS-OFI-RCCL plugin via
`NCCL_NET_PLUGIN=librccl-net.so`, `FI_MR_CACHE_MONITOR`, `FI_CXI_*` sizing/match
settings, and optionally Slurm `--network=disable_rdzv_get` with
`FI_CXI_RDZV_PROTO=alt_read` for RCCL-heavy jobs:
https://docs.olcf.ornl.gov/software/analytics/pytorch_frontier.html

## r4 Failure Being Diagnosed

`docs/FRONTIER_E97_1P3B_PRETRAINED_K160_32N_R4_RUN_20260626.md` reports Slurm
job `4904689` failed at the first post-resume K160 synchronization window with:

```text
WorkNCCL(SeqNum=153, OpType=ALLREDUCE, NumelIn=1286589072, NumelOut=1286589072, Timeout(ms)=600000)
```

That `NumelIn` matches the E97 1.3B trainable parameter count. The prior r3
failure at the same sequence number was a 1-element scalar allreduce. Therefore
r4 is the right failure surface for communication-stack diagnosis: the scalar
clock collective was removed from the merge path, and the model-state merge
itself failed to complete.

## Code Artifacts Added

- `scripts/frontier/rccl_allreduce_diag.py`: minimal NCCL/RCCL
  `torch.distributed` allreduce probe.
- `scripts/frontier/rccl_allreduce_diag.sbatch`: Frontier sbatch wrapper for
  current or recommended env.
- `scripts/frontier/rccl_allreduce_diag_alt_rdzv.sbatch`: same diagnostic with
  `#SBATCH --network=disable_rdzv_get` and `FI_CXI_RDZV_PROTO=alt_read`.
- `scripts/frontier/e97_1p3b_pretrained_canary.sbatch`: patched to record
  `FI_*`, `NCCL*`, `RCCL*`, `LD_LIBRARY_PATH`, plugin status, and to support
  opt-in `FRONTIER_RCCL_ENV=recommended`.

The diagnostic sizes were:

```text
scalar=1,medium=1048576,model=1286589072
```

The diagnostic dtype was `float32`. The model-sized tensor is
`1286589072 * 4 = 5146356288` bytes per rank.

## Current Launcher Env Audit

The r4 training artifact
`/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260626/E97_1.3B_pretrained_k160_32n_r4/4904689-20260626T145541Z/artifacts/env.txt`
captured only `RESUME_CHECKPOINT` and `TRAIN_MINUTES` because the launcher env
grep omitted `FI_*`, `LD_LIBRARY_PATH`, and module state. The launcher itself
only explicitly set:

```text
MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED:-1}
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-hsn0}
MASTER_ADDR=<first Slurm host>
MASTER_PORT=3442
OMP_NUM_THREADS=7
```

It did not explicitly set:

```text
NCCL_NET_PLUGIN
NCCL_NET_GDR_LEVEL
NCCL_CROSS_NIC
FI_CXI_DEFAULT_CQ_SIZE
FI_CXI_DEFAULT_TX_SIZE
FI_CXI_RX_MATCH_MODE
FI_CXI_RDZV_PROTO
```

The current-env 32-node diagnostic did inherit these exact relevant values:

```text
FI_CXI_ATS=0
FI_MR_CACHE_MONITOR=kdreg2
MASTER_ADDR=frontier00020
MASTER_PORT=3442
MPICH_GPU_SUPPORT_ENABLED=1
MPICH_OFI_NIC_POLICY=NUMA
NCCL_DEBUG=INFO
NCCL_SOCKET_IFNAME=hsn0
RCCL_DIAG_PLUGIN_STATUS=not-found
SLURM_JOB_NUM_NODES=32
SLURM_NTASKS=256
```

Full exact env:
`/lustre/orion/bif148/proj-shared/emender/frontier_runs/rccl_diag/20260626/current/4905861-20260626T173124Z/artifacts/rank0_results.json`

## Recommended Env Tested

The recommended/alt-rendezvous 32-node diagnostic used:

```text
FI_CXI_DEFAULT_CQ_SIZE=131072
FI_CXI_DEFAULT_TX_SIZE=2048
FI_CXI_RDZV_PROTO=alt_read
FI_CXI_RX_MATCH_MODE=hybrid
FI_MR_CACHE_MONITOR=kdreg2
MASTER_ADDR=frontier01028
MASTER_PORT=3442
MPICH_GPU_SUPPORT_ENABLED=1
MPICH_OFI_NIC_POLICY=NUMA
NCCL_CROSS_NIC=1
NCCL_DEBUG=INFO
NCCL_NET_GDR_LEVEL=3
NCCL_NET_PLUGIN=librccl-net.so
NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
RCCL_DIAG_ALT_RDZV=1
RCCL_DIAG_PLUGIN_STATUS=not-found
SLURM_NETWORK=disable_rdzv_get
SLURM_JOB_NUM_NODES=32
SLURM_NTASKS=256
```

Full exact env:
`/lustre/orion/bif148/proj-shared/emender/frontier_runs/rccl_diag/20260626/recommended/4905884-20260626T173732Z/artifacts/rank0_results.json`

Important caveat: `NCCL_NET_PLUGIN=librccl-net.so` was set, but the diagnostic's
filesystem check did not find a readable `librccl-net.so` in `LD_LIBRARY_PATH`.
The logs report the bundled PyTorch RCCL library path:

```text
/ccs/home/erikgarrison/.local/23.11.0-0/lib/python3.10/site-packages/torch/lib/librccl.so
```

I found no `aws-ofi-rccl` or `rccl` module via `module spider`, and no
`librccl-net.so` under:

```text
/lustre/orion/bif148/proj-shared
/lustre/orion/bif148/scratch/erikgarrison
```

## Slurm Evidence

| Env | Nodes | Job ID | Slurm state | Exit | Elapsed | Stdout | Stderr | Run root |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| current | 2 | `4905841` | `COMPLETED` | `0:0` | `00:00:36` | `logs/frontier/rccl_diag/rccl-current-2n-4905841.out` | `logs/frontier/rccl_diag/rccl-current-2n-4905841.err` | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/rccl_diag/20260626/current/4905841-20260626T172558Z` |
| current | 4 | `4905844` | `COMPLETED` | `0:0` | `00:00:34` | `logs/frontier/rccl_diag/rccl-current-4n-4905844.out` | `logs/frontier/rccl_diag/rccl-current-4n-4905844.err` | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/rccl_diag/20260626/current/4905844-20260626T172748Z` |
| current | 8 | `4905848` | `COMPLETED` | `0:0` | `00:00:35` | `logs/frontier/rccl_diag/rccl-current-8n-4905848.out` | `logs/frontier/rccl_diag/rccl-current-8n-4905848.err` | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/rccl_diag/20260626/current/4905848-20260626T172901Z` |
| current | 16 | `4905856` | `COMPLETED` | `0:0` | `00:00:38` | `logs/frontier/rccl_diag/rccl-current-16n-4905856.out` | `logs/frontier/rccl_diag/rccl-current-16n-4905856.err` | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/rccl_diag/20260626/current/4905856-20260626T173012Z` |
| current | 32 | `4905861` | `COMPLETED` | `0:0` | `00:00:47` | `logs/frontier/rccl_diag/rccl-current-32n-4905861.out` | `logs/frontier/rccl_diag/rccl-current-32n-4905861.err` | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/rccl_diag/20260626/current/4905861-20260626T173124Z` |
| recommended+alt-rdzv | 2 | `4905863` | `COMPLETED` | `0:0` | `00:00:31` | `logs/frontier/rccl_diag/rccl-recommended-2n-4905863.out` | `logs/frontier/rccl_diag/rccl-recommended-2n-4905863.err` | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/rccl_diag/20260626/recommended/4905863-20260626T173239Z` |
| recommended+alt-rdzv | 4 | `4905867` | `COMPLETED` | `0:0` | `00:00:38` | `logs/frontier/rccl_diag/rccl-recommended-4n-4905867.out` | `logs/frontier/rccl_diag/rccl-recommended-4n-4905867.err` | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/rccl_diag/20260626/recommended/4905867-20260626T173352Z` |
| recommended+alt-rdzv | 8 | `4905876` | `COMPLETED` | `0:0` | `00:00:34` | `logs/frontier/rccl_diag/rccl-recommended-8n-4905876.out` | `logs/frontier/rccl_diag/rccl-recommended-8n-4905876.err` | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/rccl_diag/20260626/recommended/4905876-20260626T173505Z` |
| recommended+alt-rdzv | 16 | `4905883` | `COMPLETED` | `0:0` | `00:00:38` | `logs/frontier/rccl_diag/rccl-recommended-16n-4905883.out` | `logs/frontier/rccl_diag/rccl-recommended-16n-4905883.err` | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/rccl_diag/20260626/recommended/4905883-20260626T173619Z` |
| recommended+alt-rdzv | 32 | `4905884` | `COMPLETED` | `0:0` | `00:01:01` | `logs/frontier/rccl_diag/rccl-recommended-32n-4905884.out` | `logs/frontier/rccl_diag/rccl-recommended-32n-4905884.err` | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/rccl_diag/20260626/recommended/4905884-20260626T173732Z` |

## Pass/Fail By Tensor Size

All cells below are rank-0 tensor-value checks, not just Slurm status.

| Env | Nodes | World | Scalar 1 | Medium 1048576 | Model 1286589072 | Model elapsed |
| --- | ---: | ---: | --- | --- | --- | ---: |
| current | 2 | 16 | pass | pass | pass | `3.713s` |
| current | 4 | 32 | pass | pass | pass | `3.930s` |
| current | 8 | 64 | pass | pass | pass | `4.017s` |
| current | 16 | 128 | pass | pass | pass | `4.154s` |
| current | 32 | 256 | pass | pass | pass | `4.675s` |
| recommended+alt-rdzv | 2 | 16 | pass | pass | pass | `1.297s` |
| recommended+alt-rdzv | 4 | 32 | pass | pass | pass | `2.178s` |
| recommended+alt-rdzv | 8 | 64 | pass | pass | pass | `2.236s` |
| recommended+alt-rdzv | 16 | 128 | pass | pass | pass | `2.495s` |
| recommended+alt-rdzv | 32 | 256 | pass | pass | pass | `2.102s` |

The recommended/alt-rendezvous env substantially improved model-sized allreduce
latency in this microdiagnostic even though `librccl-net.so` was not located by
the wrapper.

## Launcher Patch / Rollback Recommendation

Implemented low-risk launcher support:

```text
FRONTIER_RCCL_ENV=recommended
FRONTIER_RCCL_ALT_RDZV=1
AWS_OFI_RCCL_PLUGIN_DIR=/path/to/aws-ofi-rccl   # optional, when available
```

When `FRONTIER_RCCL_ENV=recommended`, the launcher exports:

```text
FI_MR_CACHE_MONITOR=kdreg2
FI_CXI_DEFAULT_CQ_SIZE=131072
FI_CXI_DEFAULT_TX_SIZE=2048
FI_CXI_RX_MATCH_MODE=hybrid
NCCL_NET_GDR_LEVEL=3
NCCL_CROSS_NIC=1
NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
NCCL_NET_PLUGIN=librccl-net.so
```

When `FRONTIER_RCCL_ALT_RDZV=1`, it also exports:

```text
FI_CXI_RDZV_PROTO=alt_read
```

For an actual alternative-rendezvous allocation, submit with Slurm:

```bash
sbatch --network=disable_rdzv_get ...
```

Do **not** make this the unconditional default yet. The plugin path is not
verified, and the current-env microdiagnostic already passes. Use the opt-in
recommended env for one 32-node K160 training retry, and require that retry to
complete the first post-resume K160 DiLoCo merge and write a successor
checkpoint before any 64/128/256-node training.

Recommended next training command shape:

```bash
sbatch -N 32 -J e97-1p3b-k160-32n-rccl-retry \
  --network=disable_rdzv_get \
  --export=ALL,WG_TASK_ID=run-e97-1p3b-k160-32n,SCALEOUT_VARIANT=E97_1.3B_pretrained_k160_32n_rccl_retry,FRONTIER_RCCL_ENV=recommended,FRONTIER_RCCL_ALT_RDZV=1,RESUME_CHECKPOINT=<job-4903889-K-aligned-latest.pt>,TRAIN_MINUTES=35,WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS=1200,WALLTIME_CHECK_EVERY=160,DISTRIBUTED_HEALTH_CHECK_EVERY=160,REQUESTED_NODE_HOURS=32.0 \
  scripts/frontier/e97_1p3b_pretrained_k160_scale_ladder.sbatch
```

If a readable aws-ofi-rccl build is found first, add:

```text
AWS_OFI_RCCL_PLUGIN_DIR=/path/to/aws-ofi-rccl
```

## Scalar-Collective Decision

Keep the scalar-collective removal. Do not revert it.

Rationale:

- r3 failed on `NumelIn=1`.
- r4 progressed to the model-sized `NumelIn=1286589072` merge.
- The standalone diagnostics passed 1-element scalar allreduce at every tested
  node count through 32 nodes in both env modes.

After the 32-node training comm path is fixed, the scalar-clock behavior should
remain removed or guarded as an optional compatibility/debug path. It should not
return as an unconditional collective inside the ScheduleFree DiLoCo merge.

## Stop Conditions Still Active

- No 64-node, 128-node, or 256-node training submission until a 32-node K160
  training retry completes the first post-resume model-sized DiLoCo merge.
- If the next 32-node retry fails despite this standalone allreduce pass,
  inspect the training merge sequence itself: rank participation, tensor
  flattening/chunking, dtype, stream synchronization, per-rank exception before
  the merge, and whether any health/finalization collective interleaves with the
  model merge.
