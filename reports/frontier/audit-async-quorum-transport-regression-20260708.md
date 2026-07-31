# Audit: async quorum transport regression

Task: `audit-async-quorum`  
Date: 2026-07-08  
Scope: no Slurm jobs submitted; this audit reads committed code, WG logs, reports, and existing run artifacts only.

## Executive finding

The resilient quorum rerun ladder and the 256n debug gate selected the Python TCP quorum path because their launch environment exported `ASYNC_QUORUM_TRANSPORT=tcp`. The Frontier smoke wrapper translated that value into `--actual-multinode-tcp-quorum`; the E97 train entrypoint then set `RealAsyncFileRankConfig.transport = "tcp"`; `RealAsyncFileRank.run()` therefore entered `_coordinate_network_rank_quorum()` instead of either the `mpi-dense` path or the compiled MPICH helper path.

This means the accepted 1n/8n/64n resilient rerun evidence and the failed 256n debug gate evidence validate the TCP coordinator/control-plane behavior only. They do not validate the intended Frontier production transport for async/quorum scaleout, which is MPI/compiled MPICH helper transport.

## Exact TCP selection path

The current path is:

1. `ASYNC_QUORUM_TRANSPORT=tcp` in the Slurm export/environment.
2. `scripts/frontier/trainpy_async_quorum_smoke_common.sh` selects the `tcp` case and appends `--actual-multinode-tcp-quorum`.
3. `scripts/frontier/e97_async_diloco_train.py` parses mutually exclusive transport flags and sets:
   - `compiled-cray-mpich-helper-*` when `--actual-multinode-compiled-mpich-quorum` is present;
   - `mpi-dense` when `--actual-multinode-mpi-dense-quorum` is present;
   - otherwise `tcp` when `--actual-multinode-tcp-quorum` is present.
4. `ndm/async_diloco_real.py` dispatches `transport == "tcp"` to `_coordinate_network_rank_quorum()` on node rank 0.

Relevant code anchors:

- `scripts/frontier/trainpy_async_quorum_smoke_common.sh:50-59` currently defaults `ASYNC_QUORUM_TRANSPORT` to `compiled-cray-mpich-helper-p2p`, but an explicit exported value overrides that default.
- `scripts/frontier/trainpy_async_quorum_smoke_common.sh:229-243` maps `tcp` to `--actual-multinode-tcp-quorum`, `mpi-dense` to `--actual-multinode-mpi-dense-quorum`, and `compiled-cray-mpich-helper-*` to `--actual-multinode-compiled-mpich-quorum`.
- `scripts/frontier/e97_async_diloco_train.py:104-113` defines the three transport flags and the quorum-mode default.
- `scripts/frontier/e97_async_diloco_train.py:181-199` enforces only one transport flag and requires the compiled helper binary when the compiled helper flag is used.
- `scripts/frontier/e97_async_diloco_train.py:269-273` sets `transport` to compiled helper, `mpi-dense`, or `tcp`.
- `ndm/async_diloco_real.py:219` defaults `RealAsyncFileRankConfig.transport` to `tcp`.
- `ndm/async_diloco_real.py:317-327` validates allowed transports and requires compiled MPICH for strict collectives.
- `ndm/async_diloco_real.py:405-428` dispatches compiled MPICH, `mpi-dense`, or TCP network quorum.
- `ndm/async_diloco_real.py:1693-1711` labels the TCP payload as `actual_multinode_tcp_quorum_debug` and states that dense delta exchange should target MPI P2P rather than the Python debug payload.

## Run evidence

### 1n/8n/64n resilient rerun ladder

The ladder task log reports job IDs `4956437`, `4956445`, and `4956459`, all accepted all launched ranks and reported TCP bytes. The per-job artifacts record the concrete command and environment selected by the wrapper:

| Scale | Job | Evidence path | TCP selector evidence |
| --- | ---: | --- | --- |
| 1n / 8 ranks | 4956437 | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/rerun_resilient_quorum_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_resilient_quorum_rerun_1n/4956437-20260708T104400Z/artifacts/env.txt` | line 32: `async_quorum_transport=tcp`; line 47: `bounded_debug_transport=actual_multinode_tcp_quorum`; line 99 and `artifacts/command.txt:1`: `srun -N 1 -n 8 ... --global-quorum 8 ... --timeout-s 120 ... --actual-multinode-tcp-quorum` |
| 8n / 64 ranks | 4956445 | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/rerun_resilient_quorum_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_resilient_quorum_rerun_8n/4956445-20260708T104614Z/artifacts/env.txt` | line 32: `async_quorum_transport=tcp`; line 47: `bounded_debug_transport=actual_multinode_tcp_quorum`; line 99 and `artifacts/command.txt:1`: `srun -N 8 -n 64 ... --global-quorum 64 ... --timeout-s 180 ... --actual-multinode-tcp-quorum` |
| 64n / 512 ranks | 4956459 | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/rerun_resilient_quorum_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_resilient_quorum_rerun_64n/4956459-20260708T105539Z/artifacts/env.txt` | line 32: `async_quorum_transport=tcp`; line 47: `bounded_debug_transport=actual_multinode_tcp_quorum`; line 99 and `artifacts/command.txt:1`: `srun -N 64 -n 512 ... --global-quorum 512 ... --timeout-s 300 ... --actual-multinode-tcp-quorum` |

The commands also pass `--compiled-mpich-helper-bin` and `--compiled-mpich-ipc-dir`, but those arguments are inert in these jobs because the active transport flag is `--actual-multinode-tcp-quorum`.

### 256n resilient debug gate

`reports/frontier/evaluate-rerun-resilient-quorum-256n-debug-gate-20260708.md:80-84` records the exact submit command for job `4956594`:

```bash
sbatch --parsable -N 256 -J resilient-quorum-256n-debug -t 00:20:00 -p batch -q debug --export=ALL,WG_TASK_ID=evaluate-rerun-resilient-quorum-256n-debug-gate,SMOKE_NAME=256n-resilient-debug,SMOKE_NODE_COUNT=256,SCALEOUT_VARIANT=E97_1.3B_step1065000_resilient_quorum_rerun_256n_debug,OUTPUT_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/rerun_resilient_quorum_256n_debug_gate_20260708,ASYNC_QUORUM_TRANSPORT=tcp,ASYNC_TRAINPY_RANKS=2048,ASYNC_EXPECTED_RANKS=2048,ASYNC_GLOBAL_QUORUM=2048,ASYNC_EXPECTED_MISSING_UPDATES=0,ASYNC_TIMEOUT_S=1200,REQUESTED_WALLTIME=00:20:00,REQUESTED_NODE_HOURS=85.333333 scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch
```

The job artifact `logs/frontier/trainpy_async_quorum/resilient-quorum-256n-debug-4956594.out` records:

- line 32: `async_quorum_transport=tcp`
- line 47: `bounded_debug_transport=actual_multinode_tcp_quorum`
- line 99: `srun -N 256 -n 2048 ... --global-quorum 2048 ... --timeout-s 1200 ... --actual-multinode-tcp-quorum`

The same report records the outcome:

- `reports/frontier/evaluate-rerun-resilient-quorum-256n-debug-gate-20260708.md:107-125`: 2048/2048 rank starts, only 593/2048 accepted, quorum deferred, 1455 timed out, latest checkpoint not advanced, and TCP bytes reported.

## MPI and compiled-helper integration points

The codebase has three async/quorum data-plane choices:

1. TCP debug quorum
   - Entrypoint flag: `--actual-multinode-tcp-quorum`.
   - Config transport: `tcp`.
   - Runtime path: `_coordinate_network_rank_quorum()`.
   - Metrics labels: `actual_multinode_tcp_quorum_debug`, `transport.name=tcp`, `update_bytes.tcp_payload`, `update_bytes.node_metadata`.
   - Intended use: local/small bounded debug only.

2. `mpi-dense`
   - Entrypoint flag: `--actual-multinode-mpi-dense-quorum`.
   - Config transport: `mpi-dense`.
   - Runtime path: `_coordinate_mpi_dense_rank()` calling `run_mpi_dense_quorum()`.
   - Implementation: `ndm/async_diloco_mpi.py`, using `mpi4py` over Cray MPICH when available.
   - Status: useful comparison/integration point, but prior Frontier reports show the mpi4py path was fragile and was not the selected production path. `reports/frontier/frontier-mpi-dense-async-diloco-validation-20260707.md:51` records an `ASYNC_QUORUM_TRANSPORT=mpi-dense` attempt; `reports/frontier/resolve-frontier-2n-mpi-ofi-20260707.md` narrowed the Frontier-usable substrate to compiled Cray MPICH C rather than mpi4py for this scale path.

3. Compiled MPICH helper
   - Entrypoint flag: `--actual-multinode-compiled-mpich-quorum`.
   - Wrapper selector: `ASYNC_QUORUM_TRANSPORT=compiled-cray-mpich-helper-p2p`.
   - Actual metrics transport: `compiled-cray-mpich-helper-collective-reduce`.
   - Runtime path: `_coordinate_compiled_mpich_dense_rank()` calling `run_compiled_mpich_dense_quorum()`.
   - Implementation: `ndm/async_diloco_compiled_mpich.py` plus `scripts/frontier/compiled_mpich_dense_helper.cpp`.
   - Helper evidence: `scripts/frontier/compiled_mpich_dense_helper.cpp:444` emits `transport=compiled-cray-mpich-helper-collective-reduce`; `scripts/frontier/compiled_mpich_dense_helper.cpp:693` exposes the C ABI entry point `compiled_mpich_dense_helper_run_once`.
   - Current production wrapper: `scripts/frontier/async_diloco_e97_256n12h_launch.sbatch:309-319` now refuses TCP and mpi-dense for the 256n/12h path and requires the compiled helper transport mode.

The compiled helper path has already demonstrated strict collective scaling through 256n:

- `reports/frontier/run-compiled-helper-20260708.md:80-81` records the 8n compiled-helper submit with `ASYNC_QUORUM_TRANSPORT=compiled-cray-mpich-helper-p2p`.
- `logs/frontier/trainpy_async_quorum/compiled-helper-trainpy-8n-4954290.out:999` records accepted 64/64, mode `actual_multinode_compiled_mpich_quorum`, transport `compiled-cray-mpich-helper-collective-reduce`, reducer `mpi_reduce_bucketed_weighted_sum`, collective `MPI_Reduce`, and `tcp_dense_data_plane=false`.
- `logs/frontier/trainpy_async_quorum/compiled-helper-trainpy-64n-4954317.out:7271` records accepted 512/512 with the same compiled-helper mode and transport.
- `reports/frontier/evaluate-compiled-helper-256n-debug-20260708.md:85-86` records the 256n compiled-helper submit with 2048 ranks and `ASYNC_QUORUM_TRANSPORT=compiled-cray-mpich-helper-p2p`.
- `logs/frontier/trainpy_async_quorum/compiled-helper-trainpy-256n-debug-4954634.out:28775` records accepted 2048/2048, quorum advanced, mode `actual_multinode_compiled_mpich_quorum`, `world_size=2048`, transport `compiled-cray-mpich-helper-collective-reduce`, collective `MPI_Reduce`, `tcp_dense_data_plane=false`, and latest advanced.

## Missing production integration

The missing piece is not basic compiled MPICH reachability; that exists and has 256n strict-collective evidence. The missing piece is resilient async/quorum semantics over the compiled helper transport.

Today, the resilient ladder that accepted 1n/8n/64n used TCP. The compiled helper evidence uses strict collective behavior where all launched ranks participate. That validates the dense MPI transport substrate and the strict path, but it does not by itself validate a production resilient quorum ladder that can tolerate missing/stale/late ranks without falling back to Python TCP fan-in.

The implementation gap is therefore:

- Add or finish a compiled-helper/MPI resilient quorum mode that carries the quorum metadata and dense update reduction over the compiled MPICH helper path.
- Ensure that nonunanimous quorum, timeout, stale/late/missing accounting, and catchup metadata are recorded without routing through `_coordinate_network_rank_quorum()`.
- Ensure the wrapper and approval tooling can distinguish `compiled helper strict collective`, `compiled helper resilient quorum`, and `TCP debug quorum`.

## Production invariant

Frontier scale ladders must use MPI/compiled MPICH helper transport. TCP is local/small debug only and must be explicitly labeled as not production-approval evidence when used.

Concrete invariant:

- `SMOKE_NODE_COUNT >= 8`, `ASYNC_TRAINPY_RANKS >= 64`, 64n/128n/256n gates, and all 1h/12h/production approval ladders must not run with `ASYNC_QUORUM_TRANSPORT=tcp`.
- Scale reports must require `transport.name` to start with `compiled-cray-mpich-helper` and must require `tcp_dense_data_plane=false`.
- Any intentional TCP run above local debug size must require an explicit override variable such as `ALLOW_FRONTIER_TCP_SCALE_DEBUG=1`, use a job/report name containing `tcp-debug-no-production`, and set a machine-readable `production_approval_eligible=false`.

## Production approval status

No current TCP resilient-quorum result should be used for production approval.

- The 1n/8n/64n resilient rerun ladder proves that the Python TCP debug quorum path can accept 8/8, 64/64, and 512/512 ranks under those timeouts. It does not prove the intended MPI/compiled helper async/quorum design.
- The 256n resilient debug gate proves the opposite limit: the TCP path started 2048/2048 ranks but accepted only 593/2048 before timeout. That is useful negative evidence for the TCP control-plane limit and is not production approval evidence.
- The compiled-helper 8n/64n/256n strict collective runs are usable as evidence that the compiled MPICH helper transport and dense reduction substrate work at Frontier scale. They are not, by themselves, production approval for resilient async/quorum semantics until the resilient quorum mode is wired to the compiled helper path and validated.
- The `mpi-dense` mpi4py evidence is not production approval evidence for Frontier scaleout.

## Recommended changes

1. Make TCP scale use fail closed in `scripts/frontier/trainpy_async_quorum_smoke_common.sh`.
   - If `ASYNC_QUORUM_TRANSPORT=tcp` and either `SMOKE_NODE_COUNT > 1` or `ASYNC_TRAINPY_RANKS > 8`, exit nonzero unless `ALLOW_FRONTIER_TCP_SCALE_DEBUG=1`.
   - When the override is present, write `bounded_debug_transport=actual_multinode_tcp_quorum`, `production_approval_eligible=false`, and require the run/report name to include `tcp-debug-no-production`.

2. Add an entrypoint guard in `scripts/frontier/e97_async_diloco_train.py`.
   - Add `--allow-tcp-scale-debug` or equivalent.
   - Reject `--actual-multinode-tcp-quorum` above local debug size unless that override is present.
   - Emit a metrics field such as `transport.approval_class = "tcp-debug-only"` for TCP and `"frontier-production-candidate"` only for compiled-helper transport.

3. Normalize compiled-helper naming.
   - The wrapper selector `compiled-cray-mpich-helper-p2p` is a legacy selector name, while the actual helper metrics report `compiled-cray-mpich-helper-collective-reduce`.
   - Either rename the selector to the actual collective-reduce transport or record separate `transport.selector` and `transport.actual` fields in metrics and reports.

4. Add wrapper tests.
   - Verify default `ASYNC_QUORUM_TRANSPORT` is compiled helper.
   - Verify `ASYNC_QUORUM_TRANSPORT=tcp` maps to `--actual-multinode-tcp-quorum` only for local debug size.
   - Verify TCP exits nonzero for 8n/64n/256n without the explicit override.
   - Verify compiled helper appends `--actual-multinode-compiled-mpich-quorum` and records helper binary/ipc paths.

5. Add entrypoint transport-selection tests.
   - Verify the three mutually exclusive flags produce `tcp`, `mpi-dense`, and compiled-helper config transports.
   - Verify strict collective requires compiled helper.
   - Verify TCP scale debug requires the explicit override.

6. Add report/approval parser checks.
   - A report containing `transport.name=tcp`, `async_quorum_transport=tcp`, `bounded_debug_transport=actual_multinode_tcp_quorum`, or `update_bytes.tcp_payload` at Frontier scale must be classified as `production_approval_eligible=false`.
   - A production approval report must assert compiled helper transport, `tcp_dense_data_plane=false`, expected rank acceptance, latest advancement, and no TCP payload byte accounting.

7. Implement and validate compiled-helper resilient quorum.
   - Reuse the compiled helper substrate validated by job `4954634`.
   - Add resilient quorum metadata and timeout/stale/missing accounting without using the TCP coordinator.
   - Re-run the ladder only after the code/launcher guards make accidental TCP scale tests impossible.

## Validation checklist

- Reported exact command/env/config path that selected TCP in 1n/8n/64n/256n resilient tests.
- Reported available MPI/compiled helper async/quorum integration points and the missing resilient compiled-helper integration.
- Recommended concrete implementation changes and tests.
- Stated production approval status separately for TCP evidence, compiled-helper strict collective evidence, and mpi4py `mpi-dense` evidence.
- Submitted no Slurm jobs.
