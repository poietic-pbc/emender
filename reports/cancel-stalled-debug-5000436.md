# Emergency cancellation record: Slurm job 5000436

Captured on 2026-07-15. All times below are UTC unless Slurm renders the
site-local time. This record concerns only the authorized debug allocation
5000436. No job was submitted. Production job 4980157 was only queried
read-only and was not modified, cancelled, requeued, or resubmitted.

## Cancellation and terminal state

- Final pre-cancel snapshot at 2026-07-15T07:27:43Z: job 5000436 was RUNNING
  for 00:42:29 on 256 nodes (2048 tasks/GPUs); active steps were `.0`,
  `.batch`, and `.extern`.
- Exactly one cancellation command was issued: `scancel 5000436` at
  2026-07-15T07:27:57Z. It returned exit status 0.
- Final `squeue -h -j 5000436` and `squeue --steps -h -j 5000436` returned
  zero rows after cleanup. Therefore no allocation or step remains consuming
  any of the 256 nodes.
- Final `sacct` states:
  - allocation `5000436`: `CANCELLED by 19032`, elapsed 00:42:43,
    2026-07-15T02:45:14 through 2026-07-15T03:27:57, 256 nodes;
  - step `5000436.0`: `CANCELLED`, ended 2026-07-15T03:27:58, 256 nodes;
  - step `5000436.batch`: `FAILED` after cancellation cleanup, ended
    2026-07-15T03:28:17;
  - step `5000436.extern`: `COMPLETED`, ended 2026-07-15T03:28:26,
    256 nodes.

## Preserved training state

Run root:

`/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_quorum_b4k40_ladder_20260709/20260715/E97_1.3B_step1065000_async_quorum_b4k40_ladder_256n/5000436-20260715T064518Z`

Generation 0 was finalized authoritatively with 2048/2048 accepted updates,
no missing/failed/invalid/late/timed-out updates, and loss 2.4854. Its model
checkpoint is:

`async_run/checkpoints/emender_E97_100m_20260715/checkpoint_step_1525040_loss_2.4854.pt`

- size: 15,439,252,298 bytes
- SHA-256: `8e8d133f9008525cb25cfd52a27a6699f1e7c7e2318f949f660e4844d00b7290`
- generation-0 manifest: `async_run/generations/gen_000000/manifest.json`

The latest fully published checkpoint is generation 9, step 1525400, loss
2.4184:

`async_run/checkpoints/emender_E97_100m_20260715/checkpoint_step_1525400_loss_2.4184.pt`

- size: 15,439,252,298 bytes
- SHA-256: `ee9d69d9c3efd5696042b30ad1ad57236d5035876bae5ce2e9cc2010e5017fd3`
- `async_run/latest.pt` remains a symlink to this checkpoint.
- `async_run/latest.json` remains the generation-9 publication record.
- The termination wrapper recovered `continuation/last-valid.json`, a symlink
  to `last-valid-20260715T072807Z.json`, recording this generation-9 checkpoint.
  This preservation did not advance or otherwise change terminated training.

## Final heartbeat, rank, and stall evidence

- The published state stopped at generation 9; all 2048 rank heartbeats then
  entered generation 10.
- Rank 0's final heartbeat was at stage
  `compiled_mpich_helper_send_starting`, transport
  `compiled-cray-mpich-helper-collective-reduce`.
- The per-rank compiled-helper trace contains 2048/2048 markers for
  `run_once_enter`, `request_parsed`, `bridge_enter`, `mpi_initialized`, and
  `collective_reduce_complete`, but 0/2048 `collective_reduce_reduced` and
  0/2048 `return_written` markers. This is the retained stall signature: all
  ranks completed the collective-reduce call but no reduced result was
  consumed or returned.
- Durable evidence remains under `async_run/progress/`,
  `artifacts/compiled_mpich_trace/`, `logs/trainpy_async_quorum.log`, and the
  Slurm stdout/stderr paths recorded by `scontrol show job 5000436`.

## Production exclusion

The final read-only check showed production job 4980157 still pending on 256
nodes for reason `Priority`. No command in this teardown targeted it. No
`sbatch`, requeue, or resubmit command was run.
