# Monitor: 256-node K40 E97 production probe

Task: `monitor-256-node-k40`
Job: `4936017` (`e97-b4-k40-256n12h`)
Checked: `2026-07-03T14:34:50Z` (`2026-07-03T10:34:50-04:00`)

## Status

- Slurm state: `PENDING`
- Pending reason: `Priority`
- Runtime elapsed: `00:00:00`
- Time limit: `12:00:00`
- Requested nodes: `256`
- Requested node-hours: `3072.0`
- No stdout/stderr files existed yet at the Slurm-recorded paths, consistent with the job not having started.

`squeue` reported an estimated start time of `2026-07-04T03:54:00`, while `sacct` still showed `Start=Unknown`.

## Checkpoint Inputs

The job was submitted with:

- `RESUME_CHECKPOINT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt`
- `CHAIN_LATEST_PATH=/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k40_256n_hier_g4_bucket64m_avg_12h/latest.pt`
- `CHAIN_MANIFEST_PATH=/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k40_256n_hier_g4_bucket64m_avg_12h/latest.pt.manifest.json`

Findings:

- The required input `latest.pt` did not exist.
- The input chain directory `E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg` did not exist.
- The output chain directory `E97_1.3B_step489920_b4_k40_256n_hier_g4_bucket64m_avg_12h` did not exist yet.
- No matching E97 `step489920` K80/K40 chain directory was found under `/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains`.

This is a pre-start blocker for the submitted job. The wrapper script checks:

```bash
[[ -r "$RESUME_CHECKPOINT" ]] || { echo "RESUME_CHECKPOINT is not readable: $RESUME_CHECKPOINT" >&2; exit 4; }
```

Because `RESUME_CHECKPOINT` is explicitly exported to the missing `latest.pt`, the script will not fall back to `CHAIN_SEED_CHECKPOINT`. If the path remains missing when the allocation starts, the job is expected to fail before training with exit status `4`.

## Runtime Health Checks

Because the job had not started:

- Loss blowup: not assessable; no training metric lines exist.
- Non-finite loss: not assessable; no training log exists.
- RCCL/NCCL failures: not observed; no runtime log exists.
- Checkpoint/finalization: not observed; no output chain pointer exists.

## Token Accounting

Actual processed tokens as of this pass: `0`, because the job has not started.

Configured token geometry from the Slurm export and script:

- Nodes: `256`
- Tasks per node: `8`
- World size: `2048` ranks
- `BATCH_SIZE=4`
- `CHUNK_SIZE=2048`
- Tokens per training step, assuming each rank consumes `BATCH_SIZE * CHUNK_SIZE`: `2048 * 4 * 2048 = 16,777,216`
- `DILOCO_K=40`, so one outer synchronization covers `671,088,640` tokens by that same accounting.
- `SAVE_EVERY=160`, so each save interval covers `2,684,354,560` tokens by that same accounting.

## Recommendation

Do not submit or cancel anything from this monitor pass. Before the job starts, a human or authorized operator should either create/restore the required input chain pointer or explicitly decide what to do with the pending allocation. Without that intervention, the current submission is expected to consume queue placement and then exit immediately on the unreadable `RESUME_CHECKPOINT`.
