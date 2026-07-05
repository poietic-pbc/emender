# Async DiLoCo E97 256n12h Launch Package

Task: `async-diloco-prepare-256n12h`
Date: 2026-07-05
Status: prepared only; no production Slurm job submitted by this task.

## Decision

Prepare the 256-node 12-hour async quorum DiLoCo E97 package, but keep
submission blocked on the downstream WG human-approval task
`async-diloco-submit-256n12h-human-approval`. The included Slurm wrapper,
`scripts/frontier/async_diloco_e97_256n12h_launch.sbatch`, has a hard approval
guard and exits before training unless `ASYNC_DILOCO_HUMAN_APPROVED=1` is set by
that downstream approval path.

This package is separate from the earlier non-async `e97-b4-k40-256n12h` Slurm
job `4936017`, which prior monitors found had an input-chain pointer problem and
then later hit all-rank startup fragility. Do not treat this package as
permission to submit another production job.

## Exact Target

- Training target: `E97_1.3B_step483000_async_diloco_256n12h_20260705`
- Source checkpoint:
  `/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_20260623_103742_step_483000/checkpoint_step_483000_loss_2.5431.pt`
- Source checkpoint size measured by both 32/64-node evidence files:
  `7,719,673,482` bytes.
- Output root:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/async_diloco/E97_1.3B_step483000_async_quorum_b4_k40_256n12h/`
- Script prepared:
  `scripts/frontier/async_diloco_e97_256n12h_launch.sbatch`
- Multi-node async entrypoint expected by the script:
  `scripts/frontier/async_diloco_e97_2n8n_debug.py`. This entrypoint appears
  in the 32/64-node evidence command files but is not present in this checkout;
  therefore the wrapper fails explicitly after approval if the entrypoint has
  not been restored or replaced. The downstream human-approval task must resolve
  this preflight before any production submission.

## Recipe

- Nodes: `256` Frontier nodes, one node supervisor task per node, eight GPU
  workers per node.
- Walltime: `12:00:00`.
- Expected requested node-hours: `256 * 12 = 3,072`.
- Local worker geometry: `BATCH_SIZE=4`, `CHUNK_SIZE=2048`, `DILOCO_K=40`.
- Nominal token geometry: `16,777,216` tokens per local step and
  `671,088,640` tokens per DiLoCo generation.
- Local quorum: `8/8` GPU workers per node.
- Global quorum: `240/256` node updates, matching the 93.75% quorum fraction
  used in the passing 32-node (`30/32`) and 64-node (`60/64`) evidence.
- Timeout policy: local timeout `120s`, global timeout `240s`. This is shorter
  than the 900s debug timeout because 256-node B4 K40 generations are expected
  around 1-2 minutes; a 15-minute global timeout would defeat the purpose of
  async quorum liveness.
- Staleness: reject stale updates; accepted updates must be based on the current
  open generation.
- Weighting: token-weighted accepted node deltas.
- Transport mode: `cray-mpich-gpu-aware-p2p` target with optional 16-node group
  aggregators. Dense update payloads must not use Lustre as the data plane;
  Lustre is for manifests, recovery checkpoints, and export checkpoints.
- Generation manifests: every DiLoCo generation.
- Recovery checkpoints: whichever fires first, `5` generations or `600s`.
- Export checkpoints: whichever fires first, `45` generations or `3600s`.
- Finalization buffer: initial hard buffer `1200s`; after the first real
  256-node recovery checkpoint, keep at least
  `max(900s, 3 * observed_p99_recovery_write_s + one_generation_s + 300s)`.
- Chain behavior: `CHAIN_UPDATE_ON_FAILURE=0`; failed training must not advance
  a production continuation pointer.

## Evidence Used

The dependency report artifact named in WG context was not present in this
checkout, but the machine-readable metrics artifacts are readable and sufficient:

- 32-node metrics:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/async_diloco/E97_1.3B_step483000_32n64n_config_20260705/20260705/4944228-20260705T200009Z/artifacts/async_diloco_e97_32n_metrics.json`
- 64-node metrics:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/async_diloco/E97_1.3B_step483000_32n64n_config_20260705/20260705/4944237-20260705T201032Z/artifacts/async_diloco_e97_64n_metrics.json`
- Design and local simulation evidence:
  `docs/ASYNC_QUORUM_DILOCO.md` plus the local/prototype async quorum utilities
  in `ndm/async_diloco.py`.

Measured summary:

| Evidence | Nodes | Configured global quorum | Effective global quorum | Local quorum distribution | Generation duration | Global merge duration | Checkpoint write duration | Checkpoint size in async run | Checkpoint overhead | Tokens/generation in config test | Conclusion |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 32n config job `4944228` | 32 | 30 | 30 | min/p50/p95/max all 8 | `1.2291s` | `5.9580s` | `0.000167s` | manifest `8,473B`, recovery `8,291B`, total `16,764B` | `0.01362%` | `245,760` | pass |
| 64n config job `4944237` | 64 | 60 | 60 | min/p50/p95/max all 8 | `1.2255s` | `11.0333s` | `0.000170s` | manifest `8,482B`, recovery `8,297B`, total `16,779B` | `0.01384%` | `491,520` | pass |

Both runs used the exact step-483000 source checkpoint above, induced global
lag/drop at the tail nodes, accepted quorum without local worker loss, recorded
zero stale/failed/invalid updates, wrote generation manifests and recovery
records, advanced only the debug-run latest pointer, and left the production
latest guard unchanged. The local simulation/prototype evidence establishes the
core stale-rejection, quorum advancement, durable manifest, and recovery record
schema; the 32/64-node evidence shows those mechanics scale to node-count
configuration tests.

## Checkpoint Cadence Justification

The cadence is not copied from a fixed 20-30 minute rule. It is based on the
observed checkpoint overhead and the expected 256-node generation cost.

Measured config-test writes were tiny metadata recovery records, not full
7.7GB model exports: `0.000167s` at 32 nodes and `0.000170s` at 64 nodes,
about `0.014%` overhead, with roughly `16.8KB` total manifest plus recovery
metadata. The source model checkpoint itself is `7.72GB`, so the first real
256-node full recovery/export write must be monitored separately. The low
measured overhead justifies starting with frequent recovery checkpoints, while
the difference between metadata size and full checkpoint size is the reason the
monitoring plan has an immediate checkpoint-overhead regression gate.

For nominal 256n B4 K40, each local step is `16,777,216` tokens and each
generation is `671,088,640` tokens. If 32/64-node evidence extrapolates to
roughly 1-2 minutes per generation at 256 nodes, a 20-minute recovery interval
can lose about 10-20 accepted generations, or `6.71B` to `13.42B` tokens, and
about `85.3` node-hours. A `5` generation or `600s` policy bounds the initial
loss window to roughly 5-10 minutes in the expected regime, around `3.36B` to
`6.71B` tokens, and about `21.3` to `42.7` node-hours.

Recommendation for launch: generation manifests every generation; recovery
checkpoints every `5` generations or `600s`, whichever fires first; export
checkpoints every `45` generations or `3600s`, whichever fires first; final
recovery/export checkpoint with an initial `1200s` buffer. If the first two real
256-node recovery checkpoints exceed `2%` wall overhead or `120s` write time,
switch the recovery cadence to `10` generations or `900s` only after a human
review of the added loss window.

## Expected Work

Requested budget:

- Node-hours: `3,072`.
- Gross walltime: `12h`.
- Finalization buffer: `20min`, leaving about `11.67h` of planned active
  generation time before drain.

Token estimates using the required nominal geometry:

- At 2 minutes/generation for 11.67h: about `350` generations and
  `234.9B` tokens.
- At 1 minute/generation for 11.67h: about `700` generations and
  `469.8B` tokens.
- Full 12h envelope without buffer would be about `360-720` generations and
  `241.6B-483.2B` tokens.

These are planning estimates. The monitor must report actual accepted
generations and token-weighted accepted updates from manifests, not inferred
wall-clock totals.

## Monitoring And Watchdog Plan

Start monitoring immediately after allocation and then every 10 minutes until
the first five accepted generations, every 30 minutes during stable operation,
and continuously during the final 30 minutes.

Required checks:

- Loss blowup and NaN detection: cancel if any accepted node update reports
  non-finite loss, if global loss moving average is non-finite, or if the
  generation-level loss average increases by more than 0.5 nats over the
  previous stable 10-generation window without recovery in the next two
  generations.
- Quorum collapse: page immediately if effective global quorum falls below
  `245/256` for three consecutive generations; cancel if it falls below the
  configured `240/256` threshold, if generation advancement stalls for more
  than `2 * ASYNC_GLOBAL_TIMEOUT_S`, or if more than 16 nodes are missing for
  two consecutive generations.
- Timeout/staleness: page if stale or timed-out node updates exceed 5% of
  requested nodes for three consecutive generations; cancel if stale plus
  timed-out exceeds 10% for three consecutive generations after the first five.
- Checkpoint duration/overhead regression: record write duration, checkpoint
  bytes, percent overhead, and latest advancement for every recovery/export
  checkpoint. Page if recovery write time exceeds `60s` or overhead exceeds
  `1%`; cancel or hold for human decision if write time exceeds `120s`,
  overhead exceeds `2%`, or checkpoint cadence cannot meet the 5-10 minute
  recovery objective.
- Checkpoint failure detection: cancel if any manifest or recovery checkpoint
  write fails, if `latest` points at a missing or partial path, if a recovery
  checkpoint has zero bytes, or if a `.tmp`/partial artifact remains after the
  publish window.
- Transport and merger health: page on GPU-aware MPI transport fallback to
  filesystem data plane, repeated group-aggregator failure, or merger memory
  pressure; cancel on global merger crash or unbounded backlog.
- Production pointer guard: failed training must not advance production latest.
  Verify `CHAIN_UPDATE_ON_FAILURE=0` and compare pointer/readlink/stat before
  and after the run.

Minimum monitor artifacts:

- Per-generation manifests and latest pointer status.
- Accepted/missing/stale/timed-out node counts.
- Loss moving averages and invalid-update counts.
- Generation duration, merge duration, rebase duration, checkpoint duration,
  checkpoint size, and overhead.
- Tokens per accepted generation and cumulative accepted tokens.
- Slurm state, stdout/stderr paths, run root, git commit, and environment file.

## Rollback And Cancel Criteria

Cancel rather than wait out the allocation when any of these are true:

- Source checkpoint is unreadable or does not match the exact target above.
- Job does not advance generation 0 within 15 minutes after allocation startup.
- Effective quorum cannot reach `240/256` after the configured timeout.
- Loss is NaN, Inf, or unrecoverably blows up by the criteria above.
- Checkpoint publish fails or `latest` advances to an unreadable, partial, or
  non-finalized recovery record.
- Recovery checkpoint overhead exceeds `2%` or checkpoint writes exceed `120s`
  twice in a row without human acceptance of the larger risk window.
- Dense update payloads spill to Lustre as the communication plane.
- The global merger cannot keep up and backlog grows for three generations.
- Any production chain pointer advances on failed training.

Rollback path:

1. Stop new generation admission.
2. Let in-flight accepted generation finish only if quorum has already been
   reached and checkpoint budget remains.
3. Publish a final recovery checkpoint if the finalization buffer still allows
   it; otherwise preserve the latest known-good recovery checkpoint and do not
   advance production latest.
4. Record the terminal manifest and mark the run no-go for continuation.
5. Resume only from the last finalized recovery/export checkpoint after a
   separate WG task validates the manifest and checkpoint load.

## Risk Assessment

- Transport risk: high. The 32/64-node evidence is a config/prototype path, not
  proof of production GPU-aware dense delta transport at 256 nodes. Mitigation:
  require GPU-aware MPICH transport, group aggregation, and immediate cancel on
  filesystem payload fallback.
- Checkpoint risk: medium-high. Measured checkpoint overhead is excellent for
  metadata records, but not a full 7.7GB+ production global state write.
  Mitigation: 5-generation/600s initial recovery cadence, monitor full write
  duration and overhead, and adapt only after observed 256-node writes.
- Quorum risk: medium. The prior jobs passed 30/32 and 60/64 with induced tail
  drops and no stale failures, but 240/256 may expose node-health variance.
  Mitigation: monitor effective quorum distribution and cancel before burning
  hours at threshold-only progress.
- Numerical risk: medium. Async partial quorum is not identical to full-cohort
  synchronous DiLoCo unless all workers participate; stale updates are rejected
  to keep the first production path conservative. Mitigation: token weighting,
  loss-window monitoring, and no stale-Hogwild extension in this run.
- Operational risk: high. 256 nodes for 12h is `3,072` node-hours. Mitigation:
  explicit human approval gate, short startup no-go criteria, frequent
  generation manifests, and early cancel criteria.

## Human Approval Gate

No production job was submitted by this task. The next action is not `sbatch`;
it is human review of this package and explicit resumption of
`async-diloco-submit-256n12h-human-approval`. Only that downstream task may set
`ASYNC_DILOCO_HUMAN_APPROVED=1` and decide whether to submit, revise, or reject
the package. That task must also verify or restore the multi-node async
entrypoint recorded in the 32/64-node evidence before submitting.
