# Frontier E97 GPU-Island No-DDP Synthesis

Date: 2026-06-25
Task: `synthesize-e97-gpu`

## Decision

Do not create a later 16-node or 32-node GPU-island/no-DDP task from the current
evidence.

The no-DDP GPU-island path is operationally viable: the audit showed
`DILOCO_ISLAND_SIZE=1` avoids within-island DDP wrapping/all-reduce, the 4-node
probe completed cleanly, and the 8-node probe also completed cleanly. The
quality signal does not justify escalation, though. The 4-node K80 row improved
strongly versus the 16-node avg source, but the 8-node K80 row regressed on the
same fixed source-vs-candidate gate. That makes the current no-DDP K80 avg
recipe a negative scale signal rather than a 16/32-node candidate.

Blocker: the 8-node GPU-island/no-DDP K80 continuation from the 16-node avg
source failed fixed eval with CE delta `+0.16329194` and BPB delta
`+0.06793899` versus gates of `<= +0.025` CE and `<= +0.010` BPB. It is also
worse than the best same-source node-island DDP evidence, the 32-node K80 avg
row, which still failed but only by CE `+0.07406617` and BPB `+0.03081585`.

Because the answer is no, there is no recommended 16-node or 32-node no-DDP
configuration to run next. A future no-DDP task would first need a smaller
bounded recipe investigation, such as an 8-node cadence or averaging-strength
bracket, that clears the same fixed source-vs-candidate gate before spending
16/32-node allocations.

## Evidence Compared

All same-source rows below use the same 16-node avg source checkpoint and the
same saved-basis fixed eval tensor. Negative deltas are improvements versus the
source; positive deltas are regressions.

| Track | Nodes | Topology | K / outer | Final train loss | Fixed CE delta | Fixed BPB delta | Interpretation |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| 16-node avg source | 16 | node-island DDP, 16 islands x 8 GPUs | K10 avg | `5.2531` | `0.00000000` | `0.00000000` | Common source checkpoint and baseline. |
| 32-node node-island DDP original | 32 | 32 islands x 8 GPUs | K10 avg | `5.8701` | `+0.18676377` | `+0.07770465` | Reproduced regression class. |
| 32-node node-island DDP retry | 32 | 32 islands x 8 GPUs | K10 avg | `5.8164` | `+0.17589760` | `+0.07318369` | Reproduced regression class. |
| 32-node node-island DDP K40 | 32 | 32 islands x 8 GPUs | K40 avg | `5.2822` | `+0.13129998` | `+0.05462847` | Cadence helped but still failed fixed gate. |
| 32-node node-island DDP K80 | 32 | 32 islands x 8 GPUs | K80 avg | `5.1084` | `+0.07406617` | `+0.03081585` | Best same-source 32-node row, still not green. |
| 4-node GPU-island no-DDP | 4 | 32 islands x 1 GPU | K80 avg | `5.2398` | `-0.51040077` | `-0.21235658` | Strong low-node positive signal. |
| 8-node GPU-island no-DDP | 8 | 64 islands x 1 GPU | K80 avg | `5.5594` | `+0.16329194` | `+0.06793899` | Clean systems result but failed quality gate. |

The no-DDP 4-node row is the only green same-source candidate row in this
comparison, but it did not survive the next bounded scale step. The 8-node
no-DDP row is not strictly the worst result in the whole table: it is slightly
better than the K10 32-node DDP failures on fixed deltas. That does not make it
promising, because the current node-island DDP ladder already found a better
same-source 32-node point at K80, and even that point remains below the
promotion gate.

## GPU-Island No-DDP Findings

The audit established that `--diloco` sets normal DDP off, and
`DILOCO_ISLAND_SIZE=1` skips the hybrid DiLoCo DDP wrapping branch. The
remaining communication is initialization, rank-0 initial-weight broadcast,
barriers, and periodic global model-weight averaging, not per-step within-island
gradient all-reduce.

The 4-node probe validated the intended path with 32 ranks/GPU islands,
`DILOCO_K=80`, stateless avg outer, `BATCH_SIZE=1`, `CHUNK_SIZE=2048`, and the
16-node avg source checkpoint. It completed cleanly, wrote valid final/latest
checkpoints, recorded `DILOCO_MERGES=16`, and passed fixed eval with large
negative deltas.

The 8-node probe used the same controlled family with 64 ranks/GPU islands. It
also completed cleanly and confirmed no `[DDP] wrapped model` or
`[DiLoCo-hybrid]` markers, but its training loss trended worse over the short
window and fixed eval failed. This means the operational no-DDP machinery works,
while the optimization behavior is not robust enough to scale as-is.

## Node-Island DDP Comparison

The node-island DDP ladder has a clearer scale-control signal than the no-DDP
track. K10 at 32 nodes regressed twice. K40 repaired much of the train-loss
dynamics but still failed fixed eval. K80 improved further and is currently the
best same-source 32-node avg row, but it still misses the fixed-eval gate.

Against that ladder, GPU-island/no-DDP is not a better escalation candidate:

- it does remove within-island DDP communication and changes the optimization
  geometry from 8-GPU islands to singleton GPU islands;
- it produced one strong low-node result at 4 nodes;
- it failed at 8 nodes with fixed-eval deltas worse than the best DDP K80 row;
- it gives no evidence that adding more singleton islands at 16 or 32 nodes
  will improve optimization.

The most conservative interpretation is that no-DDP changes the dynamics in a
way that can help at 32 singleton islands but degrades at 64 singleton islands
under K80 avg. Without a green 8-node result, 128 or 256 singleton islands would
be an unjustified spend.

## Schedule-Free Context

Schedule-free `sfsgd_y` probes are useful context but not a replacement for this
decision. The 4/8/16/32-node schedule-free rows are from-scratch runs with
coherent schedule-free outer state, not continuations from the 16-node avg
source checkpoint. The 32-node schedule-free from-scratch row completed cleanly
and scored CE `4.85214365` / BPB `2.01877561`, but that is same-track
schedule-free health evidence, not a same-source no-DDP or avg-ladder candidate.

The same-source schedule-free ladder remains blocked by the non-`avg`
outer-state resume guard: the common avg source checkpoint lacks
`diloco_outer_state`, so a conforming `sfsgd` continuation cannot be launched
without changing semantics. Therefore schedule-free should remain a separate
comparison arm and should not be used to justify a no-DDP 16/32-node task.

## Recommended Next Action

No 16-node or 32-node GPU-island/no-DDP training task should be added now.

If the project wants to keep investigating the no-DDP idea, the next work should
be a design or small-probe task, not a scale escalation. Its blocker to resolve
should be:

```text
The 8-node GPU-island/no-DDP K80 avg continuation regressed on fixed eval
versus the 16-node avg source; find a no-DDP recipe that passes the same
8-node fixed gate before considering 16/32 nodes.
```

A reasonable bounded investigation would keep E97-MLP, the same source
checkpoint, `DILOCO_ISLAND_SIZE=1`, stateless avg outer, export basis `x`,
`BATCH_SIZE=1`, `CHUNK_SIZE=2048`, and the same fixed eval tensor, while varying
only one scale-control knob such as K or averaging strength at <=8 nodes. That
follow-up is deliberately not a 16/32-node task.

## Scope Confirmations

- No Slurm jobs were submitted from this synthesis task. I only read existing
  reports, logs, and WG task state.
- `run-64-node-e97` remains `open (PAUSED)`.
- This synthesis preserves E97-MLP scope. It does not recommend GDN2, CMAES,
  schedule-free continuation from the avg source, 64-node work, or any
  uncontrolled-source variant.

## Source Notes

- GPU-island/no-DDP audit semantics:
  `docs/FRONTIER_E97_GPU_ISLAND_AUDIT_20260625.md`.
- 4-node no-DDP result and fixed deltas:
  `docs/FRONTIER_E97_GPU_ISLAND_4NODE_PROBE_20260625.md:8`,
  `docs/FRONTIER_E97_GPU_ISLAND_4NODE_PROBE_20260625.md:139`.
- 8-node no-DDP decision and fixed deltas:
  `docs/FRONTIER_E97_GPU_ISLAND_8NODE_PROBE_20260625.md:8`,
  `docs/FRONTIER_E97_GPU_ISLAND_8NODE_PROBE_20260625.md:147`.
- 32-node K10 fixed-eval regression:
  `docs/FRONTIER_E97_32NODE_FIXED_EVAL_20260624.md:64`.
- 32-node K40 fixed-eval row:
  `docs/FRONTIER_E97_32NODE_K40_DIAGNOSTIC_20260624.md:230`.
- 32-node K80 fixed-eval row:
  `docs/FRONTIER_E97_32NODE_K80_DIAGNOSTIC_20260625.md:213`.
- Schedule-free separation and non-unblocking status:
  `docs/FRONTIER_E97_SCHEDULEFREE_SCALE_SYNTHESIS_20260625.md:8`,
  `docs/FRONTIER_E97_32NODE_SCHEDULEFREE_FROM_SCRATCH_DIAGNOSTIC_20260625.md:8`.
