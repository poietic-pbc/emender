# E97 1.3B K160 Scale Gate Override - 2026-06-26

WG context: `run-e97-1p3b-k160-16n`, `synthesize-e97-k160-16`

## Decision

Continue the K160 node-doubling scale ladder under a blowup/systems gate.

The 16-node K160 run completed cleanly, finalized a valid consensus checkpoint,
and did not show a loss blowup. Its 500/1000/2000 local-step training-loss
window deltas were small mixed movements over a short 50-minute probe:

- 500 local steps: `-0.010672`
- 1000 local steps: `+0.007633`
- 2000 local steps: `+0.009275`

Those deltas are not sufficient evidence to stop a scale experiment whose goal
is to discover whether the no-DDP singleton-GPU-island path remains stable and
throughput-effective as node count doubles. The quality gate should catch clear
failures: non-finite loss, large sustained loss blowup, bad finalization,
collective/runtime failures, pathological sync cost, or severe throughput
collapse. Small short-window loss changes should be recorded and compared, not
used as a hard stop.

## Superseded Interpretation

`docs/FRONTIER_E97_1P3B_PRETRAINED_K160_16N_SYNTHESIS_20260625.md` recorded a
conservative no-go decision for 32-node continuation. That interpretation is
superseded by this human-reviewed scale-gate policy.

The larger deterministic eval remains secondary context. It is useful evidence
for choosing between recipes, but it is not a stop condition for this K160
systems scaling ladder unless paired with blowup or clear training instability.

## Continuing Source

The 32-node rung should resume from the 16-node final checkpoint:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_k160_16n/4902278-20260625T213328Z/train/emender_E97_1.3B_20260625_173454/latest.pt
```

## Gate For Remaining Rungs

Continue each rung if:

- Slurm job completes or otherwise produces analyzable training output.
- Loss remains finite with no large sustained blowup versus the predecessor.
- Final checkpoint and `latest.pt` are valid, or any finalization failure is
  clearly understood and recoverable.
- Topology remains singleton GPU islands with no DDP:
  `DILOCO_ISLAND_SIZE=1`, avg outer, K160, export basis `x`.
- Throughput and DiLoCo sync cost are recorded; pathological collapse should
  stop the ladder.

Do not stop solely because a 500/1000/2000-step averaged loss window moves by a
small amount over a short one-hour probe.
