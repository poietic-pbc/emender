# Direct same-allocation `train.py` RCCL restart — job 5125415

**Verdict: PASS (`full_pass=true`).** This is a deliberately narrow
**replacement research path**, not Compute Pool v1 conformance and not
async-decoupled-v2.1 conformance or scale authorization.

## Authority and applicability

Read against *Resilient DiLoCo Compute Pool*, version 1 (2026-07-17), its
required conformance checklist, and `RESILIENT_DILOCO_GAP_MATRIX.md`. The full
namespaces were reviewed: **R01–R16, NDP01–NDP17, V21S01–V21S17, and
ISP01–ISP07**. This experiment intentionally uses legacy/reference fixed-world
hierarchical `train.py` collectives, so it does not claim leased READY
membership, native point-to-point transport, v2.1 clocks/snapshots, or any
V21S/ISP closure. In particular:

- **R14 / NDP13:** every child step had a 420-second external timeout;
  the damaged step returned nonzero in 99 seconds and the allocation remained
  live. Stage timestamps, return codes, logs, and checkpoint evidence are
  retained. This is bounded failure evidence, not the missing ISP06/ISP07
  asynchronous overlap evidence.
- **R16 / NDP17:** this exact two-node direct result is a replacement research
  observation only. It does not satisfy the current-source native G2/G3/G4/G5
  chain, does not authorize 8 nodes, and creates no downstream scale task.
- **NDP13 / NDP17:** the broken RCCL communicator was destroyed with step
  `5125415.1`; recovery step `5125415.2` was a fresh process group on one node.
  No communicator shrink, failed-step tensor, or partial checkpoint was reused.
- The especially applicable replacement-path lessons are R07/R12 atomic
  checkpoint/reload, R14 bounded stages, R16 two-node evidence discipline,
  NDP02 failure-sensitive collective contrast, NDP13 containment, NDP15
  checkpoint handoff, and NDP17 qualification boundaries. The remaining R,
  NDP, V21S, and ISP requirements remain unapplied/unclaimed rather than being
  weakened by this result.

## Immutable scheduler transaction

Submitted source was commit `4bd75b0e2291202d84d4a04d60551ffed8bac0a2`,
payload digest `066e949d47b7f08987f6a307ee3b562149a7d0e1823f980dacd518427800ce1e`.
The submit wrapper sourced `scripts/frontier/activate_emender_frontier.sh` and
bound `PYTHON_BIN` to `$EMENDER_PYTHON`:
`/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python`.
All Python/preflight commands used that interpreter.

Payload 5125415 was held, scheduler-owned collector 5125416 was registered with
`afterany:5125415`, and only then was the payload released. Collector 5125416
completed `0:0`; supplemental scheduler-owned collector 5125489 retained child
step accounting after the original collector exposed that `sacct -X` omitted
steps. Scripts, command, source hashes, seed/input identity, rank logs,
stdout/stderr, checkpoint hashes, step return codes, node sets, and verdict are
under:

- run: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/direct-same-allocation/5125415/`
- terminal collection: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/direct-same-allocation/collectors/5125489/payload-5125415/`

Queued and live `squeue` records explicitly name `Partition=batch` and
`QOS=debug`. Terminal `sacct` records:

```text
5125415|COMPLETED|0:0|2|frontier[04304-04305]|batch|debug|00:09:49
5125415.0|COMPLETED|0:0|2|16|frontier[04304-04305]|00:03:18
5125415.1|CANCELLED|0:9|2|16|frontier[04304-04305]|00:01:39
5125415.2|COMPLETED|0:0|1|8|frontier04304|00:03:06
```

The collector uses `batch/normal` because Frontier permits the durable
collector to coexist independently; the actual experiment allocation is
exactly `Nodes=2`, `Partition=batch`, `QOS=debug`.

## Direct evidence

1. Step `.0`, 16 ranks on two nodes, performed multiple real hierarchical
   DiLoCo merges with 67,108,864-element buckets. It atomically published step
   489934, then an independent mmap reload verified model and optimizer state.
   SHA-256: `8d69fa8d74d31edfb75fd3e3a69d3127cd044937f8628fd94839931613de19d9`.
2. Step `.1` reloaded that checkpoint. Exactly rank 1 exited at merge 1,
   `sf_x` bucket 0 (`DILOCO_FAULT_INJECTION`, requested exit 86). Slurm killed
   the damaged child step; the supervisor observed rc 137 within 99 seconds.
   No checkpoint or `latest.pt` exists below the fault output. The batch job
   remained RUNNING on both original nodes with `batch|debug`.
3. Step `.2` launched immediately in the same allocation on `frontier04304`
   with a deliberately reduced world, 8 ranks instead of 16, and a fresh
   `MASTER_PORT`/process group. It reloaded step 489934, completed 15 finite
   64M-bucket hierarchical merges, atomically published/reloaded step 489949,
   and exited 0. SHA-256:
   `969681034bc7115879ba6917fd6e519dec82bbd1892beb55118966ab67847212`.
4. Recorded recovery downtime was 0 whole seconds; discarded work was one
   local step. No old communicator or fault-output state was used.

## Machine verdict

```json
{"allocation_survived":true,"failed_step_bounded":true,"fresh_srun_launched":true,"world_size_changed":true,"checkpoint_reloaded":true,"post_relaunch_merge_passed":true,"full_pass":true}
```

Authoritative complete verdict: `5125415/verdict.json`. Earlier payloads were
not retries of unchanged executed training: 5125259/5125279 never started;
5125292 stopped before a child/model load due missing pre-load plugin ordering;
5125325 supplied a successful baseline but fail-closed when its injection label
matched no ScheduleFree collective. Each subsequent submission changed payload
bytes and was durably collected. No unchanged failed payload was retried.

## Validation commands

```bash
source scripts/frontier/activate_emender_frontier.sh
bash -n scripts/frontier/direct_same_allocation_trainpy_restart.sbatch \
  scripts/frontier/direct_same_allocation_collector.sh \
  scripts/frontier/submit_direct_same_allocation_trainpy_restart.sh
"$EMENDER_PYTHON" -m pytest -q \
  tests/test_diloco_merge.py::test_diloco_fault_injection_exits_only_exact_collective \
  tests/test_diloco_merge.py::test_bucketed_schedulefree_merge_avoids_scalar_collectives
PYTHON_BIN="$EMENDER_PYTHON" "$EMENDER_PYTHON" -m py_compile train.py
sacct -j 5125415,5125489 -P \
  --format=JobIDRaw,JobName,State,ExitCode,DerivedExitCode,NNodes,NTasks,NodeList,Partition,QOS,Start,End,Elapsed
```
