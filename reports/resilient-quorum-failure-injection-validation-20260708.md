# Resilient Quorum Failure-Injection Validation - 2026-07-08

Task: `validate-resilient-quorum-failure-injection`

## Scope

This validation used automated tests/simulations only. No Slurm job was submitted from this task, including no 1n debug job and no 8n/64n/256n scale job.

## Evidence

- Added `tests/test_async_diloco_failure_injection.py`.
- Missing/nonjoining and stuck ranks: `test_failure_injection_missing_and_stuck_ranks_advance_without_unanimity` injects a 4-rank dense quorum with only ranks 0 and 1 submitting, ranks 2 and 3 marked timed out, quorum 2. It asserts `quorum_status == "advanced"`, `accepted_updates == 2`, `timed_out_updates == 2`, and `timed_out_ranks == (2, 3)`.
- Late old-base update policy: `test_late_base_generation_policy_accepts_current_and_rejects_old_with_metrics` submits one update from old `base_generation=1`/`staleness=1` plus two current `base_generation=2` updates for generation 2. It asserts only current updates are accepted and the late old-base rank is counted in `stale_updates`/`stale_ranks`.
- Stale worker catchup: `test_stale_worker_catchup_loads_latest_rebases_and_resets_base_generation` writes a run-local latest generation, has a behind worker observe `global_generation=5`, load the current state, preserve its local displacement by rebasing, and reset the next update base generation to 5. The test asserts the catchup metrics/log payload, not only source shape.
- Run-local latest behavior: `test_run_local_latest_is_isolated_from_production_latest_guard` proves a debug/run-local latest pointer advances under the run directory while a simulated production `latest.pt` symlink is unchanged and no top-level production latest is created.
- Strict collective availability: `test_resilient_dense_transport_and_strict_collective_paths_are_both_present` keeps the nonblocking resilient dense transport and compiled MPICH strict `MPI_Reduce` path both under test.

## Validation Checklist

- Failure-injection tests demonstrate quorum advance without all ranks present: covered by missing/stuck dense quorum test.
- Catchup behavior is evidenced with metrics/logs, not only code inspection: covered by catchup payload assertions.
- No old blocking collective is required in resilient mode for missing/stuck ranks: covered by importable dense quorum path returning with only quorum envelopes and timed-out ranks, with filesystem live quorum disabled.
- Existing strict collective path remains available and tested: covered by strict helper path assertion and existing compiled MPICH tests.
- If a 1n debug run is submitted, job id/artifacts/rank metrics/checkpoint behavior are reported: not applicable; no Slurm run submitted.
- No 8n/64n/256n Slurm jobs are submitted: satisfied; no `sbatch` command was run from this task.
