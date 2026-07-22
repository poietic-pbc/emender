# Retained two-node rank-level evidence

The executable regression `tests/test_resilient_e97_topology.py` produces the
16-rank topology/certificate surface.  The authoritative generation,
checkpoint, stale/corrupt rejection, weighted aggregation, and whole-node loss
evidence is produced in pytest temporary directories by the exact commands in
`reports/implement-rank-level-validation.md`; those tests validate atomic file
contents before teardown.  This directory is the durable index tying that
evidence to test names and commit history without pretending local simulation
is a live Frontier allocation.

- failing-first specification commit: `70ea99fd`
- topology: `test_rank_topology_certificate_enumerates_exactly_sixteen_independent_identities`
- rank containment: `test_exhausted_trainer_is_retired_without_retiring_manager_or_siblings`
- generation/certificate: `tests/test_resilient_pool_runtime.py`
- checkpoint/fencing: `tests/test_resilient_e97_runtime.py`
- unequal-weight reference: `tests/test_resilient_e97_reducer.py`
- Slurm policy: `tests/test_resilient_e97_true_2n_launcher.py`
