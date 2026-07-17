# Leased peer lifecycle validation

This implementation conforms to the **Resilient DiLoCo Compute Pool version 1**
authority in `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, specifically R02, R03,
R11, and R14 from `docs/RESILIENT_DILOCO_GAP_MATRIX.md`.

The lifecycle artifact is `ndm/resilient_peer_membership.py`. Its active
snapshot contains only unexpired `READY` incarnations whose validated base
generation equals the requested open generation. It accepts no allocation,
launch-size, rank, MPI, Slurm, tensor, or filesystem-payload input. Thus it
introduces no all-rank collective/barrier, Lustre hot path, or full-model
broker.

Focused tests in `tests/test_resilient_peer_membership.py` cover bounded slow
boot and sync, late admission to the next generation, lease expiry, stable-ID
rejoin under a new incarnation, stale-incarnation rejection, catch-up before
readmission, draining, renewal, and active snapshot membership.

Validation commands:

```sh
python3.11 -m compileall -q ndm/resilient_peer_membership.py tests/test_resilient_peer_membership.py
python3.11 -m pytest -q tests/test_resilient_peer_membership.py tests/test_resilient_node_quorum.py
git diff --check
```

On Frontier at implementation time, compileall and `git diff --check` pass.
The pytest command is the exact required focused suite, but Python 3.11 lacks
the pytest module; the system `pytest` uses unsupported Python 3.6 and lacks
the project's torch dependency. This environment limitation is not recorded
as a test pass.
