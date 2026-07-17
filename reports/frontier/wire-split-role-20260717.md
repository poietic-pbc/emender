# Split-role resilient E97 runtime wiring — 2026-07-17

## Scope and safety

This change wires the retained two-node topology to concrete CPU-manager and
GPU-trainer entrypoints. No Slurm command was submitted or used to mutate a job,
and no production checkpoint pointer was read or changed. Validation was local
and control-plane only.

## Runtime contract

- `scripts/frontier/resilient_e97_role.py manager` owns no model, optimizer, or
  dataset. It accepts six of eight fenced trainer contributions, validates the
  source/layout/checksum/finite constraints, computes the token-weighted mean,
  releases consumed trainer mappings, exchanges the node aggregate through the
  non-MPI framed manager transport, and publishes the accepted aggregate.
- `scripts/frontier/resilient_e97_role.py trainer` is the only model-bearing
  role. Production mode requires a verified seed and approved argument/data
  inputs, forces ScheduleFree, runs exactly 40 local steps, publishes tensor
  deltas, validates and applies the aggregate, and advances step/generation/loss.
- The designated node-0/trainer-0 serializes model and inner-optimizer state.
  The node-0 manager validates it and atomically publishes the immutable handoff
  manifest with byte count, SHA256, chain, membership, fences and identities.
- A seed without outer state fails closed. The sole accepted migration string,
  `initialize-zero-from-verified-generation-9`, records
  `initialized_not_restored`; it never represents absent state as restored.
- TERM is latched by trainers and takes effect only between fully applied
  generations. The manager only finalizes a designated trainer proposal for a
  completed generation, so a partial generation cannot become the handoff.

The failure-sensitive implementation imports no MPI, RCCL, TCPStore, or
all-rank collective. The launch supervisor retains independent `srun --no-kill`
steps and restarts roles separately.

## Validation evidence

Project runtime:

```text
/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python
```

Focused command:

```text
python -m pytest -q \
  tests/test_resilient_e97_runtime.py \
  tests/test_resilient_e97_split_roles.py \
  tests/test_resilient_node_transport.py \
  tests/test_resilient_node_quorum.py \
  tests/test_resilient_e97_true_2n_launcher.py \
  tests/test_resilient_e97_topology.py
```

Result: `32 passed in 38.42s`.

The end-to-end test launches eight independent trainer processes and one
model-free manager through three generations. A second control launches two
independent managers and twelve trainers, exercises the framed cross-manager
transport, and verifies the frozen two-manager membership. Existing focused
tests cover stale epochs, duplicates, corrupt and nonfinite payloads, missing
members, heartbeat eviction, replay/catch-up, bounded spool ownership, apply
identity/deadline, and independent child termination.

Syntax and hygiene commands:

```text
python -m compileall -q ndm scripts/frontier/resilient_e97_role.py
bash -n scripts/frontier/resilient_e97_true_2n.sbatch
git diff --check
```

Result: all passed with no output.

The established resilient suite was then rerun with the four new split-role
files appended to the prior 115-test command. Result:

```text
130 passed in 197.05s (0:03:17)
```

The system `python3` is Python 3.6 without Torch and failed during collection;
as in the established project validation recipe, the result above supersedes it
using the pinned ROCm/Torch environment.
