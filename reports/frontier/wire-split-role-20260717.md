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
- Generation zero is pinned to the original verified step-1,525,000 seed with
  SHA256 `1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9`.
  Its absent outer state is initialized from the approved outer configuration
  and recorded as `initialized_not_restored`. Generation 9 from job 5000436 is
  old-path evidence only and is never selected as the new harness seed.
- TERM is latched by trainers and takes effect only between fully applied
  generations. The manager only finalizes a designated trainer proposal for a
  completed generation, so a partial generation cannot become the handoff.

The failure-sensitive implementation imports no MPI, RCCL, TCPStore, or
all-rank collective. The launch supervisor retains independent `srun --no-kill`
steps and restarts roles separately. It also defaults to the scale-oriented
`node-local` launch mode: one durable `srun --no-kill` step per physical node
spawns and independently supervises its CPU manager plus eight GPU children.
`independent-step` remains available for the retained two-node fault-isolation
gate. A control test proves that both backends preserve role identity, device
binding, heartbeat/progress deadlines, eviction, and restart configuration.

No live update, tensor shard, aggregate, redistribution, heartbeat, quorum, or
replay message uses the shared run directory. `--bulk-root` defaults to
node-local `/tmp/resilient-e97`; a fail-closed guard rejects roots under the
shared run tree or on Lustre/NFS/GPFS/CIFS mounts. Each local mailbox and network
replay spool has an explicit byte limit. Managers record the configured bound,
observed high-water mark, and post-release ownership locally. Shared Lustre is
limited to the initial seed read and atomically finalized checkpoint/handoff.

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

After the node-local launch and bulk-staging architecture additions, the direct
runtime/launcher control command was rerun: `8 passed in 34.71s`. This includes
both launch backends, the bounded/high-water assertions, two-manager transport,
and the eight-trainer three-generation flow.

After the pinned-seed/restart and strict no-shared-hot-path override, the
expanded split protocol command passed `30 passed in 74.33s`. It includes a
fresh-process reload of model, inner optimizer, initialized/restored outer
state, step, generation, async chain, and fencing identities, followed by a
continuation generation that exactly equals an uninterrupted three-generation
control. It also asserts the shared run tree contains only finalized
`checkpoints/` and `handoff/` files.

The complete established-plus-new focused suite was rerun after the final user
overrides: `132 passed in 238.63s (0:03:58)`.

The final control/bulk split introduces `BulkChunkStream` and its bounded
non-MPI implementation. Control JSON headers carry only identities and byte
checksums; payloads travel as separate persistent stream frames. The window is
one bounded chunk (1 MiB by default), each chunk has a fenced shard attempt and
SHA256, identical reconnects are idempotent, conflicting duplicates fail, and
the coordinator discards participant bytes immediately after incremental exact
weighted reduction while retaining only compact receipts. The local manager
likewise validates descriptors first and reads/aggregates one chunk across the
accepted quorum at a time. The direct transport/runtime suite passed
`19 passed in 75.11s` after this change.

Final established-plus-new suite result after the streamed bulk-plane change:
`133 passed in 234.79s (0:03:54)`.

The system `python3` is Python 3.6 without Torch and failed during collection;
as in the established project validation recipe, the result above supersedes it
using the pinned ROCm/Torch environment.
