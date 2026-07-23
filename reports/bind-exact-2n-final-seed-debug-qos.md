# Exact two-node final-seed and debug-QoS binding

Date: 2026-07-23
Task: `bind-exact-2n-final-seed-debug-qos`
Scope: implementation and local validation only; **no Slurm job was
submitted**.

## Outcome

The exact pipelined two-node acceptance renderer no longer contains or exports
the obsolete step-1525000 Lustre checkpoint. It loads the seed solely from
`configs/frontier/e97_async_256.yaml` and binds the rendered plan and `sbatch`
payload to:

- immutable step: `2300930`
- accepted tokens: `150793748480`
- object size: `7719680116`
- SHA256:
  `0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2`
- immutable checkpoint URI and step-manifest URI under the step-2300930 S3
  prefix, plus the reviewed discovery pointer

Every allocated node runs
`scripts/frontier/materialize_e97_s3_seed.py` before the allocation supervisor
or a model-bearing trainer is started. The materializer:

1. rejects a non-job-scoped path, a shared-filesystem path, and any existing
   job-local file;
2. cross-checks the immutable step manifest and discovery pointer against the
   canonical URI, step, loss, byte size, and SHA256;
3. streams the immutable S3 object into a temporary node-local file while
   computing its exact byte count and SHA256;
4. promotes only the verified file atomically to
   `/tmp/emender-e97-seed-$SLURM_JOB_ID/checkpoint-step-2300930.pt`; and
5. retains a per-node JSON identity record under the phase run directory.

The trainer independently checks the job-local directory, byte size, and
SHA256 before its first `torch.load`, then checks the payload step before
cloning model state. Outer-state cold-start admission now accepts an explicitly
supplied reviewed seed identity; the historical hard-coded step-1525000
fallback was removed, and cold-start initialization now fails closed if its
caller omits the approved identity.

The iterative controller requests both `-p batch` and `--qos=debug`. While a
job is queued or running it queries `squeue -o '%T|%P|%q'`; after the job
leaves the queue it queries
`sacct --format=State,ExitCode,Partition,QOS`. A missing or different
partition/QoS fails closed. Pending and terminal JSON evidence contains
separate `partition` and `qos` fields, so the default `squeue` `PARTITION`
column is never used as a QoS proxy.

The debug and production parity fixtures carry the same final immutable seed.
The exact plan records that the only authorized smoke/production differences
are node resources, QoS, walltime, and deliberate fault injection; the
launcher, native-CXI backend, eight trainers per node, K40 payload, model,
optimizer, dataset, seed, and checkpoint contract remain identical.

## Compute-pool v1 conformance

Authority: `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, version 1 (2026-07-17),
including its required conformance checklist. Applicable gap-matrix
requirements are **R07, R09, R10, R12, R14, R16** and **NDP15, NDP16,
NDP17**.

- **READY membership, bounded waits, and no launched-rank invariant:** this
  seed/queue change does not alter pool membership or contribution admission.
  The exact launcher still uses two nodes as capacity, one model-free manager
  and eight trainers per node, global `Q_min=2`, positive
  `T_min=3934080`, and the existing bounded stage deadlines. No collective or
  launched-rank membership rule was introduced (R09, R14, R16).
- **Fenced generations, exact math, rejection, and atomic evidence:** the
  native generation protocol and weighted math are unchanged. The canonical
  seed identity is now part of the rendered source identity and is verified
  before model load. Fenced immutable checkpoint/handoff publication remains
  the only authoritative continuation path (R07, R12, NDP15).
- **Bounded non-Lustre hot path and no broker:** the seed is an initialization
  input, not a dense update path. It is materialized independently into
  node-local `/tmp`; shared Lustre checkpoint reuse is rejected. Native memfd
  plus CXI contribution, aggregation, and redistribution paths are unchanged,
  and the manager remains model-free (R09, R10).
- **Failure/deadline and recovery path:** pointer/manifest disagreement,
  inaccessible authority, wrong size, wrong hash, partial download, stale
  local file, shared-path reuse, seed-config drift, partition drift, and QoS
  drift all fail closed before model admission or acceptance advancement. The
  serial controller retains its bounded `squeue`→`sacct` propagation window
  and fresh-allocation handoff phases (R12, R14, NDP15).
- **Telemetry and evidence:** each node records the seed authorities and their
  document digests, staged size/SHA, job ID, hostname, and staged path.
  Scheduler wait/terminal evidence explicitly records both Partition and QOS.
  Existing native provider, identity, byte-bound, release, and checkpoint
  telemetry remains required (R14, NDP16).
- **Ordered scale gate:** the plan remains exactly two-node and explicitly
  forbids 4/8/32/64/256-node submission. It retains the exact-code full-layout
  G2 artifact gate and native-CXI attestation. No scale or Slurm execution is
  claimed by this implementation (R16, NDP17).

## Validation

Canonical Frontier activation selected the approved Python 3.12 environment
before Python, pytest, or compile checks:

```bash
source scripts/frontier/activate_emender_frontier.sh
```

Focused renderer and seed validation:

```bash
"$EMENDER_PYTHON" -m pytest -q \
  tests/test_resilient_e97_exact_2n_acceptance.py \
  tests/test_e97_s3_seed.py
```

Result: `25 passed`.

The final validation pass ran the focused launcher/runtime/parity tests,
Python compilation, shell syntax, guide lock-step check, and whitespace check:

```bash
"$EMENDER_PYTHON" -m pytest -q \
  tests/test_resilient_e97_exact_2n_acceptance.py \
  tests/test_e97_s3_seed.py \
  tests/test_resilient_e97_true_2n_launcher.py \
  tests/test_resilient_e97_runtime.py
"$EMENDER_PYTHON" -m py_compile \
  scripts/frontier/render_resilient_e97_exact_2n_acceptance.py \
  scripts/frontier/materialize_e97_s3_seed.py \
  scripts/frontier/resilient_e97_role.py \
  ndm/resilient_e97_runtime.py
bash -n scripts/frontier/resilient_e97_true_2n.sbatch
cmp -s AGENTS.md CLAUDE.md
git diff --check
```

Result: `120 passed`. The first focused pass exposed only a missing local native
library for the retained job-5039258 reconnect fixture. The canonical native
build was therefore run rather than suppressing that test:

```bash
PYTHON_BIN="$EMENDER_PYTHON" \
  scripts/frontier/build_native_resilient_dataplane.sh
```

The build completed, all native CTests passed (`10/10`), the installed native
manifest was recorded, and the full 120-test focused pass then completed with
no skips or failures. `bash -n`, `py_compile`, byte-identical guide comparison,
and `git diff --check` also passed.

The committed report is intentionally not a live Frontier acceptance result.
It authorizes no larger rung and records no Slurm submission.
