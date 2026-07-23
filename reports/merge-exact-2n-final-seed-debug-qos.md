# Exact two-node final-seed/debug-QoS authoritative integration

Date: 2026-07-23
Task: `merge-exact-2n-final-seed-debug-qos`

## Outcome

The evaluated exact-two-node final-seed and debug-QoS fix is integrated on top
of the fetched authoritative remote history. The old `origin/main` was
`4e35a83ab3d9bed6d950daff72034da1a45fe011`.

The reviewed implementation commit was
`e1adf7b58becdb70e9925d59ce24e7e685b1d7ad`. WG had already produced the
tree-identical squash commit
`9324af364cdf421b1fcea29d68f45f496c82336b`; the two commits have the same
parent, and `git diff e1adf7b5 9324af36` was empty. Integration therefore used
the established reviewed-ancestry pattern:

```text
9453c256 merge: reconcile authoritative main for exact 2n seed integration
1b65b798 merge: retain reviewed exact 2n seed ancestry
```

The second merge is intentionally content-neutral. It makes the exact reviewed
SHA an explicit ancestor without reapplying or modifying its patch. Both the
old remote SHA and the evaluated SHA pass `git merge-base --is-ancestor`
against the integrated history. The final evidence commit, pushed
`origin/main` SHA, `git ls-remote` value, and remote ancestry checks are
recorded in the task log after publication because a Git commit cannot embed
its own content-derived SHA.

No Slurm command was executed. In particular, this task ran no `sbatch`,
`srun`, `salloc`, `scancel`, or scheduler-mutating command.

## Exact seed and materialization contract

The canonical seed is loaded from
`configs/frontier/e97_async_256.yaml` and is identical in the debug and
production parity fixtures:

- immutable step: `2300930`
- accepted tokens: `150793748480`
- object size: `7719680116`
- SHA256:
  `0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2`
- immutable checkpoint and step-manifest URIs under the step-2300930 S3
  prefix, cross-checked against the reviewed discovery pointer

The exact implementation/config paths were scanned for `1525000`,
`step_1525000`, `step-1525000`, and `checkpoint-step-1525000`; the scan was
empty. The negative regression assertion in the test suite is not a runtime
hard-code.

The materialization tests exercise the canonical fail-closed flow: authority
agreement, atomic streaming download, exact byte/SHA verification, rejection
of stale or shared-filesystem targets, and a per-node runtime identity record.
The launcher constrains the destination to
`/tmp/emender-e97-seed-$SLURM_JOB_ID`, invokes the materializer on every
allocated node before any model-bearing role, and retains evidence under the
phase run directory. The trainer independently verifies job scoping, size,
SHA, and checkpoint step before load.

An actual 7.7-GB checkpoint download was intentionally not performed from a
login-node integration task: production materialization requires a real
`SLURM_JOB_ID` and a verified node-local mount. The unchanged materialization
tests cover the complete byte flow without weakening that admission rule.

## Queue and parity contract

The safe dry-run renderer produced
`build/merge-exact-2n-final-seed-debug-qos/exact-2n-plan.json`. Every phase is
exactly two nodes, K40, step 2300930, `partition=batch`, and `qos=debug`. The
plan explicitly forbids 4, 8, 32, 64, and 256-node submissions.

The submit argv remains:

```text
-N 2 -p batch --qos=debug
```

Queued/running evidence is queried as:

```text
squeue -h -j JOB_ID -o '%T|%P|%q'
```

Terminal accounting evidence is queried as:

```text
sacct -n -X -j JOB_ID --format=State,ExitCode,Partition,QOS -P
```

Both paths retain separate `partition` and `qos` fields and fail closed unless
they are exactly `batch` and `debug`. The test matrix also proves that the
single-job debug-QoS controller submits at most one phase, preserves resumable
pending evidence, tolerates only a bounded `squeue`-to-`sacct` propagation
window, and never treats the default `squeue` PARTITION column as QoS.

The direct debug/production parity checker returned `ok=true`, with no missing
required field and no forbidden difference. The allowlisted differences were
only nodes, QoS, walltime, and deliberate debug fault injection. Launcher,
model, dataset, optimizer, seed, K40, trainers/managers, quorum, transports,
checkpoint contract, deadlines, code identity, and payload identity remained
identical.

## Validation

All Python, pytest, native build, and renderer commands followed the canonical
Frontier activation:

```bash
source scripts/frontier/activate_emender_frontier.sh
```

The activated interpreter was Python 3.12.13 at
`$EMENDER_PYTHON`. Canonical native build, install, artifact attestation, and
CTest were:

```bash
PYTHON_BIN="$EMENDER_PYTHON" BUILD_JOBS=8 \
  bash scripts/frontier/build_native_resilient_dataplane.sh
```

Result: build and install passed; canonical CTest **10/10 passed**; the native
artifact manifest was recorded.

The unchanged focused/canonical selection covered final-seed materialization,
exact rendering, launcher and parity, explicit queue evidence, checkpoint and
fresh-process restart, native pipeline behavior, quorum/runtime boundaries,
and strict overlap/idle/cadence validation:

```bash
export TMPDIR=/tmp
"$EMENDER_PYTHON" -m pytest -q -p no:cacheprovider \
  tests/test_e97_s3_seed.py \
  tests/test_resilient_e97_exact_2n_acceptance.py \
  tests/test_resilient_e97_true_2n_launcher.py \
  tests/test_resilient_e97_runtime.py \
  tests/test_native_pool_integration.py \
  tests/test_native_pipeline.py \
  tests/test_validate_pipelined_e97_performance.py \
  tests/test_resilient_e97_topology.py \
  tests/test_resilient_node_quorum.py \
  tests/test_resilient_pool_runtime.py
```

Result: **183 passed in 151.23 seconds**, with no skips or failures.
`TMPDIR=/tmp` keeps AF_UNIX socket paths within the Linux length bound in the
long WG-managed worktree.

Safe renderer, parity, syntax, guide lock-step, and whitespace checks were:

```bash
"$EMENDER_PYTHON" \
  scripts/frontier/render_resilient_e97_exact_2n_acceptance.py \
  --repo "$PWD" \
  --native-build-manifest \
    build/native-resilient-dataplane/native-artifacts.json \
  --full-layout-gate \
    /lustre/orion/bif148/scratch/erikgarrison/emender/reports/frontier/native-dataplane/5033120/full-layout-gate.json \
  --run-root \
    build/merge-exact-2n-final-seed-debug-qos/runs \
  --output \
    build/merge-exact-2n-final-seed-debug-qos/exact-2n-plan.json \
  --allow-non-authoritative-dry-run
"$EMENDER_PYTHON" scripts/frontier/check_resilient_e97_parity.py \
  --debug configs/frontier/e97_resilient_debug_rendered.json \
  --production configs/frontier/e97_resilient_production_rendered.json \
  --output \
    build/merge-exact-2n-final-seed-debug-qos/rendered-parity.json
"$EMENDER_PYTHON" -m py_compile \
  scripts/frontier/render_resilient_e97_exact_2n_acceptance.py \
  scripts/frontier/materialize_e97_s3_seed.py \
  scripts/frontier/resilient_e97_role.py \
  ndm/resilient_e97_runtime.py \
  scripts/frontier/validate_pipelined_e97_performance.py
bash -n scripts/frontier/resilient_e97_true_2n.sbatch
cmp AGENTS.md CLAUDE.md
git diff --check
```

Result: the no-submit renderer and parity checker passed, `py_compile` and
`bash -n` passed, `git diff --check` passed, and `AGENTS.md`/`CLAUDE.md` were
byte-identical at 4,155 bytes with SHA256
`03df3144163ca85584ae598a64a54c4c09b705a81fb1be34cc04e1bf4a2706da`.

## Compute-pool v1 conformance

This integration applies the mandatory checklist in *Resilient DiLoCo Compute
Pool*, version 1 (2026-07-17), and the companion gap matrix. Applicable
requirements are **R07, R09, R10, R12, R14, R16** and **NDP15, NDP16,
NDP17**.

- **R07 / R12 / NDP15:** the final immutable seed is verified before model
  load; the exact acceptance sequence retains fenced atomic global
  checkpoint/latest publication, reviewed outer-state initialization, and
  fresh-allocation restart from the prior authoritative handoff.
- **R09 / R10:** the manager remains model-free. Seed staging is node-local
  initialization rather than a dense hot path; native memfd/CXI contribution,
  aggregation, and redistribution remain non-Lustre and broker-free.
- **R14 / NDP16:** materialization and scheduler evidence bind explicit
  identities, sizes, digests, Partition, QOS, stage deadlines, and retained
  telemetry. The strict validator continues to require background overlap,
  less than 10% foreground control-plane idle, and at most 1.25x cadence when
  background work fits.
- **R16 / NDP17:** retained exact-code full-layout G2 evidence remains the hard
  predecessor. This integration authorizes only the next exact two-node
  validation; it neither executes that rung nor authorizes 4+ nodes.

READY membership, bounded waits, current-fence generation identities,
deterministic token-weighted math, stale/corrupt rejection, bounded
point-to-point transport, prompt release, and no launched-rank/all-rank
invariant remain unchanged. The minimum floor remains two READY managers,
`Q_min=2`, and `T_min=3934080` accepted tokens.
