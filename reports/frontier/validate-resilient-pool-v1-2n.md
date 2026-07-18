# Resilient DiLoCo Compute Pool v1 — two-node Frontier validation

Date: 2026-07-18

Task: `validate-resilient-pool-v1-2n`

Authority: `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, version 1, and companion
requirements R01–R16 in `docs/RESILIENT_DILOCO_GAP_MATRIX.md`.

## Status

The local pre-submit gate passed at source commit
`87355275976184898fc3fb1975473ce69952f49f`. The ordered live ladder remains
stage-gated: this report first retains the immutable startup-smoke payload; the
resilience and newer-fence restart payloads may be rendered and submitted only
after their preceding live rung passes.

No production allocation, normal-QoS allocation, 4+ node allocation, or
two-hour allocation is authorized by this report.

## Authoritative integration and queue gate

- Integrated v1 commit `ae2e6f26046fb7a6b348e845fb4615092a7c37e0` is an
  ancestor of local `HEAD` and fetched authoritative `origin/main`.
- The local lifecycle reliability fix is
  `87355275976184898fc3fb1975473ce69952f49f`; local `HEAD` and fetched
  `origin/main` were identical before payload rendering.
- `squeue -u "$USER"` returned no rows at the initial and final pre-submit
  checks. The retained submit script repeats this check and fails closed if any
  user job appears before `sbatch`.
- The superseded job `5027064` used code `8d0b6a5`, broad 900-second stage
  maxima, and failed before a first training heartbeat. This payload uses the
  integrated pool runtime and 180/420/180/720 hard stage bounds, so it is not an
  unchanged retry.

## Local pre-submit validation

The first exact suite invocation exposed an unreliable two-second healthy
Python-launch allowance in the bounded step-supervisor test: 119 passed and one
failed. The standalone reproducer failed on the loaded login node; direct cold
approved-Python launches reached 1.77 seconds before pytest/fork overhead. The
test now gives healthy process launch a bounded 30 seconds while retaining the
independent 0.2-second stuck-child eviction proof. The focused reproducer then
passed, and the exact full suite passed:

```text
/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python -m pytest -q -p no:cacheprovider tests/test_fenced_admission.py tests/test_resilient_peer_membership.py tests/test_resilient_node_quorum.py tests/test_resilient_e97_reducer.py tests/test_resilient_shard_owner.py tests/test_resilient_pool_runtime.py tests/test_resilient_e97_split_roles.py tests/test_resilient_node_transport.py tests/test_resilient_e97_runtime.py tests/test_resilient_e97_true_2n_launcher.py tests/test_async_diloco_real_trainer.py tests/test_train_helpers.py
120 passed in 220.14s
```

Additional passing gates:

- approved Python 3.12 `compileall` over the integrated runtime, launcher, and
  selected tests;
- `python -m json.tool` on the integration metrics and `git diff --check`;
- no `.tolist(`, all-reduce/all-gather/process-group/barrier/MPI collective
  token in the live integrated files;
- exact weighted/reference, changing-membership, idempotence, stale/corrupt
  rejection, owner-loss replay, newer-fence restart, trainer parity, and
  launcher topology tests within the 120-test suite;
- representative integration metrics: two retained spool files, zero
  per-microchunk files, zero retained bytes after release, no central
  full-model broker, and no Lustre tensor hot path;
- the exact `/lustre/orion/bif148/proj-shared/emender/runs` control-store parent
  passed independent-connection loser admission, atomic bundle visibility,
  monotonic fence `1 -> 2`, and stale-publication rejection. The batch repeats
  real allocation admission before any role/model load.

## Immutable identity

Runtime attested locally and is re-attested before roles start:

```json
{"executable":"/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python","python":"3.12.13","torch":"2.10.0+rocm7.1","torch_hip":"7.1.25424","triton":"3.6.0"}
```

Inputs and payload files:

| Artifact | SHA-256 |
|---|---|
| generation-0 E97 seed (7,719,679,924 bytes) | `1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9` |
| pinned p50k tokenizer | `94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069` |
| flat batch-size-4 E97 arguments | `afc2a65fd8c73499e74e21cb9531c978206c3a9c898e42d18cc58bb93eb9fe9c` |
| two-node sbatch launcher | `3c4e41ef28e6cecc42b68a68fdbc111e26ba1c304f6d785e7110ea4d49ef6ed6` |
| allocation supervisor | `75722c09b9329fd0f4741d9b0b5182403b0bd9f8cc57545f2d59e4a05dd8f88b` |
| live role entrypoint | `dde43114e04a0e0a7540c8ebd07e33a96b293d8887a1c8b31a5bb288cf57a7a6` |
| fenced admission | `4fb942149d9ebb7dc8e25300f55e612fc2c3c704c73141729781dabac7ab01bb` |
| exact weighted reducer | `1cb4c30cc23d20d4c893ac047371d441b9c61c4a70f6da5dbcda86a9ec6f743c` |
| split roles | `4ac5abff8980fb2f59606dd5ccdeaf2668c0a4d4a3b9486acdc9418baff09d0f` |
| fenced runtime | `eed968906587451b1371150e68d445f316532e6cd220db6d9e69667fef7f9c77` |
| pool owner/control runtime | `4fdae65ccd5a9ec10cc2c88acb8edbf368fd16230711414dfab1f3e54603de97` |
| exact startup submit script | `f73a8c7246c305770fab5397700af5f6e3097cdf5e11a248974277f271ee845c` |

The source seed identity is
`step-1525000-sha256-1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9`.
The startup payload identity is
`8735527-20260718T143538Z-pool-v1-startup-2n20m-k40`.

## Startup command and acceptance

The exact retained command is:

```bash
bash reports/frontier/validate-resilient-pool-v1-2n-startup-submit-20260718T143538Z.sh
```

The script resolves to one `sbatch --parsable` request for exactly two nodes,
debug QoS, `00:20:00`, 16 GPUs, 2 model-free managers, 16 real trainers, batch
size 4, K=40, global node quorum 2, one generation, no injections, no resume
handoff, node-local `/tmp` bulk/kernel/tokenizer paths, and bounded 1 MiB
chunks under a 32 GiB shared spool ledger. It refuses a nonempty queue, an
existing run directory, non-authoritative checkout, or runtime-code drift.

Run identity:
`validate-resilient-pool-v1-2n-startup-20260718T143538Z-8735527`.

Required observed stages from allocation start are READY by 180 seconds, local
K=40 by 420 seconds after trainer start, exchange/commit within a further 180
seconds, and first atomic generation by 720 seconds. TERM@300 provides the
handoff lead. Success additionally requires runtime/topology attestation,
accepted-token/weighted-manifest evidence, byte throughput/high-water/release
telemetry, no forbidden transport, and a reloadable immutable checkpoint.

## Version-1 conformance checklist

- **R01–R03:** exclusive expiring fenced admission precedes model load; two
  leased READY node managers define active membership, not 16 launched ranks.
- **R04–R07:** fresh fence/generation identities, duplicate/stale/corrupt
  rejection, exact token weighting, explicit quorum/deadlines, and one atomic
  immutable commit/checkpoint/latest bundle are live acceptance conditions.
- **R08–R10:** deterministic distributed shard owners use bounded checksummed
  point-to-point chunks and prompt release; managers are model-free; tensor,
  membership, heartbeat, spool, and redistribution hot paths are node-local or
  network, never Lustre. Only bounded durable lease/checkpoint evidence uses the
  shared run directory.
- **R11–R12:** this startup rung establishes the fenced checkpoint. Manager and
  trainer loss/rejoin plus outer-state recovery are reserved for the ordered
  resilience and fresh-allocation rungs.
- **R13–R16:** the scheduler-neutral pool is mapped through the Frontier
  adapter; stage deadlines and immutable metrics are mandatory; reference math
  passed; this two-node gate must pass before any 4+ node consumer proceeds.

Minimum startup progress floor: two complete READY node-peer contributions,
positive accepted-token weight, and one atomic committed generation. No
all-rank collective, full-model broker, Lustre tensor exchange, microfile spool,
unbounded wait, unchanged retry, or production mutation is permitted.
