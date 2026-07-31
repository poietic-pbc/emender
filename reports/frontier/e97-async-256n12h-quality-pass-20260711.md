# E97 async 256-node smoke/production task quality pass

Date: 2026-07-11

Task: `quality-pass-e97`

Source audit:
`reports/frontier/e97-async-256n12h-smoke-production-parity-plan-20260711.md`
(audit commits `f95353b`, `948e424`).

## Verdict

The downstream graph is tightened to the only safe sequence:

`audit-e97-async-256` -> `quality-pass-e97` ->
`implement-canonical-e97` -> `run-exact-256-node` ->
`promote-exact-successful-256`.

The permitted profile difference is exactly job duration (Slurm walltime and
its deterministic trainer stop budget) plus queue selection (partition and
QoS). Account and reservation are not queue fields. Every other resolved
field and artifact must be canonically identical. The legacy tasks
`submit-and-monitor` and `retry-refreshed-e97` remain paused and are explicitly
superseded; neither is production authority.

## BEGIN implement-canonical-e97
## Objective
Implement the audit plan at
`reports/frontier/e97-async-256n12h-smoke-production-parity-plan-20260711.md`.
Replace the divergent launch paths with one immutable, content-addressed E97
async bundle rendered from `configs/frontier/e97_async_256.yaml`, with strict
policy in `configs/frontier/e97_async_256_parity_policy.json`. Implement
`scripts/frontier/render_e97_async_256.py` and
`scripts/frontier/check_e97_async_promotion.py`; update submission entrypoints
so no production `sbatch` is reachable without the exact successful-smoke
fingerprint and atomic `promotion.json`.

## Required behavior
- One schema-validated base contains account, absent reservation, topology,
  launcher/entrypoint, fully resolved trainer and `srun` argv, environment,
  modules/runtime/helper manifests, code revision/clean-tree identity,
  seed/data/tokenizer hashes, model/optimizer/DiLoCo/transport settings, and
  checkpoint/publication semantics. Prefer the known-successful job `4962400`
  resolved payload; do not mix in failed job `4963853` settings.
- Profiles may contain only (1) walltime plus a deterministic trainer
  duration/stop budget and (2) partition/QoS queue selection. Account,
  reservation, job topology, all runtime/training fields, and all hashes are
  forbidden profile overrides. Unknown profile keys and unresolved `${...}`
  fail closed.
- Emit `rendered.sbatch`, `runtime.json`, `inputs.json`, `code.json`, runtime
  and helper manifests, and `fingerprint.sha256`. Normalize only typed allowed
  fields and renderer-proven derived metadata. Compare every other scheduler
  directive, script byte, canonical JSON value, environment value, and input
  hash.
- Enforce 256 nodes, 8 ranks/GPUs per node, 2048 launched/participant/worker/
  global-quorum ranks, explicit unique rank IDs `0..2047`, and one GPU/rank.
- The old independent production wrapper and WG procedures must no longer be
  submission authority. `submit-and-monitor` and `retry-refreshed-e97` remain
  paused and are superseded by `promote-exact-successful-256`.

## Exact verification commands
```bash
python scripts/frontier/render_e97_async_256.py --profile smoke --out build/e97-256/smoke
python scripts/frontier/render_e97_async_256.py --profile production --out build/e97-256/production
python scripts/frontier/check_e97_async_promotion.py --smoke build/e97-256/smoke --production build/e97-256/production --policy configs/frontier/e97_async_256_parity_policy.json
```

## Validation
- [ ] A failing regression fixture reproduces the forbidden drift between jobs
  `4962400` and `4963853` before the implementation.
- [ ] Deterministic golden smoke/production renders pass the exact commands
  above and normalized output differs only in typed duration and queue fields.
- [ ] Individual negative mutation tests reject account, reservation,
  node/rank/GPU topology, seed/checkpoint/data, model/optimizer/DiLoCo quorum
  and timeout, environment/module/runtime, launcher/entrypoint, code revision,
  helper hash, and checkpoint/publication policy changes.
- [ ] Tests reject missing/unknown directives or profile keys, arbitrary job
  names/paths, unresolved variables, dirty or changed code/inputs, and generic
  environment ignores.
- [ ] Production dry-run/submission without the exact successful-smoke
  fingerprint, successful exact-topology evidence, and atomic `promotion.json`
  exits nonzero before `sbatch`.
- [ ] Relevant project build/tests and smoke gate pass; worker records artifacts,
  checks messages, commits surgically, and pushes.
## END implement-canonical-e97

## BEGIN run-exact-256-node
## Objective
After `implement-canonical-e97`, perform a real 256-node smoke using the exact
immutable production-capable bundle. Do not reconstruct YAML, command lines,
or environment and do not use `sbatch --export` overrides. The only smoke
profile changes are walltime/deterministically derived stop budget and concrete
queue partition/QoS.

## Mandatory pre-submit gate
Run, retain, and require exit zero from:
```bash
python scripts/frontier/render_e97_async_256.py --profile smoke --out build/e97-256/smoke
python scripts/frontier/render_e97_async_256.py --profile production --out build/e97-256/production
python scripts/frontier/check_e97_async_promotion.py --smoke build/e97-256/smoke --production build/e97-256/production --policy configs/frontier/e97_async_256_parity_policy.json
```
Submit only the content-addressed smoke `rendered.sbatch`. On the allocation,
re-capture argv, sorted environment, modules, code/input/runtime/helper hashes,
and compare them with the bundle before `srun`; abort on drift.

## Validation
- [ ] A real exact-topology smoke is submitted from the immutable bundle at
  256 nodes x 8 GPU ranks = 2048 ranks; job ID, fingerprint, exact `sbatch`
  command, queue, walltime, and all manifests are recorded.
- [ ] Machine parity passes before allocation and its canonical JSON evidence
  proves the only differences are duration/stop budget and partition/QoS;
  account is identical and reservation remains absent.
- [ ] On-node preflight passes before `srun` and confirms the same fingerprint,
  code, inputs, environment/modules, helper, launcher, argv, and topology.
- [ ] Slurm reports `COMPLETED 0:0`; exactly 2048/2048 unique rank IDs
  `0..2047` start and contribute accepted updates; loss is finite; at least one
  DiLoCo merge completes with the expected transport.
- [ ] Metrics/manifest/rank-start evidence exists; a checkpoint is finalized
  and successfully reloaded; the external production latest pointer is
  unchanged.
- [ ] Only after every prior check passes, the validator atomically writes
  `promotion.json` into the same immutable bundle naming the job, artifacts,
  fingerprint, and acceptance evidence. Failure produces no promotion record
  and no production submission.
- [ ] Worker preserves failure evidence, checks messages, records artifacts,
  commits/pushes reports, and completes only on exact-topology success.
## END run-exact-256-node

## BEGIN promote-exact-successful-256
## Objective
After successful completion of `run-exact-256-node`, promote the exact
content-addressed bundle fingerprint recorded by that smoke. Never reconstruct
the training configuration. Production changes only walltime/deterministically
derived stop budget and concrete partition/QoS queue selection. This task does
not inherit historical approval: a real production submission also requires a
current explicit human approval recorded for this exact fingerprint.

## Fail-closed promotion procedure
Re-run and retain:
```bash
python scripts/frontier/render_e97_async_256.py --profile production --out build/e97-256/production
python scripts/frontier/check_e97_async_promotion.py --smoke build/e97-256/smoke --production build/e97-256/production --policy configs/frontier/e97_async_256_parity_policy.json
```
Before `sbatch`, verify the immutable smoke fingerprint and signed/atomic
`promotion.json`; re-hash code, seed/data/tokenizer, runtime/modules, and helper;
require successful smoke evidence (`COMPLETED 0:0`, exact 2048-rank topology and
updates, finite loss/merge, metrics, finalized/reloaded checkpoint, pointer
guard); and require explicit human approval for this fingerprint. Any mismatch,
missing evidence, missing approval, or changed/expired input stops before
submission and returns the workflow to implementation/smoke as appropriate.

## Validation
- [ ] Dependency is exactly `run-exact-256-node`; no production `sbatch` is
  attempted before its machine-checked parity, successful real smoke, and
  atomic promotion record exist.
- [ ] Canonical diff evidence contains only walltime/derived stop budget and
  partition/QoS. Account is identical, reservation absent, and every other
  resolved scheduler/runtime/training/input field matches the smoke.
- [ ] Production consumes the exact promoted smoke fingerprint, revalidates all
  hashes, and rejects regeneration, drift, missing evidence, or missing current
  human approval before `sbatch`.
- [ ] If approved, exactly one concrete production submission is attempted;
  record human approval, job ID, exact command, `scontrol show job`, rendered
  script, manifests, and fingerprint. On-node preflight repeats before `srun`.
- [ ] Startup confirms 256 x 8 = 2048 ranks and the identical launcher, argv,
  environment, code, inputs, helper, checkpoint, model/optimizer/DiLoCo config;
  initial loss and merge health are finite/valid.
- [ ] `submit-and-monitor` and `retry-refreshed-e97` remain paused and their
  descriptions/logs explicitly state they are superseded and confer no
  submission authority.
- [ ] Worker checks messages, records artifacts, commits/pushes reports, and
  completes the task.
## END promote-exact-successful-256

## Acceptance review

- Audit findings and exact paths/commands are incorporated above.
- The difference allowlist is closed: duration/derived stop budget and
  partition/QoS only.
- The graph sequence prevents promotion before implementation and the real
  exact-topology smoke.
- Promotion itself has three independent pre-submit gates: parity result,
  successful exact-fingerprint smoke attestation, and current human approval.
- Historical production submitters remain paused and are explicitly
  superseded.
