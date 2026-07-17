# Resilient E97 startup smoke 5023212 — terminal evidence

## Identity and exact allocation

- Job: `5023212`, real `sbatch` submission (not `--test-only`).
- Run: `run-resilient-e97-2-smoke-20260717T232905Z-0b0e952`.
- Payload: `0b0e952-20260717T232905Z-startup-smoke-flat-e97-config`.
- Code: `0b0e95279e7b550beab5a136c5c624ba7259cd03`.
- Scheduler envelope: `batch`, debug QoS, exactly 2 nodes and 16 GPUs,
  `00:20:00`, no failure injection.
- Submitted `2026-07-17T19:30:22-04:00`, started `19:30:44`, ended
  `19:36:12`. Queue time was 22 seconds; runtime was 5 minutes 28 seconds.
- Slurm terminal result: `FAILED`, `ExitCode=1:0`; maximum step RSS was
  `87626308K`.

The retained `exact-command.sh` and `rendered-parity.json` are in the immutable
run directory under
`/lustre/orion/bif148/proj-shared/emender/runs/run-resilient-e97-2-smoke-20260717T232905Z-0b0e952`.
Rendered parity was `ok=true` with no forbidden or missing fields.

## Observed roles and terminal diagnosis

The launcher emitted `managers=2 real_trainers=16 trainers_per_node=8
local_steps=40 collective=none`. The supervision event stream records both
model-free managers and all sixteen distinct node/local-rank trainers. It then
records bounded `heartbeat_deadline` eviction and at most two restarts per
trainer. No generation finalized, so this run is not startup-smoke acceptance
evidence and cannot admit a full resilience allocation.

The decisive trainer traceback is a network timeout while tiktoken tries to
fetch `p50k_base.tiktoken` from `openaipublic.blob.core.windows.net`. Frontier
compute nodes cannot depend on that public download. The changed launcher now
requires the immutable p50k cache artifact, verifies SHA256
`94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069`,
stages it once to a payload-specific `/tmp` cache on each allocated node, and
exports `TIKTOKEN_CACHE_DIR` before roles start. The tiktoken URL-key filename
is `ec7223a39ce59f226a68acc30dc1af2788490e15`. The retained source input is
`/lustre/orion/bif148/proj-shared/emender/tokenizers/tiktoken/p50k_base/ec7223a39ce59f226a68acc30dc1af2788490e15`.

This is a changed payload. Job 5023212 will not be retried unchanged, and a
new short two-node startup smoke remains mandatory before any full `02:00:00`
gate.

## Validation and conformance

Conformance was checked against *Resilient DiLoCo Compute Pool*, version 1.
Applicable requirements are R02, R03, R06, R08, R09, R10, R14, and R16. The
run proves bounded role supervision but does not close R16 because there is no
committed generation. The cache fix keeps tokenizer configuration input on
Lustre only for bounded startup staging; live heartbeat, update, aggregate,
quorum, and redistribution traffic remains node-local/network-only.

