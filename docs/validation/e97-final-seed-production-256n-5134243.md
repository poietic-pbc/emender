# Frontier E97 final-seed 256-node continuation — job 5134243

**Date:** 2026-08-02  
**Task:** `monitor-existing-256-node`  
**Verdict:** **PASS** (`full_pass=true`)

## Authority and conformance boundary

This is the human-reviewed 256-node production observation under **ADR-003,
production same-allocation execution epochs (2026-07-31)** in
`docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, using the production crosswalk and
conformance checklist in `docs/RESILIENT_DILOCO_GAP_MATRIX.md`.

- **R07:** `train.py` synchronously published through temporary-file/rename and
  temporary-symlink/rename. Exactly two complete checkpoints remain, no partial
  file remains, and canonical stable `RUN_DIR/train/latest.pt` resolves to the
  independently readable final checkpoint.
- **R12:** the stable run identity and directory are independent of the Slurm
  job ID. All 2,048 ranks cold-resumed step 2300930, and the same stable pointer
  now authorizes future continuation from step 2303840.
- **R14 / NDP13:** one fixed-world child completed under the bounded production
  launcher and shut down cleanly. No damaged communicator was reused or shrunk;
  progress-aware relaunch was not needed.
- **R16:** the attended launch explicitly reviewed 256 nodes and bound exact
  source, payload, final seed, and scheduler fields. This evidence does not
  authorize another job.
- **NDP15:** only synchronous checkpoint atomicity applies. No background
  checkpoint, overlap, mailbox, hashing-in-training, or later-apply claim is
  made.
- **NDP02** and the elastic/native clause of **NDP17** are explicitly retired
  for this production path. R02-R06/R08-R11 dynamic-pool semantics,
  NDP01/NDP03-NDP12/NDP14/NDP16, V21S01-V21S17, and ISP01-ISP07 are not
  claimed.

The rendered compute closure added no SQLite, database, filesystem lock,
metadata heartbeat, membership service, cell, owner tree, or central full-model
broker. The fixed-world child used GPU/RCCL hierarchical collectives; Lustre
held the exact seed, synchronous atomic checkpoints, and durable evidence.

## Exact inherited transaction and scheduler evidence

This task inspected only the already-submitted payload and submitted no job,
retry, cancellation, or changed payload:

```text
job_id=5134243
collector_job_id=null by design
source_sha=c7f4600a4698e426846fa2af18c3809217d60374
payload_digest=841b0494657e31ea788ca5987b6c17188577b8aaf51c44902eee3ebc26ca9c0e
run_id=e97-final-seed-production-256n
run_dir=/lustre/orion/bif148/proj-shared/emender/frontier_runs/final-seed-production-256n/runs/e97-final-seed-production-256n
```

Independent `sha256sum -c` passed for all nine immutable payload/source assets.
The retained live `identity/squeue-live.txt` records literal
`5134243|RUNNING|256|...|batch|debug|02:00:00|...`.
`identity/scontrol-live.txt` separately records `JobState=RUNNING`,
`NumNodes=256`, `NumTasks=2048`, `Partition=batch`, `QOS=debug`, and
`TimeLimit=02:00:00`.

Terminal accounting is preserved in the stable run at
`terminal/sacct-5134243.txt` and committed at
`reports/frontier/e97-final-seed-production-256n-5134243-sacct.txt`:

```text
JobIDRaw|JobName|State|ExitCode|DerivedExitCode|NNodes|...|Partition|QOS|...|Start|End|Elapsed
5134243|e97-final-seed-256n|COMPLETED|0:0|0:0|256|...|batch|debug|...|2026-08-01T14:16:09|2026-08-01T16:02:58|01:46:49
```

Thus the live and terminal authorities separately and explicitly preserve
`Nodes=256`, `Partition=batch`, and `QOS=debug`; terminal state/exit/elapsed are
`COMPLETED`, `0:0`, and `01:46:49`.

## Exact final-seed cold start and all-rank initialization

The independently rehashed seed was exactly 7,719,680,116 bytes with SHA-256
`0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2`,
step 2300930, loss 2.4365, and accepted-token count 150,793,748,480. Its
attestation SHA-256 was
`27e234891df02b64b9db77fc784c341e5a3ae6e87418b8f1af167776d1d710bb`.
All 256 unique offline node manifests record this same size/SHA/step/token and
attestation identity with `network_fetches=0`.

The execution-epoch record binds the original node-local seed target
`/tmp/emender-e97-seed-5134243/checkpoint-step-2300930.pt`. Raw output contains
exactly 2,048 `Resuming from .../train/latest.pt` and 2,048 `Resumed at step
2300930` records. Unique rank sets for the post-RCCL `[DiLoCo] rank
R/2048 bound to cuda:0` and fused-kernel guard markers are both exactly
`0..2047`; the Libfabric RCCL plugin loaded 2,048 times. Rank 0 also recorded
the identical-start broadcast across all 2,048 ranks.

## K40/64M RCCL progress and finite metrics

The raw topology record is `topology=hierarchical`, `group_size=4`, and
`bucket_numel=67108864`. The child completed **73 real K40 merges**, numbered
1-73 at the exact steps 2300960, 2301000, ..., 2303840. Every merge duration was
finite and positive:

| Metric | Observation |
|---|---:|
| K40 cadence, median (range) | 72 s (71-87 s) |
| hierarchical RCCL merge, median (range) | 6,320 ms (5,068-15,227 ms) |
| finite metric records | 291 |
| last logged step | 2303840 |
| last logged loss | 2.2665 |
| last global throughput | 7,683,935 tokens/s |
| last-10 median global throughput | 9,410,375.5 tokens/s |

All parsed loss and global-throughput values were finite, and every throughput
was positive. The final extra merge was correctly skipped because step 2303840
was already a K-aligned consensus step.

## Atomic checkpoint, independent reload, and continuation authority

All 2,048 ranks entered walltime finalization at step 2303840 and emitted 2,048
matching readiness receipts. Retention left exactly:

```text
checkpoint_step_2303800_loss_2.4602.pt  7,719,679,988 bytes
checkpoint_step_2303840_loss_2.3178.pt  7,719,680,116 bytes  <- latest.pt
```

No checkpoint temporary/partial file remained. Using canonical
`$EMENDER_PYTHON` (Python 3.12.13, PyTorch 2.10.0+rocm7.1), the final target was
independently SHA-256 hashed and `torch.load(..., map_location="cpu",
mmap=True, weights_only=False)` reopened it across the production serialization
boundary:

```text
path=/lustre/orion/bif148/proj-shared/emender/frontier_runs/final-seed-production-256n/runs/e97-final-seed-production-256n/train/checkpoint_step_2303840_loss_2.3178.pt
size=7719680116
sha256=39c13e6a54dd6609d033cc118cde3a0bd42a2b1e128eea64925d387b961f9913
step=2303840
loss=2.317775578260422
keys=checkpoint_metadata,loss,model_state_dict,optimizer_state_dict,step
```

Canonical stable `RUN_DIR/train/latest.pt` is an atomic symlink to that exact
file. Step 2303840 is 2,910 steps beyond seed step 2300930, so this stable
pointer—not the original node-local seed or any temporary file—is the future
continuation authority.

The single execution epoch records
`1|0|256|2048|34244|1|2303840`; launcher return code is zero, raw output ends
with `Training complete! Final step: 2303840`, and no traceback, runtime error,
NCCL warning, or nonfinite marker was found. This is clean shutdown; recovery
was not invoked.

## Machine verdict and commands

The committed machine verdict is
`reports/frontier/e97-final-seed-production-256n-5134243-verdict.json`
(SHA-256 `e06f996ac475be42f6cec995063b66911cfce0f63856c28ae6d1be66a0ec26d7`).
It records all mandatory conditions true, `errors=[]`, and literal
`full_pass=true`.

Validation sourced the canonical Frontier environment and used only
`$EMENDER_PYTHON` for Python work:

```bash
source scripts/frontier/activate_emender_frontier.sh
"$EMENDER_PYTHON" --version
(cd "$PAYLOAD_ROOT" && sha256sum -c SHA256SUMS)
sha256sum "$SEED_CACHE"
sacct -j 5134243 -X -P \
  --format=JobIDRaw,JobName,State,ExitCode,DerivedExitCode,NNodes,NTasks,NodeList,Partition,QOS,Account,Timelimit,Submit,Start,End,Elapsed
"$EMENDER_PYTHON" - "$RUN_DIR" "$RUN_DIR/terminal/independent-validation.json.tmp" <<'PY'
# Stream-parse exact raw output and all receipts; independently SHA-256 and
# mmap-reload canonical latest.pt; fail unless every condition passes.
PY
```

The durable validator result and repository copy both report
`initialized_ranks=2048`, `seed_resume_ranks=2048`, `k40_merges=73`,
`checkpoint_reload_step=2303840`, `clean_shutdown=true`, and `full_pass=true`.
