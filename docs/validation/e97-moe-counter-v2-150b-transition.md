# E97 MoE boundary-relative counter sampler at the 150.134B authority

**Status:** implementation and real-corpus CPU preflight PASS; GPU continuation
not launched by this change.

**Date:** 2026-08-08

## Finding and correction

The initial `emender-byte-window-counter-v1` contract derives its per-rank
cursor from `accepted_tokens / (data_world * context)`. The real 150B MoE
authority is not divisible by the 2,048-rank, 2,048-token denominator:

```text
accepted_tokens=150134063104
data_world * context=2048 * 2048=4194304
remainder=3145728
```

Every 256-node B4 step adds 16,777,216 accepted tokens, a multiple of that
denominator, so the historical remainder can never become aligned. Counter-v1
therefore correctly failed closed but could never perform the intended legacy
transition. Synthetic tests had used divisible clocks and did not expose the
real authority.

This change adds `emender-byte-window-counter-v2`. Its immutable identity adds
`stream_origin_accepted_tokens`; cursor arithmetic is boundary relative:

```text
absolute_rank_sample_index =
  (total_accepted_tokens - stream_origin_accepted_tokens)
  / (data_world_size * context_size)
```

For the intended transition:

```text
stream_origin_accepted_tokens=150134063104
initial absolute_rank_sample_index=0
samples per rank per B4 step=4
```

The origin participates in the canonical SHA-256 sample identity together with
schema, corpus digest, tokenizer digest, fixed key, world, rank, absolute
per-rank sample index, and bounded retry index. It therefore creates a new,
explicitly named stream rather than relabelling legacy samples. Manifest,
sidecar, and tensor-shard sampler authorities retain the origin, total accepted
clock, relative cursor, and legacy-to-counter boundary.

Counter-v1 remains byte-compatible for the literally identical paper arms: its
serialized identity omits the new field and remains anchored at token zero.
Historical checkpoints and jobs remain explicitly legacy.

## Distribution

V2 changes stream identity and cursor origin, not the sampling distribution.
Candidate starts remain a SHA-256-derived uniform draw modulo the same valid
CommaPile byte-offset range. The corpus, `p50k_base` tokenizer, UTF-8 replacement
handling, six-byte/token read envelope, boundary-token discard, and 2,048
prediction-token context are unchanged.

Sampling remains with replacement. V2 guarantees deterministic stream identity,
uninterrupted/resume equality, and replay of unaccepted work; it does not claim
that two distinct sample identities can never resolve to the same byte offset.
A no-replacement guarantee would require a separately reviewed permutation
sampler.

## Bound production transition

The reviewed stage-2 identity, if separately authorized, is:

```text
schema=emender-byte-window-counter-v2
corpus_sha256=44f4c33471e0d49686453d81850380532bdc4a09e15c71b78eb8ec2d71bbcaa9
tokenizer_sha256=94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069
sampler_key=42
data_world_size=2048
context_size=2048
stream_origin_accepted_tokens=150134063104
transition_from_legacy=1
```

The checkpoint validator requires the origin to equal the complete legacy
manifest's accepted-token boundary and requires a complete K40-aligned step.
The actual authority at step 2,332,080 passes those checks. Wrong origin,
world, context, corpus, tokenizer, key, schema, cursor, or accepted clock fails
before any MoE checkpoint tensor restore.

## Validation

Authority is `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, ADR-003 production
same-allocation execution epochs (2026-07-31), and the production crosswalk in
`docs/RESILIENT_DILOCO_GAP_MATRIX.md`. Applicable safety intent is **R07** and
**R12** atomic checkpoint/restart authority, **R14/NDP13** fail-closed launch,
**R16** evidence discipline, and **NDP15** checkpoint atomicity. This change
adds no live membership, database, SQLite, filesystem lock, metadata heartbeat,
or hot-path transport.

Explicitly retired and unclaimed are **R02-R06, R08-R11; NDP01,
NDP03-NDP12, NDP14, NDP16-NDP17; V21S01-V21S17; and ISP01-ISP07**. No elastic,
native-data-plane, asynchronous-overlap, background-checkpoint, or
communicator-shrink claim is made.

Exact static/unit validation:

```bash
source scripts/frontier/activate_emender_frontier.sh
bash -n \
  scripts/frontier/e97_35b_moe_production.sbatch \
  scripts/frontier/submit_e97_35b_moe_scale.sh
"$EMENDER_PYTHON" -m py_compile \
  train.py ndm/data/tokenized_dataset.py ndm/e97_moe_checkpoint.py \
  scripts/frontier/e97_35b_moe_train.py
"$EMENDER_PYTHON" -m pytest -q \
  tests/test_tokenized_counter_sampler.py \
  tests/test_train_helpers.py \
  tests/test_e97_moe_checkpoint.py \
  tests/test_e97_moe_production_launcher.py
```

Result: **56 passed**.

A real-corpus/login-node preflight then constructed the exact identity above,
validated the actual 150B manifest as `legacy-transition`, sampled rank zero at
cursor zero, advanced one B4 global step to cursor four, and compared an
uninterrupted next B4 batch with a newly constructed resumed stream. Result:

```text
transition_status=legacy-transition
origin_cursor=0
next_cursor=4
resume_equal=true
```

No Slurm job or mutable checkpoint was created by this validation.
