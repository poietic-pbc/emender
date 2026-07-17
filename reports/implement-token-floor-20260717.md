# Token-floor generation-close evidence

This implementation conforms to `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`,
architecture decision and design authority **version 1 (2026-07-17)**, for
companion-matrix requirements **R04** (fresh fenced contribution identity,
strict stale rejection, and idempotent duplicate receipts) and **R06**
(explicit contribution/token floors and a bounded generation deadline).

## Conformance checklist

- Admission uses the generation fence plus stable worker identity, boot
  incarnation, and contribution sequence. Identical replay returns the original
  receipt; conflicting identity reuse, stale attempts, non-READY incarnations,
  corrupt payloads, and digest mismatches are rejected.
- `Q_min`, `T_min`, and an optional READY fraction are computed from one sorted,
  immutable leased-READY snapshot. The fraction threshold is capped at that
  snapshot's size; neither launch ranks nor a fixed world appears in the API.
- Close freezes accepted identities in identity sort order. Once commit-ready,
  the result is immutable and late input is rejected. A missed floor at the
  finite generation deadline produces explicit defer evidence before the run
  deadline and abort evidence at the bounded run deadline, with no commit.
- The implementation is metadata-only protocol logic. It adds no collective,
  fixed-world assumption, central model broker, or Lustre tensor/payload path.
  Optional durable evidence contains identities and counters, never tensors.

## Validation commands

```text
/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python -m pytest -q tests/test_resilient_node_quorum.py
/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python -m pytest -q tests/test_resilient_node_transport.py
python3.11 -m compileall -q ndm/resilient_node_quorum.py tests/test_resilient_node_quorum.py
git diff --check
```

The focused results were 11/11 quorum tests and 10/10 transport regression
tests passing. The quorum-collapse test inspects its JSONL pause/defer/abort
evidence and asserts that no `latest.json` commit exists.
