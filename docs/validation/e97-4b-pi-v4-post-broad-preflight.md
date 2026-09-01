# E97 4B Pi v4 post-broad evaluation preflight

**Status:** frozen before public post-training payload download; evaluation-only

V4 is the replacement structural holdout after the first v3 evaluation became
diagnostic. No v4 record, generator output, path, value, payload, or expected
trajectory may enter SFT, teacher prompts, rejection sampling, or RL.

## Authority

- root: `/mnt/nvme1n1/erikg/sft/pi-core-eval-v4-post-broad-heldout`;
- records: 240, 40 per family;
- expected calls: 960;
- manifest SHA-256:
  `8d0d5d350a39c9f5c007d98e4a0010f4a06f39e5a7fb89ecd9ccc4e37e66b8d8`;
- metadata SHA-256:
  `fc391133e82cc85f05e0e29a00bd521d3d2fdac526f14def1893e166db363dac`.

Families:

1. three-way comparison;
2. pointer-conditioned single-target edit;
3. typed CSV aggregation and JSON write;
4. test-guided implementation rename;
5. search-selected link update;
6. checksum manifest construction and verification.

## Mechanical validation

`scripts/validate_e97_pi_eval_authority.py` reconstructed every sandbox and
executed all 960 declared calls. It checked declared failing-command exit status,
exact writes/edits, successful terminal verifiers, postconditions, and required
final evidence. Result: 240/240 tasks and 960/960 expected calls passed.

The first model evaluation is reserved for a behaviorally selected broad
post-training checkpoint that has already cleared smoke, v2, broad instruction,
and real-repository development gates. After that first evaluation, v4 becomes
diagnostic and a new holdout is required for any later blind claim.
