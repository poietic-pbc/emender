# E97 4B Pi SFT 64K local qualification

## Scope

This report qualifies one local eight-rank, 65,536-token masked-SFT
forward/backward/update/checkpoint/resume path. It does not establish long-context
behavior, elastic execution, or resilient/native/async conformance. Production
architecture authority remains ADR-003 and
[`../RESILIENT_DILOCO_COMPUTE_POOL.md`](../RESILIENT_DILOCO_COMPUTE_POOL.md).

## Immutable inputs

- parent checkpoint SHA-256:
  `3ace004251643acf2e7c7f720e8f29968ad0a483441553c0c885b87b3df84568`;
- action-only Open-SWE authority SHA-256:
  `a0e8336fe17463f98330464a6abf56cb850eb6559bbc94ba082fd28e3b9fe939`;
- 65,536-token pack manifest SHA-256:
  `bddce065f73e7dddcfbfca00ed602b9784ed37f4f0f06a8f3e6cb8f02ddd297e`;
- source commit: `ed3a234ddbb05738c95a75eae1e2762e6cbe312e`;
- world: eight local RTX 6000 Ada ranks;
- optimizer state: pinned-CPU BF16 Schedule-Free;
- layer activation-checkpoint group size: two.

The complete-record 64K authority retains 82,954,991 assistant targets. It
never splits a record.

## Group-one negative control

One-layer checkpoint groups reached approximately 47.2--47.3 GiB of device
memory and failed closed during MLP recomputation, chunked cross-entropy, and
backward. This configuration is not qualified at 64K on 48 GB devices.

## Group-two update 1

Run:
`e97-4b-pi-sft-64k-group2-systems-probe-ed3a234d`

- global input tokens: 325,287;
- assistant targets: 49,483;
- loss: 1.4870434;
- gradient norm: 5.125;
- step duration: 84.246 seconds, including a 17.830-second fixed-world merge;
- maximum HBM allocated: 39,837,257,728 bytes;
- maximum HBM reserved: 48,834,281,472 bytes;
- checkpoint SHA-256:
  `da392cbda5b3980374aef5057e1e52ad21c9cf4b81e27b522120a6f9eb89f472`;
- independent mmap reload: passed.

## Exact resume and update 2

Run:
`e97-4b-pi-sft-64k-group2-resume-u2-ed3a234d`

The run restored update 1 model, optimizer, sampler, and token clocks, then
executed update 2.

- resumed from update: 1;
- update-2 global input tokens: 379,073;
- update-2 assistant targets: 68,040;
- cumulative input tokens: 704,360;
- cumulative assistant targets: 117,523;
- loss: 1.4133558;
- gradient norm: 4.21875;
- step duration: 82.797 seconds, including a 15.766-second merge;
- maximum HBM allocated: 42,383,423,488 bytes;
- maximum HBM reserved: 49,517,953,024 bytes;
- checkpoint SHA-256:
  `6b3e798791e4f4d72bf7b86b1acd7dcb27f1f2079825e3c2513d798625751385`;
- independent mmap reload: passed.

## Verdict

The group-two 64K path is systems-qualified for bounded local experiments. The
margin in reserved HBM is small, so changes to kernels, shapes, loss chunking,
or allocator behavior require a fresh probe. A sustained campaign additionally
requires behavioral long-range-memory gates and multi-update throughput and
thermal observations.

This result is fixed-world local evidence only. It does not close resilient
DiLoCo gap requirements and makes no claims against the native/elastic/async
requirement IDs in `RESILIENT_DILOCO_GAP_MATRIX.md`.
