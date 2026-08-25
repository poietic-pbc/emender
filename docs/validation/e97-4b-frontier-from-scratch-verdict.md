# E97 4B full-scale from-scratch verdict

**Status:** stopped by attended operator decision; no normal-QoS continuation authorized.

## Decision

The 256-node / 2,048-rank B5/K25 path is mechanically correct but rejected as
an efficient from-scratch foundation-training strategy. The operator requested
a graceful stop after comparing it with the same 4B foundation trained on the
Lambda eight-GPU seed workflow. No further job may treat this run as an
authorized production predecessor without a new explicit review.

The correct successor direction is:

1. train a competent seed on the smaller Lambda world, where the optimizer
   receives many more updates per accepted token;
2. verify the seed and sampler/checkpoint provenance;
3. use fixed-world scaled DiLoCo as continuation, not random-init bootstrap.

## Why aggregate scaling failed scientifically

The operator-reported Lambda reference uses eight GPUs, B32, context 2,048 and
K32, and reaches approximately 3 nats by 2B aggregate tokens. Its geometry is:

```text
8 * 32 * 2048 = 524,288 tokens/optimizer step
2B tokens ~= 3,815 optimizer steps ~= 119 K32 merges
```

The Frontier geometry used 2,048 ranks, B5, context 2,048 and K25:

```text
2048 * 5 * 2048 = 20,971,520 tokens/optimizer step
2B tokens ~= 95 optimizer steps ~= 4 K25 merges
```

The effective aggregate batch is 40 times larger while the validated inner LR
remains unchanged. Consequently the Frontier path receives 40 times fewer
optimizer updates per accepted token. Its collectives, optimizer, checkpoint,
and restart paths worked; its token efficiency did not. Counting aggregate
replica exposure as if it were the small-world seed's sequential optimization
exposure was the invalid assumption.

## Exact-world bootstrap authority

Job `5339920` completed on `Partition=batch`, `QOS=debug`, `Requeue=0`:

- source `6380f436b993995bf534e9fc4364f87294baa2a6`;
- 256 nodes / 2,048 ranks;
- B5/K25;
- step 50 / 1,048,576,000 accepted tokens;
- loss 7.6901;
- merges 47.919s and 56.907s;
- peak allocated HBM 47,721 MiB; reserved 57,562--57,564 MiB;
- checkpoint bytes 24,276,098,175;
- SHA-256 `b34680870ec241f4d7213c722cd838939d9bfc928b8392de9c9d881e3f2dd83a`;
- atomic latest publication and collector mmap-reload receipt passed.

This is the only collector-complete exact-world checkpoint in the stopped
lineage.

## Attended diagnostic continuation and stop

Job `5340377` resumed step 50 on `Partition=batch`, `QOS=debug`, `Requeue=0`.
The operator requested a graceful stop after the large-batch inefficiency was
identified. The payload completed `0:0` in 01:20:05:

- terminal step 528 / 11,072,962,560 cumulative tokens;
- 20 model consensuses including the final partial-K consensus;
- regular K25 merge median 39.229s (31.415--59.236s);
- effective sustained aggregate throughput approximately 2.536M tokens/s;
- terminal loss summary 5.5880;
- peak allocated HBM 47,729 MiB; reserved 49,982 MiB on all ranks;
- no traceback, OOM, NaN, or Inf;
- all 2,048 finalization receipts present.

The final step-528 diagnostic checkpoint is complete and hashed:

- bytes 24,276,098,367;
- SHA-256 `d33a25e596181e8bb8aaf52ecccd204c5ffb5acc1fcb7364a22196a09cf70d72`;
- `latest.pt` points to it;
- it is **not promotion authority** because `528 % 25 != 0`.

The collector correctly failed closed rather than issuing a reload receipt for
that non-K-aligned checkpoint. The last retained K-aligned diagnostic
checkpoint is step 500 / 10,485,760,000 tokens, loss 5.4711, SHA-256
`aebed77b0731184ea95a2ec3608e148cca01e1e3b3965a87a3fef619945e2b9d`.
It is retained for analysis only and is not authorized for a normal-QoS
continuation.

## Architecture conformance boundary

This work follows ADR-003 fixed-world execution epochs. Applicable safety
intent is R07, R12, R14/NDP13, R16, and NDP15 checkpoint atomicity. The payload
failed no collective or checkpoint safety invariant; the scientific scaling
assumption was rejected. R02--R06/R08--R11, NDP02 and other native/elastic
clauses, and V21S01--V21S17/ISP01--ISP07 remain retired or unclaimed for this
path.
