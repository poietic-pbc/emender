# E97 dense recurrent-cache qualification (jobs 5307047, 5307175, 5307227)

## Verdict

Dense E97 recurrent state can be retained safely across HTTP/tool turns with constant per-session GPU cache memory. The qualified correctness path ingests every committed token through the same one-token model shape, retains all recurrent matrices in FP32, generates transactionally into a shadow cache, includes the emitted stop/RS token in the resulting state, and reuses a cache only for an exact append-only token prefix.

On the real dense-agent update-128 checkpoint, eight independent ranks and eight caller chunk patterns produced:

- exact recurrent-state equality across caller boundaries;
- exact boundary-logit equality;
- exact 64-token greedy continuation equality;
- exact post-generation state and logit equality;
- exact reset reproducibility;
- successful branch/truncation rejection;
- no mutation of the committed cache during shadow generation;
- 9,832,658 bytes of recurrent state plus boundary logits per active session.

M1 of `docs/E97_DENSE_PI_AGENT_EXECUTION_PLAN.md` passes. M2, the minimal Pi-compatible model server, may begin.

## Authorities

Checkpoint:

`/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-dense-agent/train-128u-lr1e5-v1/checkpoints/checkpoint_agent_sft_u000128.pt`

SHA-256:

`1864f1c9138b17e99da698f3272017c7962da676f0a21d68a42ad98868bbbced`

Source commits:

- `42e43f36`: initial fused-cache qualification;
- `18a3a61a`: 64-token continuation gate;
- `08a28ffb`: canonical tokenwise ingestion fix.

Final authority:

`/lustre/orion/bif148/proj-shared/emender/evaluations/e97-dense-recurrent-cache-qualification-v3`

Final summary:

`/lustre/orion/bif148/proj-shared/emender/evaluations/e97-dense-recurrent-cache-qualification-v3/results/summary.json`

## Job 5307047: initial bounded qualification

The initial eight-rank run compared one fused full-prefix call with multiple segmented calls and generated eight greedy tokens from both states.

Result:

- status `pass` on 8/8 ranks;
- FP32 recurrent states on 8/8 ranks;
- identical boundary argmax and eight-token greedy continuation on 8/8 ranks;
- maximum hidden-state absolute difference `0.027432918548583984`;
- maximum boundary-logit absolute difference `0.203125`;
- cache bytes per session `9,832,658`;
- peak allocated HBM per rank `2,770,609,664` bytes.

Scheduler evidence:

- `Partition=batch`;
- `QOS=debug`;
- `Requeue=0`;
- state `COMPLETED`;
- exit `0:0`;
- elapsed `00:07:44`.

This established practical short-continuation stability, but the nonzero differences showed that FP32 recurrent matrices alone do not make arbitrary BF16 projection shapes bitwise identical.

## Job 5307175: extended gate exposed a real boundary problem

The second run forced 64 tokens rather than stopping at RS. Seven ranks remained equal, but rank 2 diverged from uninterrupted replay after 20 generated tokens:

- pre-generation hidden maximum difference `0.02089560031890869`;
- pre-generation logit maximum difference `0.1875`;
- post-divergence hidden maximum difference `1.2838678359985352`;
- post-divergence logit maximum difference `14.8515625`;
- greedy continuation equality `false`;
- terminal job state `FAILED`, exit `1:0`.

Rank 2 and rank 5 used the same prompt with different caller segmentation. Rank 5 matched the uninterrupted authority. The first rank-2 mismatch was generated-token index 20, before either trajectory reached RS. Therefore the failure could not be dismissed as irrelevant generation after the assistant boundary.

### Root cause

The recurrence matrices were retained in FP32 and the valid-length fused kernel returned the correct real-token boundary. However, q/k/v/g projections remained BF16 GEMMs. Changing the input time dimension changes GEMM shape and rounding. Small projection differences entered the nonlinear recurrent stack and eventually crossed a greedy decision boundary.

A server that processed an HTTP suffix as one variable-length tensor would therefore make results depend on where Pi request boundaries happened to occur. It could also fail to reproduce an evicted session by replaying the transcript with different chunking.

## Fix: canonical tokenwise ingestion

Commit `08a28ffb` changed `advance_e97_cache()` so all committed input is processed with one-token model calls, regardless of how the caller groups the input token list. This establishes one canonical execution shape for:

- uninterrupted prefill;
- incremental Pi turns;
- tool-result appends;
- session reconstruction after eviction or restart;
- arbitrary network/request boundaries.

Generation was already tokenwise. The public cache remains immutable from the caller's perspective: generation returns a shadow cache and the caller decides whether to commit it.

This correctness path is intentionally conservative. Faster chunk-invariant projections or replay using persisted original segmentation may be evaluated later, but neither is required for the bounded agent.

The production checkpoint has `use_conv=0`. The cache API fails closed for convolutional E97 variants because convolution buffers are not yet represented in `E97RecurrentCache`.

## Job 5307227: final qualification

Eight ranks exercised three canonical prompts and eight different caller chunk patterns. Each rank compared full caller input, segmented caller input, clean reset, 64 forced greedy tokens, a synthetic stop token, branch mismatch, truncation mismatch, and transactional non-mutation.

Final summary:

```json
{
  "all_checks": true,
  "cache_state_bytes_per_session": [9832658],
  "generation_tokens_per_shard": [64],
  "hidden_max_abs_difference": 0.0,
  "next_logits_max_abs_difference": 0.0,
  "peak_allocated_bytes_per_rank": 2749940736,
  "post_generation_hidden_max_abs_difference": 0.0,
  "post_generation_logits_max_abs_difference": 0.0,
  "shards": 8,
  "status": "pass"
}
```

All checks passed on every shard:

- token prefix equality;
- boundary greedy equality;
- 64-token greedy continuation equality;
- post-generation argmax equality;
- clean-reset greedy equality;
- FP32 state dtype;
- finite state;
- committed-cache preservation;
- stop token consumed into the shadow cache;
- exact append suffix recognition;
- branch rejection;
- truncation rejection.

Per-rank prompt lengths were 80, 85, or 94 tokens. Full canonical prefill, including first-use kernel effects, took 4.33-4.65 seconds. The repeated segmented pass took 1.60-1.94 seconds. Complete qualification after model load took approximately 21 seconds per rank. These timings establish viability, not optimized serving throughput.

Scheduler evidence:

- `Partition=batch`;
- `QOS=debug`;
- `Requeue=0`;
- state `COMPLETED`;
- exit `0:0`;
- elapsed `00:12:18`;
- node `frontier08496`.

Terminal accounting is retained at:

`/lustre/orion/bif148/proj-shared/emender/evaluations/e97-dense-recurrent-cache-qualification-v3/identity/sacct-5307227.txt`

## Claims and limitations

Qualified claims:

1. Recurrent cache memory is constant with transcript length per active session.
2. The current checkpoint needs approximately 9.38 MiB for recurrent matrices plus boundary logits per session.
3. Canonical tokenwise replay is exactly invariant to caller/HTTP chunk boundaries in the tested fused path.
4. RS/stop tokens can be retained in the committed model prefix even when hidden from Pi display.
5. Branches and shortened/edited histories can be detected before state reuse.

Not yet claimed:

1. Constant network request size: standard OpenAI-compatible Pi requests may still resend full history.
2. Arbitrarily many simultaneous sessions: total cache memory scales with active sessions.
3. Persistence across server restart without replay: states are not yet serialized.
4. Convolutional E97 support.
5. Optimized long-prompt prefill.
6. Semantic agent reliability: this qualification tests inference correctness, not task success.
