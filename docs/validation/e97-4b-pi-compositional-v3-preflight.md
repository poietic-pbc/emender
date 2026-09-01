# E97 4B Pi compositional v3 preflight

**Status:** curriculum and blind evaluation are frozen; candidate training has
not started.

## Gap evidence

On the new 240-task compositional v2 panel, both the promoted live-aligned u8
checkpoint and its update-64 parent passed 0/240 full tasks. The promoted model
still improved protocol completion (32/240 versus 0/240) and no-cycle behavior
(70/240 versus 3/240), but neither model produced an exact accepted action
trajectory. Traces showed strong original-template substitution: user-specified
paths were replaced by memorized `docs/release-*`, `config/config.json`, and
similar patterns. The original 119/120 panel is therefore a core smoke result,
not broad instruction-following evidence.

## Live compositional source

A new live-aligned authority contains 60,000 trajectories, evenly divided across
six v2 gap families. All assistant actions and the grounded final/newline are
targeted; successful empty tool results use Pi's exact `(no tool output)` context.
Frozen v2 records are excluded by identity, index domain, seed, values, paths,
and payloads.

- records: 60,000;
- input tokens: 24,922,100;
- assistant targets: 12,031,923;
- authority SHA-256:
  `5b2f27ec0713cce04681327a958c597c2e82be61ffb0b648c8b5af4f57ce4dc7`;
- pure-source pack SHA-256:
  `7acc8a06425ccae13bdc37b3ac1b830d47fec7734b1e3228889f70495b304c4b`;
- complete source-pack validation: passed.

## Replay mixture

The training mixture retains previous live Pi behavior and broad instruction
replay instead of repeating the failed final-only strategy:

| Source | Assistant targets | Fraction |
|---|---:|---:|
| compositional live | 12,000,218 | 65.21% |
| prior live-aligned Pi | 2,400,059 | 13.04% |
| Tulu 3 replay | 4,001,084 | 21.74% |

Mixture identity:

- records: 86,224;
- input tokens: 36,461,270;
- assistant targets: 18,401,361;
- authority SHA-256:
  `af31177410a0e818e7d2247d00c4783667ae34a738f4048fc45542bec7178133`;
- 4K pack SHA-256:
  `808dfee68941b953d91220ae38f8b89dc3fa6d73c3f3395341e6fd70254b322c`;
- train packs: 8,954; validation packs: 102;
- complete pack validation: passed.

The packer excluded 161 pre-existing oversize records (932,242 assistant target
tokens) rather than truncating them. The compositional source records are bounded;
the exclusions come from replay material.

## Blind family-held-out panel

A separate v3 panel was frozen before candidate training. Its six families are
not present in the new compositional source: configuration comparison,
pointer-following edit, search-to-write, non-unique edit recovery, stale-command
recovery, and two-input aggregation. It has 240 tasks and 840 expected calls.
Mechanical clean-room execution validated 240/240 terminal sandboxes and 840/840
tool contracts.

- manifest SHA-256:
  `ef481c637fde5916b8b0fe1f80cc2b4f0a6b88262088cbb33aefe0fed6bd6d09`;
- metadata SHA-256:
  `9ae13ee4d19075e8ab581a78b1edce15086398731823ab7b6db5a516037f96f2`.

## Candidate ladder

The first candidate branch starts from the promoted live-aligned u8 saved `x`
weights with a fresh Schedule-Free optimizer. It uses LR `2e-6`, K8 synchronization,
and retained checkpoints every eight updates through update 64. Selection order
is: v2 behavior, original 120-task smoke retention, then untouched blind v3.
Loss alone cannot authorize continuation or promotion.
