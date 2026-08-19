# E97 dense-agent v1 baseline (job 5306907)

## Verdict

The dense 1.3B recurrent E97 model learned a strong bounded action grammar and largely preserved arguments after one inexpensive 128-update SFT pass. It did **not** become a reliable end-to-end agent: final answers were weakly grounded in teacher-provided tool observations, and the count task design required the model to count a returned filename list rather than delegating counting to the tool.

This checkpoint is the immutable integration parent for the first Pi vertical slice. It is not a promoted reliable agent.

## Authorities

Dense seed:

`/lustre/orion/bif148/proj-shared/emender/frontier_runs/final-seed-production-256n/milestones/step-2322520-tokens-513013841920/checkpoint_step_2322520_loss_2.2798.pt`

SHA-256: `e559df3e8c540aef59ce8c9d73338f255cbe2fb9c7301ab45c7ef36a5b0fb857`

Dense-agent authority:

`/lustre/orion/bif148/proj-shared/emender/sft/dense-agent-v1`

Authority manifest SHA-256: `01382bea60c02c09b62250a17f7865cce26c2994d817522674eee252eda8a065`

Pack manifest SHA-256: `cfb6c4ccdf3079752d3c02c6e8eb1f5f8bf87b7a9f8cce528fa45f1c255c21e6`

Update-128 checkpoint:

`/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-dense-agent/train-128u-lr1e5-v1/checkpoints/checkpoint_agent_sft_u000128.pt`

SHA-256: `1864f1c9138b17e99da698f3272017c7962da676f0a21d68a42ad98868bbbced`

Evaluation root:

`/lustre/orion/bif148/proj-shared/emender/evaluations/e97-dense-agent-128u-v1`

Source commit: `4d4872d7dea5b2758cc4a63f616a942cf9ea2b36`

## Training result

Frontier job 5306854 completed 128 synchronized full-model updates:

- 4,100,794 packed tokens;
- 951,111 assistant targets;
- loss `3.2663 -> 0.2428`;
- gradient norm `10.19 -> 0.93`;
- peak HBM 13.15 GB;
- runtime 15m35s;
- terminal exit `0:0`.

These are optimization diagnostics, not held-out capability evidence.

## Held-out evaluation

Job 5306907 evaluated all 308 task-level excluded examples (720 assistant turns) on eight rank-local GCDs. Scheduler evidence recorded both required queue fields:

- `Partition=batch`;
- `QOS=debug`;
- `Requeue=0`;
- terminal return code `0`.

Machine-readable result:

`/lustre/orion/bif148/proj-shared/emender/evaluations/e97-dense-agent-128u-v1/results/summary.json`

Exact metrics:

| Metric | Result |
|---|---:|
| Tasks | 308 |
| Assistant turns | 720 |
| Action syntax validity | 1.000000 |
| Exact action turns | 0.868932 |
| Correct tool name (post-hoc audit) | 0.881068 |
| Exact JSON arguments (post-hoc audit) | 0.978155 |
| RS stopping | 0.951389 |
| Exact final response | 0.006494 |
| Strict whole-task exact | 0.006494 |

Strict task exact by family:

| Family | Tasks | Exact |
|---|---:|---:|
| calculator | 108 | 0.000000 |
| lookup | 104 | 0.019231 |
| count | 96 | 0.000000 |

A post-hoc semantic final-value audit, which ignored harmless wording/format differences when the requested value was correct, found:

| Family | Correct | Total | Accuracy |
|---|---:|---:|---:|
| calculator | 47 | 108 | 43.5% |
| lookup | 41 | 104 | 39.4% |
| count | 0 | 96 | 0.0% |

The semantic audit is useful diagnosis but was not emitted by the original evaluator; future evaluators must make semantic criteria executable and store per-example judgments.

## Critical evaluation limitation

This was a **teacher-forced observation evaluation**, not a true agent rollout. For every assistant turn, the evaluator appended the expected assistant output to the transcript before continuing. Therefore:

- an incorrect predicted action did not determine the next tool result;
- the evaluator supplied the expected tool observation;
- later turns were scored from an oracle trajectory rather than the model's trajectory;
- strict whole-task exact is still meaningful as a sequence of per-turn comparisons, but it does not measure recovery or compounding errors during actual execution.

Promotion now requires Pi to execute the model's own parsed actions and return the resulting observations.

## Representative traces

`<RS>` below denotes byte `0x1e`, token 218.

### Exact held-out lookup final

Task `agent-lookup-00013651`:

Expected:

```text
Final: The owner of Project-013651 is Amina.<RS>
```

Generated:

```text
Final: The owner of Project-013651 is Amina.<RS>
```

This shows that exact observation-to-final copying is possible, but it occurred too rarely.

### Valid grammar and argument with wrong tool name

Task `agent-calculator-00000078`:

Expected:

```text
Action: calculator
Arguments: {"expression":"413 * 188"}<RS>
```

Generated:

```text
Action: calculation
Arguments: {"expression":"413 * 188"}<RS>
```

The syntax and argument generalized while the closed tool vocabulary did not. Pi must reject unknown tools rather than interpret them heuristically.

### Ungrounded calculator final

For the same task, after the evaluator supplied the teacher tool observation, expected was:

```text
Final: 413 * 188 = 77644.<RS>
```

Generated began:

```text
Final: 7.5 - 0.0 = 7.5

Scoring:
```

This is a grounding failure despite correct arithmetic being available in the transcript.

### Semantically correct lookup with harmless wording difference

Task `agent-lookup-00002803`:

Expected:

```text
Final: The budget of Project-002803 is $622,000.<RS>
```

Generated:

```text
Final: The budget for Project-002803 is $622,000.<RS>
```

Strict exact scoring marks this wrong; semantic scoring correctly recognizes the value. Both metrics are needed.

### Count contract failure

Task `agent-count-00002165`:

Expected:

```text
Final: There are 15 .txt files in data/.<RS>
```

Generated:

```text
Final:

<RS>
```

The v1 `list` tool returned all filenames and expected the recurrent model to count them. Dense-agent v2 must instead return a compact typed observation such as `{"count":15}`.

### Ambiguous lookup observation

Task `agent-lookup-00008908` asked for the owner. Expected final named Elena, but the generated final reported the budget `$865,000`. The read observation contained both owner and budget. Dense-agent v2 must return the requested typed field only, for example `{"field":"owner","value":"Elena"}`.

## Decision

1. Preserve update 128 as the v1 integration authority.
2. Do not spend further compute on unchanged v1 continuation.
3. First qualify persistent recurrent-state inference and run this checkpoint end to end through Pi.
4. Build v2 around exact Pi schemas and compact typed observations.
5. Use end-to-end tool execution, stopping, loop rate, and semantic task success as promotion gates.
