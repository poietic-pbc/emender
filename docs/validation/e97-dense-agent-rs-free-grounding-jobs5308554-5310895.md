# Dense E97 RS-free grounded Pi-agent qualification

Final training job: 5310683  
Canary job: 5310771  
Full real-Pi evaluation job: 5310895  
Final evaluation authority: `/lustre/orion/bif148/proj-shared/emender/evaluations/e97-dense-agent-pi-v3-warm32u-full291`

## Verdict

The bounded dense E97 agent now grounds typed tool results reliably through real Pi execution without RS inside a conversation.

The promoted checkpoint completed 285/291 held-out tasks under the strict all-fields-exact criterion (`97.94%`, Wilson 95% CI `[95.58%, 99.05%]`). It selected the correct operational tool on all 291 tasks, executed exactly one operational tool followed by exactly one terminating `submit_answer` on all 291, and submitted a value accepted as grounded in the actual tool observation on all 291 (`100%`, Wilson lower bound `98.70%`).

Five tasks used a wrong calculator operand and therefore returned a wrong but genuinely tool-grounded answer. One additional calculator expression contained harmless leading whitespace and was counted as a strict argument mismatch despite returning the expected value. Count passed 107/107 and lookup passed 96/96. Calculator passed 82/88 under strict exact arguments.

This reaches the full-promotion evaluation stage for the bounded synthetic agent. It does not establish general repository-agent capability.

## Protocol correction

Pretraining used RS between unrelated streaming records. Dense-agent v1 incorrectly placed RS between coherent assistant and tool turns. The promoted protocol:

- contains no RS within a conversation;
- stops generation when one structured `Action` JSON object is complete;
- commits recurrent state before RS can be emitted or consumed;
- resets recurrent state externally between unrelated tasks;
- uses typed deterministic tool observations;
- terminates through provenance-checked `submit_answer` rather than free-form `Final:` text.

The unchanged v1 checkpoint immediately retained a calculator result under RS-free serving, but repeatedly called calculator with the result as a new expression. This supported training the explicit observation-to-`submit_answer` transition.

## Data and tools

Final authority:

`/lustre/orion/bif148/proj-shared/emender/sft/dense-agent-v3-rs-free-simple`

Authority manifest SHA-256:

`ec3af4ecfbd745e068e338a6a2dfb0b04058bdcdc839746d9eb0830c5f4e004c`

Counts:

- records: 30,000;
- training records: 29,709;
- validation records: 291;
- assistant targets: 953,337;
- RS tokens in serialized trajectories: zero.

Pack authority:

`/lustre/orion/bif148/proj-shared/emender/sft/dense-agent-v3-rs-free-simple-packs-4096`

Pack manifest SHA-256:

`500556aca21419501b8a83ddf4e61291831644a052cc95434727e609f8b26633`

Exhaustive pack validation passed all 30,000 records across 788 packs.

The bounded v3 tool vocabulary is:

- `calculator({expression})`;
- `count({path,suffix})`;
- `lookup_owner({project})`;
- `lookup_budget({project})`;
- `submit_answer({value})`.

Splitting lookup by requested field removed ambiguous field selection and the v1 search/read schema interference. All file operations remain confined to the disposable task root. No shell, child process, write, or edit capability is exposed.

## Training trajectory

### Fresh v2, job 5309091

A 64-update fresh-base arm was mechanically stable, reducing masked training loss from 2.237 at update 1 to 0.214 at update 64. Its 24-task Pi canary scored 0/24: it over-selected `submit_answer` before obtaining a tool result. This confirmed that low masked loss was not behavioral evidence.

### Warm v2, jobs 5309699 and 5309750

The v1 update-128 checkpoint stores schedule-free averaged weights without optimizer state. An initial warm qualification correctly failed closed when loaded as generation/train weights. The trainer was extended with explicit `--source-weight-mode saved`; averaged warm starts are now intentional and recorded in checkpoint lineage.

The 32-update warm arm scored 15/24 (`62.5%`) on the real-Pi canary:

- calculator: 8/9;
- count: 7/7;
- lookup: 0/8.

All lookup tasks selected the lookup tool, but v1 search/read argument patterns interfered with the two-field lookup schema.

### Simplified v3, jobs 5309997 and 5310683

Lookup was simplified to field-specific tools with one project argument. The 8-update qualification passed. The promoted 32-update warm checkpoint is:

`/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-dense-agent/v3-simple-warm-v1-32u/checkpoints/checkpoint_agent_sft_u000032.pt`

SHA-256:

`b74bf1fa2234007d6b4369ba62f9b90f2a6e47398f2ebaf23064aa0df51f7690`

Job 5310683 completed with `Partition=batch`, `QOS=debug`, `Requeue=0`, and exit `0:0`. Masked training loss reached 0.0426 at update 32. Behavioral promotion depended only on subsequent real-Pi execution.

## Real-Pi canary: job 5310771

All 24 held-out tasks passed every check:

- calculator: 9/9;
- count: 10/10;
- lookup: 5/5;
- correct first tool and exact arguments: 24/24;
- grounded terminating submission: 24/24;
- bounded two-tool trajectory: 24/24.

Every sequence was exactly one of:

```text
calculator -> submit_answer
count -> submit_answer
lookup_owner -> submit_answer
lookup_budget -> submit_answer
```

## Full held-out real-Pi evaluation: job 5310895

All 291 validation identities were executed exactly once through independent real Pi CLI sessions. Pi owned tool execution; no expected action or observation was injected.

Results:

| Metric | Result |
|---|---:|
| Strict complete success | 285/291 (97.94%) |
| Correct operational tool | 291/291 (100%) |
| Exact operational arguments | 285/291 (97.94%) |
| Exactly one terminating submit | 291/291 (100%) |
| Submitted expected task value | 286/291 (98.28%) |
| Submission accepted from actual tool result | 291/291 (100%) |
| Bounded turns | 291/291 (100%) |
| Pi completed normally | 291/291 (100%) |

Family results:

| Family | Strict result |
|---|---:|
| Calculator | 82/88 (93.18%) |
| Count | 107/107 (100%) |
| Lookup | 96/96 (100%) |

The six strict failures were all calculator argument transcription:

- one leading-space difference with the correct expression and value;
- five changed first operands, followed by correct execution and correct copying of the resulting tool value.

There were no fabricated values accepted by `submit_answer`, no loops, no extra tools, and no malformed terminal paths.

Terminal scheduler evidence:

- `Partition=batch`;
- `QOS=debug`;
- `Requeue=0`;
- `COMPLETED`;
- exit `0:0`;
- elapsed `00:12:21`;
- node `frontier07043`.

Accounting artifact:

`/lustre/orion/bif148/proj-shared/emender/evaluations/e97-dense-agent-pi-v3-warm32u-full291/identity/sacct-5310895.txt`

## Decision

Promote the v3 warm-32 checkpoint for the bounded calculator/count/lookup demonstration. The grounded finalization requirement passes. The remaining error mode is upstream calculator-argument transcription, not observation grounding.

Before broadening autonomy:

1. preserve this checkpoint and evaluation as immutable authorities;
2. add calculator argument verification or correction for the five semantic operand errors;
3. retain read-only, disposable task roots;
4. build a separate held-out repository-reading panel before claiming repository-agent capability;
5. do not enable edit or write tools yet.
