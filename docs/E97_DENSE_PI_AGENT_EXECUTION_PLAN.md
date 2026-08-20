# E97 Dense Recurrent Pi Agent Execution Plan

Status: active implementation authority  
Scope: dense E97 agent serving, recurrent-state retention, Pi integration, bounded evaluation, and targeted follow-up SFT  
Initial model authority: 513B-token dense E97-derived agent checkpoint described below

Progress:

- M0 complete: v1 baseline preserved in `docs/validation/e97-dense-agent-v1-baseline-job5306907.md`.
- M1 complete: canonical tokenwise FP32 recurrent caching qualified on the real checkpoint in job 5307227; see `docs/validation/e97-dense-recurrent-cache-jobs5307047-5307175-5307227.md`.
- M2 complete: the real Pi CLI performed a structured tool roundtrip with a recurrent-cache miss then hit on all eight ranks in job 5308186; see `docs/validation/e97-dense-pi-roundtrip-jobs5307403-5307492-5307670-5307866-5308186.md`.
- M3 complete: after removing intra-conversation RS and simplifying typed tool contracts, the warm-32 v3 checkpoint passed 285/291 strict held-out tasks and grounded 291/291 actual tool observations through real Pi in job 5310895; see `docs/validation/e97-dense-agent-rs-free-grounding-jobs5308554-5310895.md`.
- M4 is now active: preserve the bounded direct-CLI promotion authority and build a verifier-driven continual-learning gym rather than continue undifferentiated help-discovery SFT. Job 5312544 qualified the cwd-only Frontier Apptainer boundary; the portable task, Docker/Harbor, Frontier/Apptainer, recurrent-forking, and champion/challenger program is specified in `docs/EMENDER_AUTONOMOUS_CLI_GYM_PLAN.md`. See also `docs/validation/e97-cli-apptainer-cwd-isolation-job5312544.md`.

## 1. Objective

Turn the dense 1.3B E97 recurrent model into a bounded single-GPU tool agent that runs through Pi, retains recurrent state across conversation and tool turns, uses constant GPU cache memory with respect to transcript length, and fails safely when it cannot complete a task.

Pi is the execution harness and evaluation surface. We will not build a competing general-purpose agent loop. The only separate service is the Python/GPU model server needed to expose E97 through a protocol Pi can call.

The first promoted artifact is a reliable read-oriented agent, not an open-ended general assistant or unrestricted autonomous coding agent.

## 2. Immutable starting authorities

Dense seed:

`/lustre/orion/bif148/proj-shared/emender/frontier_runs/final-seed-production-256n/milestones/step-2322520-tokens-513013841920/checkpoint_step_2322520_loss_2.2798.pt`

Dense seed SHA-256:

`e559df3e8c540aef59ce8c9d73338f255cbe2fb9c7301ab45c7ef36a5b0fb857`

Dense-agent v1 corpus:

`/lustre/orion/bif148/proj-shared/emender/sft/dense-agent-v1`

Authority manifest SHA-256:

`01382bea60c02c09b62250a17f7865cce26c2994d817522674eee252eda8a065`

Pack manifest SHA-256:

`cfb6c4ccdf3079752d3c02c6e8eb1f5f8bf87b7a9f8cce528fa45f1c255c21e6`

Dense-agent v1 checkpoint:

`/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-dense-agent/train-128u-lr1e5-v1/checkpoints/checkpoint_agent_sft_u000128.pt`

Checkpoint SHA-256:

`1864f1c9138b17e99da698f3272017c7962da676f0a21d68a42ad98868bbbced`

Held-out evaluation root:

`/lustre/orion/bif148/proj-shared/emender/evaluations/e97-dense-agent-128u-v1`

The v1 evaluation established strong action-format transfer but weak final grounding: syntax 100%, correct tool 88.1%, exact arguments 97.8%, RS stopping 95.1%, calculator semantic final 43.5%, lookup semantic final 39.4%, and count semantic final 0%.

## 3. Architectural decision

The system has three layers:

1. **E97 model server**: checkpoint loading, canonical serialization, generation, action parsing, recurrent-state cache, and an OpenAI-compatible transport.
2. **Pi**: conversation lifecycle, structured tool calls, tool execution, sessions, user interaction, and trajectory recording.
3. **Safety and verification boundary**: path confinement, command policy, disposable worktrees, resource limits, loop limits, and mechanical task verification.

Pi owns the agent loop. The model server must not independently execute tools.

## 4. Recurrent-state retention contract

Retaining hidden state is the normal execution path and a central demonstration requirement. Full replay exists only as recovery and verification.

For every active Pi session, the server maintains a committed cache containing:

- canonical token prefix identity;
- FP32 recurrent state for every recurrent layer;
- next-token logits at the committed boundary when useful;
- serialization/tool-schema identity;
- generation and branch identifiers;
- optional turn-boundary recovery checkpoints.

### 4.1 Append-only fast path

For each request, the server deterministically serializes the complete Pi-visible conversation. If the committed token sequence is an exact prefix, only the new suffix is evaluated from the cached state.

### 4.2 Fail-closed mismatch path

If the prefix differs because of branching, history editing, compaction, system-prompt changes, tool-definition changes, normalization changes, or cache corruption, the server must not reuse the incompatible state. It replays from a verified compatible turn checkpoint or from the beginning.

### 4.3 Transactional generation

Generation advances a shadow copy of state. The server commits only the canonical completed assistant turn. Cancellation, transport failure, malformed tool output, or rejected generation discards the shadow state.

### 4.4 Canonical structured tool calls

E97 v1 emits textual `Action:` and `Arguments:` records, while Pi stores structured tool calls. The bridge must define one reversible canonical mapping. The committed recurrent prefix must be the same token sequence reconstructed on the next request. RS (`\x1e`, token 218) is included in the model-side prefix after every assistant action/final even when it is not shown by Pi.

### 4.5 Memory claim

GPU recurrent cache memory is constant with transcript length **per active session**. Total server memory scales with the number of active sessions. Inactive caches may be offloaded or evicted and reconstructed from the Pi transcript.

The initial OpenAI-compatible integration may still receive the complete transcript over HTTP. It avoids repeat GPU evaluation on a cache hit, but does not claim constant network or CPU request size. A delta-only custom Pi provider is optional later work.

## 5. Safety contract

The initial system exposes registered, schema-validated tools only. It must provide:

- project-root path confinement;
- no arbitrary host filesystem access;
- no credential or secret access;
- deterministic tools where possible;
- bounded reads and output truncation;
- per-call timeouts;
- bounded agent turns;
- repeated-call and cycle detection;
- rejection of destructive shell commands;
- disposable worktrees or a stronger sandbox for shell execution;
- explicit failure instead of fabricated results;
- complete action/result traces for evaluation.

Initial promotion does not require file mutation. `edit` and `write` remain disabled until read-only repository tasks pass their gate.

## 6. Ordered implementation milestones

Milestones are sequential gates. Do not begin production SFT or broad autonomous rollout merely because an earlier component compiles.

### M0. Preserve the v1 baseline

Deliverables:

- concise validation report with checkpoint and manifest hashes;
- exact and semantic metrics;
- representative successful and failed trajectories;
- explicit statement that the old evaluation teacher-forced tool observations and was not a true end-to-end agent test.

Gate: the report and immutable evaluation paths are committed.

### M1. Qualify cached dense E97 inference

Deliverables:

- public incremental inference/cache API separated from one-shot text generation;
- fused FP32 state retention when supported by the corrected valid-length kernel path;
- tests for replay-versus-incremental logits and greedy continuation;
- tests across arbitrary chunk and turn boundaries;
- tests for RS inclusion, clean reset, two-session isolation, branch mismatch, and aborted-generation rollback;
- cache metadata/state-size reporting.

Required result: cached and uninterrupted execution have exact greedy continuation and bounded documented logit drift. Any mismatch must be characterized before proceeding.

### M2. Implement the minimal model server

Deliverables:

- `GET /v1/models`;
- `POST /v1/chat/completions` with streaming SSE;
- deterministic Pi-message serialization;
- textual E97 action parser and structured tool-call response mapping;
- session-affinity support;
- transactional cache manager with prefix verification and replay fallback;
- bounded tokens, time, sessions, and cache eviction;
- structured diagnostics for cache hit, suffix length, replay reason, and state bytes.

Gate: protocol tests plus a local Pi request complete without executing a real host-mutating tool.

### M3. Run dense-agent v1 end to end through Pi

Use tools matching the v1 training contract first: calculator, lookup/search/read, and count/list. Pi must execute the model's own predicted actions and return actual observations. Do not substitute the teacher action or observation after a model error.

Deliverables:

- Pi model configuration;
- bounded Pi extension/tool definitions;
- automated isolated-session evaluator;
- full traces for the 308 excluded tasks;
- comparison with the teacher-forced v1 baseline;
- recurrent-cache hit/miss and GPU-memory measurements.

Gate: interface correctness, session isolation, and deterministic evaluation. Capability is measured but is not required to improve without retraining.

### M4. Add the safety boundary and real read-only Unix tools

Deliverables:

- Pi extension that gates tools and records policy decisions;
- disposable-worktree launcher;
- bounded `read`, `grep`, `find`, `ls`, and restricted `bash` policy;
- adversarial tests for traversal, secrets, destructive commands, timeouts, large output, and repeated loops.

Gate: all prohibited operations fail closed and leave the host/repository unchanged.

### M5. Build and train dense-agent v2

The v2 corpus uses exact promoted Pi tool names and schemas, compact typed observations, task-level split isolation, prompt paraphrases, final-value grounding, direct count/extract tools, empty-result recovery, tool errors, malformed-output repair, and loop avoidance.

Train two bounded one-node arms at LR `1e-5`:

1. fresh from the 513B dense base;
2. warm-started from dense-agent v1 update 128.

Retain checkpoints near updates 64, 128, 256, and one corpus epoch where applicable.

Promotion gates on excluded tasks:

- action syntax validity >= 99%;
- correct tool selection >= 95%;
- valid/exact arguments >= 98%;
- RS stopping >= 99%;
- end-to-end synthetic success >= 90%;
- repeated-action loops < 1%.

Training-stream loss and teacher-forced likelihood are diagnostic only.

### M6. Read-only repository agent

Begin with bounded, mechanically checkable tasks:

- symbol and definition lookup;
- configuration-default lookup;
- manifest comparison;
- checkpoint provenance;
- failing-log explanation;
- exact file/line citations.

Collect teacher trajectories inside Pi, retain only mechanically verified successes, and add corrections for states encountered by E97 rollouts. This is verified trajectory distillation/dataset aggregation, not unconstrained self-training.

Gate: a preregistered held-out repository panel passes its accuracy, citation, safety, stopping, and loop thresholds.

### M7. Optional mutation capability

Only after M6 passes, consider enabling `edit`, `write`, and test commands in disposable worktrees. Promotion requires clean-diff checks, tests, mutation-path confinement, rollback, and human review for consequential actions.

## 7. Evaluation rules

- Reset recurrent state between unrelated tasks.
- Preserve state across turns within one task.
- Never teacher-force a correct tool result after an incorrect model action in end-to-end scoring.
- Record raw model text, parsed call, policy decision, tool result, cache event, final response, and verifier result.
- Report exact and semantic success by task family.
- Report uncertainty for aggregate capability metrics.
- Distinguish protocol failures, safety-policy rejections, tool failures, loops, grounding failures, and ordinary wrong answers.
- Compare cached versus forced-replay outputs during qualification and periodically thereafter.
- Do not infer semantic capability from NLL or stopping alone.

## 8. Resource policy

The demo path is bounded to tens, not thousands, of node-hours. Most remaining allocation is reserved for foundation/renewal work. M0-M4 are implementation and small qualification work. M5 begins with the already planned cheap one-node arms. No large production submission is implied by this plan.

Frontier Python, pytest, builds, and submission preflight must source `scripts/frontier/activate_emender_frontier.sh` and use `$EMENDER_PYTHON`. Scheduler evidence must name both Partition and QOS explicitly. Jobs are never stopped or modified without operator approval.

## 9. Stop and pivot conditions

Pause before additional training if:

- cached inference is not reproducible against full replay;
- canonical Pi serialization cannot round-trip structured calls;
- safety enforcement depends on model compliance;
- end-to-end v1 failures are dominated by serving/protocol defects;
- v2 improves teacher-forced metrics without end-to-end grounding;
- the bounded demo path approaches the resource cap without passing a stated gate.

If a targeted v2 arm passes, stop broad synthetic expansion and move directly to read-only repository tasks.

## 10. Immediate work order

M0-M3 are complete. The active order is now:

1. Preserve the RS-free v3 warm-32 checkpoint and job 5310895 as bounded promotion authorities.
2. Treat the five changed calculator operands as argument-transcription failures; grounded finalization itself passed 291/291.
3. Add a small correction/verification path for operational arguments without weakening provenance checks.
4. Build and preregister a separate read-only repository-task panel in disposable task roots.
5. Keep edit and write disabled until that panel meets its accuracy and safety gates.
