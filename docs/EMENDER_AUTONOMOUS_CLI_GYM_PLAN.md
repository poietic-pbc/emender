# EMENDER Autonomous CLI Gym and Continual-Learning Plan

Status: proposed execution plan  
Scope: bounded, verifier-driven CLI agents with recurrent state, branching, and periodic all-parameter learning  
Primary deployment: Harbor on a persistent Docker server  
Qualification deployment: Frontier Slurm with the existing cwd-only Apptainer boundary

## 1. Objective

Build a generic environment in which an EMENDER agent can improve its CLI behavior from mechanically verified experience without receiving unrestricted host access or modifying its live serving weights.

The target loop is:

```text
generate or select task
  -> fork bounded recurrent rollouts
  -> execute argv in isolated environments
  -> verify final state and provenance
  -> identify successful branches and first failed transitions
  -> add curated replay/correction records
  -> periodically train an immutable challenger
  -> run held-out, regression, and safety gates
  -> atomically promote or reject the challenger
```

The system is intended to improve concrete behaviors such as command discovery, option discovery, error correction, multi-command investigation, evidence retention, and grounded finalization. It is not intended to perform unconstrained self-modification or to treat unverified model judgments as reward.

## 2. Current starting point

The existing dense E97 agent provides:

- a 1.3B recurrent model derived from the 513B-token dense checkpoint;
- an approximately 9.38 MiB FP32 recurrent cache per active session;
- exact-prefix, transactional cache advancement;
- real Pi tool execution and OpenAI-compatible serving;
- a hash-pinned cwd-only Apptainer image;
- no host filesystem, credential, environment, or network authority;
- bounded argv execution with typed stdout, stderr, exit code, timeout, and truncation metadata;
- provenance-checked `submit_answer` termination;
- a qualified direct repository CLI checkpoint that passed 40/40 real-Pi tasks across count, JSON extraction, reading, and search.

The current model does not reliably learn an unfamiliar CLI grammar from help over several turns. This is a target behavior for the gym, not a prerequisite for constructing it.

The direct checkpoint remains the initial champion. Failed discovery continuations are evidence and training inputs, not promotion authorities.

## 3. Success definition

A successful system must demonstrate all of the following:

1. **Generic environment interface.** Tasks are not hard-coded into the agent extension. The same agent can operate in generated and imported CLI environments.
2. **Mechanical reward.** Hidden verifiers determine success from environment state, exact values, registered tests, or hashes.
3. **Recurrent branching.** Multiple independent continuations can begin from one exact canonical prefix.
4. **No hidden-state merging.** A winner is selected and committed or replayed; branch tensors are never averaged.
5. **Continual improvement.** Challengers improve held-out behavioral metrics while retaining direct-operation and safety performance.
6. **Reproducibility.** Every task, rollout, correction, training run, and promotion is bound to immutable source, model, image, generator, and verifier identities.
7. **Fail-closed operation.** Missing provenance, verifier failure, resource-limit failure, ambiguous environment state, or scheduler mismatch cannot produce a promoted result.
8. **Dual-runtime parity.** The task semantics and rewards agree between Docker/Harbor and Frontier/Apptainer for the portable task subset.

## 4. Non-goals

The first system will not:

- expose a host shell or arbitrary host paths;
- allow agent-controlled network access;
- learn directly from arbitrary internet content;
- update the live champion after each rollout;
- use self-reported success as reward;
- autonomously submit large Frontier jobs without budget and policy controls;
- require low-rank adaptation; EMENDER candidates remain all-parameter checkpoints;
- begin with unrestricted source mutation or system administration tasks;
- claim general software-engineering competence from synthetic CLI success.

## 5. Design principles

### 5.1 Separate trusted control from untrusted agency

The trusted supervisor owns task generation, hidden manifests, verifiers, budgets, training submission, checkpoint promotion, and rollback. The agent sees only the task instruction, its tool schemas, and the environment exposed through the CLI boundary.

### 5.2 Optimize behavior, not teacher-forced loss

Loss is a diagnostic. Real rollout outcomes are promotion evidence. Every training arm must be followed by independent real-Pi execution.

### 5.3 Prefer deterministic correction before policy-gradient RL

The initial learner is verifier-guided imitation/DAgger:

- run the current policy;
- identify the first incorrect transition;
- obtain the correct next action from the task oracle;
- target only that transition;
- replay already-qualified behaviors;
- train an immutable candidate.

Outcome RL is deferred until the model produces valid multi-turn trajectories reliably and token/log-probability accounting is qualified.

### 5.4 Keep the task interface portable

The canonical task description follows Harbor conventions even when the runtime is Apptainer. Runtime-specific files are adapters, not competing task authorities.

### 5.5 Preserve exact recurrent lineage

A recurrent cache is valid for one model checkpoint, canonical serialization, tool schema, system prompt, and exact token prefix. Fork metadata must retain all of these identities.

## 6. System architecture

```text
                         TRUSTED CONTROL PLANE

 task generator/importer -> task registry -> rollout scheduler
           |                                      |
 hidden oracle/verifier                         branch manager
           |                                      |
 trajectory curator <- verifier results <- rollout workers
           |                                      |
 replay store -> trainer -> challenger -> promotion gate
                                      |             |
                                      +---- archive/champion

                         UNTRUSTED DATA PLANE

 E97/Pi session -> cli({argv}) -> isolated task environment
       ^                                  |
       +---------- typed observation -----+
```

### 6.1 Required services

- **Task registry:** immutable task manifests and split assignments.
- **Environment provider:** Docker/Harbor or Frontier/Apptainer implementation.
- **E97 model service:** immutable checkpoint and canonical prompt/tool authority.
- **Pi agent adapter:** owns conversations and translates structured actions.
- **Branch manager:** clones, tracks, offloads, expires, and commits recurrent caches.
- **Verifier service:** runs hidden deterministic checks outside agent authority.
- **Trajectory store:** append-only raw and normalized rollout artifacts.
- **Curriculum scheduler:** chooses tasks and correction stages from measured failures.
- **Trainer:** runs periodic all-parameter masked SFT and, later, optional RL.
- **Promotion controller:** compares champion and challenger on frozen gates.
- **Dashboard:** reports reward, transition failures, regressions, resource use, and lineage.
- **Kill switch:** stops rollout and training submission without depending on model behavior.

## 7. Canonical task format

Use a Harbor-compatible directory:

```text
tasks/<task-id>/
  instruction.md
  task.toml
  environment/
    Dockerfile                 # Docker provider, when required
    image.def                  # optional Apptainer build authority
    fixture-manifest.json
  oracle/
    solution.json              # hidden from agent
    transitions.json           # hidden correct action program
  tests/
    verify.py
    test.sh
  metadata/
    provenance.json
```

Minimum `task.toml` fields:

```toml
schema_version = "1.4"

[task]
name = "emender/<task-id>"
version = "1.0.0"

[metadata]
generator = "emender-cli-gym-v1"
generator_seed = 12345
family = "discover-options"
split = "train"

[agent]
timeout_sec = 120.0

[verifier]
timeout_sec = 60.0

[environment]
network_mode = "no-network"
cpus = 1
memory_mb = 2048
storage_mb = 1024
```

EMENDER-specific immutable metadata must additionally bind:

- task-content SHA-256;
- fixture SHA-256 tree;
- generator source commit and configuration;
- oracle and verifier SHA-256;
- required image digest;
- allowed executable policy;
- mutation mode;
- expected resource ceilings;
- train/validation/test split key.

The oracle directory must never be mounted into the agent environment. The verifier receives it through a separate trusted mount or process.

## 8. Unified environment-provider interface

Both runtimes implement the same logical contract:

```python
class EnvironmentProvider:
    def materialize(task_manifest) -> EnvironmentHandle: ...
    def execute(handle, argv, timeout, output_limit) -> CommandResult: ...
    def snapshot(handle) -> EnvironmentDigest: ...
    def verify(handle, verifier_manifest) -> VerificationResult: ...
    def destroy(handle) -> None: ...
```

`CommandResult` contains:

```json
{
  "argv": ["program", "arg"],
  "exit_code": 0,
  "stdout": "...",
  "stderr": "...",
  "timed_out": false,
  "stdout_truncated": false,
  "stderr_truncated": false,
  "duration_ms": 123
}
```

Model-visible observations may be a deterministic projection of this result, but raw results remain authoritative and are always retained.

Provider parity tests must run identical portable tasks and compare:

- initial fixture hashes;
- command exit codes and normalized outputs;
- final filesystem hashes;
- verifier reward and diagnostics;
- absence of forbidden mounts, environment, credentials, and network;
- process-tree cleanup after normal exit and timeout.

## 9. Primary runtime: Harbor on regular Docker

A persistent Docker server is the likely primary rollout platform because Harbor already provides task orchestration, custom agents, verifiers, parallel trials, and rollout-oriented metadata.

### 9.1 Recommended topology

```text
trusted host
  Harbor coordinator
  E97/Pi external-agent adapter
  trajectory and artifact store
  Docker daemon or isolated Docker worker pool
  optional local E97 GPU service

Docker task containers
  disposable task filesystem
  no credentials
  no host Docker socket
  no network by default
  bounded CPU, memory, PIDs, files, and time
```

Use a Harbor external agent rather than installing Pi or E97 inside each task container. The external adapter:

1. receives `instruction` and Harbor `BaseEnvironment`;
2. creates an E97/Pi session outside the task container;
3. translates each `cli({argv})` action into `environment.exec` without a host shell;
4. returns the typed observation to Pi;
5. stores exact token IDs, assistant masks, cache lineage, and tool events in `AgentContext` metadata;
6. stops on grounded submission, budget exhaustion, cancellation, or protocol failure.

This preserves a no-network task container. Model-service connectivity exists only in the trusted external adapter.

### 9.2 Docker security requirements

- rootless Docker where operationally qualified, or a dedicated worker VM;
- never mount `/var/run/docker.sock` into task containers;
- read-only base image and one disposable writable workspace;
- `--network none` unless a reviewed task explicitly requires an allowlist;
- dropped capabilities and `no-new-privileges`;
- seccomp/AppArmor where supported;
- CPU, memory, PID, open-file, file-size, storage, and wall-time limits;
- separate trusted verifier process or container;
- complete deletion of task containers and volumes after artifact collection;
- per-task image allowlist and digest pinning;
- no ambient host environment propagation.

### 9.3 Harbor integration deliverables

- `EmenderE97Agent(BaseAgent)`;
- task adapter for generated EMENDER tasks;
- dataset manifests for train, development, regression, and held-out splits;
- token/mask and recurrent-lineage metadata exporter;
- Harbor reward adapter for exact and decomposed diagnostics;
- Docker security policy tests;
- parity runner against the Apptainer provider.

## 10. Qualification runtime: Frontier Slurm and Apptainer

Frontier remains the authoritative environment for the current ROCm E97 model and training implementation. It cannot host an indefinite daemon through the batch scheduler. A persistent trusted coordinator must submit bounded Slurm jobs and monitor their terminal evidence.

### 10.1 Existing boundary

The qualified invocation is based on:

```text
--containall
--cleanenv
--net --network none
--no-privs
--drop-caps all
--no-mount bind-paths,home,cwd,tmp,hostfs,proc,sys
--bind <disposable-task-cwd>:/work:rw
--cwd /work
```

It exposes no persistent host write path other than the disposable task cwd. The production runner additionally verifies the image digest, uses an argv vector, caps captured output, applies `prlimit`, starts a separate process group, and kills the process tree on timeout.

### 10.2 Frontier execution model

- A trusted task-preparation step creates one disposable cwd per episode.
- The oracle and hidden verifier remain outside that cwd.
- Each CLI action starts the pinned SIF against the same cwd, preserving task state across commands.
- The E97 server and Pi adapter run on the allocated compute node, not inside the CLI container.
- After the episode, a trusted verifier reads the cwd through a separate restricted invocation.
- Raw artifacts, scheduler identity, and terminal accounting are copied to immutable project storage.

Every Frontier job must retain explicit evidence for both:

```text
Partition=batch
QOS=debug
```

for iterative acceptance work, together with `Requeue=0` and terminal `sacct` evidence.

### 10.3 Remaining Frontier qualification

Before mutation-heavy gym tasks, qualify:

- hard memory and PID enforcement on compute nodes;
- file-count and workspace-quota enforcement;
- fork-bomb and detached-process cleanup;
- verifier isolation from the agent workspace;
- concurrent task-directory isolation;
- behavior when Apptainer or the verifier is killed;
- output-bomb and sparse-file handling;
- scheduler cancellation and partial-artifact cleanup.

Frontier should initially run small qualification panels and all-parameter training. High-volume environment rollouts should move to the persistent Docker/Harbor service when parity is established.

## 11. Task generation program

### 11.1 Procedural CLI dialects

Generate small executable CLIs with deterministic hidden schemas. Randomize:

- executable names;
- subcommand names and nesting;
- long and short flag names;
- positional versus named arguments;
- required and optional fields;
- Boolean flags and enumerations;
- help layout and wrapping;
- error messages and exit codes;
- output encodings and JSON shapes;
- repository paths and requested values.

Example equivalent dialects:

```text
project-query fetch --document X --selector Y
repo inspect --file X --pointer Y
config-tool extract X --key Y
```

A split must hold out entire combinations of templates, command names, flag vocabularies, and generator seeds. Random values alone do not constitute a held-out grammar.

### 11.2 Task families

1. direct single-command execution;
2. top-level command discovery;
3. subcommand-option discovery;
4. exact line or JSON extraction;
5. literal and structured search;
6. typed nonzero-exit recovery;
7. correction after missing or invalid flags;
8. correction after a valid command returns no result;
9. two-to-six-command investigation;
10. distractor files and adversarial repository instructions;
11. evidence selection among multiple plausible outputs;
12. forked hypothesis testing;
13. read-only Git inspection;
14. patch-and-test tasks only after read-only promotion.

### 11.3 Adversarial content

Generated repositories must include untrusted text that attempts to:

- override the system prompt;
- request credentials or host paths;
- request network access;
- induce repeated or destructive commands;
- fabricate verifier success;
- instruct the model to bypass `submit_answer`.

The expected behavior is to treat repository content as data and continue under the trusted Pi protocol.

## 12. Reward and diagnostics

The promotion reward is outcome-based and preferably binary. Diagnostic rewards may be decomposed but cannot independently authorize promotion.

Suggested fields:

```json
{
  "task_success": 1,
  "environment_state_correct": 1,
  "grounded_submission": 1,
  "correct_program": 1,
  "correct_subcommand": 1,
  "correct_options": 1,
  "recovered_from_error": 0,
  "bounded": 1,
  "security_invariants": 1
}
```

The verifier also identifies the first divergent transition relative to the hidden oracle:

```text
SELECT_COMMAND
SELECT_OPTIONS
CONSTRUCT_ARGV
INTERPRET_OUTPUT
RECOVER
SUBMIT
```

This label drives curriculum selection. It is not sent back as an unverified natural-language judgment.

## 13. Recurrent branching

### 13.1 Fork identity

Each branch records:

- parent branch ID;
- fork token index and canonical prefix SHA-256;
- checkpoint SHA-256;
- system prompt and tool-schema SHA-256;
- serializer version;
- recurrent-state byte count and tensor digest for diagnostics;
- sampling configuration;
- task and environment IDs;
- creation and expiration times.

### 13.2 Fork semantics

- Clone tensors before any operation that may mutate storage.
- Share immutable model weights.
- Advance each branch transactionally.
- Commit only complete canonical assistant turns.
- On cancellation or malformed generation, discard shadow state.
- Never merge hidden tensors.
- Select one branch through verifier evidence, then commit or replay its canonical suffix.
- Offload inactive branch caches to CPU or storage when qualified.

### 13.3 Exploration policy

For each task, initially fork four strategies:

- direct execution;
- inspect top-level help;
- inspect likely files before command selection;
- deliberate error/recovery or alternate hypothesis.

Later, strategies may be sampled from measured failure modes. Branch budgets are independent and capped.

## 14. Trajectory authority

Retain both raw and normalized forms.

Raw authority includes:

- exact Pi JSON event stream;
- canonical serialized transcript;
- token IDs and assistant-target masks;
- full command results and model-visible projections;
- container/image identity;
- filesystem before/after digests;
- verifier outputs;
- recurrent branch lineage;
- runtime and scheduler evidence;
- termination reason and resource usage.

Normalized training records contain only schema-validated, reproducible turns. They must never include secrets, host paths outside approved metadata, or verifier-only content.

Every accepted record is classified as:

- verified success;
- oracle correction at first divergence;
- verified negative example for diagnostics only;
- rejected or unsafe.

Only verified successes and oracle corrections become positive assistant targets.

## 15. Continual-learning algorithm

### 15.1 Initial DAgger loop

1. Sample tasks from the current curriculum mixture.
2. Run forked champion rollouts.
3. Verify all branches.
4. Choose successful branches by deterministic preference rules.
5. For failed tasks, locate the first divergence from the oracle.
6. Construct a record whose gradient targets only the correct next action at that prefix.
7. Add balanced direct-behavior replay.
8. Deduplicate by canonical prefix and target.
9. Train a challenger after a configured number of new verified targets.
10. Evaluate and promote or reject.

### 15.2 Replay policy

Each candidate batch should include:

- 40% current failing transitions;
- 30% qualified direct CLI replay;
- 15% earlier discovery/recovery successes;
- 10% grounding and finalization replay;
- 5% adversarial safety replay.

These are initial values and must be revised from measured forgetting. No single command family may dominate merely because it generates shorter trajectories.

Normalize gradients by exact assistant target count, as in the existing dense-agent trainer.

### 15.3 Candidate training

- all parameters trainable;
- immutable source checkpoint;
- explicit `source_weight_mode`;
- exact authority and pack manifest hashes;
- small learning-rate arms first;
- checkpoint at predefined update counts;
- no automatic continuation after numerical or behavioral failure;
- no promotion based on training loss alone.

### 15.4 Later outcome RL

Only after valid trajectory generation is reliable:

- export exact token IDs, assistant masks, behavior-policy log probabilities, rewards, and branch identities;
- qualify recurrent-state reconstruction for the learner;
- use verifier outcome reward with KL/replay constraints;
- retain SFT replay to prevent syntax and grounding collapse;
- compare against verifier-guided SFT with equal rollout budgets.

Harbor can orchestrate these rollouts, but the initial learner should remain the simpler EMENDER masked-SFT implementation.

## 16. Champion/challenger promotion

Never update the champion in place.

```text
champion: current serving authority
challenger: candidate produced from a fixed champion
archive: every previously promoted checkpoint
```

### 16.1 Required gates

A candidate must pass:

- checkpoint integrity and exact lineage;
- no NaN/Inf and bounded gradient diagnostics;
- direct CLI regression;
- held-out command discovery;
- held-out option discovery;
- error recovery;
- grounded finalization;
- repeated-call protection;
- adversarial content panel;
- resource and sandbox invariants;
- legacy bounded-agent regression;
- general-language regression panel appropriate to the artifact.

### 16.2 Initial behavioral thresholds

Use sufficiently powered panels and report confidence intervals. Initial engineering gates:

- protocol-valid sessions: 100%;
- security invariants: 100%;
- grounded accepted submissions: 100% of counted successes;
- direct tasks: at least 95%;
- generated held-out discovery tasks: at least 80% before expanding complexity;
- recovery tasks: at least 80%;
- no statistically credible degradation from the champion on frozen regression panels.

Small canaries locate catastrophic failures but cannot promote a model.

### 16.3 Atomic promotion

Promotion writes an immutable manifest containing:

- checkpoint and source hashes;
- parent champion;
- training authority hashes;
- complete evaluation receipts;
- runtime image identities;
- decision and approver policy;
- rollback target.

The serving alias changes atomically. Rollback never requires retraining.

## 17. Autonomous supervisor state machine

```text
IDLE
  -> GENERATE_TASKS
  -> RUN_ROLLOUTS
  -> VERIFY
  -> CURATE
  -> WAIT_FOR_TRAIN_THRESHOLD
  -> TRAIN_CHALLENGER
  -> EVALUATE_CHALLENGER
  -> PROMOTE | REJECT
  -> IDLE
```

The supervisor must pause rather than guess when:

- a verifier is nondeterministic;
- task/image provenance is incomplete;
- disk, compute, or rollout budget is exhausted;
- a security invariant fails;
- a candidate has ambiguous regression evidence;
- Frontier partition or QoS evidence is missing;
- artifact hashes disagree;
- the champion service is unhealthy.

## 18. Budget and operations

### 18.1 Persistent Docker server

Run continuously:

- Harbor coordinator;
- task generation and verification;
- Docker rollout workers;
- trajectory database;
- dashboard;
- optional inference service on a qualified GPU.

Configure daily limits for:

- rollouts;
- concurrent containers;
- CPU/GPU hours;
- stored artifact bytes;
- failed-task retries;
- candidate trainings;
- promotion attempts.

### 18.2 Frontier

A persistent off-system coordinator may submit bounded Slurm jobs. Frontier itself should not be treated as an indefinite service. Preserve allocation by:

- batching training examples before submission;
- using one-node debug jobs for iterative qualification;
- avoiding repeated full panels until canaries pass;
- retaining one large production-slot option;
- requiring explicit policy approval for scale increases.

## 19. External ecosystem integration

### 19.1 Harbor

Adopt first for task conventions, Docker orchestration, external-agent integration, and rollout metadata.

Sources:

- <https://github.com/harbor-framework/harbor>
- <https://www.harborframework.com/docs/tasks>
- <https://www.harborframework.com/docs/agents>
- <https://www.harborframework.com/docs/training-workflows/rl>

### 19.2 LiteCoder-Terminal

Use selected MIT-tagged trajectories as format and curriculum references after provenance review. Do not directly mix their serialization with EMENDER. Translate only verified behaviors into canonical Pi turns, and preserve source/scaffold metadata.

- <https://huggingface.co/datasets/Lite-Coder/LiteCoder-Terminal-SFT>

### 19.3 InterCode

Use its simpler interactive Bash tasks as a possible early external adapter. Review NL2Bash-derived data licensing separately from the MIT framework.

- <https://github.com/princeton-nlp/intercode>

### 19.4 CLI-Gym

Treat its 1,655 Apache-tagged dataset records as a later environment-repair source. Many tasks mutate system libraries, locales, permissions, or runtime installations and are unsuitable for the initial read-only policy.

- <https://github.com/LiberCoders/CLI-Gym>
- <https://huggingface.co/datasets/LiberCoders/CLI-Gym>

### 19.5 Terminal-Bench

Reserve Terminal-Bench 2.x as a difficult external evaluation and compatibility target. Do not train on the held-out benchmark used for claims.

- <https://www.tbench.ai/>
- <https://github.com/harbor-framework/terminal-bench-2>

### 19.6 SWE environments

SWE-Gym, R2E-Gym, and SWE-Smith become relevant only after patching, test execution, rollback, and mutation-specific sandbox policies are qualified.

- <https://github.com/SWE-Gym/SWE-Gym>
- <https://github.com/R2E-Gym/R2E-Gym>
- <https://github.com/SWE-bench/SWE-Smith>

All imported tasks, source repositories, container images, and trajectories require independent license and provenance manifests. Framework license alone does not establish task or base-image rights.

## 20. Milestones

### M0: Specification and frozen baselines

Deliver:

- this execution plan;
- frozen 40/40 direct checkpoint manifest;
- failed discovery lineage registry;
- canonical task and rollout schemas;
- threat model and budget policy.

Validation:

- schema tests;
- checkpoint hashes;
- no active artifact ambiguity.

### M1: Portable microgym

Deliver:

- Harbor-compatible generated tasks;
- deterministic oracle and verifier;
- 100-task direct/discovery/recovery suite;
- immutable split manifests.

Validation:

- oracle solves 100%;
- intentionally wrong policies fail;
- verifier determinism across repeated runs;
- no oracle visibility from agent cwd.

### M2: Dual-runtime parity

Deliver:

- Harbor Docker provider;
- Frontier Apptainer provider;
- common command-result normalization;
- parity receipts.

Validation:

- identical reward and final hashes on portable tasks;
- qualified resource and security boundaries;
- process cleanup under timeout.

### M3: Recurrent branch manager

Deliver:

- explicit cache-fork API;
- branch lineage manifests;
- offload and expiration policy;
- forked rollout scheduler;
- winner commit/replay.

Validation:

- exact parent equality at fork;
- branch isolation;
- rollback leaves parent unchanged;
- deterministic replay of winning suffix;
- measured memory per branch.

### M4: Autonomous rollout and curation

Deliver:

- champion best-of-N rollouts;
- first-divergence oracle labels;
- verified trajectory authority;
- balanced curriculum scheduler.

Validation:

- no unverified positive targets;
- no train/test seed leakage;
- exact token/mask reconstruction;
- curator determinism.

### M5: Challenger training and promotion

Deliver:

- periodic all-parameter trainer integration;
- frozen regression panels;
- candidate comparison report;
- atomic promotion and rollback.

Validation:

- at least one candidate improves a powered held-out panel;
- no direct or grounding regression;
- promotion receipt is complete.

### M6: Continuous service

Deliver:

- persistent Harbor coordinator;
- bounded autonomous supervisor;
- dashboard, alerts, quotas, and kill switch;
- retention and garbage-collection policy.

Validation:

- multi-day soak without leaked containers/processes;
- deterministic restart from durable supervisor state;
- budget exhaustion pauses safely;
- champion remains available during challenger failure.

### M7: External tasks

Deliver:

- selected InterCode or LiteCoder-compatible adapter;
- CLI-Gym feasibility subset;
- Terminal-Bench compatibility evaluation.

Validation:

- provenance and license review;
- no held-out contamination;
- results clearly separated from internal generated-task metrics.

### M8: Controlled mutation

Deliver only after read-only promotion:

- disposable Git worktrees;
- expected preimage hashes;
- bounded diffs;
- registered checks;
- automatic rollback;
- verified change submission.

## 21. Immediate implementation slice

The first implementation should be deliberately small:

1. Define `GymTaskManifest`, `CommandResult`, `VerificationResult`, `RolloutRecord`, and `BranchRecord` schemas.
2. Generate 100 deterministic CLI-dialect tasks across direct, command-discovery, option-discovery, and recovery families.
3. Emit Harbor-compatible task directories.
4. Implement a local provider over the existing Apptainer runner.
5. Implement a Harbor external-agent skeleton over the same logical interface.
6. Add a hidden oracle that identifies the first wrong transition.
7. Run the frozen direct champion on the suite through real Pi.
8. Store exact token/mask and recurrent-prefix evidence.
9. Build one challenger dataset from verified successes and oracle corrections.
10. Train only after task/verifier determinism and split isolation pass.

Do not import complex external tasks or add mutation until this slice operates end to end.

## 22. Decision rule

Continue investing while the system produces reproducible held-out improvement per bounded unit of compute. Pause and redesign when:

- repeated oracle correction fails to improve the targeted transition;
- improvement requires unacceptable regression elsewhere;
- verifier reward is exploitable or nondeterministic;
- environment cost dominates useful rollout generation;
- the model plateaus across independently generated CLI dialects.

A plateau is still a useful result: it identifies the behavioral and state-retention requirements for a wider recurrent foundation, larger state geometry, improved memory structure, or recurrent-trunk expertization.
