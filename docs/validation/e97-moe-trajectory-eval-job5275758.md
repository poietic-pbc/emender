# E97-MoE four-checkpoint trajectory evaluation — job 5275758

Date: 2026-08-15  
Verdict: **282B is the provisional masked-SFT parent; reject further all-token instruction continuation; no checkpoint demonstrates robust retrieval or assistant behavior.**

## Execution authority

Job `5275758` completed `0:0` in 17m05s on four Frontier nodes. Each checkpoint
used one independent eight-GCD node-local expert-parallel island.

- `Partition=batch`
- `QOS=debug`
- `Requeue=0`
- source commit: `d730590d1b6b162019950438041299180e7314be`
- panel schema: `emender-e97-moe-paired-eval-panel-v2`
- panel SHA-256: `fcb1fbd09ca38c27fec03be945c7cbfcb45cff4dbcaf8a4bf4afcb2ef013018f`

Results:

`/lustre/orion/bif148/proj-shared/emender/evaluations/e97-moe-paired-v2/runs/job-5275758/results/`

The four checkpoint runners and final trajectory postprocessor all returned zero.

## Checkpoints

| Label | Accepted tokens | Role |
|---|---:|---|
| 250B | 250,797,359,104 | base authority |
| 282B | 282,070,089,728 | long-context continuation, mostly 32K at LR `1e-4` |
| 300B | 300,860,571,648 | 18.790B instruction-shaped all-token continuation |
| 304B | 304,618,668,032 | 22.549B instruction-shaped all-token continuation |

## Aggregate results

| Checkpoint | Wiki 2K NLL | Wiki 8K NLL | Wiki 32K NLL | External assistant-response NLL | MMLU | HellaSwag normalized |
|---|---:|---:|---:|---:|---:|---:|
| 250B | 3.359619 | 3.277222 | 3.256958 | 2.075413 | 21.48% | 44.53% |
| 282B | **3.356934** | **3.272949** | **3.254517** | **2.072752** | 19.92% | 44.53% |
| 300B | 3.376465 | 3.293579 | 3.273071 | 2.100993 | 18.75% | 43.75% |
| 304B | 3.378418 | 3.295410 | 3.277466 | 2.102958 | 19.53% | 44.92% |

Absolute WikiText NLL values across different context panels must not be compared
as a context-length effect because the panels contain different window sets.
Within-context checkpoint deltas are paired and valid.

## 250B to 282B: conservative, small general improvement; no long retrieval

Paired WikiText NLL changes were small but consistently favorable:

- 2K: `-0.002686`, bootstrap 95% interval `[-0.004150, -0.001221]`;
- 8K: `-0.004272`, interval `[-0.006104, -0.002563]`;
- 32K: `-0.002441`, interval `[-0.004639, -0.000610]`.

External assistant-response NLL also improved slightly. The paired per-example
mean change was `-0.004223`, interval `[-0.006738, -0.001795]`; token-weighted
aggregate NLL changed from `2.075413` to `2.072752`.

This is evidence that the low-LR continuation was not useless or divergent. It
made a small conservative LM update. It did **not**, however, create measurable
retrieval ability. Position-bucket changes were not preferentially concentrated
at late positions; the improvement looks like general continued training rather
than learned use of distant context.

MMLU fell by 1.56 points on this small diagnostic, with bootstrap interval
`[-3.52, 0.00]` points. HellaSwag normalized accuracy was exactly unchanged.
MMLU absolute accuracy is below four-choice chance under this raw-model prompt,
so its relative movement is diagnostic rather than a standalone capability
estimate.

## Long-context retrieval verdict

Each task had eight answer choices, so chance accuracy is 12.5%. Natural-text
filler results were noisy around chance at every distance and checkpoint:

| Checkpoint | 2K | 4K | 8K | 16K | 24K | 30K |
|---|---:|---:|---:|---:|---:|---:|
| 250B | 15.63% | 6.25% | 21.88% | 12.50% | 3.13% | 9.38% |
| 282B | 12.50% | 3.13% | 15.63% | 12.50% | 6.25% | 9.38% |
| 300B | 9.38% | 3.13% | 12.50% | 12.50% | 9.38% | 12.50% |
| 304B | 9.38% | 3.13% | 12.50% | 12.50% | 9.38% | 12.50% |

All mean correct-answer margins were negative. No checkpoint preferred the
correct value over its strongest distractor at any distance. Single-key,
RS-separated, and multikey variants were all near chance.

The earlier v1 result of perfect 2K retrieval with repeated synthetic filler did
not reproduce under varied natural filler. That result was task/filler-specific,
not robust model memory. The defensible conclusion is that none of these
checkpoints currently demonstrates usable passkey retrieval, including at 2K.

The long-context continuation therefore receives this narrow verdict:

- small general NLL improvement;
- no demonstrated long-distance retrieval improvement;
- no basis for claiming a useful 32K memory capability.

Targeted long-supervision canaries remain justified, but blind generic 32K
continuation does not.

## Instruction continuation verdict

From 282B to 300B:

- WikiText regressed by `+0.0195` to `+0.0206` nats/token across contexts, with
  paired bootstrap intervals entirely above zero;
- external assistant-response NLL regressed: paired mean `+0.03155`, interval
  `[+0.02187, +0.04191]`;
- retrieval margins generally moved downward;
- MMLU and HellaSwag did not provide compensating evidence.

From 300B to 304B:

- WikiText regressed another `+0.0018` to `+0.0044` nats/token;
- assistant-response NLL regressed another paired `+0.00438`, interval
  `[+0.00142, +0.00801]`;
- small MMLU/HellaSwag movements were mixed and too narrow to establish a new
  capability;
- retrieval remained chance.

Thus the all-token instruction phase made external reference responses **less**
likely while damaging general LM likelihood. Its low training loss measured
predictability of the training mixture, not assistant alignment.

No further all-token instruction continuation is scientifically authorized.

## Native-template generation

The v2 panel corrected formatting to native blank-line-separated
`System/User/Assistant` records, added `Assistant reasoning:` prompts, stopped on
RS token 218 or EOT, and tested greedy plus fixed-seed top-p decoding.

Greedy generation:

- all 16 prompts at every checkpoint ran to the 128-token cap;
- no checkpoint emitted RS or EOT;
- outputs were short generic fragments followed by repeated newlines;
- neither direct, reasoning, code, JSON, nor tool prompts were followed.

Sampled generation:

- content was incoherent mixtures of prose, code fragments, citations, numbers,
  and markup for every checkpoint;
- 250B and 300B never stopped;
- 282B stopped on two of 16 prompts;
- 304B stopped on five of 16 prompts, but the preceding content remained
  incoherent.

More frequent RS emission is not instruction following. Exact native templates
therefore do not reveal a hidden assistant capability in 300B or 304B.

## Routing, state, and memory health

All sampled recurrent states were finite. Maximum absolute state was 1.0, as
expected from a tanh-bounded recurrence, but the current diagnostic does not
measure the fraction saturated near ±1 and should be extended before the long
canary.

Routing remained stable rather than collapsed:

| Checkpoint | Worst max/mean expert load | Worst load CV | Minimum normalized entropy |
|---|---:|---:|---:|
| 250B | 4.788 | 1.082 | 0.894 |
| 282B | 4.709 | 1.058 | 0.897 |
| 300B | 4.764 | 1.088 | 0.888 |
| 304B | 4.755 | 1.101 | 0.885 |

Instruction continuation modestly worsened aggregate imbalance/entropy, but not
to the point of collapse. Peak HBM was approximately 13GB for every checkpoint.

## Parent decision

Use **282B as the provisional masked-SFT parent** because it has the best paired
WikiText and external assistant-response likelihood, stable routing, and no
HellaSwag regression relative to 250B. Do not describe it as long-context
capable; it is simply the strongest conservative LM parent in this panel.

Keep 250B as the rollback/control. The corrected evidence does not justify using
300B or 304B as the primary SFT parent. If masked SFT from 282B fails, compare a
small 250B control before revisiting 304B. Spending the first parent-comparison
arm on 304B is no longer the highest-value experiment.

## Authorized next experiment

1. Complete and validate the immutable stock Tülu 3 token-plus-mask authority.
2. Implement assistant-masked loss and deterministic record packing.
3. Qualify the objective at one node and K8 multinode.
4. Run matched `2e-6` and `5e-6` masked-SFT canaries from 282B.
5. Evaluate coherent response generation, RS termination, assistant NLL, general
   regression, and routing before extending token budget.
6. Add targeted long examples only after basic assistant behavior works.

The first canary should answer whether correct masking can turn the viable 282B
LM parent into an assistant. It should not attempt to solve alignment, long
memory, and tool use simultaneously.
