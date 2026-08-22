# E97 4B learning-rate rapid-interference scan

Status: approved for execution on the local 8x RTX 6000 Ada host.

## Question

For the qualified square-readout E97 shape (`d=3840`, `L=18`, `H=60`,
`n=64`, MLP ratio 2.5), what Schedule-Free AdamW learning-rate region gives
fast early optimization at per-island batch 32 without the destructive
interference observed at `1.007e-3`?

This is a hyperparameter screen, not an architecture comparison and not an
elastic/resilient-DiLoCo conformance test.

## Evidence motivating the scan

Two fixed-world eight-island runs used identical model, seed, counter-sampler
identity, batch, context, and DiLoCo settings; only learning rate differed.

* `4.7431158698290157e-4` completed 1,000,341,504 tokens with final reported
  training loss 3.7267.
* `1.007e-3` was stopped at step 91. At matched step 88 its loss was 7.5442,
  versus 6.5192 for the lower-LR control. The recent ten-report paired gap was
  +0.9348 nats for the higher LR.

The trajectories crossed. At step 24 the ultimately losing `1.007e-3` arm
looked better (8.1126 versus 9.0176), while by step 88 it was worse by 1.0250.
Therefore the historical 15-minute CMA candidate budget is invalid for this
shape: it would rank an initialization transient rather than sustained early
optimization.

## Phase A: deterministic interference bracket

Eight exact 4B candidates run concurrently, one per GPU:

| arm | learning rate |
|---|---:|
| lr0400 | 0.0004000 |
| lr0474 | 0.00047431158698290157 (completed-run anchor) |
| lr0550 | 0.0005500 |
| lr0630 | 0.0006300 |
| lr0720 | 0.0007200 |
| lr0820 | 0.0008200 |
| lr0920 | 0.0009200 |
| lr1007 | 0.0010070 (failed-run anchor) |

Every arm uses:

* exact shape `d3840-L18-H60-n64-mlp2.5`, 4,045,972,080 parameters;
* BF16 fused E97 recurrence, fail closed on eager fallback;
* Schedule-Free AdamW with pinned-CPU BF16 `z` and second moments;
* batch 32, context 2,048, gradient accumulation 1;
* gradient checkpoint groups of two, projection chunk 512, loss chunk 128;
* weight decay 0.01, gradient clipping 1.0, no warmup;
* seed 42;
* the frozen local Pile corpus and p50k tokenizer identities;
* counter sampler schema `emender-byte-window-counter-v1`, key 42, declared
  data world one;
* 96 optimizer steps and logging every four steps.

Candidates intentionally run as eight independent world-one processes. This
isolates the inner optimizer and makes all arms consume the same samples. It
does **not** test DiLoCo merge interaction. The selected LR must subsequently
pass a fresh fixed-world eight-way DiLoCo gate.

The exact 4B shape is retained rather than using a small proxy because
learning-rate transfer across width and batch is the uncertainty under test.

### Fitness

Primary fitness is the arithmetic mean of the seven reported training losses
at steps 72, 76, 80, 84, 88, 92, and 96. Lower is better. Because seed and data
are paired, candidate differences are paired differences rather than
independent-corpus noise.

The receipt also records:

* each step/loss/gradient/throughput trajectory;
* mean loss at steps 4-32 and 40-64;
* maximum reported gradient norm;
* exit status and non-finite/OOM evidence;
* source commit, full command, GPU, and NUMA node.

A crash, missing step 96, non-finite report, or missing fused-path evidence is
an infinite-fitness failure. Initial minimum loss is not a fitness because it
would reproduce the observed crossover error.

Candidate final checkpoints are not model releases. After metrics and commands
are durably recorded, loser checkpoint payloads may be pruned; logs, args, and
the aggregate receipt remain evidence.

## Phase A result and lower-bound reposition

Phase A executed from commit `170bf6c7`; all eight arms completed and passed
the fused guard. Fitness was:

| rank | LR | steps 72-96 mean |
|---:|---:|---:|
| 1 | 0.0004000 | 6.113814 |
| 2 | 0.00047431158698290157 | 6.685557 |
| 3 | 0.0006300 | 7.132157 |
| 4 | 0.0005500 | 7.166186 |
| 5 | 0.0007200 | 7.306214 |
| 6 | 0.0008200 | 7.410514 |
| 7 | 0.0009200 | 7.628100 |
| 8 | 0.0010070 | 7.708186 |

The winner hit the lower configured boundary by a material margin. Following
the same boundary-reposition rule used by the earlier per-architecture CMA
protocol, CMA refinement is deferred: fitting covariance around a clipped
boundary winner would be invalid. A second deterministic population moves the
range downward while retaining `4.0e-4` as the overlap anchor:

`[1.0, 1.3, 1.7, 2.2, 2.8, 3.3, 3.65, 4.0]e-4`.

Absolute losses from this world-one scan must not be compared directly with
the earlier world-eight trajectories because the counter-sampler world changes
the data stream. Rankings inside each paired population are the evidence.

The lower bracket also completed with every arm valid:

| rank | LR | steps 72-96 mean |
|---:|---:|---:|
| 1 | 0.000400 | 6.151371 |
| 2 | 0.000330 | 6.154643 |
| 3 | 0.000280 | 6.158914 |
| 4 | 0.000220 | 6.183414 |
| 5 | 0.000365 | 6.207643 |
| 6 | 0.000170 | 6.310014 |
| 7 | 0.000130 | 6.351286 |
| 8 | 0.000100 | 6.501300 |

This finds a broad stable basin around `2.2e-4` through `4.0e-4`, not a
CMA-resolvable point optimum. The repeated `4.0e-4` anchor changed from
6.113814 to 6.151371 across populations (0.037557), while the top four lower
bracket arms span only 0.032043. Optimizing inside that spread with one
replicate would optimize execution noise.

A confirmation rung therefore precedes CMA: two same-seed, same-stream
replicates at each of `[2.2, 2.8, 3.3, 4.0]e-4`, trained from scratch for 256
steps. Primary confirmation fitness is mean loss at steps 224-256. Replicates
are assigned to different physical GPUs. This longer horizon tests whether the
near-tie resolves as Schedule-Free averaging matures and measures the observed
cross-run noise directly.

The confirmation rung completed with all eight candidates valid:

| LR | replicate A | replicate B | mean |
|---:|---:|---:|---:|
| 0.000220 | 5.420989 | 5.406856 | 5.413922 |
| 0.000280 | 5.332322 | 5.341311 | 5.336817 |
| **0.000330** | **5.232133** | **5.268278** | **5.250206** |
| 0.000400 | 5.436789 | 5.295978 | 5.366383 |

Both `3.3e-4` replicates beat every replicate of the other three rates. Its
0.086611 mean advantage over `2.8e-4` exceeds its 0.036145 replicate spread.
The confirmation therefore selects `3.3e-4` for the world-eight production
gate. Narrower world-one CMA refinement is deferred because restoring the real
DiLoCo/data-world interaction now has greater decision value than optimizing
the single-island proxy further.

## Phase B: CMA-ES refinement

Phase A is a deterministic bracket, not itself CMA-ES. A refinement is run only
once boundary repositioning finds a stable interior region. CMA-ES operates in
bounded log-LR space around the best three bracket arms, population eight,
using the same 96-step fitness. The bracket anchors prevent a one-dimensional
stochastic population from omitting the known-good and known-bad controls.

Warmup is not mixed into Phase A. If the best no-warmup arm lies at the upper
stable boundary, a separate two-dimensional refinement may search log-LR and
warmup in `[0,64]` steps. This avoids confounding learning rate with protection
from the initialization transient.

## Production gate

The selected candidate starts fresh under the normal eight-way fixed-world
DiLoCo launcher and runs through step 256 (134,217,728 aggregate tokens):

* DiLoCo `K=32`, outer optimizer `avg`;
* identical seed and world-eight counter-sampler stream;
* merge-boundary checkpointing and interruptible final consensus publication;
* compare matched-token trajectory and a fixed held-out evaluation before
  extending to 1B tokens.

The production gate retains the applicable safety intent of gap-matrix R07
(atomic checkpoint/latest publication) and R12 (exact inner-state resume).
No elastic, native-dataplane, changing-world, or asynchronous conformance is
claimed.

## Stop and retention policy

The scan driver owns all candidate subprocesses. SIGINT or SIGTERM is forwarded
to every live candidate so each can take its normal safe final-checkpoint path.
A candidate failure terminates the scan rather than silently shrinking the
population. GPU leases are held for the complete scan and released by the
existing detached-run wrapper.

Artifacts live under:

`/mnt/nvme1n1/erikg/diloco_8gpu/e97_4b_lr_interference_scan`

The aggregate `results.json` is the decision receipt. No production run is
automatically extended solely from a rapid-screen result.
