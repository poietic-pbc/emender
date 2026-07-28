# Emender

Emender is a family of pure recurrent language models built from many small
nonlinear matrix memories. The models implement a nonlinear delta-memory
mechanism, historically called **Nonlinear Delta Memory** (NDM) in parts of the
codebase and older notes. Emender is the public repository and release identity;
nonlinear delta memory describes the recurrent update mechanism.

Emender is designed to test a simple systems and modeling hypothesis:

> A serial recurrent model can train at billion-parameter scale if the
> recurrence is organized as many independent memory programs running in
> parallel across the GPU.

Modern sequence models usually get hardware efficiency by making the sequence
operation itself parallel: attention, convolution, linear scans, or hybrids.
Emender takes the opposite route. Each memory head scans time sequentially, but
the model contains hundreds of heads per layer and many layers per network. The
training workload is therefore a large collection of small recurrent programs,
parallelized across batch elements, heads, and state tiles.

The current optimized implementation is **Emender/E88**: an emender layer with a
fused Triton kernel for the nonlinear delta-memory state update. It has been
used in 1.3B-class language-model training runs.

## v0.2 Release

The v0.2 public-release hub is
[`docs/RELEASE_V02_PUBLIC_RELEASE_HUB.md`](docs/RELEASE_V02_PUBLIC_RELEASE_HUB.md).
It links the paper, GitHub repository, exact Hugging Face revisions, Docker
smoke evidence, and public-release checklist. The earlier v0.1 tags remain
available as immutable historical checkpoints.

- GitHub release target: <https://github.com/poietic-pbc/emender>
- Paper PDF target: <http://hypervolu.me/~erik/ndm/Garrison_2026_Emender.pdf>
- Paper source: <https://github.com/poietic-pbc/emender/blob/main/paper/main.typ>
- Emender/E88 model: <https://huggingface.co/poietic-pbc/emender-e88-1.3b/tree/v0.2>
- GDN model: <https://huggingface.co/poietic-pbc/gdn-1.3b/tree/v0.2>
- M2RNN-CMA model: <https://huggingface.co/poietic-pbc/m2rnn-cma-1.3b/tree/v0.2>

The HF models are raw/base recurrent language models for research and paper
reproduction. They are not instruction-tuned or safety-tuned.

## The Memory Update

For one Emender head, the runtime state is a matrix `S`. Given key `k`, value
`v`, query `q`, output gate `g`, and decay `d`, the production nonlinear
delta-memory recurrence is:

```text
k_t       = l2_norm(silu(k_t))
q_t       = l2_norm(silu(q_t))
r_t       = S_{t-1}^T k_t
delta_t   = silu(v_t) - r_t
S_t       = tanh(d_t S_{t-1} + k_t delta_t^T)
y_t       = silu(g_t) * (S_t^T q_t)
```

The distinctive operation is the **delta correction**. The write is not a raw
outer product. The head first reads what the memory already returns at `k_t`,
then writes the activated error `silu(v_t) - r_t` back into the matrix state.
The nonlinearity bounds the recurrent state itself.

This makes Emender different from nearby matrix-state RNNs that use raw writes,
and different from linear recurrent models whose temporal state update remains
linear even when the gates are input-dependent.

## Experimental Program

This repository supports three connected evaluations.

### 1. Geometry Search

Emender was developed through repeated architecture search over recurrent
geometry: head count, state size, depth, expansion, normalization, gates, and
decay. The same search pressure is used for comparison models where possible.

The important comparison is not only "which update equation is expressive." It
is also which shape is trainable. The current evidence separates:

| Model | Role |
| --- | --- |
| `Emender/E88` | optimized nonlinear delta-memory implementation |
| `FLA-GDN` | strong linear-time gated delta baseline |
| `Mamba2` | strong selective state-space baseline |
| `M2RNN-paper` | published nonlinear matrix-state geometry |
| `M2RNN-CMA` | M2RNN-style model reshaped into the many-head regime found by search |

The M2RNN contrast is especially useful. Matrix state and temporal
nonlinearity are not sufficient by themselves. The update rule, addressing
geometry, and multi-programmed shape all matter.

### 2. Billion-Scale Language Modeling

The production comparison trains 1.3B-class recurrent language models at
2K context and then extends the context curriculum. The main quantity of
interest is not raw step count. It is loss versus wallclock compute under
matched, tuned training conditions.

The working result is that optimized pure recurrent Emender can enter the same
language-modeling loss regime as leading linear recurrent baselines. That is
the central scaling claim: pure nonlinear recurrence is not ruled out by
hardware if it is implemented as a multi-programmed workload.

Large checkpoints and raw campaign logs are not stored in git. Frozen
checkpoints should be released separately with model cards, dataset notes, and
exact commit references.

### 3. State-Tracking Expressivity

The expressivity suite tests controlled finite-state and algorithmic tasks. The
main witness is S5 permutation composition, a noncommutative state-tracking task
that stresses transition composition rather than text memorization.

In matched 8M-parameter S3/S5 experiments, Emender/E88 separates from FLA-GDN,
M2RNN, and paper-shaped M2RNN at the training length and remains ahead under
length extrapolation. The result supports a narrower mechanism claim: the
delta-correcting update is the useful resource, not matrix state alone.

## Formal Core

The Lean formalization lives in `formal/lean`. The trusted paper root checks:

- The nonlinear delta-memory update and M2RNN are distinct one-step update
  families.
- M2RNN can embed a nonlinear delta-memory step only when given an extra
  read-then-delta resource.
- Fixed-precision recurrent recognizers have a finite-state ceiling.
- The S5 tracker and exact finite transition-memory realization are formalized.

The Lean core does **not** claim a lower bound for all linear scan models, and
does **not** claim that fixed-width nonlinear delta memories exceed NC1. The
formal boundary is checked in CI and rejects `sorry`, `admit`, explicit
`axiom`, `opaque`, and `native_decide` in the trusted import closure.

The independent Std-only resilient coordination workspace lives in
`formal/resilient`. It pins its own exact Lean/Lake toolchain and dependency
lock, implements the pure resilient DiLoCo transition oracle, and replays the
shared canonical JSON trace without requiring Python, Slurm, a GPU, model
bytes, or a network peer. Its scope and native/runtime boundary are documented
in `formal/resilient/README.md`.

## Repository Layout

- `ndm/`: model code, Emender/E88 Triton kernels, and comparison baselines.
- `train.py`: byte-level and tokenized language-model training entry point.
- `experiments/expressivity_tasks/`: parity, counters, finite-state tracking,
  and S5 permutation-composition experiments.
- `scripts/`: checkpoint evaluation, sampling, dataset-stream preparation, and
  knowledge/reasoning probe builders.
- `formal/lean/`: trusted Lean proof surface.
- `formal/resilient/`: pure resilient DiLoCo coordination kernel and trace
  oracle.
- `docs/`: architecture, stability, systems, distributed training, and M2RNN
  comparison notes.
- `paper/`: working paper design and result notes.
- `provenance/`: commit anchors for the original development repositories.

## Checks

CPU Python smoke:

```bash
python -m pytest tests -m "not gpu"
python -m py_compile train.py scripts/*.py ndm/triton/*.py ndm/models/e88_fused.py
```

CUDA/Triton recurrence parity smoke:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_e88_triton.py -m gpu
```

Lean trusted-core check:

```bash
cd formal/lean
scripts/check_paper_core.sh
scripts/check_trusted_no_placeholders.sh ElmanProofs.lean
```

Lean resilient-coordination smoke:

```bash
cd formal/resilient
scripts/smoke.sh
```

The public CI runs the CPU Python and Lean trusted-core checks. The Triton test
is kept as a GPU-local parity smoke because standard GitHub-hosted runners do
not provide CUDA GPUs.

## Provenance

The nonlinear delta-memory mechanism behind Emender was developed in public in
two historical repositories:

- `ekg/elman` at `6f0724feae9fc82bd235408ac5c3ae61f2b17c79`
- `ekg/elman-proofs` at `5082610c9cdabf0b31e11dd14ee078273d486333`

Those repositories preserve the development trail. This repository is the clean
implementation, experiment, and proof artifact for Emender.
