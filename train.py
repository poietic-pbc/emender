#!/usr/bin/env python3
"""
Training script for NDM and recurrent baseline models.

Simple single-GPU training with:
- Document-aware data loading
- Optional TBPTT with hidden state tracking (--tbptt flag)
- Checkpointing and logging
- Support for all 4 ladder levels (0-3)

Usage:
    python train.py --data /path/to/data.txt --level 3 --params 100m
    python train.py --data /path/to/data.txt --level 3 --params 100m --tbptt  # Enable TBPTT
    # Explicit geometry overrides are reflected in run labels by actual params,
    # e.g. E97 + 1.29B params -> emender_E97_1.3B_<timestamp>.
"""

import os
import sys
import time
STARTUP_TIME = time.time()
import argparse
import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
import schedulefree
from pathlib import Path
import json
import datetime
import glob
import re
import tempfile
import ctypes.util
from contextlib import nullcontext

_SHUTDOWN_REQUEST = {
    'requested': False,
    'signal': None,
    'received_at': None,
}

# Add elman package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Add CUDA extension directory for hasty_pytorch_lib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'elman', 'cuda'))

from ndm.models import LadderLM, create_ladder_model

# Optional: swap E88FLAHybridCUDAFunction for PararnnHybridE88V2/V3 Triton kernel.
# V3 ([B,T,H,N]-native, no permutes) is preferred when available.
# See experiments/pararnn_kernel/tree_scan/install_hybrid{,_v3}.py
_hybrid_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'experiments/pararnn_kernel/tree_scan')
if os.environ.get('ELMAN_PARARNN_HYBRID_V3') == '1':
    sys.path.insert(0, _hybrid_root)
    from install_hybrid_v3 import install as _install_hybrid_v3
    _install_hybrid_v3()
elif os.environ.get('ELMAN_PARARNN_HYBRID') == '1':
    sys.path.insert(0, _hybrid_root)
    from install_hybrid import install as _install_hybrid
    _install_hybrid()

from ndm.data import DocumentStreamDataset, BatchedStreamDataset, create_dataloader
from ndm.data.tokenized_dataset import (
    BOUNDARY_COUNTER_SAMPLER_SCHEMA,
    COUNTER_SAMPLER_SCHEMA,
    COUNTER_SAMPLER_SCHEMAS,
    LEGACY_SAMPLER_SCHEMA,
    CounterSamplerIdentity,
    TokenizedStreamDataset,
    restore_sampler_checkpoint_metadata,
    sampler_checkpoint_metadata,
)
from ndm.models.gru_baseline import GRULM
from ndm.models.lstm_baseline import LSTMLM
from ndm.models.min_rnn_baseline import MinGRULM, MinLSTMLM
from ndm.models.cuda_gru import CudaGRULM
from ndm.models.cuda_lstm import CudaLSTMLM
from ndm.models.e88_fused import E88FusedLM
from ndm.schedulefree_offload import CPUOffloadAdamWScheduleFree

# Pin Triton autotune config selection BEFORE any fused kernel runs (kill the
# init-wedge / autotune storm that deadlocks fresh inits under box contention,
# and that would multiply across ~512 Frontier GCDs). Covers BOTH race arms:
# the E97/emender norms+gates and the gdn2-mlp FLA gated-delta/conv kernels.
# Env-gated, default ON; NDM_PIN_TRITON_AUTOTUNE=0 restores the original sweep.
try:
    from ndm.triton import pin_autotune as _pin_autotune
    _pin_autotune.maybe_install_from_env()
except Exception as _e:  # never let pinning break a real training run
    print(f"[pin-autotune] install skipped: {_e}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Train NDM and recurrent baseline models')

    # Data
    parser.add_argument('--data', type=str, required=True,
                        help='Path to training data file')
    parser.add_argument('--val_data', type=str, default=None,
                        help='Path to validation data file')
    parser.add_argument('--tokenizer', type=str, default=None,
                        choices=[None, 'gpt2', 'cl100k_base', 'r50k_base', 'p50k_base', 'o200k_base'],
                        help='tiktoken encoding name. If set, use BPE tokens instead of raw bytes. '
                             'Default (None) = byte-level, vocab_size=256.')

    # Model
    parser.add_argument('--use_triton', type=int, default=None,
                        help='For E88: use Triton fwd+bwd kernels instead of CUDA register-owned '
                             '(0=CUDA, 1=Triton). Default None = AUTO: enabled for E97/split-edit/'
                             'raw_write under bf16 (their ONLY fused path is Triton; CUDA '
                             'register-owned does not support split-edit/raw-write, so without this '
                             'they fall back to the slow eager T-scan). Parity-verified vs the eager '
                             'reference (paper/review/E97_FUSED_LM_KERNEL_NOTE.md). Pass 0 to force eager.')
    parser.add_argument('--level', type=str, default='3',
                        help='Ladder level: 0-6 (linear) or log_0 to log_5 (log-space)')
    parser.add_argument('--params', type=str, default='100m',
                        help='Target parameter count (e.g., 100m, 500m, 1b)')
    parser.add_argument('--embed_dim', type=int, default=1024,
                        help='Bottleneck embedding dim for E94 (default: 1024)')
    parser.add_argument('--dim', type=int, default=None,
                        help='Model dimension (overrides --params)')
    parser.add_argument('--depth', type=int, default=None,
                        help='Number of layers (overrides --params)')
    parser.add_argument('--expansion', type=float, default=1.0,
                        help='Hidden state expansion factor')
    parser.add_argument('--state_expansion', type=int, default=2,
                        help='State expansion for E16 (d_state = d_inner * state_expansion)')
    parser.add_argument('--n_groups', type=int, default=32,
                        help='Number of groups for compete softmax')
    parser.add_argument('--n_state', type=int, default=64,
                        help='Matrix state size for E70-E73 (S is n_state x n_state)')
    parser.add_argument('--n_slots', type=int, default=64,
                        help='Number of tape memory slots for E23 DualMemoryElman (default=64)')
    parser.add_argument('--n_heads', type=int, default=None,
                        help='Number of heads for E88 FLA Hybrid')
    parser.add_argument('--m2rnn_paper_shape', action='store_true',
                        help='For M2RNN: use grouped paper-style heads (q/k=1, v/f/g/W=n_heads, K=64, V=n_state)')
    parser.add_argument('--m2rnn_k_head_dim', type=int, default=None,
                        help='For M2RNN: key/query head dimension override')
    parser.add_argument('--m2rnn_v_head_dim', type=int, default=None,
                        help='For M2RNN: value head dimension override')
    parser.add_argument('--m2rnn_q_heads', type=int, default=None,
                        help='For M2RNN: query head count override')
    parser.add_argument('--m2rnn_k_heads', type=int, default=None,
                        help='For M2RNN: key head count override')
    parser.add_argument('--m2rnn_v_heads', type=int, default=None,
                        help='For M2RNN: value head count override')
    parser.add_argument('--m2rnn_f_heads', type=int, default=None,
                        help='For M2RNN: forget head count override')
    parser.add_argument('--m2rnn_g_heads', type=int, default=None,
                        help='For M2RNN: output gate head count override')
    parser.add_argument('--m2rnn_weight_heads', type=int, default=None,
                        help='For M2RNN: recurrent weight head count override')
    parser.add_argument('--m2rnn_output_norm', type=int, default=0,
                        help='For M2RNN: RMSNorm recurrent output before output projection')
    parser.add_argument('--m2rnn_normalize_qk', type=int, default=0,
                        help='For M2RNN: L2-normalize query/key vectors before the recurrent update')
    parser.add_argument('--m2rnn_use_residual', type=int, default=1,
                        help='For M2RNN: include D*v direct residual in recurrent output')
    parser.add_argument('--m2rnn_freeze_state_weight', type=int, default=0,
                        help='For M2RNN: keep recurrent state_weight fixed at identity')
    parser.add_argument('--m2rnn_state_grad_clip', type=float, default=None,
                        help='For M2RNN/XMA: clip recurrent state gradients inside the custom op')
    parser.add_argument('--require_m2rnn_xma', action='store_true',
                        help='Fail if --level m2rnn runs on CUDA without the XMA Triton backend')
    parser.add_argument('--hybrid_pattern', type=str, default=None,
                        help='For --level hybrid: comma-separated layer pattern, '
                             'e.g. fla-gdn,fla-gdn,fla-gdn,m2rnn-paper')
    parser.add_argument('--hybrid_m2rnn_heads', type=int, default=None,
                        help='For --level hybrid: override n_heads only for m2rnn/m2rnn-paper layers')
    parser.add_argument('--top_k', type=int, default=None,
                        help='Number of active heads per token for MoM E88 (sparse routing)')
    parser.add_argument('--k_fast', type=int, default=None,
                        help='Fast state dimension for E90 Dual-Rate (default=16)')
    parser.add_argument('--k_slow', type=int, default=None,
                        help='Slow state dimension for E90 Dual-Rate (default=48)')
    parser.add_argument('--use_gate', type=int, default=1,
                        help='Use output gating for E88/E94 (0=no gate, 1=gate, default=1)')
    parser.add_argument('--use_permutation', type=int, default=1,
                        help='E94: cross-head permutation between layers (0=off, 1=on, default=1)')
    parser.add_argument('--gate_activation', type=str, default='sigmoid',
                        help='Gate activation for E88 (sigmoid=E88 original, silu=FLA-GDN style)')
    parser.add_argument('--linear_state', type=int, default=0,
                        help='Use linear state update for E88 (0=tanh, 1=linear)')
    parser.add_argument('--use_write_gate', type=int, default=0,
                        help='Use write gate (beta) for E88 (0=no, 1=yes). Gates delta before writing to memory.')
    parser.add_argument('--e88_decay_mode', type=str, default='mamba',
                        choices=['mamba', 'simple', 'none', 'constant'],
                        help='E88 decay mode: mamba=input-dependent exponential, simple=sigmoid, none=1, constant=learned per-head constant')
    parser.add_argument('--e88_value_residual', type=int, default=0,
                        help='Add direct D*v value residual to E88 output before output gating (0=no, 1=yes)')
    parser.add_argument('--e88_raw_write', type=int, default=0,
                        help='Ablate E88 delta correction: write raw v instead of v - S^T k')
    parser.add_argument('--use_chunked_e97', type=int, default=0,
                        help='For E97 split-edit linear-state runs, use the fused chunked '
                             'E97 Triton fwd/bwd kernel instead of the sequential split-edit '
                             'Triton scan (0=no, 1=yes). Requires --use_triton 1, '
                             '--linear_state 1, and --e88_raw_write 0.')
    parser.add_argument('--e97_chunk_size', type=int, default=32,
                        help='Chunk size for --use_chunked_e97. ROCm/HIP debug starts at 32.')
    parser.add_argument('--mlp_ratio', type=float, default=0.0,
                        help='If >0, wrap every mixer layer with a post-mixer RMSNorm + SwiGLU MLP '
                             '(hidden = dim*mlp_ratio). 0 = mixer-only (default). Mirrors gdn2-mlp plumbing.')
    parser.add_argument('--mlp_multiple', type=int, default=64,
                        help='Round SwiGLU MLP hidden width to this multiple (default 64).')
    parser.add_argument('--r_h_mode', type=str, default='auto',
                        help='W_h constraint mode (spectral_norm, learned, none, auto)')
    # auto: spectral_norm for models with full W_h (1,33,42,51,52,53,56), none for diagonal/scalar
    parser.add_argument('--use_conv', type=int, default=0,
                        help='Use Conv1d before recurrence (0=no, 1=yes)')
    parser.add_argument('--d_conv', type=int, default=4,
                        help='Conv kernel size (if use_conv=1)')
    parser.add_argument('--gdn2_mlp_ratio', type=float, default=6208 / 2304,
                        help='For gdn2-mlp: SwiGLU hidden ratio (official gdn2_1.3B is 6208/2304)')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (0.0 to 0.3)')

    # Typed-Emender / E98-CMA candidate knobs (levels typed-gdn2-lm / e98-cma-lm).
    # These are forwarded to the layer as layer_kwargs; absent => layer defaults.
    parser.add_argument('--head_type_logits', type=str, default=None,
                        help='typed-gdn2-lm: comma logits [gdn2_recall,e97_track,count,latch,nonlin] '
                             '(softmax->largest-remainder head counts). E99 winner: 4.0,-1.9008,-0.9211,-2.8866,2.4146')
    parser.add_argument('--corner_mixture', type=str, default=None,
                        help='e98-cma-lm: comma per-corner head fractions [track,count,latch,nonlin]. '
                             'E98-CMA winner: 0.4015,0.2821,0.0089,0.3075')
    parser.add_argument('--lam_max', type=float, default=None,
                        help='Unified/typed cell free-gain cap (E99/E98 winner: 1.585)')
    parser.add_argument('--beta_max', type=float, default=None,
                        help='Unified/typed cell reflection-depth cap (E99/E98 winner: 2.747)')
    parser.add_argument('--igain_max', type=float, default=None,
                        help='Unified/typed cell input-gain cap (default 2.0)')
    parser.add_argument('--knob_lr_mult', type=float, default=1.0,
                        help='Separate LR multiplier for recurrence knobs (lam/beta/igain/gamma_raw). '
                             'E98-CMA winner: 5.38. 1.0 == single param group.')
    parser.add_argument('--layer_kwargs', type=str, default=None,
                        help='JSON dict of extra per-layer kwargs merged into layer_kwargs '
                             '(e.g. {"nonlin_subset_frac":0.125} for level=complex-eig-lm).')
    parser.add_argument('--gdn_allow_neg_eigval', type=int, default=1,
                        help='typed-gdn2-lm: allow negative along-key eigenvalue in GDN heads (1=yes, GDN-2 tracking)')

    # Mamba2-specific
    parser.add_argument('--mamba_expand', type=int, default=2,
                        help='Mamba2/Mamba3 expansion factor (expand)')
    parser.add_argument('--mamba_d_state', type=int, default=64,
                        help='Mamba2/Mamba3 state dimension (d_state)')
    parser.add_argument('--mamba3_headdim', type=int, default=64,
                        help='Mamba3 head dimension')
    parser.add_argument('--mamba3_mimo', type=int, default=0,
                        help='Mamba3: use MIMO kernels (1=yes, 0=SISO)')
    parser.add_argument('--mamba3_mimo_rank', type=int, default=4,
                        help='Mamba3 MIMO rank')

    # Training
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--chunk_size', type=int, default=512,
                        help='Sequence chunk size (TBPTT)')
    parser.add_argument('--checkpoint_interval', type=int, default=16,
                        help='E88 state checkpoint interval (larger=less memory, more recompute)')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Recompute layer forward during backward (saves ~16GB, enables larger batch/seq)')
    parser.add_argument('--loss_chunk_size', type=int, default=0,
                        help='Chunk T dim when computing lm_head + cross_entropy (saves T*V*2 bytes at long T)')
    parser.add_argument('--projection_chunk_size', type=int, default=0,
                        help='E88 projection recomputation chunk size (0=disabled, 512=recommended for T>=8K). Saves ~5GB/layer.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--grad_accum', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping (0 to disable)')
    parser.add_argument('--steps', type=int, default=100000,
                        help='Total training steps')
    parser.add_argument('--train_minutes', type=float, default=None,
                        help='Train for N minutes (overrides --steps)')
    parser.add_argument('--compile_warmup_steps', type=int, default=0,
                        help='Untimed fwd+bwd steps before training/probe to pay compile/autotune cost')
    parser.add_argument('--timer_after_compile_warmup', action='store_true',
                        help='For --train_minutes, start the training clock after compile_warmup_steps')
    parser.add_argument('--final_heldout_eval', action='store_true',
                        help='After training, run ONE final held-out eval on the '
                             'schedule-free AVERAGED weights (leaderboard methodology) '
                             'and print FINAL_HELDOUT_CE / FINAL_HELDOUT_BPB. Opt-in; '
                             'requires --val_data. (task e97-within-layer LM screens)')
    parser.add_argument('--final_val_batches', type=int, default=200,
                        help='Batches for the --final_heldout_eval pass.')
    parser.add_argument('--final_train_eval', action='store_true',
                        help='After training, ALSO run ONE clean eval on a slice of '
                             'the TRAIN distribution (--data) using the same averaged '
                             'weights + clean machinery as --final_heldout_eval, and '
                             'print FINAL_TRAIN_CE / FINAL_TRAIN_BPB. Lets the audit '
                             'isolate the train->held generalization gap (same units, '
                             'same weights) from units / running-average artifacts. '
                             '(task e97-audit2)')
    parser.add_argument('--heldout_bytes_per_token', type=float, default=3.783,
                        help='Bytes/token for BPB = (CE_nats/ln2)/bytes_per_token. '
                             'Default 3.783 = p50k on commapile (Study B).')
    parser.add_argument('--heldout_tensor', type=str, default=None,
                        help='Path to a FIXED pre-tokenized held-out tensor (.pt with '
                             "keys 'chunks' [N, chunk+1], 'bytes_per_token', "
                             "'scored_tokens'). When set with --final_heldout_eval, the "
                             'final held-out CE/BPB is computed on EXACTLY these chunks '
                             '(byte-for-byte identical slice across models, tokenizer-'
                             'correct), overriding --val_data / --heldout_bytes_per_token. '
                             '(task lb-compare: apples-to-apples leaderboard)')
    parser.add_argument('--heldout_curve_every', type=int, default=0,
                        help='If >0, score --heldout_tensor every N optimizer steps '
                             'and append step,tokens,train_loss,heldout_bpb to CSV.')
    parser.add_argument('--heldout_curve_path', type=str, default=None,
                        help='CSV path for --heldout_curve_every. Default: '
                             '<run output dir>/heldout_curve.csv')
    parser.add_argument('--heldout_eval_mode', type=str, default='x',
                        choices=['x', 'avg', 'y', 'train'],
                        help='ScheduleFree weight mode for fixed held-out evals: x/avg '
                             '= averaged eval weights, y/train = train weights.')
    parser.add_argument('--warmup_steps', type=int, default=0,
                        help='Warmup steps for learning rate. For adamw: linear '
                             'ramp then cosine decay (see lr_scale_at). For schedulefree: '
                             "passed to AdamWScheduleFree's built-in linear LR warmup "
                             '(critical for long-horizon stability — without it the '
                             'schedule-free x-average degrades under high constant LR; '
                             'task fix-long-horizon).')
    parser.add_argument('--optimizer', type=str, default='schedulefree',
                        choices=['adamw', 'schedulefree'],
                        help='Optimizer: adamw (warmup + cosine-decay LR schedule) or '
                             'schedulefree (built-in warmup via --warmup_steps; NO decay)')
    parser.add_argument('--offload_schedulefree_state', action='store_true',
                        help='Keep Schedule-Free z and second moments in pinned CPU RAM, '
                             'staging bounded tensors to the parameter device for updates. '
                             'Requires --optimizer schedulefree.')
    parser.add_argument('--schedulefree_offload_pin_memory', type=int, choices=[0, 1],
                        default=1,
                        help='Pin CPU-offloaded Schedule-Free state for PCIe DMA (default 1).')
    parser.add_argument('--schedulefree_offload_release_gradients', type=int,
                        choices=[0, 1], default=1,
                        help='Release each gradient after its streamed optimizer update to '
                             'bound HBM (default 1).')
    parser.add_argument('--schedulefree_offload_bucket_numel', type=int,
                        default=67_108_864,
                        help='Maximum elements in each reusable GPU state-staging bucket '
                             '(default 67,108,864; 128 MiB per BF16 state tensor).')
    parser.add_argument('--min_lr_frac', type=float, default=0.1,
                        help='AdamW cosine-decay floor as a fraction of --lr (the LR '
                             'decays from lr to min_lr_frac*lr over --steps). 0.1 = decay '
                             'to 10%% of peak. Only used by --optimizer adamw.')

    # DiLoCo periodic-sync parallelism (task implement-diloco-periodic). Opt-in,
    # only meaningful under torchrun (WORLD_SIZE>1). When --diloco is set, ranks
    # train INDEPENDENTLY (no per-step DDP gradient all-reduce) and only average
    # MODEL WEIGHTS every --diloco_k local optimizer steps. This recovers the
    # ~62k tok/s independent ceiling that vanilla per-step DDP halves to ~31k on a
    # no-NVLink PCIe box. Outer step (DiLoCo): W_{r+1} = W_r + outer_lr*outer_mom,
    # outer_mom = outer_beta*outer_mom + (mean_i(W_{r,i}) - W_r). Defaults
    # (outer_lr=1, outer_beta=0) reduce to plain periodic weight averaging
    # (local-SGD), the conservative first production path per
    # docs/SCHEDULEFREE_DILOCO_FRONTIER_DESIGN.md.
    parser.add_argument('--diloco', action='store_true',
                        help='Enable DiLoCo periodic model-weight averaging instead of '
                             'per-step DDP gradient all-reduce (requires torchrun WORLD_SIZE>1)')
    parser.add_argument('--diloco_k', type=int, default=250,
                        help='DiLoCo sync interval: average model weights every K local '
                             'optimizer steps (design recommends 250; 100 conservative)')
    parser.add_argument('--diloco_outer_lr', type=float, default=1.0,
                        help='DiLoCo outer learning rate (1.0 = plain averaging)')
    parser.add_argument('--diloco_outer_beta', type=float, default=0.0,
                        help='DiLoCo outer momentum beta (0.0 = local-SGD, no outer momentum)')
    parser.add_argument('--diloco_outer_optimizer', type=str, default='avg',
                        choices=['avg', 'momentum', 'sfsgd'],
                        help='DiLoCo outer optimizer: avg = stateless periodic averaging; '
                             'momentum = fixed-momentum DiLoCo outer update; '
                             'sfsgd = separate manual ScheduleFree-SGD outer state machine.')
    parser.add_argument('--diloco_export_basis', type=str, default='x',
                        choices=['x', 'y'],
                        help='Basis exported to the outer optimizer at a DiLoCo boundary. '
                             'x preserves current eval-weight semantics; y exports the '
                             'inner ScheduleFree train point for ablations.')
    parser.add_argument('--diloco_bootstrap_outer_state', type=str, default='none',
                        choices=['none', 'from-loaded-model'],
                        help='Explicit non-avg DiLoCo resume guard. Default none preserves '
                             'fail-closed behavior when the checkpoint lacks compatible '
                             'diloco_outer_state. from-loaded-model initializes fresh outer '
                             'bookkeeping from the already-loaded model/optimizer tensors '
                             'without changing live model parameters.')
    parser.add_argument('--diloco_island_size', type=int, default=0,
                        help='HYBRID mode (task diloco-loss-parity-longhorizon): >1 forms '
                             'islands of this many consecutive ranks that do per-step DDP '
                             'gradient all-reduce WITHIN the island (tight sync, exact SGD), '
                             'while DiLoCo periodic averaging runs ACROSS islands every K '
                             'steps. world_size must be divisible by island_size. 0/1 = pure '
                             'DiLoCo (no intra-island DDP). Trades some throughput for '
                             'sample-efficiency when pure-DiLoCo lags DDP at matched tokens.')
    parser.add_argument('--diloco_merge_bucket_numel', type=int,
                        default=_env_int('NDM_DILOCO_MERGE_BUCKET_NUMEL', 0),
                        help='Optional DiLoCo merge all-reduce bucket size in elements. '
                             '0/unset preserves the default monolithic flat all-reduce. '
                             'Can also be set with NDM_DILOCO_MERGE_BUCKET_NUMEL.')
    parser.add_argument('--diloco_merge_topology', type=str,
                        default=os.environ.get('DILOCO_MERGE_TOPOLOGY', 'global'),
                        choices=['global', 'hierarchical'],
                        help='DiLoCo merge communication topology. global preserves the '
                             'existing all-rank all-reduce. hierarchical is opt-in and '
                             'reduces exact sums within rank groups, all-reduces group '
                             'roots, then broadcasts the exact global average back '
                             '(env: DILOCO_MERGE_TOPOLOGY).')
    parser.add_argument('--diloco_merge_group_size', type=int,
                        default=_env_int('NDM_DILOCO_MERGE_GROUP_SIZE', 8),
                        help='Ranks per first-level group for '
                             '--diloco_merge_topology=hierarchical. Frontier srun ranks '
                             'are consecutive by node with 8 GPU tasks/node, so the '
                             'default forms node-local groups. The final group may be '
                             'smaller; weighted SUM/count averaging keeps semantics exact. '
                             '(env: NDM_DILOCO_MERGE_GROUP_SIZE).')
    parser.add_argument('--diloco_merge_group_create_barrier_every', type=int,
                        default=_env_int('NDM_DILOCO_MERGE_GROUP_CREATE_BARRIER_EVERY', 8),
                        help='Hierarchical merge process-group construction pacing. '
                             'When >0, all ranks enter a default-group barrier after '
                             'the root communicator and after each batch of this many '
                             'first-level groups. This avoids a large burst of RCCL/NCCL '
                             'communicator construction at scale while preserving '
                             'collective order. 0 disables the pacing barriers. '
                             '(env: NDM_DILOCO_MERGE_GROUP_CREATE_BARRIER_EVERY).')
    parser.add_argument('--diloco_merge_completion_barrier', type=int,
                        default=_env_bool_int('NDM_DILOCO_MERGE_COMPLETION_BARRIER', True),
                        help='For hierarchical DiLoCo merges, run one all-rank completion '
                             'barrier after the tree reduce/broadcast finishes. This keeps '
                             'local subgroups from racing ahead while another subgroup is '
                             'still completing its final broadcast. 1=enabled, 0=disabled '
                             '(env: NDM_DILOCO_MERGE_COMPLETION_BARRIER).')
    parser.add_argument('--diloco_merge_debug', type=int, choices=[0, 1],
                        default=_env_bool_int('NDM_DILOCO_MERGE_DEBUG', False),
                        help='Enable rank-filtered DiLoCo merge entry/exit logging '
                             '(0/1; env: NDM_DILOCO_MERGE_DEBUG).')
    parser.add_argument('--diloco_merge_debug_ranks', type=str,
                        default=os.environ.get('NDM_DILOCO_MERGE_DEBUG_RANKS', '0'),
                        help='Comma-separated ranks to log when --diloco_merge_debug=1, '
                             'or "*" for all ranks (env: NDM_DILOCO_MERGE_DEBUG_RANKS).')

    # Checkpointing
    parser.add_argument('--output', type=str, default='./output',
                        help='Output directory')
    parser.add_argument('--exact_output_dir', type=str, default=None,
                        help='Use this exact run/checkpoint directory instead of creating a '
                             'timestamped child. Intended for externally supervised restart '
                             'epochs that must share one atomic latest.pt and retention set.')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='Save checkpoint every N steps')
    parser.add_argument('--log_every', type=int, default=10,
                        help='Log every N steps')
    parser.add_argument('--val_every', type=int, default=500,
                        help='Validate every N steps')
    parser.add_argument('--keep_checkpoints', type=int, default=5,
                        help='Number of checkpoints to keep')
    parser.add_argument('--walltime_minutes', type=float, default=None,
                        help='Optional scheduler walltime budget in minutes, measured from '
                             'process start. If unset, SLURM_JOB_END_TIME or SLURM_TIMELIMIT '
                             'is used when available.')
    parser.add_argument('--walltime_final_checkpoint_margin_seconds', type=float, default=600.0,
                        help='When a walltime deadline is known, stop training and run the '
                             'normal final checkpoint path once this many seconds remain. '
                             'Default 600s leaves room for DiLoCo final merge and rank-0 save.')
    parser.add_argument('--walltime_check_every', type=int, default=1,
                        help='Check coordinated walltime/shutdown finalization every N optimizer '
                             'steps. By default distributed runs perform one scalar all-reduce at '
                             'each check so a signal or deadline observed by any rank brings all '
                             'ranks into finalization together.')
    parser.add_argument('--distributed_health_check_every', type=int, default=1,
                        help='Check distributed non-finite loss/grad health every N optimizer '
                             'steps. Defaults to every step. Pure DiLoCo scaleout jobs can set '
                             'this to the DiLoCo K interval to avoid per-step scalar collectives; '
                             'a local non-finite value between checks raises on that rank and '
                             'fails the job rather than entering a mismatched collective.')
    parser.add_argument('--distributed_init_timeout_seconds', type=float,
                        default=(
                            float(os.environ['NDM_DISTRIBUTED_INIT_TIMEOUT_SECONDS'])
                            if os.environ.get('NDM_DISTRIBUTED_INIT_TIMEOUT_SECONDS') else None
                        ),
                        help='Optional explicit torch.distributed init_process_group timeout. '
                             'Unset preserves PyTorch defaults; set a finite debug value such as '
                             '900 to expose rendezvous failures without waiting indefinitely '
                             '(env: NDM_DISTRIBUTED_INIT_TIMEOUT_SECONDS).')
    parser.add_argument('--disable_scalar_status_collectives', action='store_true',
                        help='Disable steady-state default-process-group scalar all-reduces used '
                             'for health, train-budget, and walltime status checks. DiLoCo jobs '
                             'then use local non-finite guards plus filesystem stop requests at '
                             'safe optimizer-step boundaries; model-weight DiLoCo collectives are '
                             'unchanged.')
    parser.add_argument('--status_request_poll_seconds', type=float, default=1.0,
                        help='When --disable_scalar_status_collectives is set, poll this many '
                             'seconds for a filesystem stop request at each walltime/budget check '
                             'boundary. This lets one rank publish a stop without a NCCL/RCCL '
                             'scalar all-reduce.')
    parser.add_argument('--disable_walltime_final_checkpoint', action='store_true',
                        help='Disable walltime-aware pre-shutdown final checkpointing. Signal '
                             'handling still requests a graceful final checkpoint.')

    # System
    parser.add_argument('--bf16', action='store_true',
                        help='Use bfloat16 mixed precision')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile')
    parser.add_argument('--compile_mode', type=str, default='max-autotune',
                        help='torch.compile mode (default, reduce-overhead, max-autotune)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--total_tokens', type=int, default=None,
                        help='Trusted total-token bootstrap for a legacy --resume checkpoint. '
                             'For checkpoints with embedded total_tokens this is optional and '
                             'must match exactly; newly written checkpoints use the embedded '
                             'value as sole authority.')
    parser.add_argument('--sampler_schema', type=str, default=None,
                        help='Versioned accepted-token sampler schema. Unset is explicitly '
                             'legacy mutable-RNG mode; scientific runs must pass a member of '
                             f'{sorted(COUNTER_SAMPLER_SCHEMAS)}.')
    parser.add_argument('--sampler_corpus_sha256', type=str, default=None,
                        help='Frozen canonical corpus SHA-256 for the counter sampler.')
    parser.add_argument('--sampler_tokenizer_sha256', type=str, default=None,
                        help='Frozen tokenizer-cache SHA-256 for the counter sampler.')
    parser.add_argument('--sampler_key', type=int, default=None,
                        help='Fixed nonnegative counter sampler key (paper study: 42).')
    parser.add_argument('--sampler_data_world_size', type=int, default=None,
                        help='Declared fixed data world; must exactly match launched WORLD_SIZE.')
    parser.add_argument('--sampler_stream_origin_accepted_tokens', type=int, default=None,
                        help='Counter-v2 immutable accepted-token stream origin; v1 is zero.')
    parser.add_argument('--tbptt', action='store_true',
                        help='Enable TBPTT (carry hidden state across chunks)')
    parser.add_argument('--orth_reg', type=float, default=0.0,
                        help='Orthogonality regularization weight for E79 (0=disabled)')
    parser.add_argument('--orth_sep', type=float, default=0.01,
                        help='Weight for k/m separation in orthogonality loss')
    parser.add_argument('--orth_orth', type=float, default=0.001,
                        help='Weight for key orthogonality in orthogonality loss')

    # Memory probing
    parser.add_argument('--probe_memory', action='store_true',
                        help='Run 1 fwd+bwd step, print peak GPU memory in MB, then exit')
    parser.add_argument('--probe_optimizer_steps', type=int, default=1,
                        help='With --probe_memory, run this many optimizer updates. '
                             'Updates after the first use synthetic gradients to measure '
                             'steady-state offload cost without another fwd/bwd (default 1).')

    return parser.parse_args()


def _handle_shutdown_signal(signum, _frame):
    _SHUTDOWN_REQUEST['requested'] = True
    _SHUTDOWN_REQUEST['signal'] = signal.Signals(signum).name
    _SHUTDOWN_REQUEST['received_at'] = time.time()


def install_shutdown_signal_handlers():
    """Install lightweight handlers that let the training loop checkpoint."""
    for signame in ('SIGTERM', 'SIGINT', 'SIGHUP', 'SIGUSR1'):
        sig = getattr(signal, signame, None)
        if sig is not None:
            signal.signal(sig, _handle_shutdown_signal)


def parse_slurm_timelimit_seconds(value):
    """Parse common SLURM time limit strings to seconds.

    SLURM commonly exposes limits as minutes, HH:MM:SS, MM:SS, or
    D-HH:MM:SS. Unlimited/unknown values return None.
    """
    if value is None:
        return None
    raw = str(value).strip()
    if not raw or raw.upper() in {'UNLIMITED', 'INFINITE', 'NOT_SET', 'NONE'}:
        return None
    if raw.isdigit():
        return int(raw) * 60

    days = 0
    clock = raw
    if '-' in raw:
        day_text, clock = raw.split('-', 1)
        if not day_text.isdigit():
            return None
        days = int(day_text)

    parts = clock.split(':')
    if len(parts) == 2 and all(p.isdigit() for p in parts):
        minutes, seconds = [int(p) for p in parts]
        return days * 86400 + minutes * 60 + seconds
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        hours, minutes, seconds = [int(p) for p in parts]
        return days * 86400 + hours * 3600 + minutes * 60 + seconds
    return None


def resolve_walltime_deadline(args, now=None):
    """Return (deadline_epoch, source) for explicit/SLURM walltime, if known."""
    if getattr(args, 'disable_walltime_final_checkpoint', False):
        return None, None
    if now is None:
        now = time.time()
    if getattr(args, 'walltime_minutes', None) is not None:
        return STARTUP_TIME + float(args.walltime_minutes) * 60.0, '--walltime_minutes'

    slurm_end = os.environ.get('SLURM_JOB_END_TIME')
    if slurm_end:
        try:
            end_epoch = float(slurm_end)
        except ValueError:
            end_epoch = None
        if end_epoch and end_epoch > now:
            return end_epoch, 'SLURM_JOB_END_TIME'

    slurm_limit_s = parse_slurm_timelimit_seconds(os.environ.get('SLURM_TIMELIMIT'))
    if slurm_limit_s is not None:
        return STARTUP_TIME + float(slurm_limit_s), 'SLURM_TIMELIMIT'
    return None, None


def model_variant_label(args):
    bits = [f"level={args.level}", f"params_arg={args.params}"]
    if getattr(args, '_model_param_slug', None):
        bits.append(f"derived_params={args._model_param_slug}")
    if getattr(args, '_model_total_params', None) is not None:
        bits.append(f"total_params={args._model_total_params}")
    if getattr(args, 'mlp_ratio', 0.0):
        bits.append(f"mlp_ratio={args.mlp_ratio}")
    if str(args.level).lower() in ('gdn2', 'gdn2-mlp'):
        bits.append(f"gdn2_mlp_ratio={getattr(args, 'gdn2_mlp_ratio', None)}")
    return ','.join(bits)


def count_model_parameters(model):
    """Return exact total/trainable counts from the instantiated model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_params': int(total),
        'trainable_params': int(trainable),
    }


def human_param_slug(total_params):
    """Format a compact parameter-count slug for run names."""
    total_params = int(total_params)
    if total_params >= 1_000_000_000:
        return f"{total_params / 1_000_000_000:.1f}B"
    if total_params >= 1_000_000:
        return f"{total_params / 1_000_000:.0f}M"
    if total_params >= 1_000:
        return f"{total_params / 1_000:.0f}K"
    return str(total_params)


def model_family_slug(args):
    """Return a grounded family label for visible run names."""
    level = str(args.level)
    level_lower = level.lower()
    if level in ('E97', '97', 'E97-M2') or bool(getattr(args, 'e88_raw_write', 0)):
        return 'emender'
    if level_lower in ('typed-gdn2-lm', 'e98-cma-lm'):
        return 'emender'
    if level_lower in ('gdn2', 'gdn2-mlp'):
        return 'gdn2'
    return 'level'


def _safe_slug_part(value):
    text = str(value)
    text = re.sub(r'[^A-Za-z0-9_.-]+', '-', text)
    return text.strip('-') or 'model'


def build_run_label_prefix(args, total_params):
    """Build the timestamp-free run label prefix from resolved model facts."""
    family = _safe_slug_part(model_family_slug(args))
    level = _safe_slug_part(args.level)
    param_slug = _safe_slug_part(human_param_slug(total_params))
    if family == 'level':
        return f"level{level}_{param_slug}"
    return f"{family}_{level}_{param_slug}"


def attach_model_run_metadata(args, model):
    """Attach derived run-name and exact parameter metadata to parsed args."""
    counts = count_model_parameters(model)
    total_params = counts['total_params']
    param_slug = human_param_slug(total_params)
    run_label_prefix = build_run_label_prefix(args, total_params)
    metadata = {
        'model_family': model_family_slug(args),
        'level': str(args.level),
        'params_arg': str(args.params),
        'derived_param_slug': param_slug,
        'run_label_prefix': run_label_prefix,
        **counts,
    }
    args._model_family = metadata['model_family']
    args._model_param_slug = param_slug
    args._model_run_label_prefix = run_label_prefix
    args._model_total_params = metadata['total_params']
    args._model_trainable_params = metadata['trainable_params']
    args._model_metadata = metadata
    args._model_variant = model_variant_label(args)
    return metadata


class FinalCheckpointController:
    """Coordinates walltime/signal-triggered final checkpoint entry."""

    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.rank = int(getattr(args, '_rank', 0))
        self.world_size = int(getattr(args, '_world_size', 1))
        self.request_dir = Path(getattr(args, 'output', '.'))
        self.request_path = self.request_dir / '.final_checkpoint_request'
        self.ready_dir = self.request_dir / '.final_checkpoint_ready'
        self.deadline, self.deadline_source = resolve_walltime_deadline(args)
        self.margin_s = max(0.0, float(getattr(args, 'walltime_final_checkpoint_margin_seconds', 0.0)))
        self.check_every = max(1, int(getattr(args, 'walltime_check_every', 1)))
        self.triggered = False
        self.reason = None
        self.remaining_s = None
        self.logged_probe = False

    @property
    def walltime_active(self):
        return self.deadline is not None

    def _local_request(self):
        now = time.time()
        if _SHUTDOWN_REQUEST['requested']:
            return f"signal:{_SHUTDOWN_REQUEST['signal']}", (
                self.deadline - now if self.deadline is not None else None)
        if self.deadline is None:
            return None, None
        remaining = self.deadline - now
        if remaining <= self.margin_s:
            return f"walltime:{self.deadline_source}", remaining
        return None, remaining

    def reset_coordination_files(self):
        """Clear stale non-collective final-checkpoint coordination files."""
        try:
            self.request_path.unlink()
        except FileNotFoundError:
            pass
        if self.ready_dir.exists():
            for path in self.ready_dir.glob('rank_*.ready'):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
        else:
            self.ready_dir.mkdir(parents=True, exist_ok=True)

    def rebind_coordination_dir(self, request_dir):
        """Move future final-checkpoint coordination files to a run directory."""
        self.request_dir = Path(request_dir)
        self.request_path = self.request_dir / '.final_checkpoint_request'
        self.ready_dir = self.request_dir / '.final_checkpoint_ready'

    def _write_stop_request(self, reason, step):
        self.request_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            'reason': reason,
            'step': int(step),
            'rank': self.rank,
            'world_size': self.world_size,
            'time': time.time(),
        }
        fd, tmp_name = tempfile.mkstemp(
            prefix='.final_checkpoint_request.',
            suffix='.tmp',
            dir=str(self.request_dir),
            text=True,
        )
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(payload, f, sort_keys=True)
                f.write('\n')
            os.replace(tmp_name, self.request_path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    def _read_peer_request_reason(self):
        try:
            with open(self.request_path) as f:
                payload = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        return payload.get('reason') or 'peer_final_checkpoint_request'

    def _poll_peer_request_reason(self, timeout_s=0.0, poll_s=0.05):
        deadline = time.time() + max(0.0, float(timeout_s))
        while True:
            reason = self._read_peer_request_reason()
            if reason:
                return reason
            if time.time() >= deadline:
                return None
            time.sleep(min(poll_s, max(0.0, deadline - time.time())))

    def maybe_request_stop(self, step, dist_enabled=False, peer_poll_s=0.0,
                           extra_local_reason=None, extra_remaining=None):
        """Return (stop, reason, remaining_s) at safe optimizer-step boundaries."""
        if self.triggered:
            return True, self.reason, self.remaining_s
        if step % self.check_every != 0:
            return False, None, None

        local_reason, remaining = self._local_request()
        if local_reason is None and extra_local_reason:
            local_reason = extra_local_reason
            remaining = extra_remaining
        if local_reason:
            self._write_stop_request(local_reason, step)
            reason = local_reason
        else:
            reason = None

        if dist_enabled:
            peer_reason = self._poll_peer_request_reason(peer_poll_s)
            if peer_reason and not reason:
                reason = 'peer_final_checkpoint_request'
        elif not reason:
            reason = self._read_peer_request_reason()

        if reason:
            self.triggered = True
            self.reason = reason
            self.remaining_s = remaining
            return True, self.reason, self.remaining_s
        return False, None, remaining

    def mark_finalization_ready(self, step):
        self.ready_dir.mkdir(parents=True, exist_ok=True)
        ready_path = self.ready_dir / f'rank_{self.rank:06d}.ready'
        payload = {
            'step': int(step),
            'rank': self.rank,
            'world_size': self.world_size,
            'reason': self.reason or 'training_complete',
            'time': time.time(),
        }
        fd, tmp_name = tempfile.mkstemp(
            prefix=f'.rank_{self.rank:06d}.',
            suffix='.tmp',
            dir=str(self.ready_dir),
            text=True,
        )
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(payload, f, sort_keys=True)
                f.write('\n')
            os.replace(tmp_name, ready_path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    def wait_for_all_finalization_ready(self, timeout_s=1800.0, poll_s=0.05):
        if self.world_size <= 1:
            return
        deadline = time.time() + timeout_s
        expected = self.world_size
        while True:
            ready = len(list(self.ready_dir.glob('rank_*.ready'))) if self.ready_dir.exists() else 0
            if ready >= expected:
                return
            if time.time() > deadline:
                raise TimeoutError(
                    f"timed out waiting for final-checkpoint readiness: "
                    f"{ready}/{expected} ranks reported ready in {self.ready_dir}")
            time.sleep(poll_s)


def distributed_any(local_flag, device, dist_enabled=True):
    """Return whether any distributed rank reported local_flag=True."""
    if not dist_enabled or not dist.is_initialized():
        return bool(local_flag)
    flag = torch.tensor(
        [1 if local_flag else 0],
        device=device,
        dtype=torch.int32,
    )
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def consensus_final_checkpoint_stop(controller, local_stop, reason, remaining, device, dist_enabled):
    """Propagate final-checkpoint stop decisions at a collective-safe boundary."""
    if not dist_enabled or not dist.is_initialized():
        return local_stop, reason, remaining
    if distributed_any(local_stop, device, dist_enabled=True):
        if not local_stop:
            reason = controller._read_peer_request_reason() or 'peer_final_checkpoint_request'
            controller.triggered = True
            controller.reason = reason
            controller.remaining_s = remaining
        return True, reason, remaining
    return False, None, remaining


def parse_level(level_str):
    """Parse level string to int or keep as string for log-space levels."""
    if level_str.startswith('log_'):
        return level_str  # Keep as string for log-space levels
    try:
        return int(level_str)
    except ValueError:
        return level_str  # Keep as string for any other format


def _resolve_library_from_ld_path(name):
    paths = []
    for root_var in ('OLCF_OFI_NCCL_ROOT', 'AWS_OFI_RCCL_PLUGIN_DIR'):
        root = os.environ.get(root_var)
        if root:
            paths.extend([
                os.path.join(root, 'lib', name),
                os.path.join(root, 'lib64', name),
            ])
            if root_var == 'OLCF_OFI_NCCL_ROOT':
                rocm_module = os.environ.get('FRONTIER_ROCM_MODULE')
                if rocm_module:
                    paths.append(os.path.join(root, rocm_module, 'lib', name))
                paths.extend(glob.glob(os.path.join(root, 'rocm', '*', 'lib', name)))
    for part in os.environ.get('LD_LIBRARY_PATH', '').split(':'):
        if part:
            paths.append(os.path.join(part, name))
    for path in paths:
        if os.path.isfile(path) and os.access(path, os.R_OK):
            return path
    found = ctypes.util.find_library(name[:-3] if name.endswith('.so') else name)
    return found or None


def _selected_env(prefixes, names):
    keys = sorted(
        key for key in os.environ
        if key in names or any(key.startswith(prefix) for prefix in prefixes)
    )
    return {key: os.environ.get(key) for key in keys}


def frontier_runtime_manifest():
    """Capture Frontier communication/runtime state for post-run diagnosis."""
    try:
        import triton
        triton_version = getattr(triton, '__version__', None)
    except Exception as exc:
        triton_version = f"import-error:{exc!r}"

    return {
        'python_version': sys.version.split()[0],
        'python_executable': sys.executable,
        'torch_version': getattr(torch, '__version__', None),
        'torch_version_hip': getattr(torch.version, 'hip', None),
        'triton_version': triton_version,
        'olcf_ofi_nccl_root': os.environ.get('OLCF_OFI_NCCL_ROOT'),
        'librccl_net_path': _resolve_library_from_ld_path('librccl-net.so') or 'not-found',
        'env': _selected_env(
            prefixes=('FI_CXI', 'NCCL'),
            names={
                'HSA_FORCE_FINE_GRAIN_PCIE',
                'OLCF_OFI_NCCL_ROOT',
                'NCCL_NET_PLUGIN',
                'FRONTIER_RUNTIME_PROFILE',
                'FRONTIER_ENABLE_OLCF_RCCL_PLUGIN',
                'FRONTIER_ROCM_MODULE',
                'FRONTIER_RCCL_NET_PLUGIN_MODULE',
                'NDM_DISTRIBUTED_INIT_TIMEOUT_SECONDS',
            },
        ),
    }


def setup_output_dir(args, model_metadata=None):
    """Create output directory with run info."""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    prefix = getattr(args, '_model_run_label_prefix', None)
    if prefix is None and model_metadata:
        prefix = model_metadata.get('run_label_prefix')
    if prefix is None:
        prefix = f"level{args.level}_{args.params}"
    exact_output_dir = getattr(args, 'exact_output_dir', None)
    if exact_output_dir:
        output_dir = Path(exact_output_dir)
        run_name = output_dir.name
    else:
        run_name = f"{prefix}_{timestamp}"
        output_dir = Path(args.output) / run_name

    output_dir.mkdir(parents=True, exist_ok=True)

    run_manifest = {
        'run_name': run_name,
        'run_label_prefix': prefix,
        'output_dir': str(output_dir),
        'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        'resume': getattr(args, 'resume', None),
        'diloco_bootstrap_outer_state': getattr(args, 'diloco_bootstrap_outer_state', 'none'),
        'model': model_metadata or getattr(args, '_model_metadata', None),
        'sampler': (
            {"schema": LEGACY_SAMPLER_SCHEMA, "status": "legacy"}
            if counter_sampler_identity_from_args(
                args, world_size=int(getattr(args, '_world_size', 1))) is None
            else counter_sampler_identity_from_args(
                args, world_size=int(getattr(args, '_world_size', 1))).to_metadata()
        ),
        'runtime': frontier_runtime_manifest(),
    }

    # Save args
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    with open(output_dir / 'run_manifest.json', 'w') as f:
        json.dump(run_manifest, f, indent=2)

    return output_dir


def lr_scale_at(step, warmup_steps, total_steps, min_lr_frac=0.1):
    """Warmup + cosine-decay LR multiplier in [min_lr_frac, 1.0].

    Linear warmup over `warmup_steps`, then a SINGLE cosine decay from 1.0 down to
    `min_lr_frac` across the remaining steps to `total_steps`. Returns a scale that
    the caller multiplies onto each param group's base LR (preserving per-group
    multipliers such as --knob_lr_mult).

    The previous `get_lr` used `step/warmup_steps` as the cosine phase, so the LR
    oscillated with period 2*warmup_steps and collapsed to min right after warmup
    instead of decaying over the run — it never referenced total_steps. Constant-LR
    schedule-free has been MEASURED (task fix-long-horizon) to roll the held-out
    x-average over at long horizon even WITH warmup; a real decay to a small final
    LR is what keeps held-out BPB monotone. See docs/SCALE_PLAN.md §2.2.
    """
    import math
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    denom = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / denom))
    return min_lr_frac + (1.0 - min_lr_frac) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None or value == '':
        return default
    return int(value)


def _env_bool_int(name, default=False):
    value = os.environ.get(name)
    if value is None or value == '':
        return 1 if default else 0
    return 1 if value.strip().lower() in ('1', 'true', 'yes', 'on') else 0


def _infer_visible_device_count():
    visible = (
        os.environ.get('CUDA_VISIBLE_DEVICES')
        or os.environ.get('HIP_VISIBLE_DEVICES')
        or os.environ.get('ROCR_VISIBLE_DEVICES')
    )
    if visible is not None:
        entries = [x for x in visible.split(',') if x.strip()]
        return len(entries)
    try:
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except Exception:
        pass
    return None


def resolve_distributed_env_from_slurm(device_count=None):
    """Fill torch.distributed env from Slurm when an srun wrapper forgot it.

    Frontier debug jobs use ``--gpus-per-task=1 --gpu-bind=closest``. In that
    mode each task sees one rank-local GPU as device 0, so ``LOCAL_RANK`` must
    be 0 even when ``SLURM_LOCALID`` ranges over tasks on the node. Returning a
    status string keeps this logic unit-testable without initializing dist.
    """
    slurm_ntasks = os.environ.get('SLURM_NTASKS')
    if not slurm_ntasks:
        return 'no-slurm'
    try:
        ntasks = int(slurm_ntasks)
    except ValueError:
        return 'invalid-slurm-ntasks'
    if ntasks <= 1:
        return 'single-slurm-task'
    if os.environ.get('WORLD_SIZE'):
        return 'world-size-present'

    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = os.environ.get('SLURM_PROCID', '0')
    if 'LOCAL_RANK' not in os.environ:
        if device_count is None:
            device_count = _infer_visible_device_count()
        if device_count == 1:
            os.environ['LOCAL_RANK'] = '0'
        else:
            os.environ['LOCAL_RANK'] = os.environ.get('SLURM_LOCALID', '0')
    return 'derived-from-slurm'


def _validated_total_tokens(value, source):
    if type(value) is not int:  # bool is intentionally not an integer authority.
        raise TypeError(f"{source} total_tokens must be an integer")
    if value < 0:
        raise ValueError(f"{source} total_tokens must be nonnegative")
    return value


def tokens_per_global_optimizer_step(world_size, batch_size, chunk_size, grad_accum):
    """Return authoritative fixed-world tokens committed by one optimizer step."""
    values = {
        'world_size': world_size,
        'batch_size': batch_size,
        'chunk_size': chunk_size,
        'grad_accum': grad_accum,
    }
    for name, value in values.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer for token accounting")
    return world_size * batch_size * chunk_size * grad_accum


def e97_fused_runtime_contract_failures(args):
    """Return configuration clauses that would route BF16 E97 to eager recurrence."""
    e97_family = (
        str(getattr(args, 'level', '')) in ('E97', '97', 'E97-M2')
        or bool(getattr(args, 'e88_raw_write', 0))
    )
    if not e97_family or not bool(getattr(args, 'bf16', False)):
        return []
    contract = {
        'use_triton': int(getattr(args, 'use_triton', 0)) == 1,
        'use_gate': bool(getattr(args, 'use_gate', 0)),
        'gate_activation_silu': getattr(args, 'gate_activation', None) == 'silu',
        'no_write_gate': not bool(getattr(args, 'use_write_gate', 0)),
        'no_value_residual': not bool(getattr(args, 'e88_value_residual', 0)),
    }
    return [name for name, satisfied in contract.items() if not satisfied]


def advance_total_tokens(total_tokens, world_size, batch_size, chunk_size, grad_accum):
    """Advance the in-memory clock after one successfully completed optimizer step."""
    current = _validated_total_tokens(total_tokens, 'current')
    return current + tokens_per_global_optimizer_step(
        world_size, batch_size, chunk_size, grad_accum)


def resolve_checkpoint_total_tokens(checkpoint, explicit_total_tokens=None,
                                    checkpoint_path=None):
    """Resolve the restart clock, rejecting ambiguity or contradictory input.

    Legacy checkpoints have no authoritative clock and therefore require one
    explicit trusted bootstrap.  Once a checkpoint embeds the clock, that
    embedded value is authoritative across changed-world restarts and requeues.
    """
    source = str(checkpoint_path or '<checkpoint>')
    explicit = None
    if explicit_total_tokens is not None:
        explicit = _validated_total_tokens(explicit_total_tokens, 'explicit')

    if 'total_tokens' not in checkpoint:
        if explicit is None:
            raise ValueError(
                f"legacy checkpoint {source} lacks total_tokens; "
                "provide trusted --total_tokens explicitly"
            )
        return explicit

    embedded = _validated_total_tokens(checkpoint['total_tokens'], 'embedded checkpoint')
    metadata = checkpoint.get('checkpoint_metadata')
    if isinstance(metadata, dict) and 'total_tokens' in metadata:
        mirrored = _validated_total_tokens(metadata['total_tokens'], 'checkpoint metadata')
        if mirrored != embedded:
            raise ValueError(
                f"checkpoint metadata total_tokens {mirrored} contradicts "
                f"top-level total_tokens {embedded} in {source}"
            )
    if explicit is not None and explicit != embedded:
        raise ValueError(
            f"explicit total_tokens {explicit} contradicts embedded "
            f"total_tokens {embedded} in {source}"
        )
    return embedded


def counter_sampler_identity_from_args(args, *, world_size):
    """Resolve a complete scientific sampler identity or explicit legacy mode."""
    schema = getattr(args, 'sampler_schema', None)
    fields = {
        'corpus_sha256': getattr(args, 'sampler_corpus_sha256', None),
        'tokenizer_sha256': getattr(args, 'sampler_tokenizer_sha256', None),
        'sampler_key': getattr(args, 'sampler_key', None),
        'data_world_size': getattr(args, 'sampler_data_world_size', None),
    }
    stream_origin = getattr(args, 'sampler_stream_origin_accepted_tokens', None)
    if schema is None:
        provided = [name for name, value in fields.items() if value is not None]
        if stream_origin is not None:
            provided.append('stream_origin_accepted_tokens')
        if provided:
            raise ValueError(
                "sampler identity fields require --sampler_schema: " + ", ".join(provided))
        return None
    if schema not in COUNTER_SAMPLER_SCHEMAS:
        raise ValueError(
            f"unsupported --sampler_schema {schema!r}; expected one of "
            f"{sorted(COUNTER_SAMPLER_SCHEMAS)!r}")
    missing = [name for name, value in fields.items() if value is None]
    if schema == BOUNDARY_COUNTER_SAMPLER_SCHEMA and stream_origin is None:
        missing.append('stream_origin_accepted_tokens')
    if schema == COUNTER_SAMPLER_SCHEMA and stream_origin not in (None, 0):
        raise ValueError("counter-v1 cannot use a nonzero stream origin")
    if missing:
        raise ValueError("counter sampler identity missing fields: " + ", ".join(missing))
    if int(fields['data_world_size']) != int(world_size):
        raise ValueError(
            f"sampler data world {fields['data_world_size']} does not match launched "
            f"WORLD_SIZE {world_size}")
    tokenizer_name = getattr(args, 'tokenizer', None)
    if not tokenizer_name:
        raise ValueError("counter sampler requires an explicit tokenizer")
    return CounterSamplerIdentity(
        schema=schema,
        corpus_sha256=fields['corpus_sha256'],
        tokenizer_sha256=fields['tokenizer_sha256'],
        sampler_key=int(fields['sampler_key']),
        data_world_size=int(fields['data_world_size']),
        context_size=int(args.chunk_size),
        stream_origin_accepted_tokens=int(stream_origin or 0),
    )


def dense_sampler_checkpoint_metadata(args, *, world_size, total_tokens):
    """Build the rank-independent sampler authority persisted atomically."""
    identity = counter_sampler_identity_from_args(args, world_size=world_size)
    if identity is None:
        return {"schema": LEGACY_SAMPLER_SCHEMA, "status": "legacy"}
    return sampler_checkpoint_metadata(identity, total_accepted_tokens=total_tokens)


def validate_dense_checkpoint_sampler(checkpoint, args, *, world_size,
                                      checkpoint_path=None):
    """Fail closed on sampler drift before model/optimizer state is loaded."""
    total_tokens = resolve_checkpoint_total_tokens(
        checkpoint, getattr(args, 'total_tokens', None), checkpoint_path)
    expected = counter_sampler_identity_from_args(args, world_size=world_size)
    checkpoint_metadata = checkpoint.get('checkpoint_metadata')
    persisted = (
        checkpoint_metadata.get('sampler')
        if isinstance(checkpoint_metadata, dict) else None
    )
    if expected is None:
        if persisted is None or persisted == {
                "schema": LEGACY_SAMPLER_SCHEMA, "status": "legacy"}:
            return total_tokens
        raise ValueError(
            "checkpoint contains a counter sampler identity but launch arguments "
            "select legacy mode")
    if persisted is None:
        raise ValueError(
            "legacy checkpoint lacks counter sampler metadata and cannot be "
            "silently relabelled")
    if persisted.get('schema') == LEGACY_SAMPLER_SCHEMA:
        raise ValueError("legacy sampler checkpoint cannot resume as counter sampler")
    _identity, persisted_tokens, _cursor = restore_sampler_checkpoint_metadata(
        persisted, expected_identity=expected)
    if persisted_tokens != total_tokens:
        raise ValueError("sampler accepted-token clock contradicts checkpoint total_tokens")
    return total_tokens


def save_checkpoint(model, optimizer, step, loss, output_dir, keep_n=5, outer_state=None,
                    metadata=None, *, total_tokens):
    """Atomically save model/optimizer state and its authoritative token clock."""
    total_tokens = _validated_total_tokens(total_tokens, 'checkpoint')
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f'checkpoint_step_{step:06d}_loss_{loss:.4f}.pt'

    checkpoint_metadata = dict(metadata or {})
    if ('total_tokens' in checkpoint_metadata
            and checkpoint_metadata['total_tokens'] != total_tokens):
        raise ValueError("checkpoint metadata total_tokens contradicts checkpoint authority")
    checkpoint_metadata['total_tokens'] = total_tokens
    payload = {
        'step': step,
        'total_tokens': total_tokens,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'checkpoint_metadata': checkpoint_metadata,
    }
    if outer_state is not None:
        payload['diloco_outer_state'] = outer_state

    with tempfile.NamedTemporaryFile(
            prefix=f'.{ckpt_path.name}.',
            suffix='.tmp',
            dir=output_dir,
            delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, ckpt_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    # Update latest symlink atomically so readers never see a missing/latest half
    # update. Under distributed training only rank 0 calls this function.
    latest_path = output_dir / 'latest.pt'
    tmp_latest = output_dir / f'.latest.pt.{os.getpid()}.tmp'
    try:
        if tmp_latest.exists() or tmp_latest.is_symlink():
            tmp_latest.unlink()
        tmp_latest.symlink_to(ckpt_path.name)
        os.replace(tmp_latest, latest_path)
    finally:
        if tmp_latest.exists() or tmp_latest.is_symlink():
            tmp_latest.unlink()

    # Clean up old checkpoints by numeric training step.  Lexicographic filename
    # ordering breaks across digit-width boundaries such as 999500 -> 1000000.
    def _checkpoint_sort_key(path):
        match = re.search(r'checkpoint_step_(\d+)_loss_', os.path.basename(path))
        return int(match.group(1)) if match else -1

    ckpts = sorted(glob.glob(str(output_dir / 'checkpoint_step_*.pt')),
                   key=_checkpoint_sort_key)
    # A periodic and final save can legitimately share one training step but
    # have different loss strings.  Numeric-step sorting alone is then tied and
    # used to delete the checkpoint that ``latest.pt`` was just advanced to.
    # Always retain the checkpoint published by this call and evict older peers
    # around it.  ``latest.pt`` must never become a broken symlink.
    retain = max(1, int(keep_n))
    others = [path for path in ckpts if Path(path) != ckpt_path]
    while len(others) + 1 > retain:
        os.remove(others.pop(0))

    return ckpt_path


def load_checkpoint(path, model, optimizer=None, return_checkpoint=False,
                    preflight=None):
    """Load a checkpoint without privately materializing its full CPU storage.

    Frontier launches several GPU trainers per physical node.  A conventional
    ``torch.load`` allocates one anonymous CPU copy of every model and optimizer
    tensor in every trainer, multiplying a large restart image by the local GPU
    count.  File-backed storages let the kernel share clean checkpoint pages
    across those independently supervised processes; ``load_state_dict`` still
    gives each model the normal owned parameter storage it requires.
    """
    ckpt = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    if preflight is not None:
        preflight(ckpt)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    result = (ckpt.get('step', 0), ckpt.get('loss', float('inf')))
    if return_checkpoint:
        return (*result, ckpt)
    return result


@torch.no_grad()
def validate(model, val_loader, device, max_batches=100):
    """Run validation."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    for i, (chunk, is_doc_end, actual_lengths) in enumerate(val_loader):
        if i >= max_batches:
            break

        chunk = chunk.to(device)
        loss = model(chunk, return_loss=True)

        # Weight by actual tokens
        batch_tokens = actual_lengths.sum().item()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

    model.train()
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def score_heldout_tensor(model, heldout_chunks, bytes_per_token, device, eval_bs=8, bf16=False):
    """Score a fixed token tensor and return (CE nats/token, BPB, scored tokens)."""
    import math as _math
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    eval_bs = max(1, min(int(eval_bs), heldout_chunks.shape[0]))
    for i in range(0, heldout_chunks.shape[0], eval_bs):
        batch = heldout_chunks[i:i + eval_bs].to(device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                            enabled=bool(bf16) and device.type == 'cuda'):
            loss = model(batch, return_loss=True)
        loss = loss[0] if isinstance(loss, tuple) else loss
        scored = batch.shape[0] * (batch.shape[1] - 1)
        total_nll += float(loss.item()) * scored
        total_tokens += scored
    model.train()
    ce = total_nll / max(total_tokens, 1)
    bpb = (ce / _math.log(2.0)) / bytes_per_token
    return ce, bpb, total_tokens


def prepare_schedulefree_eval_mode(optimizer, args):
    """Put schedule-free params in the requested eval basis."""
    if args.optimizer != 'schedulefree':
        return
    if args.heldout_eval_mode in ('y', 'train'):
        optimizer.train()
    else:
        optimizer.eval()


def heldout_eval_mode_label(args):
    """Canonical label for the schedule-free held-out weight basis."""
    return 'y' if args.heldout_eval_mode in ('y', 'train') else 'x'


def normalize_training_args(args, *, announce=False):
    """Apply train.py's import-safe argument normalization in-place."""
    args.level = parse_level(args.level)
    if (bool(getattr(args, 'offload_schedulefree_state', False))
            and getattr(args, 'optimizer', 'schedulefree') != 'schedulefree'):
        raise ValueError(
            "--offload_schedulefree_state requires --optimizer schedulefree")
    if (bool(getattr(args, 'offload_schedulefree_state', False))
            and bool(getattr(args, 'diloco', False))
            and getattr(args, 'diloco_outer_optimizer', 'avg') != 'avg'):
        raise ValueError(
            "CPU-offloaded Schedule-Free currently requires "
            "--diloco_outer_optimizer avg; momentum/sfsgd need an additional "
            "offloaded-z translation path")
    if getattr(args, 'use_triton', None) is None:
        _e97_family = str(args.level) in ('E97', '97', 'E97-M2')
        _needs_triton_fused = _e97_family or bool(getattr(args, 'e88_raw_write', 0))
        if _needs_triton_fused and getattr(args, 'bf16', False):
            args.use_triton = 1
            if announce:
                print(f"[fused] level={args.level!r} raw_write={bool(getattr(args, 'e88_raw_write', 0))}: "
                      f"AUTO-enabling Triton split-edit kernel (--use_triton 1). Pass --use_triton 0 to force eager.",
                      flush=True)
        else:
            args.use_triton = 0
    return args


def resolve_training_r_h_mode(args, *, announce=False):
    """Resolve the effective recurrent W_h constraint mode used by train.py."""
    r_h_mode = getattr(args, 'r_h_mode', 'auto')
    if r_h_mode == 'auto' and args.level != 'mamba2':
        full_wh_levels = {1, 33, 42, 51, 52, 53, 56, 57, 58, 60}
        matrix_state_levels = {70, 71, 72, 73}
        level_int = int(args.level) if str(args.level).isdigit() else 0
        if level_int in full_wh_levels:
            r_h_mode = 'spectral_norm'
            if announce:
                print(f"Auto r_h_mode: spectral_norm (level {level_int} has full W_h)")
        elif level_int in matrix_state_levels:
            r_h_mode = 'none'
            if announce:
                print(f"Auto r_h_mode: none (level {level_int} is matrix state - gated update is bounded)")
        else:
            r_h_mode = 'none'
            if announce:
                print(f"Auto r_h_mode: none (level {level_int} has bounded/no W_h)")
    return r_h_mode


def resolve_training_vocab_size(args, *, announce=False):
    """Return the token vocabulary size for byte-level or tiktoken training."""
    if getattr(args, 'tokenizer', None):
        import tiktoken
        enc = tiktoken.get_encoding(args.tokenizer)
        vocab_size = enc.n_vocab
        if announce:
            print(f"Tokenizer: {args.tokenizer}, vocab_size={vocab_size}")
        return vocab_size
    return 256


def build_training_model(args, *, vocab_size=None, r_h_mode=None):
    """Build an unwrapped train.py LadderLM model without invoking the CLI.

    The helper intentionally exposes the real E97/Ladder path needed by the
    async DiLoCo orchestrator. Legacy special-case baseline branches remain in
    train(args) so existing CLI behavior stays unchanged.
    """
    normalize_training_args(args)
    if vocab_size is None:
        vocab_size = resolve_training_vocab_size(args)
    if r_h_mode is None:
        r_h_mode = resolve_training_r_h_mode(args)

    if args.dim is not None and args.depth is not None:
        layer_kwargs = {}
        if getattr(args, 'head_type_logits', None) is not None:
            layer_kwargs['head_type_logits'] = [float(x) for x in args.head_type_logits.split(',')]
            layer_kwargs['gdn_allow_neg_eigval'] = bool(getattr(args, 'gdn_allow_neg_eigval', 1))
        if getattr(args, 'corner_mixture', None) is not None:
            layer_kwargs['corner_mixture'] = [float(x) for x in args.corner_mixture.split(',')]
        if getattr(args, 'lam_max', None) is not None:
            layer_kwargs['lam_max'] = args.lam_max
        if getattr(args, 'beta_max', None) is not None:
            layer_kwargs['beta_max'] = args.beta_max
        if getattr(args, 'igain_max', None) is not None:
            layer_kwargs['igain_max'] = args.igain_max
        if getattr(args, 'layer_kwargs', None) is not None:
            layer_kwargs.update(json.loads(args.layer_kwargs))
        return LadderLM(
            vocab_size=vocab_size,
            dim=args.dim,
            depth=args.depth,
            level=args.level,
            layer_kwargs=(layer_kwargs or None),
            expansion=getattr(args, 'expansion', 1.0),
            n_groups=getattr(args, 'n_groups', 32),
            n_state=getattr(args, 'n_state', 64),
            n_slots=getattr(args, 'n_slots', 64),
            n_heads=getattr(args, 'n_heads', None),
            top_k=getattr(args, 'top_k', None),
            k_fast=getattr(args, 'k_fast', None),
            k_slow=getattr(args, 'k_slow', None),
            use_gate=bool(getattr(args, 'use_gate', 1)),
            gate_activation=getattr(args, 'gate_activation', 'sigmoid'),
            linear_state=bool(getattr(args, 'linear_state', 0)),
            use_write_gate=bool(getattr(args, 'use_write_gate', 0)),
            e88_decay_mode=getattr(args, 'e88_decay_mode', 'mamba'),
            e88_value_residual=bool(getattr(args, 'e88_value_residual', 0)),
            e88_raw_write=bool(getattr(args, 'e88_raw_write', 0)),
            use_chunked_e97=bool(getattr(args, 'use_chunked_e97', 0)),
            e97_chunk_size=getattr(args, 'e97_chunk_size', 32),
            state_expansion=getattr(args, 'state_expansion', 2),
            r_h_mode=r_h_mode,
            use_conv=bool(getattr(args, 'use_conv', 0)),
            d_conv=getattr(args, 'd_conv', 4),
            gdn2_mlp_ratio=getattr(args, 'gdn2_mlp_ratio', 6208 / 2304),
            dropout=getattr(args, 'dropout', 0.0),
            checkpoint_interval=getattr(args, 'checkpoint_interval', 16),
            gradient_checkpointing=bool(getattr(args, 'gradient_checkpointing', False)),
            projection_chunk_size=getattr(args, 'projection_chunk_size', 0),
            loss_chunk_size=getattr(args, 'loss_chunk_size', 0),
            use_triton=bool(getattr(args, 'use_triton', 0)),
            mlp_ratio=getattr(args, 'mlp_ratio', 0.0),
            mlp_multiple=getattr(args, 'mlp_multiple', 64),
        )

    return create_ladder_model(
        target_params=args.params,
        level=args.level,
        vocab_size=vocab_size,
        expansion=getattr(args, 'expansion', 1.0),
        n_groups=getattr(args, 'n_groups', 32),
        state_expansion=getattr(args, 'state_expansion', 2),
        r_h_mode=r_h_mode,
    )


def build_training_optimizer(core_model, args, *, announce=False):
    """Create train.py-compatible optimizer and per-group base_lr metadata."""
    knob_suffixes = ('lam_raw', 'beta_raw', 'igain_raw', 'gamma_raw')
    knob_params, base_params = [], []
    for name, p in core_model.named_parameters():
        if not p.requires_grad:
            continue
        if any(name.endswith(s) for s in knob_suffixes):
            knob_params.append(p)
        else:
            base_params.append(p)

    knob_lr_mult = float(getattr(args, 'knob_lr_mult', 1.0))
    use_knob_group = knob_lr_mult != 1.0 and len(knob_params) > 0
    if use_knob_group:
        param_groups = [
            {'params': base_params, 'lr': args.lr},
            {'params': knob_params, 'lr': args.lr * knob_lr_mult},
        ]
        if announce:
            print(f"Knob-LR group: {len(knob_params)} knob params at lr="
                  f"{args.lr * knob_lr_mult:.2e} ({knob_lr_mult}x base); "
                  f"{len(base_params)} base params at lr={args.lr:.2e}")
    else:
        param_groups = core_model.parameters()

    if args.optimizer == 'schedulefree':
        if bool(getattr(args, 'offload_schedulefree_state', False)):
            optimizer = CPUOffloadAdamWScheduleFree(
                param_groups,
                lr=args.lr,
                weight_decay=args.weight_decay,
                betas=(0.9, 0.95),
                warmup_steps=getattr(args, 'warmup_steps', 0),
                pin_memory=bool(getattr(args, 'schedulefree_offload_pin_memory', 1)),
                release_gradients=bool(getattr(
                    args, 'schedulefree_offload_release_gradients', 1)),
                bucket_numel=int(getattr(
                    args, 'schedulefree_offload_bucket_numel', 67_108_864)),
            )
            if announce:
                print(f"Using schedule-free AdamW with pinned-CPU z/second-moment "
                      f"offload (lr={args.lr}, warmup_steps={args.warmup_steps})")
        else:
            optimizer = schedulefree.AdamWScheduleFree(
                param_groups,
                lr=args.lr,
                weight_decay=args.weight_decay,
                betas=(0.9, 0.95),
                warmup_steps=getattr(args, 'warmup_steps', 0),
            )
            if announce:
                print(f"Using schedule-free AdamW (lr={args.lr}, warmup_steps={args.warmup_steps})")
    else:
        optimizer = AdamW(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
        )
        if announce:
            print(f"Using AdamW with warmup={args.warmup_steps} steps + cosine decay to "
                  f"{args.min_lr_frac:.0%} of lr over {args.steps} steps")

    for pg in optimizer.param_groups:
        pg['base_lr'] = pg['lr']
    return optimizer


def build_training_dataset(args, *, rank=0, dist_enabled=False, start_step=0,
                           world_size=None, total_accepted_tokens=None):
    """Create one rank's stream from an accepted-token authority or legacy seed."""
    data_seed = args.seed + int(start_step) + (rank if dist_enabled else 0)
    actual_world_size = int(
        world_size if world_size is not None else (getattr(args, '_world_size', 1)))
    sampler_identity = counter_sampler_identity_from_args(
        args, world_size=actual_world_size)
    if args.tbptt:
        if sampler_identity is not None:
            raise ValueError(
                "counter accepted-token sampler is not implemented for TBPTT")
        return BatchedStreamDataset(
            data_path=args.data,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size + 1,
            seed=data_seed,
        )
    if getattr(args, 'tokenizer', None):
        if sampler_identity is not None and total_accepted_tokens is None:
            raise ValueError(
                "counter sampler dataset requires total_accepted_tokens authority")
        return TokenizedStreamDataset(
            data_path=args.data,
            chunk_size=args.chunk_size + 1,
            rank=rank,
            world_size=actual_world_size,
            seed=data_seed,
            tokenizer_name=args.tokenizer,
            sampler_identity=sampler_identity,
            total_accepted_tokens=(
                total_accepted_tokens if sampler_identity is not None else None),
            accepted_tokens_per_sample=args.chunk_size,
        )
    return DocumentStreamDataset(
        data_path=args.data,
        chunk_size=args.chunk_size + 1,
        seed=data_seed,
    )


def get_training_batch(train_dataset, args, device):
    """Fetch one train.py-style batch from a streaming dataset."""
    if args.tbptt:
        chunks, is_doc_end = train_dataset.get_batch(device=device)
        actual_lengths = torch.full((args.batch_size,), args.chunk_size + 1, device=device)
    else:
        chunks, is_doc_end, actual_lengths = train_dataset.get_batch(args.batch_size, device=device)
    return chunks, is_doc_end, actual_lengths


def compute_training_loss(model, chunks, args, *, prev_hiddens=None):
    """Compute train.py's autoregressive loss for one chunk batch."""
    if getattr(args, 'bf16', False):
        autocast_ctx = torch.autocast(
            device_type=chunks.device.type,
            dtype=torch.bfloat16,
            enabled=True,
        )
    else:
        autocast_ctx = nullcontext()
    with autocast_ctx:
        if args.tbptt:
            result = model(
                chunks,
                return_loss=True,
                return_prev_hiddens=True,
                prev_hiddens=prev_hiddens,
            )
        else:
            result = model(
                chunks,
                return_loss=True,
            )

        if isinstance(result, tuple):
            loss, (next_hidden, _) = result
        else:
            loss = result
            next_hidden = None

    return loss, next_hidden


def train_one_optimizer_step(model, optimizer, args, *, batch_iter=None, device=None,
                             step=0, hidden_state=None, phase_callback=None):
    """Run one real train.py optimizer step, including grad accumulation."""
    if device is None:
        device = next(model.parameters()).device
    model.train()
    if args.optimizer == 'schedulefree':
        optimizer.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    tokens_processed = 0
    def phase(name, **details):
        if phase_callback is not None:
            phase_callback(name, {"step": step, **details})

    phase("optimizer_step_start")
    for micro_step in range(max(1, int(getattr(args, 'grad_accum', 1)))):
        if batch_iter is None:
            raise ValueError("batch_iter is required for train_one_optimizer_step")
        phase("data_load_start", micro_step=micro_step)
        chunks, is_doc_end, actual_lengths = next(batch_iter)
        chunks = chunks.to(device)
        actual_lengths = actual_lengths.to(device)
        phase("data_load_end", micro_step=micro_step,
              tokens=int(actual_lengths.sum().item()))
        if args.tbptt and hidden_state is not None:
            reset_mask = is_doc_end.to(device).view(-1, 1)
            hidden_state = [h * (~reset_mask) if h is not None else None for h in hidden_state]
        phase("forward_start", micro_step=micro_step)
        loss, next_hidden = compute_training_loss(
            model, chunks, args, prev_hiddens=hidden_state)
        phase("forward_end", micro_step=micro_step)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at step {step}: {loss.item()}")
        phase("backward_start", micro_step=micro_step)
        (loss / max(1, int(getattr(args, 'grad_accum', 1)))).backward()
        phase("backward_end", micro_step=micro_step)
        if args.tbptt and next_hidden is not None:
            hidden_state = [h.detach() if h is not None else None for h in next_hidden]
        phase("loss_sync_start", micro_step=micro_step)
        loss_value = float(loss.item())
        phase("loss_sync_end", micro_step=micro_step, loss=loss_value)
        total_loss += loss_value
        tokens_processed += int(actual_lengths.sum().item())

    if getattr(args, 'grad_clip', 0) > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    else:
        grad_norm = sum(
            p.grad.norm().item() ** 2 for p in model.parameters()
            if p.grad is not None
        ) ** 0.5
    if not torch.isfinite(torch.as_tensor(grad_norm)):
        optimizer.zero_grad(set_to_none=True)
        grad_value = grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm)
        raise FloatingPointError(f"non-finite grad norm at step {step}: {grad_value}")

    if args.optimizer == 'adamw':
        scale = lr_scale_at(step, args.warmup_steps, args.steps, args.min_lr_frac)
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['base_lr'] * scale
        lr = optimizer.param_groups[0]['lr']
    else:
        lr = args.lr

    phase("optimizer_update_start")
    optimizer.step()
    phase("optimizer_update_end")
    optimizer.zero_grad()
    grad_value = grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm)
    result = {
        'step': step + 1,
        'loss': total_loss / max(1, int(getattr(args, 'grad_accum', 1))),
        'grad_norm': grad_value,
        'lr': lr,
        'tokens_processed': tokens_processed,
        'hidden_state': hidden_state,
    }
    phase("optimizer_step_end", loss=result['loss'], tokens=tokens_processed)
    return result


def _clone_param_list_like(params, source=None, fill_zeros=False):
    out = []
    for i, p in enumerate(params):
        if source is None:
            t = torch.zeros_like(p.data) if fill_zeros else p.data.detach().clone()
        else:
            t = source[i].detach().to(device=p.device, dtype=p.dtype).clone()
            if fill_zeros:
                t.zero_()
        out.append(t)
    return out


def _snapshot_model_params(core_model):
    return [p.data.detach().clone() for p in core_model.parameters()]


def _restore_model_params(core_model, snapshot):
    for p, saved in zip(core_model.parameters(), snapshot):
        p.data.copy_(saved)


def _model_params_equal_snapshot(core_model, snapshot):
    return all(torch.equal(p.data, saved) for p, saved in zip(core_model.parameters(), snapshot))


def _current_git_commit():
    try:
        import subprocess
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _diloco_outer_bootstrap_metadata(args, ckpt=None, reason=None,
                                     inner_optimizer_state_loaded=None,
                                     source_basis='loaded',
                                     source_has_outer_state=None,
                                     model_tensors_equal_after_restore=None):
    ckpt = ckpt or {}
    if source_has_outer_state is None:
        source_has_outer_state = 'diloco_outer_state' in ckpt
    metadata = {
        'performed': True,
        'guard': 'from-loaded-model',
        'requested_cli': '--diloco_bootstrap_outer_state=from-loaded-model',
        'source_checkpoint': getattr(args, 'resume', None),
        'source_checkpoint_step': ckpt.get('step'),
        'source_checkpoint_has_diloco_outer_state': bool(source_has_outer_state),
        'missing_or_incompatible_reason': reason,
        'target_outer_optimizer': getattr(args, 'diloco_outer_optimizer', None),
        'target_export_basis': getattr(args, 'diloco_export_basis', None),
        'target_outer_lr': getattr(args, 'diloco_outer_lr', None),
        'target_outer_beta': getattr(args, 'diloco_outer_beta', None),
        'bootstrap_source': 'loaded_model_weights',
        'model_weight_mutation': 'none; restored byte-identical after bootstrap',
        'model_tensors_equal_after_restore': bool(model_tensors_equal_after_restore),
        'inner_optimizer_state_loaded': bool(inner_optimizer_state_loaded),
        'inner_schedulefree_basis_used_for_anchor': source_basis,
        'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        'code_commit': _current_git_commit(),
    }
    return metadata


def checkpoint_metadata_with_diloco_bootstrap(metadata, bootstrap_metadata):
    if bootstrap_metadata:
        out = dict(metadata or {})
        out['diloco_outer_state_bootstrap'] = bootstrap_metadata
        return out
    return metadata


def bootstrap_diloco_outer_state_from_loaded_model(core_model, optimizer, args,
                                                  ckpt=None,
                                                  reason=None,
                                                  inner_optimizer_state_loaded=None,
                                                  source_has_outer_state=None):
    """Create fresh non-avg DiLoCo outer state from already-loaded tensors.

    The helper may switch ScheduleFree modes to derive the requested basis, but
    it restores the live model parameters byte-for-byte before returning.
    """
    if args.diloco_outer_optimizer == 'avg':
        return None, None

    params = list(core_model.parameters())
    pre_bootstrap = _snapshot_model_params(core_model)
    pre_schedulefree_train_modes = None
    if getattr(args, 'optimizer', None) == 'schedulefree':
        pre_schedulefree_train_modes = [
            group.get('train_mode') for group in optimizer.param_groups
        ]
    source_basis = 'loaded'
    try:
        if args.diloco_outer_optimizer == 'momentum':
            if args.optimizer == 'schedulefree' and inner_optimizer_state_loaded:
                optimizer.eval()
                source_basis = 'x'
                anchor = [p.data.detach().clone() for p in params]
            else:
                anchor = [t.clone() for t in pre_bootstrap]
            state = {
                'mode': 'momentum',
                'anchor': anchor,
                'moment': [torch.zeros_like(t) for t in anchor],
            }
        elif args.diloco_outer_optimizer == 'sfsgd':
            if args.optimizer != 'schedulefree':
                raise ValueError("--diloco_outer_optimizer sfsgd requires --optimizer schedulefree")
            export_basis = getattr(args, 'diloco_export_basis', 'x')
            if inner_optimizer_state_loaded:
                if export_basis == 'y':
                    optimizer.train()
                    source_basis = 'y'
                else:
                    optimizer.eval()
                    source_basis = 'x'
                outer_basis = [p.data.detach().clone() for p in params]
            else:
                outer_basis = [t.clone() for t in pre_bootstrap]
            outer_lr = float(args.diloco_outer_lr)
            state = {
                'mode': 'sfsgd',
                'x': [t.clone() for t in outer_basis],
                'z': [t.clone() for t in outer_basis],
                'y': [t.clone() for t in outer_basis],
                'k': 0,
                'weight_sum': 0.0,
                'lr_max': outer_lr,
            }
        else:
            raise ValueError(f"unknown DiLoCo outer optimizer: {args.diloco_outer_optimizer}")
    finally:
        _restore_model_params(core_model, pre_bootstrap)
        if pre_schedulefree_train_modes is not None:
            for group, train_mode in zip(optimizer.param_groups, pre_schedulefree_train_modes):
                if train_mode is not None:
                    group['train_mode'] = train_mode

    restored_equal = _model_params_equal_snapshot(core_model, pre_bootstrap)
    if not restored_equal:
        raise RuntimeError("DiLoCo outer-state bootstrap changed live model parameters")
    metadata = _diloco_outer_bootstrap_metadata(
        args,
        ckpt=ckpt,
        reason=reason,
        inner_optimizer_state_loaded=inner_optimizer_state_loaded,
        source_basis=source_basis,
        source_has_outer_state=source_has_outer_state,
        model_tensors_equal_after_restore=restored_equal,
    )
    state['bootstrap_metadata'] = metadata
    return state, metadata


def resolve_diloco_outer_state_for_resume(core_model, optimizer, args,
                                          loaded_state=None,
                                          ckpt=None,
                                          inner_optimizer_state_loaded=False):
    """Fail-closed DiLoCo resume guard plus explicit bootstrap opt-in."""
    if args.diloco_outer_optimizer == 'avg':
        return initialize_diloco_outer_state(
            core_model, optimizer, args, loaded_state=loaded_state), None

    if not getattr(args, 'resume', None):
        return initialize_diloco_outer_state(core_model, optimizer, args), None

    bootstrap_guard = getattr(args, 'diloco_bootstrap_outer_state', 'none')
    reason = None
    source_has_outer_state = loaded_state is not None
    if loaded_state is None:
        reason = 'missing_diloco_outer_state'
    else:
        mode = loaded_state.get('mode') if isinstance(loaded_state, dict) else None
        if mode != args.diloco_outer_optimizer:
            reason = f"incompatible_diloco_outer_state_mode:{mode}->{args.diloco_outer_optimizer}"

    if reason is None:
        if bootstrap_guard == 'from-loaded-model':
            raise ValueError(
                "checkpoint already contains compatible diloco_outer_state; "
                "--diloco_bootstrap_outer_state from-loaded-model is only allowed "
                "when outer state is missing or incompatible")
        return initialize_diloco_outer_state(
            core_model, optimizer, args, loaded_state=loaded_state), None

    if bootstrap_guard != 'from-loaded-model':
        raise ValueError(
            f"checkpoint {getattr(args, 'resume', None)} lacks compatible "
            f"diloco_outer_state for --diloco_outer_optimizer "
            f"{args.diloco_outer_optimizer} ({reason}); rerun with "
            "--diloco_bootstrap_outer_state from-loaded-model only if you intend "
            "to start fresh outer state from the loaded model weights")

    return bootstrap_diloco_outer_state_from_loaded_model(
        core_model,
        optimizer,
        args,
        ckpt=ckpt,
        reason=reason,
        inner_optimizer_state_loaded=inner_optimizer_state_loaded,
        source_has_outer_state=source_has_outer_state,
    )


def initialize_diloco_outer_state(core_model, optimizer, args, loaded_state=None):
    """Create or restore the explicit DiLoCo outer optimizer state."""
    if args.diloco_outer_optimizer == 'avg':
        return None

    params = list(core_model.parameters())
    if loaded_state is not None:
        mode = loaded_state.get('mode')
        if mode != args.diloco_outer_optimizer:
            raise ValueError(
                f"checkpoint outer optimizer mode {mode!r} does not match "
                f"--diloco_outer_optimizer {args.diloco_outer_optimizer!r}")
        restored = {'mode': mode}
        for key, value in loaded_state.items():
            if key == 'mode':
                continue
            if isinstance(value, list) and value and torch.is_tensor(value[0]):
                restored[key] = _clone_param_list_like(params, value)
            else:
                restored[key] = value
        return restored

    if args.diloco_outer_optimizer == 'momentum':
        if args.optimizer == 'schedulefree':
            optimizer.eval()
        state = {
            'mode': 'momentum',
            'anchor': [p.data.detach().clone() for p in params],
            'moment': [torch.zeros_like(p.data) for p in params],
        }
        if args.optimizer == 'schedulefree':
            optimizer.train()
        return state

    if args.diloco_outer_optimizer == 'sfsgd':
        if args.optimizer != 'schedulefree':
            raise ValueError("--diloco_outer_optimizer sfsgd requires --optimizer schedulefree")
        # The outer ScheduleFree system is separate from the inner one. It starts
        # at the current inner train point; subsequent merges translate the inner
        # x/z/y geometry to the new outer_y.
        optimizer.train()
        outer_y = [p.data.detach().clone() for p in params]
        outer_lr = float(args.diloco_outer_lr)
        return {
            'mode': 'sfsgd',
            'x': [t.clone() for t in outer_y],
            'z': [t.clone() for t in outer_y],
            'y': outer_y,
            'k': 0,
            'weight_sum': 0.0,
            'lr_max': outer_lr,
        }

    raise ValueError(f"unknown DiLoCo outer optimizer: {args.diloco_outer_optimizer}")


def _diloco_merge_debug_rank_enabled(args):
    if int(getattr(args, 'diloco_merge_debug', 0) or 0) != 1:
        return False
    rank = int(getattr(args, '_rank', os.environ.get('RANK', '0')) or 0)
    ranks = str(getattr(args, 'diloco_merge_debug_ranks', '0') or '0').strip()
    if ranks == '*':
        return True
    enabled = set()
    for item in ranks.split(','):
        item = item.strip()
        if not item:
            continue
        try:
            enabled.add(int(item))
        except ValueError:
            continue
    return rank in enabled


def _diloco_merge_log(args, step, merge_index, label, bucket_index, numel, dtype,
                      phase, elapsed_s=None):
    if not _diloco_merge_debug_rank_enabled(args):
        return
    rank = int(getattr(args, '_rank', os.environ.get('RANK', '0')) or 0)
    elapsed = "" if elapsed_s is None else f" elapsed_s={elapsed_s:.6f}"
    print(
        f"[DiLoCoMergeDebug] rank={rank} step={step} merge={merge_index} "
        f"label={label} bucket={bucket_index} numel={numel} dtype={dtype} "
        f"phase={phase}{elapsed}",
        flush=True,
    )


def _diloco_sync_accelerator():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _build_diloco_hierarchical_merge_groups(world_size, rank, group_size,
                                            create_barrier_every=8):
    """Create exact two-level DiLoCo merge groups.

    The first level is consecutive rank groups, which matches Frontier's
    8-tasks-per-node srun layout used by the scaleout scripts. Groups may be
    unequal: the merge reduces SUMs, never averages local averages, so the final
    division by world_size preserves the current all-rank mean exactly.
    """
    if group_size <= 0:
        raise ValueError("--diloco_merge_group_size must be >0 for hierarchical merge")
    if not dist.is_initialized():
        raise RuntimeError("hierarchical DiLoCo merge requires torch.distributed")
    groups = []
    local_group = None
    local_group_ranks = None
    local_root = None
    n_groups = (world_size + group_size - 1) // group_size
    root_ranks = [
        group_idx * group_size
        for group_idx in range(n_groups)
    ]
    root_group = None
    create_barrier_every = int(create_barrier_every or 0)
    if len(root_ranks) > 1:
        # Create the overlapping second-level communicator first, before local
        # subgroups. Frontier RCCL/NCCL is sensitive to subgroup creation order
        # when later collectives involve non-contiguous root ranks.
        root_group = dist.new_group(ranks=root_ranks)
        if create_barrier_every > 0:
            dist.barrier()

    for group_idx in range(n_groups):
        start = group_idx * group_size
        group_ranks = list(range(start, min(start + group_size, world_size)))
        root = group_ranks[0]
        g = dist.new_group(ranks=group_ranks)
        groups.append((group_ranks, g))
        if rank in group_ranks:
            local_group = g
            local_group_ranks = group_ranks
            local_root = root
        if create_barrier_every > 0 and (
            (group_idx + 1) % create_barrier_every == 0 or
            (group_idx + 1) == n_groups
        ):
            dist.barrier()

    if local_group is None or local_group_ranks is None or local_root is None:
        raise RuntimeError(f"rank {rank} was not assigned to a DiLoCo merge group")

    return {
        'topology': 'hierarchical',
        'group_size': int(group_size),
        'groups': groups,
        'root_ranks': root_ranks,
        'root_group': root_group,
        'local_group': local_group,
        'local_group_ranks': local_group_ranks,
        'local_root': int(local_root),
        'local_count': int(len(local_group_ranks)),
        'is_root': bool(rank == local_root),
        'n_groups': int(len(root_ranks)),
    }


def _diloco_hierarchical_sum_average_flat(flat, world_size, args):
    meta = getattr(args, '_diloco_merge_groups', None)
    if not meta or meta.get('topology') != 'hierarchical':
        raise RuntimeError("missing hierarchical DiLoCo merge group metadata")
    local_group = meta['local_group']
    local_root = int(meta['local_root'])
    is_root = bool(meta['is_root'])
    root_group = meta.get('root_group')

    dist.reduce(flat, dst=local_root, op=dist.ReduceOp.SUM, group=local_group)
    # The next stage uses a different process group, and ProcessGroupNCCL/RCCL
    # CUDA collectives may return after enqueue. Force the local reduce to finish
    # before the root all-reduce consumes the same buffer on another communicator.
    _diloco_sync_accelerator()
    if is_root:
        if root_group is not None:
            dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=root_group)
            _diloco_sync_accelerator()
        flat.div_(world_size)
    dist.broadcast(flat, src=local_root, group=local_group)
    _diloco_sync_accelerator()


def _warm_diloco_hierarchical_merge_groups(args, device):
    meta = getattr(args, '_diloco_merge_groups', None)
    if not meta or meta.get('topology') != 'hierarchical':
        raise RuntimeError("missing hierarchical DiLoCo merge group metadata")
    _w = torch.zeros(1, device=device)
    dist.all_reduce(_w, group=meta['local_group'])
    if meta['is_root'] and meta.get('root_group') is not None:
        dist.all_reduce(_w, group=meta['root_group'])
    dist.barrier()


def _maybe_inject_diloco_collective_rank_exit(*, label, bucket_index,
                                               merge_index):
    """Test-only, fail-closed rank exit immediately before a DiLoCo collective.

    The surrounding ``srun`` owns failure containment.  Exiting here strands the
    other ranks inside the collective, exercising the real RCCL failure path;
    the batch-shell supervisor must terminate that step and start a fresh one.
    No handler or communicator recovery is attempted in this process.
    """
    requested_rank = os.environ.get('EMENDER_DILOCO_EXIT_RANK')
    if requested_rank is None:
        return
    try:
        rank = dist.get_rank()
        target_rank = int(requested_rank)
        target_merge = int(os.environ['EMENDER_DILOCO_EXIT_MERGE'])
        target_bucket = int(os.environ.get('EMENDER_DILOCO_EXIT_BUCKET', '0'))
        exit_code = int(os.environ.get('EMENDER_DILOCO_EXIT_CODE', '86'))
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError('invalid EMENDER_DILOCO_EXIT_* fault injection') from exc
    if not 1 <= exit_code <= 255:
        raise RuntimeError('EMENDER_DILOCO_EXIT_CODE must be in [1,255]')
    if (rank != target_rank or merge_index != target_merge
            or bucket_index != target_bucket
            or label != os.environ.get('EMENDER_DILOCO_EXIT_LABEL', 'model')):
        return
    print('DILOCO_FAULT_INJECTION ' + json.dumps({
        'rank': rank, 'merge_index': merge_index, 'bucket_index': bucket_index,
        'label': label, 'exit_code': exit_code,
    }, sort_keys=True), flush=True)
    # Give peers a bounded head start into the same collective, making this a
    # communicator/process-failure experiment rather than a pre-collective stop.
    time.sleep(float(os.environ.get('EMENDER_DILOCO_EXIT_DELAY_SECONDS', '0.25')))
    os._exit(exit_code)


def _diloco_allreduce_average_flat(flat, world_size, args, label, step=None,
                                   merge_index=None, bucket_index_base=0):
    topology = str(getattr(args, 'diloco_merge_topology', 'global') or 'global').lower()
    bucket_numel = int(getattr(args, 'diloco_merge_bucket_numel', 0) or 0)
    if bucket_numel <= 0 or bucket_numel >= flat.numel():
        _diloco_merge_log(args, step, merge_index, label, bucket_index_base,
                          int(flat.numel()), flat.dtype, 'enter')
        t0 = time.time()
        if topology == 'hierarchical':
            _diloco_hierarchical_sum_average_flat(flat, world_size, args)
        elif topology == 'global':
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            flat.div_(world_size)
        else:
            raise ValueError(f"unknown DiLoCo merge topology: {topology}")
        _diloco_merge_log(args, step, merge_index, label, bucket_index_base,
                          int(flat.numel()), flat.dtype, 'exit', time.time() - t0)
        return

    bucket_index = bucket_index_base
    for start in range(0, flat.numel(), bucket_numel):
        chunk = flat.narrow(0, start, min(bucket_numel, flat.numel() - start))
        _diloco_merge_log(args, step, merge_index, label, bucket_index,
                          int(chunk.numel()), chunk.dtype, 'enter')
        t0 = time.time()
        _maybe_inject_diloco_collective_rank_exit(
            label=label, bucket_index=bucket_index, merge_index=merge_index)
        if topology == 'hierarchical':
            _diloco_hierarchical_sum_average_flat(chunk, world_size, args)
        elif topology == 'global':
            dist.all_reduce(chunk, op=dist.ReduceOp.SUM)
            chunk.div_(world_size)
        else:
            raise ValueError(f"unknown DiLoCo merge topology: {topology}")
        _diloco_merge_log(args, step, merge_index, label, bucket_index,
                          int(chunk.numel()), chunk.dtype, 'exit',
                          time.time() - t0)
        bucket_index += 1


@torch.no_grad()
def _diloco_allreduce_average_tensor_list_(
        tensors, world_size, args, label, *, staging_device, step=None,
        merge_index=None):
    """Average a tensor list through one bounded reusable staging buffer.

    This path is required when tensors live in CPU-offloaded optimizer state:
    NCCL cannot reduce them directly, and flattening the full state would
    recreate an optimizer-sized HBM allocation.  It is also suitable for GPU
    tensors when a bounded merge footprint is required.
    """
    tensors = list(tensors)
    if not tensors:
        return
    dtype = tensors[0].dtype
    if any(tensor.dtype != dtype for tensor in tensors):
        raise ValueError(f"{label} DiLoCo tensor list must have one dtype")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError(f"{label} DiLoCo tensor list must be contiguous")
    total_numel = sum(int(tensor.numel()) for tensor in tensors)
    configured = int(getattr(args, 'diloco_merge_bucket_numel', 0) or 0)
    # CPU-offloaded z must never use a full-model staging allocation.  Match the
    # production 64M-element merge bound when no explicit bound was supplied.
    bucket_numel = configured if configured > 0 else 67_108_864
    bucket_numel = max(1, min(bucket_numel, total_numel))
    staging = torch.empty(bucket_numel, dtype=dtype, device=staging_device)

    tensor_index = 0
    tensor_offset = 0
    bucket_index = 0
    while tensor_index < len(tensors):
        filled = 0
        destinations = []
        while filled < bucket_numel and tensor_index < len(tensors):
            tensor = tensors[tensor_index]
            available = int(tensor.numel()) - tensor_offset
            count = min(bucket_numel - filled, available)
            source = tensor.view(-1).narrow(0, tensor_offset, count)
            staging.narrow(0, filled, count).copy_(source, non_blocking=False)
            destinations.append((tensor, tensor_offset, filled, count))
            filled += count
            tensor_offset += count
            if tensor_offset == tensor.numel():
                tensor_index += 1
                tensor_offset = 0

        active = staging.narrow(0, 0, filled)
        _diloco_allreduce_average_flat(
            active, world_size, args, label, step=step,
            merge_index=merge_index, bucket_index_base=bucket_index)
        for tensor, destination_offset, source_offset, count in destinations:
            tensor.view(-1).narrow(0, destination_offset, count).copy_(
                staging.narrow(0, source_offset, count), non_blocking=False)
        bucket_index += 1


@torch.no_grad()
def diloco_merge(core_model, optimizer, args, world_size, outer_state,
                 step=None, merge_index=None):
    """DiLoCo outer step: average model weights across ranks (every K local steps).

    task implement-diloco-periodic. This is the inter-worker synchronization that
    replaces vanilla DDP's per-step gradient all-reduce. Each rank trains K local
    optimizer steps INDEPENDENTLY, then here we average the model weights once.

    ScheduleFree interaction (docs/SCHEDULEFREE_DILOCO_FRONTIER_DESIGN.md): the
    optimizer evolves both the running-average eval weights x (held implicitly in
    p.data while in eval mode) and the base iterate z (held in per-param state).
    DiLoCo therefore merges both tensor states: x is averaged across replicas
    using the same model-weight semantics, z is averaged across replicas, and the
    live train weights y are rebuilt from the merged x/z pair. ScheduleFree's
    scalar clocks (weight_sum, k, lr_max) are local optimizer bookkeeping and are
    preserved as-is. That avoids fragile 1-element process-group collectives at
    large scale; under the normal equal-step DiLoCo schedule they are already
    identical, and under uneven ranks equal-weight tensor averaging is acceptable.
    Adam second moments (exp_avg_sq) stay per-rank (independent preconditioning
    -> independent exploration, the point of DiLoCo).

    Outer optimizer (general DiLoCo):
        delta       = mean_i(W_{r,i}) - W_r
        outer_mom   = outer_beta * outer_mom + delta
        W_{r+1}     = W_r + outer_lr * outer_mom
    Defaults outer_lr=1.0, outer_beta=0.0 reduce to plain periodic weight
    averaging (local-SGD): W_{r+1} = mean_i(W_{r,i}). In that case we skip the
    anchor/momentum buffers entirely (no extra memory).

    Returns the all-reduce wall-clock seconds (for sync-cost accounting).
    """
    if world_size <= 1:
        return 0.0

    sf = (args.optimizer == 'schedulefree')
    outer_optimizer = getattr(args, 'diloco_outer_optimizer', None)
    if outer_optimizer is None:
        outer_optimizer = 'momentum' if outer_state is not None else 'avg'
    elif outer_optimizer == 'avg' and outer_state is not None and 'anchor' in outer_state:
        # Backward-compatible direct callers/tests may pass the legacy explicit
        # anchor/moment state without setting the new routing flag.
        outer_optimizer = 'momentum'
    export_basis = getattr(args, 'diloco_export_basis', 'x')
    if outer_optimizer == 'sfsgd' and not sf:
        raise ValueError("--diloco_outer_optimizer sfsgd requires --optimizer schedulefree")

    export_y_by_group = {}
    if sf and outer_optimizer == 'sfsgd' and export_basis == 'y':
        for group in optimizer.param_groups:
            group_params = list(group['params'])
            if group_params:
                export_y_by_group[id(group)] = torch._utils._flatten_dense_tensors(
                    [p.data for p in group_params])

    # 1. y-mode swap: switch schedulefree to eval so p.data holds the averaged x_i.
    if sf:
        optimizer.eval()

    params = list(core_model.parameters())
    t0 = time.time()
    # 2. all-reduce the model weights to the cross-rank consensus: p.data <-
    #    mean_i(W_{r,i}) on every rank. Coalesce into one flat bucket so the
    #    1.29B-param sync is a SINGLE collective launch (the whole point: ONE
    #    all-reduce per K steps, not K). SUM+divide rather than ReduceOp.AVG so
    #    the path is backend-agnostic (gloo, used by the CPU unit test, has no AVG).
    if sf:
        # Average ScheduleFree tensor state only. Scalar clocks are intentionally
        # left local: reducing them would add tiny NCCL collectives that are not
        # worth the fragility for DiLoCo's equal-step scaleout path.
        for group in optimizer.param_groups:
            group_params = list(group['params'])
            if not group_params:
                continue

            if getattr(optimizer, 'state_storage', None) == 'pinned-cpu':
                _diloco_allreduce_average_tensor_list_(
                    [p.data for p in group_params], world_size, args, 'sf_x',
                    staging_device=group_params[0].device,
                    step=step, merge_index=merge_index)
            else:
                flat_x = torch._utils._flatten_dense_tensors(
                    [p.data for p in group_params])
                _diloco_allreduce_average_flat(
                    flat_x, world_size, args, 'sf_x',
                    step=step, merge_index=merge_index)
                for p, merged in zip(
                        group_params,
                        torch._utils._unflatten_dense_tensors(
                            flat_x, [p.data for p in group_params])):
                    p.data.copy_(merged)

            z_params = [p for p in group_params if 'z' in optimizer.state.get(p, {})]
            if z_params:
                z_tensors = [optimizer.state[p]['z'] for p in z_params]
                if any(z.device != z_params[index].device
                       for index, z in enumerate(z_tensors)):
                    _diloco_allreduce_average_tensor_list_(
                        z_tensors, world_size, args, 'sf_z',
                        staging_device=z_params[0].device,
                        step=step, merge_index=merge_index)
                else:
                    flat_z = torch._utils._flatten_dense_tensors(z_tensors)
                    _diloco_allreduce_average_flat(
                        flat_z, world_size, args, 'sf_z',
                        step=step, merge_index=merge_index)
                    for p, merged_z in zip(
                            z_params,
                            torch._utils._unflatten_dense_tensors(flat_z, z_tensors)):
                        optimizer.state[p]['z'].copy_(merged_z)

            if outer_optimizer == 'sfsgd' and export_basis == 'y':
                flat_y = export_y_by_group[id(group)]
                _diloco_allreduce_average_flat(flat_y, world_size, args, 'sf_y',
                                               step=step, merge_index=merge_index)
                export_y_by_group[id(group)] = flat_y
    else:
        flat = torch._utils._flatten_dense_tensors([p.data for p in params])
        _diloco_allreduce_average_flat(flat, world_size, args, 'params',
                                       step=step, merge_index=merge_index)
        for p, merged in zip(params, torch._utils._unflatten_dense_tensors(flat, [p.data for p in params])):
            p.data.copy_(merged)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    merge_topology = str(getattr(args, 'diloco_merge_topology', 'global') or 'global').lower()
    completion_barrier = int(getattr(args, 'diloco_merge_completion_barrier', 1) or 0)
    if merge_topology == 'hierarchical' and completion_barrier != 0:
        # Hierarchical merging ends with independent local-group broadcasts. A
        # fast local group can otherwise leave the merge while another group is
        # still in its final broadcast, and later K-step collectives may overlap.
        # One all-rank barrier per merge restores a single completion point.
        dist.barrier()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    sync_s = time.time() - t0

    # 3. Outer optimizer (momentum). Skipped for the local-SGD default.
    #
    # SF GEOMETRY INVARIANT (sf-diloco-p1): the outer step displaces the merged
    # ScheduleFree EVAL weight x (held in p.data while in eval mode) by a server
    # displacement s = x_new - x_bar. ScheduleFree's live train point is
    # y = beta1*x + (1-beta1)*z, so x and z together define the inner geometry.
    # Translation invariance demands that ANY displacement applied to x is also
    # applied to z (hence to y): x<-x+s, z<-z+s => y<-y+s. The previous code
    # shifted only x, leaving z fixed; train() then rebuilt y = ybar + beta1*s
    # (only a beta1 fraction of the server update reached the next train point)
    # and stretched the x-z gap by -s. That corrupted inner SF geometry on EVERY
    # nontrivial outer update (any outer_lr!=1 or any outer momentum), not just
    # beta>0 -- which is why nonzero outer beta looked catastrophic. We now shift
    # z by the same s. For non-ScheduleFree optimizers there is no z and the
    # x-only move is already correct.
    outer_lr = args.diloco_outer_lr
    outer_beta = args.diloco_outer_beta
    debug_assert = os.environ.get('NDM_DILOCO_DEBUG_ASSERT', '0') == '1'
    if outer_optimizer == 'momentum' and outer_state is not None:
        anchor = outer_state['anchor']   # W_r (x-basis)
        moment = outer_state['moment']   # outer momentum buffer
        # Per-merge geometry instrumentation (task sf-diloco-p2): accumulate the
        # whole-model squared norms needed for three diagnostic ratios. All ranks
        # hold identical x_bar / z / anchor / moment after the all-reduce (delta is
        # rank-invariant), so every rank computes the same numbers; only rank 0
        # prints. float64 accumulators on the param device avoid per-param syncs.
        _idev = params[0].data.device
        _sq_land_num = torch.zeros((), device=_idev, dtype=torch.float64)  # ||x+  - x_r||^2
        _sq_land_den = torch.zeros((), device=_idev, dtype=torch.float64)  # ||xbar - x_r||^2
        _sq_anchor   = torch.zeros((), device=_idev, dtype=torch.float64)  # ||x_r||^2
        _sq_gap      = torch.zeros((), device=_idev, dtype=torch.float64)  # ||z - x+||^2
        _sq_xnew     = torch.zeros((), device=_idev, dtype=torch.float64)  # ||x+||^2
        for p, w_r, m in zip(params, anchor, moment):
            x_bar = p.data                             # merged eval x = mean_i(x_i)
            delta = x_bar - w_r                        # mean_i(W_{r,i}) - W_r
            m.mul_(outer_beta).add_(delta)             # outer_mom = beta*mom + delta
            x_new = torch.add(w_r, m, alpha=outer_lr)  # W_{r+1} = W_r + lr*mom
            shift = x_new - x_bar                      # server displacement on x
            z = optimizer.state.get(p, {}).get('z') if sf else None
            gap_before = (z - x_bar).detach().clone() if (debug_assert and z is not None) else None
            # Accumulate squared norms BEFORE overwriting p.data / anchor.
            _sq_land_num += (x_new - w_r).double().pow(2).sum()
            _sq_land_den += delta.double().pow(2).sum()
            _sq_anchor   += w_r.double().pow(2).sum()
            _sq_xnew     += x_new.double().pow(2).sum()
            p.data.copy_(x_new)                        # x  <- x + s
            if z is not None:
                z.add_(shift)                          # z  <- z + s  (preserve SF geometry)
                _sq_gap += (z - x_new).double().pow(2).sum()
            w_r.copy_(x_new)                           # advance anchor to W_{r+1} (x-basis)
            if gap_before is not None:
                # Post-condition: the x-z gap (and therefore y's offset from x)
                # is preserved by the outer rebase. (z+s)-(x+s) == z-x.
                gap_drift = (z - p.data - gap_before).abs().max().item()
                # In bf16 production runs, z.add_(shift) and p.data.copy_(x_new)
                # round through bf16 storage. The invariant is exact in real
                # arithmetic, but the runtime assert must allow one low-precision
                # storage quantum; fp32 CPU regression tests still use the tighter
                # tolerance.
                if p.data.dtype in (torch.bfloat16, torch.float16):
                    gap_tol = 1.5 * torch.finfo(p.data.dtype).eps
                else:
                    gap_tol = 1e-4
                assert gap_drift < gap_tol, (
                    f"DiLoCo outer update broke SF x-z geometry: gap drift {gap_drift} "
                    f"(tol {gap_tol}; z displacement must equal x displacement s)")
        # Emit the three ratios (single host sync for all six accumulators). These
        # are the load-bearing diagnostics for P2's "does matched-gain momentum
        # behave post-fix?" question:
        #   land_frac = |x+ - x_r| / |xbar - x_r|  fraction of the raw K-step drift
        #               that the outer step actually lands (lr*mom vs delta; >1 means
        #               momentum is amplifying the server update -> watch for spirals)
        #   disp_mag  = |xbar - x_r| / |x_r|       relative magnitude of the merge displacement
        #   gap_health= |z - x| / |x|              SF inner gap; must stay BOUNDED post-fix
        eps = 1e-30
        land_num = float(_sq_land_num.sqrt().item())
        land_den = float(_sq_land_den.sqrt().item())
        anchor_n = float(_sq_anchor.sqrt().item())
        gap_n    = float(_sq_gap.sqrt().item())
        xnew_n   = float(_sq_xnew.sqrt().item())
        land_frac  = land_num / (land_den + eps)
        disp_mag   = land_den / (anchor_n + eps)
        gap_health = gap_n / (xnew_n + eps)
        # Stash for programmatic access (regression tests + callers); the print is
        # the human/log-facing surface.
        outer_state['last_metrics'] = {
            'land_frac': land_frac,
            'disp_mag': disp_mag,
            'gap_health': gap_health,
        }
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            print(f"  [DiLoCo-geom] land_frac(|x+-xr|/|xbar-xr|)={land_frac:.4f} "
                  f"disp_mag(|xbar-xr|/|xr|)={disp_mag:.3e} "
                  f"gap_health(|z-x|/|x|)={gap_health:.3e} "
                  f"(outer_lr={outer_lr} outer_beta={outer_beta})", flush=True)
    elif outer_optimizer == 'sfsgd':
        if outer_state is None or outer_state.get('mode') != 'sfsgd':
            raise ValueError("sfsgd DiLoCo merge requires an sfsgd outer_state")
        outer_state['lr_max'] = max(float(outer_state['lr_max']), float(outer_lr))
        weight = float(outer_state['lr_max']) ** 2
        outer_state['weight_sum'] = float(outer_state['weight_sum']) + weight
        outer_state['k'] = int(outer_state['k']) + 1
        c = weight / outer_state['weight_sum'] if outer_state['weight_sum'] > 0.0 else 0.0
        beta_out = float(args.diloco_outer_beta)
        outer_x = outer_state['x']
        outer_z = outer_state['z']
        outer_y = outer_state['y']
        beta1_by_param = {
            p: float(group['betas'][0])
            for group in optimizer.param_groups
            for p in group['params']
        }

        param_to_export = {}
        if export_basis == 'y':
            for group in optimizer.param_groups:
                group_params = list(group['params'])
                if not group_params:
                    continue
                for p, merged_y in zip(
                        group_params,
                        torch._utils._unflatten_dense_tensors(
                            export_y_by_group[id(group)], [p.data for p in group_params])):
                    param_to_export[p] = merged_y

        for p, ox, oz, oy in zip(params, outer_x, outer_z, outer_y):
            z = optimizer.state.get(p, {}).get('z')
            beta1 = beta1_by_param.get(p)
            current_inner_train_point = (
                p.data if z is None or beta1 is None
                else torch.add(p.data, z, alpha=(1.0 - beta1) / beta1).mul(beta1)
            )
            endpoint = param_to_export[p] if export_basis == 'y' else p.data
            delta = endpoint - oy
            oz.add_(delta, alpha=outer_lr)
            ox.lerp_(oz, c)
            oy.copy_(oz).lerp_(ox, beta_out)
            shift = oy - current_inner_train_point
            p.data.copy_(current_inner_train_point).add_(shift)
            if z is not None:
                z.add_(shift)
        for group in optimizer.param_groups:
            group['train_mode'] = True

    # 4. Rebuild ScheduleFree train weights y from the merged full state. With
    #    AdamWScheduleFree, train() maps eval-mode x and state z to
    #    y = beta1*x + (1-beta1)*z. The averaging denominator and clocks are left
    #    intact so x remains a full-run cumulative average across future steps.
    #    Because step 3 shifted x and z by the same s, train() yields the
    #    translation-invariant y+ = ybar + s (whole inner geometry preserved).
    if sf and outer_optimizer != 'sfsgd':
        optimizer.train()
    return sync_s


def train(args):
    """Main training loop."""
    install_shutdown_signal_handlers()

    # AUTO-resolve --use_triton (default None). E97 (split-edit) and raw-write have
    # their fused fwd/bwd ONLY in the Triton kernel — the CUDA register-owned path
    # rejects both, so without Triton they silently fall back to the eager T-scan
    # (~40-260x slower). Default those families to the fused Triton path under bf16
    # (parity-verified, paper/review/E97_FUSED_LM_KERNEL_NOTE.md). Everything else
    # keeps the historical default (CUDA register-owned, use_triton=0).
    normalize_training_args(args, announce=True)

    # --- Distributed setup (opt-in, backward compatible) -----------------------
    # Activates ONLY under torchrun (WORLD_SIZE>1). Single-GPU/no-torchrun runs are
    # byte-identical to before: dist_enabled=False, rank=0, world_size=1, is_main=True.
    #
    # Two distributed modes share the same process-group / data-sharding / rank-0
    # gating plumbing but differ in HOW gradients/weights are synchronized:
    #   * DDP (default):  per-step gradient all-reduce. use_ddp=True. Exact SGD
    #                     equivalence but the 1.29B bf16 all-reduce dominates on a
    #                     no-NVLink PCIe box (preflight-100b: 52% scaling eff).
    #   * DiLoCo (--diloco): each rank trains INDEPENDENTLY (no per-step comm) and
    #                     model weights are averaged every --diloco_k steps. Recovers
    #                     the ~62k tok/s independent ceiling. use_ddp=False.
    slurm_env_status = resolve_distributed_env_from_slurm()
    dist_enabled = int(os.environ.get('WORLD_SIZE', '1')) > 1
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    is_main = (rank == 0)
    use_ddp = dist_enabled and not args.diloco
    use_diloco = dist_enabled and args.diloco
    # _ddp_enabled retained for back-compat with downstream references; it now means
    # "wrapped in torch DDP" (per-step sync), NOT merely "distributed".
    args._ddp_enabled = use_ddp
    args._dist_enabled = dist_enabled
    args._use_diloco = use_diloco
    args._rank = rank
    args._world_size = world_size
    args._is_main = is_main
    args._model_variant = model_variant_label(args)
    # Validate the complete launch identity before model, optimizer, data, or
    # checkpoint state can be mutated.
    launch_sampler_identity = counter_sampler_identity_from_args(
        args, world_size=world_size)

    # Setup
    torch.manual_seed(args.seed)
    if dist_enabled:
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        if not dist.is_initialized():
            init_kwargs = {}
            if getattr(args, 'distributed_init_timeout_seconds', None) is not None:
                init_timeout_s = max(1.0, float(args.distributed_init_timeout_seconds))
                init_kwargs['timeout'] = datetime.timedelta(seconds=init_timeout_s)
                if is_main:
                    print(f"[distributed-init] timeout={init_timeout_s:.1f}s", flush=True)
            try:
                dist.init_process_group(backend='nccl', device_id=device, **init_kwargs)
            except TypeError:
                dist.init_process_group(backend='nccl', **init_kwargs)
        _mode = 'DiLoCo' if use_diloco else 'DDP'
        if is_main:
            print(f"[{_mode}] world_size={world_size} backend=nccl; this is rank {rank} "
                  f"on {device}", flush=True)
            if slurm_env_status == 'derived-from-slurm':
                print("[distributed-env] derived RANK/WORLD_SIZE/LOCAL_RANK from "
                      "Slurm because WORLD_SIZE was not exported by the launcher",
                      flush=True)
            if use_diloco:
                print(f"[DiLoCo] periodic model-weight averaging: K={args.diloco_k} "
                      f"outer_lr={args.diloco_outer_lr} outer_beta={args.diloco_outer_beta} "
                      f"(no per-step gradient all-reduce)", flush=True)
        print(f"[{_mode}] rank {rank}/{world_size} bound to {device}", flush=True)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

    status_collectives_enabled = (
        dist_enabled
        and dist.is_initialized()
        and not bool(getattr(args, 'disable_scalar_status_collectives', False))
    )
    status_request_poll_s = max(0.0, float(getattr(args, 'status_request_poll_seconds', 0.0)))
    if is_main and dist_enabled and not status_collectives_enabled:
        print("[status] scalar default-process-group status collectives disabled; "
              f"filesystem stop-request polling={status_request_poll_s:.2f}s at "
              "walltime/budget check boundaries", flush=True)

    final_ckpt = FinalCheckpointController(args, device)
    if is_main:
        final_ckpt.reset_coordination_files()
    if dist_enabled and dist.is_initialized():
        # This early barrier is before any DDP/island process groups are created,
        # so it cannot race with subgroup gradient collectives. It only makes the
        # stale-file cleanup visible before ranks start polling for stop requests.
        dist.barrier()
    if is_main:
        if final_ckpt.walltime_active:
            remaining = final_ckpt.deadline - time.time()
            print(f"[final-checkpoint] armed: source={final_ckpt.deadline_source} "
                  f"remaining_s={remaining:.1f} margin_s={final_ckpt.margin_s:.1f} "
                  f"check_every={final_ckpt.check_every} model_variant={args._model_variant} "
                  f"head_rank={rank}/{world_size}", flush=True)
        else:
            print(f"[final-checkpoint] walltime deadline not detected; signal-triggered "
                  f"final checkpoint remains armed. margin_s={final_ckpt.margin_s:.1f} "
                  f"model_variant={args._model_variant} head_rank={rank}/{world_size}",
                  flush=True)

    # Per-rank FUSED guard (preflight-100b). For E97 split-edit/raw-write BF16,
    # ``use_triton=1`` alone is NOT sufficient: E88FLAHybrid selects its optimized
    # recurrence only when the output gate is SiLU and the unsupported ablations
    # below are absent. A sigmoid gate silently selects the eager T-scan even with
    # use_triton=1 (observed 40-260x slowdown). Fail closed on the complete runtime
    # predicate instead of printing a false no-eager claim.
    _e97_family = str(args.level) in ('E97', '97', 'E97-M2') or bool(getattr(args, 'e88_raw_write', 0))
    if _e97_family and getattr(args, 'bf16', False):
        _failed_fused_contract = e97_fused_runtime_contract_failures(args)
        if _failed_fused_contract:
            raise RuntimeError(
                f"[fused-guard] rank {rank}: E97/raw-write BF16 would leave the "
                f"optimized Triton path; failed={_failed_fused_contract} "
                f"(use_triton={getattr(args, 'use_triton', None)}, "
                f"use_gate={getattr(args, 'use_gate', None)}, "
                f"gate_activation={getattr(args, 'gate_activation', None)}, "
                f"use_write_gate={getattr(args, 'use_write_gate', None)}, "
                f"e88_value_residual={getattr(args, 'e88_value_residual', None)})")
        print(f"[fused-guard] rank {rank}/{world_size}: level={args.level} bf16 "
              f"gate=silu use_triton=1 -> optimized split-edit Triton kernel, "
              f"NO eager fallback", flush=True)
    if str(args.level).lower() in ('gdn2', 'gdn2-mlp'):
        from ndm.models.external_gdn2 import probe_gdn2_external_dependencies

        gdn2_probe = probe_gdn2_external_dependencies()
        assert gdn2_probe.get('ok'), (
            f"[fused-guard] rank {rank}: GDN-2 external fused path preflight failed: "
            f"{gdn2_probe.get('failure', gdn2_probe)}")
        gdn2_path = gdn2_probe.get('gdn2_path')
        backend = gdn2_probe.get('torch_backend')
        hip = gdn2_probe.get('torch_version_hip')
        chunk_ops = gdn2_probe.get('chunk_ops_module')
        print(f"[fused-guard] rank {rank}/{world_size}: level={args.level} "
              f"bf16={bool(args.bf16)} backend={backend} torch.version.hip={hip} "
              f"GDN2_PATH={gdn2_path} chunk_ops={chunk_ops} -> "
              "FLA chunked GDN-2 fused kernel import path, NO eager fallback", flush=True)

    output_dir = None
    heldout_curve = None
    heldout_curve_path = None

    # Resolve 'auto' r_h_mode based on model architecture.
    r_h_mode = resolve_training_r_h_mode(args, announce=True)

    # Resolve vocab size: 256 for byte-level (default) or tokenizer vocab size.
    vocab_size = resolve_training_vocab_size(args, announce=True)

    # Create model
    if args.level == 'mamba2':
        # Special handling for Mamba2 - use Mamba2LM directly
        from ndm.models.mamba2_baseline import Mamba2LM
        if args.dim is not None and args.depth is not None:
            model = Mamba2LM(
                vocab_size=vocab_size,
                dim=args.dim,
                depth=args.depth,
                d_state=args.mamba_d_state,
                expand=args.mamba_expand,
                headdim=64,
                loss_chunk_size=args.loss_chunk_size,
            )
        else:
            from ndm.models.mamba2_baseline import create_mamba2_model
            model = create_mamba2_model(target_params=args.params, vocab_size=vocab_size, expand=args.mamba_expand)
    elif args.level == 'mamba3':
        from ndm.models.mamba3_baseline import Mamba3LM
        if args.dim is None or args.depth is None:
            raise ValueError("--level mamba3 requires explicit --dim and --depth")
        mamba3_chunk_size = min(args.chunk_size, 64)
        if args.mamba3_mimo:
            mamba3_chunk_size = min(mamba3_chunk_size, max(16, 64 // max(1, args.mamba3_mimo_rank)))
        model = Mamba3LM(
            vocab_size=vocab_size,
            dim=args.dim,
            depth=args.depth,
            d_state=args.mamba_d_state,
            expand=args.mamba_expand,
            headdim=args.mamba3_headdim,
            is_mimo=bool(args.mamba3_mimo),
            mimo_rank=args.mamba3_mimo_rank,
            mamba_chunk_size=mamba3_chunk_size,
            loss_chunk_size=args.loss_chunk_size,
        )
    elif args.level == 'hybrid':
        if args.dim is None or args.depth is None:
            raise ValueError("--level hybrid requires explicit --dim and --depth")
        if not args.hybrid_pattern:
            raise ValueError("--level hybrid requires --hybrid_pattern")
        from ndm.models.hybrid_ladder import HybridLadderLM

        layer_pattern = [part.strip() for part in args.hybrid_pattern.split(',') if part.strip()]
        layer_kwargs = []
        for level in layer_pattern:
            kw = {}
            if args.hybrid_m2rnn_heads is not None and level in ('m2rnn', 'm2rnn-paper'):
                kw['n_heads'] = args.hybrid_m2rnn_heads
            if level in ('m2rnn', 'm2rnn-paper'):
                kw.update(
                    k_head_dim=args.m2rnn_k_head_dim,
                    v_head_dim=args.m2rnn_v_head_dim,
                    num_q_heads=args.m2rnn_q_heads,
                    num_k_heads=args.m2rnn_k_heads,
                    num_v_heads=args.m2rnn_v_heads,
                    num_f_heads=args.m2rnn_f_heads,
                    num_g_heads=args.m2rnn_g_heads,
                    num_weight_heads=args.m2rnn_weight_heads,
                    use_conv=bool(args.use_conv) or level == 'm2rnn-paper',
                    d_conv=args.d_conv,
                    output_norm=bool(args.m2rnn_output_norm) or level == 'm2rnn-paper',
                    normalize_qk=bool(args.m2rnn_normalize_qk),
                    use_residual=bool(args.m2rnn_use_residual),
                    state_weight_trainable=not bool(args.m2rnn_freeze_state_weight),
                    gradient_clipping=(
                        args.m2rnn_state_grad_clip
                        if args.m2rnn_state_grad_clip is not None
                        else (1.0 if level == 'm2rnn-paper' else None)
                    ),
                )
            layer_kwargs.append(kw)

        model = HybridLadderLM(
            vocab_size=vocab_size,
            dim=args.dim,
            depth=args.depth,
            layer_pattern=layer_pattern,
            layer_kwargs=layer_kwargs,
            n_state=args.n_state,
            n_heads=args.n_heads,
            expansion=args.expansion,
            use_gate=bool(args.use_gate),
            gate_activation=args.gate_activation,
            dropout=args.dropout,
        )
    elif args.level == 'm2rnn':
        # M2RNN baseline: nonlinear matrix-to-matrix RNN with matrix-valued state.
        from ndm.models.m2rnn_baseline import (
            M2RNNLM,
            XMA_M2RNN_AVAILABLE,
            create_m2rnn_model,
        )
        print(f"M2RNN XMA Triton backend: {XMA_M2RNN_AVAILABLE}")
        if args.require_m2rnn_xma and device.type == 'cuda' and not XMA_M2RNN_AVAILABLE:
            raise RuntimeError(
                "M2RNN XMA backend is required but unavailable. "
                "Set XMA_PATH=/home/erikg/xma or pass a valid XMA checkout."
            )
        if args.dim is not None and args.depth is not None:
            model = M2RNNLM(
                vocab_size=vocab_size,
                dim=args.dim,
                depth=args.depth,
                n_heads=args.n_heads,
                n_state=args.n_state,
                expansion=args.expansion,
                paper_shape=args.m2rnn_paper_shape,
                k_head_dim=args.m2rnn_k_head_dim,
                v_head_dim=args.m2rnn_v_head_dim,
                num_q_heads=args.m2rnn_q_heads,
                num_k_heads=args.m2rnn_k_heads,
                num_v_heads=args.m2rnn_v_heads,
                num_f_heads=args.m2rnn_f_heads,
                num_g_heads=args.m2rnn_g_heads,
                num_weight_heads=args.m2rnn_weight_heads,
                use_gate=bool(args.use_gate),
                use_residual=bool(args.m2rnn_use_residual),
                state_weight_trainable=not bool(args.m2rnn_freeze_state_weight),
                use_conv=bool(args.use_conv),
                d_conv=args.d_conv,
                output_norm=bool(args.m2rnn_output_norm),
                normalize_qk=bool(args.m2rnn_normalize_qk),
                dropout=args.dropout,
                gradient_clipping=args.m2rnn_state_grad_clip,
                gradient_checkpointing=args.gradient_checkpointing,
                loss_chunk_size=args.loss_chunk_size,
            )
        else:
            model = create_m2rnn_model(target_params=args.params, vocab_size=vocab_size)
    elif args.level == 'gru':
        # GRU baseline
        if args.dim is not None and args.depth is not None:
            model = GRULM(
                vocab_size=vocab_size,
                dim=args.dim,
                depth=args.depth,
                expansion_factor=args.expansion,
            )
        else:
            from ndm.models.gru_baseline import create_gru_model
            model = create_gru_model(target_params=args.params, vocab_size=vocab_size)
    elif args.level == 'lstm':
        # LSTM baseline
        if args.dim is not None and args.depth is not None:
            model = LSTMLM(
                vocab_size=vocab_size,
                dim=args.dim,
                depth=args.depth,
                expansion_factor=args.expansion,
            )
        else:
            from ndm.models.lstm_baseline import create_lstm_model
            model = create_lstm_model(target_params=args.params, vocab_size=vocab_size)
    elif args.level == 'mingru':
        # minGRU baseline (parallel)
        if args.dim is not None and args.depth is not None:
            model = MinGRULM(
                vocab_size=vocab_size,
                dim=args.dim,
                depth=args.depth,
                expansion_factor=args.expansion,
            )
        else:
            from ndm.models.min_rnn_baseline import create_mingru_model
            model = create_mingru_model(target_params=args.params, vocab_size=vocab_size)
    elif args.level == 'minlstm':
        # minLSTM baseline (parallel)
        if args.dim is not None and args.depth is not None:
            model = MinLSTMLM(
                vocab_size=vocab_size,
                dim=args.dim,
                depth=args.depth,
                expansion_factor=args.expansion,
            )
        else:
            from ndm.models.min_rnn_baseline import create_minlstm_model
            model = create_minlstm_model(target_params=args.params, vocab_size=vocab_size)
    elif args.level == 'cudagru':
        # CUDA GRU (avoids cuDNN bfloat16 regression)
        model = CudaGRULM(
            vocab_size=vocab_size,
            dim=args.dim,
            depth=args.depth,
            expansion_factor=args.expansion,
        )
    elif args.level == 'cudalstm':
        # CUDA LSTM (avoids cuDNN bfloat16 regression)
        model = CudaLSTMLM(
            vocab_size=vocab_size,
            dim=args.dim,
            depth=args.depth,
            expansion_factor=args.expansion,
        )
    elif args.level in ('E94', 'E94r'):
        # E94 — canonical: per-head 16x16 (or 32x32) W_h_time, fixed per-layer head
        # permutation, dim-wide residual stream, tied embed/lm_head.
        # 'E94r' kept as alias for in-flight runs; semantically identical to 'E94'.
        # --use_gate 1: silu output gating (E88-style depth nonlinearity).
        # --use_permutation 0: ablation, disable cross-head info flow.
        # --gradient_checkpointing: per-layer activation checkpointing.
        from ndm.models.e94 import E94Model
        model = E94Model(
            vocab_size=vocab_size,
            dim=args.dim,
            n_heads=args.n_heads,
            head_dim=args.n_state,
            depth=args.depth,
            dropout=args.dropout,
            tie_embedding=True,
            use_gate=bool(args.use_gate),
            use_permutation=bool(getattr(args, 'use_permutation', 1)),
            gradient_checkpointing=args.gradient_checkpointing,
        )
    elif args.level == 'E94nr':
        # ABLATION ONLY: original no-residual E94. Doesn't scale beyond ~100M.
        from ndm.models.e94 import E94NoResidualModel
        model = E94NoResidualModel(
            vocab_size=vocab_size,
            n_heads=args.n_heads,
            head_dim=args.n_state,
            depth=args.depth,
            dropout=args.dropout,
            share_layer_weights=False,
        )
    elif args.level == 'E94oh':
        # E94-OneHot: pure architecture, no learned input/output projections.
        # Token → one-hot tile K times → residual stream of dim K·vocab.
        # Per layer: LayerNorm → reshape to per-head → permute heads → time recurrence
        # via Triton (W_h_time + tanh) → reshape back → residual add.
        # Output: mean across K tiles → vocab logits. NO learned head.
        # K is set via --n_heads (overloaded), head_dim via --n_state.
        from ndm.models.e94 import E94OneHotModel
        model = E94OneHotModel(
            vocab_size=vocab_size,
            K=args.n_heads,                          # overload: --n_heads = K (tile factor)
            head_dim=args.n_state,
            depth=args.depth,
            dropout=args.dropout,
            gradient_checkpointing=args.gradient_checkpointing,
        )
    elif isinstance(args.level, str) and args.level.lower() == 'e88_fused':
        # E88 Fused: optimized kernel with [B, T, H, dim] layout (no transpose overhead)
        model = E88FusedLM(
            vocab_size=vocab_size,
            dim=args.dim,
            depth=args.depth,
            n_heads=args.n_heads,
            n_state=args.n_state,
            expansion=args.expansion,
            use_gate=bool(args.use_gate),
            checkpoint_interval=args.checkpoint_interval,
        )
    elif args.dim is not None and args.depth is not None:
        # Build extra per-layer kwargs for the typed-gdn2-lm / e98-cma-lm
        # candidate levels (None for every other level => default behaviour).
        layer_kwargs = {}
        if args.head_type_logits is not None:
            layer_kwargs['head_type_logits'] = [float(x) for x in args.head_type_logits.split(',')]
            layer_kwargs['gdn_allow_neg_eigval'] = bool(args.gdn_allow_neg_eigval)
        if args.corner_mixture is not None:
            layer_kwargs['corner_mixture'] = [float(x) for x in args.corner_mixture.split(',')]
        if args.lam_max is not None:
            layer_kwargs['lam_max'] = args.lam_max
        if args.beta_max is not None:
            layer_kwargs['beta_max'] = args.beta_max
        if args.igain_max is not None:
            layer_kwargs['igain_max'] = args.igain_max
        if args.layer_kwargs is not None:
            # Generic JSON passthrough of extra per-layer kwargs (e.g.
            # nonlin_subset_frac for level=complex-eig-lm). Merged last so it
            # overrides the specific knobs above.
            import json as _json
            layer_kwargs.update(_json.loads(args.layer_kwargs))
        model = LadderLM(
            vocab_size=vocab_size,
            dim=args.dim,
            depth=args.depth,
            level=args.level,
            layer_kwargs=(layer_kwargs or None),
            expansion=args.expansion,
            n_groups=args.n_groups,
            n_state=args.n_state,
            n_slots=args.n_slots,
            n_heads=args.n_heads,
            top_k=args.top_k,
            k_fast=args.k_fast,
            k_slow=args.k_slow,
            use_gate=bool(args.use_gate),
            gate_activation=args.gate_activation,
            linear_state=bool(args.linear_state),
            use_write_gate=bool(args.use_write_gate),
            e88_decay_mode=args.e88_decay_mode,
            e88_value_residual=bool(args.e88_value_residual),
            e88_raw_write=bool(args.e88_raw_write),
            use_chunked_e97=bool(args.use_chunked_e97),
            e97_chunk_size=args.e97_chunk_size,
            state_expansion=args.state_expansion,
            r_h_mode=r_h_mode,
            use_conv=bool(args.use_conv),
            d_conv=args.d_conv,
            gdn2_mlp_ratio=args.gdn2_mlp_ratio,
            dropout=args.dropout,
            checkpoint_interval=args.checkpoint_interval,
            gradient_checkpointing=args.gradient_checkpointing,
            projection_chunk_size=args.projection_chunk_size,
            loss_chunk_size=args.loss_chunk_size,
            use_triton=bool(args.use_triton),
            mlp_ratio=args.mlp_ratio,
            mlp_multiple=args.mlp_multiple,
        )
    else:
        model = create_ladder_model(
            target_params=args.params,
            level=args.level,
            vocab_size=vocab_size,
            expansion=args.expansion,
            n_groups=args.n_groups,
            state_expansion=args.state_expansion,
            r_h_mode=r_h_mode,
        )

    model_metadata = attach_model_run_metadata(args, model)
    if is_main:
        print(
            f"Run label prefix: {model_metadata['run_label_prefix']} "
            f"(params_arg={model_metadata['params_arg']}, "
            f"total_params={model_metadata['total_params']:,}, "
            f"trainable_params={model_metadata['trainable_params']:,})",
            flush=True,
        )

    # Under distributed training, rank 0 owns the run directory / checkpoints.
    # Broadcast the rank-0 path so non-main ranks can coordinate finalization
    # without racing to create their own timestamped run directories.
    if dist_enabled and not is_main:
        path_box = [None]
    else:
        output_dir = setup_output_dir(args, model_metadata=model_metadata)
        print(f"Output directory: {output_dir}")
        path_box = [str(output_dir)]
    if dist_enabled and dist.is_initialized():
        dist.broadcast_object_list(path_box, src=0)
        output_dir = Path(path_box[0])

    final_ckpt.rebind_coordination_dir(output_dir)
    if is_main:
        final_ckpt.reset_coordination_files()
    if dist_enabled and dist.is_initialized():
        dist.barrier()

    heldout_curve = None
    heldout_curve_path = None
    if is_main and args.heldout_curve_every > 0:
        if not args.heldout_tensor:
            raise ValueError("--heldout_curve_every requires --heldout_tensor")
        ho = torch.load(args.heldout_tensor, map_location='cpu')
        if 'chunks' not in ho or 'bytes_per_token' not in ho:
            raise ValueError("--heldout_tensor must contain 'chunks' and 'bytes_per_token'")
        heldout_curve = {
            'chunks': ho['chunks'],
            'bytes_per_token': float(ho['bytes_per_token']),
            'eval_bs': int(os.environ.get('HELDOUT_EVAL_BS', '8')),
        }
        heldout_curve_path = Path(args.heldout_curve_path) if args.heldout_curve_path else output_dir / 'heldout_curve.csv'
        heldout_curve_path.parent.mkdir(parents=True, exist_ok=True)
        if not heldout_curve_path.exists() or heldout_curve_path.stat().st_size == 0:
            heldout_curve_path.write_text(
                'step,tokens,train_loss,heldout_ce,heldout_bpb,heldout_tokens,'
                'heldout_bytes_per_token,mode,wall_time_utc\n'
            )
        print(f"Held-out curve: every {args.heldout_curve_every} steps, "
              f"mode={args.heldout_eval_mode}, tensor={args.heldout_tensor}, "
              f"csv={heldout_curve_path}", flush=True)

    # Convert dtype while moving each parameter so large BF16 models never
    # transiently materialize a complete FP32 copy in HBM before conversion.
    if args.bf16:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device)

    if args.compile and hasattr(torch, 'compile'):
        print(f"Compiling model (mode={args.compile_mode})...")
        model = torch.compile(model, mode=args.compile_mode)

    # core_model = the unwrapped module (attribute access, checkpoint state_dict,
    # eval forwards). model = the (optionally DDP-wrapped) module used for the timed
    # train fwd/bwd so gradients all-reduce across ranks. DDP broadcasts rank-0
    # weights at construction => replicas start identical regardless of init seed.
    core_model = model
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False, gradient_as_bucket_view=True)
        if is_main:
            print(f"[DDP] wrapped model in DistributedDataParallel "
                  f"(device_ids=[{local_rank}])", flush=True)
    elif use_diloco:
        # DiLoCo: NOT wrapped in DDP (no per-step gradient all-reduce). Each rank
        # trains its own replica independently. DDP normally broadcasts rank-0
        # weights at construction to guarantee identical replicas; we replicate
        # that guarantee explicitly so every island starts from the SAME W_0 even
        # if any init op were non-deterministic across ranks. Broadcast both
        # parameters and buffers (running stats etc.) in-place from rank 0.
        with torch.no_grad():
            for p in core_model.parameters():
                dist.broadcast(p.data, src=0)
            for b in core_model.buffers():
                dist.broadcast(b.data, src=0)
        dist.barrier()
        if is_main:
            print(f"[DiLoCo] broadcast rank-0 W_0 to all {world_size} ranks "
                  f"(identical start)", flush=True)

        merge_topology = str(getattr(args, 'diloco_merge_topology', 'global') or 'global').lower()
        args._diloco_merge_topology = merge_topology
        args._diloco_merge_groups = None
        if merge_topology == 'hierarchical':
            merge_group_size = int(getattr(args, 'diloco_merge_group_size', 8) or 8)
            create_barrier_every = int(
                getattr(args, 'diloco_merge_group_create_barrier_every', 8) or 0)
            if is_main:
                n_merge_groups = (world_size + merge_group_size - 1) // merge_group_size
                print(f"[DiLoCo-merge] building hierarchical process groups: "
                      f"group_size={merge_group_size} n_groups={n_merge_groups} "
                      f"create_barrier_every={create_barrier_every}",
                      flush=True)
            args._diloco_merge_groups = _build_diloco_hierarchical_merge_groups(
                world_size, rank, merge_group_size, create_barrier_every)
            if is_main:
                print("[DiLoCo-merge] hierarchical process groups built; warming "
                      "communicators", flush=True)
            # Warm subgroup communicators without interleaving default-group
            # barriers against nonmember ranks. Each rank first warms exactly
            # the group it will use for first-level local reductions; then roots
            # warm the second-level root group.
            _warm_diloco_hierarchical_merge_groups(args, device)
            if is_main:
                group_counts = [
                    len(group_ranks)
                    for group_ranks, _ in args._diloco_merge_groups['groups']
                ]
                print(f"[DiLoCo-merge] topology=hierarchical group_size={merge_group_size} "
                      f"groups={group_counts} roots={args._diloco_merge_groups['root_ranks']} "
                      f"(exact weighted SUM/world_size; bucket_numel="
                      f"{getattr(args, 'diloco_merge_bucket_numel', 0)})",
                      flush=True)
        elif merge_topology != 'global':
            raise ValueError(f"unknown DiLoCo merge topology: {merge_topology}")
        elif is_main:
            print(f"[DiLoCo-merge] topology=global bucket_numel="
                  f"{getattr(args, 'diloco_merge_bucket_numel', 0)}", flush=True)

        # HYBRID (task diloco-loss-parity-longhorizon): if --diloco_island_size > 1,
        # form islands of consecutive ranks that do per-step DDP gradient all-reduce
        # WITHIN the island (tight, exact-SGD sync over a cheap 2-GPU collective),
        # while the periodic diloco_merge keeps averaging ACROSS ALL ranks every K
        # steps. Because intra-island ranks are kept bit-identical by their DDP, a
        # GLOBAL all-reduce mean over all ranks equals the per-ISLAND mean (each
        # island's weights are summed island_size times, then divided by world_size)
        # -> the existing global diloco_merge is the correct cross-island outer step,
        # unchanged. This trades some throughput (per-step intra-island comm) for the
        # sample-efficiency of a larger effective per-island batch (island_size*bs).
        island_size = int(getattr(args, 'diloco_island_size', 0) or 0)
        if island_size > 1:
            assert world_size % island_size == 0, (
                f"--diloco_island_size {island_size} must divide world_size {world_size}")
            n_islands = world_size // island_size
            # new_group is collective: every rank must call it for EVERY island group,
            # in the same order, even ones it does not join.
            island_groups = []
            island_group = None
            for isl in range(n_islands):
                grp_ranks = list(range(isl * island_size, (isl + 1) * island_size))
                g = dist.new_group(ranks=grp_ranks)
                island_groups.append((grp_ranks, g))
                if rank in grp_ranks:
                    island_group = g
            # SEQUENTIALLY initialize each island's NCCL subgroup communicator. The
            # NCCL comm for a subgroup is created lazily on first use; if all islands
            # init concurrently (each DDP construction firing a collective on its own
            # 2-rank comm at once) they deadlock on a no-NVLink box (observed: a 600 s
            # BROADCAST timeout at DDP construction). A tiny all-reduce per island with
            # a GLOBAL barrier between forces creation one island at a time.
            for grp_ranks, g in island_groups:
                if rank in grp_ranks:
                    _w = torch.zeros(1, device=device)
                    dist.all_reduce(_w, group=g)
                dist.barrier()
            model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                        find_unused_parameters=False, gradient_as_bucket_view=True,
                        process_group=island_group)
            args._diloco_island_size = island_size
            args._diloco_n_islands = n_islands
            if is_main:
                print(f"[DiLoCo-hybrid] {n_islands} islands x {island_size} GPUs: per-step "
                      f"DDP gradient all-reduce WITHIN island + DiLoCo periodic averaging "
                      f"ACROSS islands every K={args.diloco_k} (subgroup comms warmed "
                      f"sequentially)", flush=True)

    if is_main:
        print(f"Model: Level {args.level}, {core_model.get_num_params():,} parameters")

    optimizer = build_training_optimizer(core_model, args, announce=True)

    # Resume if requested
    start_step = 0
    loaded_outer_state = None
    loaded_checkpoint = None
    loaded_optimizer_state = False
    total_tokens = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        preflight_result = {}

        def _checkpoint_preflight(checkpoint):
            preflight_result['total_tokens'] = validate_dense_checkpoint_sampler(
                checkpoint, args, world_size=world_size,
                checkpoint_path=args.resume)

        start_step, _, ckpt = load_checkpoint(
            args.resume, core_model, optimizer, return_checkpoint=True,
            preflight=_checkpoint_preflight)
        loaded_checkpoint = ckpt
        loaded_outer_state = ckpt.get('diloco_outer_state')
        loaded_optimizer_state = 'optimizer_state_dict' in ckpt
        total_tokens = preflight_result['total_tokens']
        # Optimizer state dicts carry their original param-group LR. For
        # continuation runs we want the explicit CLI LR to be authoritative.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
        print(f"Resumed at step {start_step} with total_tokens={total_tokens}")
    elif args.total_tokens is not None:
        explicit_fresh_tokens = _validated_total_tokens(args.total_tokens, 'explicit')
        if explicit_fresh_tokens != 0:
            raise ValueError(
                "nonzero --total_tokens requires --resume so model state and token "
                "authority cannot diverge"
            )

    # DDP data sharding: each rank reads a DISTINCT stream so the world processes
    # different real tokens every step (true data parallelism, not replicated work).
    # Offsetting the dataset seed by rank gives disjoint sampling positions across
    # the corpus. Model weights stay identical (DDP broadcast); only the data differs.
    # Create dataset - use BatchedStreamDataset for TBPTT (persistent per-batch streams)
    if args.tbptt:
        print("TBPTT enabled: using BatchedStreamDataset (persistent streams)")
    train_dataset = build_training_dataset(
        args, rank=rank, dist_enabled=dist_enabled, start_step=start_step,
        world_size=world_size, total_accepted_tokens=total_tokens)
    if is_main:
        if launch_sampler_identity is None:
            print(f"[data-stream] schema={LEGACY_SAMPLER_SCHEMA} "
                  f"seed_base={args.seed + start_step} starting_step={start_step} "
                  "rank_offset=global-rank", flush=True)
        else:
            print(f"[data-stream] schema={launch_sampler_identity.schema} "
                  f"accepted_tokens={total_tokens} world={world_size} "
                  f"context={args.chunk_size} key={launch_sampler_identity.sampler_key} "
                  f"corpus={launch_sampler_identity.corpus_sha256} "
                  f"tokenizer={launch_sampler_identity.tokenizer_sha256}", flush=True)

    val_loader = None
    if args.val_data:
        val_loader = create_dataloader(
            args.val_data,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size + 1,
            device=device,
        )

    def get_training_batch():
        if args.tbptt:
            chunks, is_doc_end = train_dataset.get_batch(device=device)
            actual_lengths = torch.full((args.batch_size,), args.chunk_size + 1, device=device)
        else:
            chunks, is_doc_end, actual_lengths = train_dataset.get_batch(args.batch_size, device=device)
        return chunks, is_doc_end, actual_lengths

    def compute_training_loss(chunks, prev_hiddens=None):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.bf16):
            if args.tbptt:
                result = model(
                    chunks,
                    return_loss=True,
                    return_prev_hiddens=True,
                    prev_hiddens=prev_hiddens,
                )
            else:
                result = model(
                    chunks,
                    return_loss=True,
                )

            if isinstance(result, tuple):
                loss, (next_hidden, _) = result
            else:
                loss = result
                next_hidden = None

        return loss, next_hidden

    model.train()
    if args.optimizer == 'schedulefree':
        optimizer.train()
    if (getattr(optimizer, 'state_storage', None) == 'pinned-cpu'
            and not loaded_optimizer_state):
        init_stats = optimizer.initialize_state_()
        print(f"[sf-offload] initialized "
              f"{init_stats['state_bytes'] / (1024 ** 3):.3f} GiB pinned CPU state "
              f"in {init_stats['seconds']:.3f}s", flush=True)

    # DiLoCo outer-optimizer state. avg is explicitly stateless. momentum keeps
    # the P1/P2 geometry-fixed anchor/moment buffers. sfsgd is a separate manual
    # ScheduleFree-SGD state machine (outer_x/outer_z/outer_y) and never wraps the
    # live parameters with schedulefree.SGDScheduleFree.
    outer_state = None
    diloco_bootstrap_metadata = None
    if use_diloco:
        outer_state, diloco_bootstrap_metadata = resolve_diloco_outer_state_for_resume(
            core_model,
            optimizer,
            args,
            loaded_state=loaded_outer_state,
            ckpt=loaded_checkpoint,
            inner_optimizer_state_loaded=loaded_optimizer_state,
        )
        if is_main:
            if args.diloco_outer_optimizer == 'avg':
                print("[DiLoCo] outer optimizer: avg (stateless periodic averaging)",
                      flush=True)
            elif diloco_bootstrap_metadata:
                print(
                    "[DiLoCo] bootstrapped missing outer state from loaded model weights: "
                    f"mode={args.diloco_outer_optimizer} "
                    f"export_basis={args.diloco_export_basis} "
                    f"reason={diloco_bootstrap_metadata['missing_or_incompatible_reason']} "
                    f"guard={diloco_bootstrap_metadata['guard']} "
                    f"source={args.resume} "
                    f"step={diloco_bootstrap_metadata['source_checkpoint_step']}; "
                    "model tensors restored byte-identical after bootstrap",
                    flush=True,
                )
            elif loaded_outer_state is not None:
                print(f"[DiLoCo] restored outer optimizer state "
                      f"({args.diloco_outer_optimizer}) from checkpoint", flush=True)
            else:
                print(f"[DiLoCo] outer optimizer: {args.diloco_outer_optimizer} "
                      f"(outer_lr={args.diloco_outer_lr}, "
                      f"outer_beta={args.diloco_outer_beta}, "
                      f"export_basis={args.diloco_export_basis})", flush=True)
    diloco_merges = 0
    diloco_sync_total_s = 0.0

    if args.compile_warmup_steps > 0:
        print(f"Compile/autotune warmup: {args.compile_warmup_steps} untimed fwd+bwd step(s)")
        warmup_start = time.time()
        for warmup_step in range(args.compile_warmup_steps):
            chunks, _, _ = get_training_batch()
            loss, _ = compute_training_loss(chunks)
            if not torch.isfinite(loss):
                print(f"Non-finite compile warmup loss at warmup step {warmup_step + 1}: {loss.item()}")
                break
            loss.backward()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            print(f"  warmup {warmup_step + 1}/{args.compile_warmup_steps} | loss {loss.item():.4f}")
        print(f"Compile/autotune warmup took {time.time() - warmup_start:.1f}s")

    # Memory probe mode: run 1 fwd+bwd step, report peak memory, exit
    if args.probe_memory:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        print(f"PROBE_BASELINE_MEMORY_MB: "
              f"{torch.cuda.memory_allocated(device) / (1024 * 1024):.1f}")
        chunks, _, _ = get_training_batch()
        probe_phase_started = time.perf_counter()
        loss, _ = compute_training_loss(chunks)
        torch.cuda.synchronize(device)
        print(f"PROBE_FORWARD_SECONDS: "
              f"{time.perf_counter() - probe_phase_started:.3f}")
        print(f"PROBE_AFTER_FORWARD_MEMORY_MB: "
              f"{torch.cuda.memory_allocated(device) / (1024 * 1024):.1f}")
        probe_phase_started = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize(device)
        print(f"PROBE_BACKWARD_SECONDS: "
              f"{time.perf_counter() - probe_phase_started:.3f}")
        print(f"PROBE_AFTER_BACKWARD_MEMORY_MB: "
              f"{torch.cuda.memory_allocated(device) / (1024 * 1024):.1f}")
        optimizer.step()
        if int(getattr(args, 'probe_optimizer_steps', 1)) < 1:
            raise ValueError("--probe_optimizer_steps must be positive")
        if getattr(optimizer, 'state_storage', None) == 'pinned-cpu':
            print(f"PROBE_SF_OFFLOAD_STEP_SECONDS_1: "
                  f"{optimizer.last_step_stats.get('seconds', 0.0):.3f}")
        for probe_update in range(2, int(args.probe_optimizer_steps) + 1):
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.grad = torch.full_like(parameter, 1.0e-4)
            optimizer.step()
            if getattr(optimizer, 'state_storage', None) == 'pinned-cpu':
                print(f"PROBE_SF_OFFLOAD_STEP_SECONDS_{probe_update}: "
                      f"{optimizer.last_step_stats.get('seconds', 0.0):.3f}")
        optimizer.zero_grad()
        torch.cuda.synchronize(device)
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        print(f"PROBE_PEAK_MEMORY_MB: {peak_mb:.1f}")
        if getattr(optimizer, 'state_storage', None) == 'pinned-cpu':
            stats = optimizer.last_step_stats
            print(f"PROBE_SF_OFFLOAD_STATE_GIB: "
                  f"{optimizer.offloaded_state_bytes() / (1024 ** 3):.3f}")
            print(f"PROBE_SF_OFFLOAD_H2D_GIB: "
                  f"{stats.get('h2d_bytes', 0) / (1024 ** 3):.3f}")
            print(f"PROBE_SF_OFFLOAD_D2H_GIB: "
                  f"{stats.get('d2h_bytes', 0) / (1024 ** 3):.3f}")
            print(f"PROBE_SF_OFFLOAD_STEP_SECONDS: {stats.get('seconds', 0.0):.3f}")
        sys.exit(0)

    # Training state
    hidden_state = None  # Only used if --tbptt
    accumulated_steps = 0
    running_loss = 0
    tokens_processed = 0
    avg_loss = 0.0  # Initialize to avoid UnboundLocalError if training ends before first log
    last_train_loss = None
    start_time = time.time()

    # Track last 100 step losses for reliable final metric
    from collections import deque
    last_100_losses = deque(maxlen=100)

    print(f"\nStarting training from step {start_step}...")
    print(f"Batch size: {args.batch_size}, Chunk size: {args.chunk_size}")
    print(f"Gradient accumulation: {args.grad_accum}, Effective batch: {args.batch_size * args.grad_accum}")
    print()

    step = start_step

    # Time-based training setup defaults to process startup so slow init remains
    # part of fitness.  Some external kernels need an explicit untimed compile
    # warmup; in that mode we start the clock here, after warmup completes.
    train_start_time = time.time() if args.timer_after_compile_warmup else STARTUP_TIME
    train_end_time = None
    if args.train_minutes is not None:
        train_end_time = train_start_time + args.train_minutes * 60
        elapsed_init = time.time() - STARTUP_TIME
        elapsed_clock = time.time() - train_start_time
        remaining = max(0.0, args.train_minutes * 60 - elapsed_clock)
        clock_origin = "after compile warmup" if args.timer_after_compile_warmup else "from process start"
        print(f"Time-based training: {args.train_minutes} min budget ({clock_origin}). "
              f"Init took {elapsed_init:.1f}s, {remaining:.1f}s remaining for training.")

    stopped_nonfinite = False
    stop_training = False

    # Prefetch data function
    import threading
    import queue
    prefetch_queue = queue.Queue(maxsize=2)
    prefetch_stop = threading.Event()

    def prefetch_worker():
        """Background thread to prefetch batches."""
        while not prefetch_stop.is_set():
            try:
                if args.tbptt:
                    chunks, is_doc_end = train_dataset.get_batch(device=device)
                    actual_lengths = torch.full((args.batch_size,), args.chunk_size + 1, device=device)
                else:
                    chunks, is_doc_end, actual_lengths = train_dataset.get_batch(args.batch_size, device=device)
                prefetch_queue.put((chunks, is_doc_end, actual_lengths), timeout=1.0)
            except queue.Full:
                continue
            except Exception as e:
                print(f"Prefetch error: {e}")
                break

    prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
    prefetch_thread.start()

    while True:
        if stop_training:
            break
        if train_end_time is None and step >= args.steps:
            break
        if train_end_time is not None and not dist_enabled and time.time() >= train_end_time:
            break

        # Get prefetched batch
        try:
            chunks, is_doc_end, actual_lengths = prefetch_queue.get(timeout=5.0)
        except queue.Empty:
            print("Warning: prefetch queue empty, fetching synchronously")
            if args.tbptt:
                chunks, is_doc_end = train_dataset.get_batch(device=device)
                actual_lengths = torch.full((args.batch_size,), args.chunk_size + 1, device=device)
            else:
                chunks, is_doc_end, actual_lengths = train_dataset.get_batch(args.batch_size, device=device)

        # Reset hidden state at document boundaries (only if TBPTT enabled)
        if args.tbptt and hidden_state is not None:
            reset_mask = is_doc_end.view(-1, 1)
            hidden_state = [h * (~reset_mask) if h is not None else None for h in hidden_state]

        # Forward pass
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.bf16):
            if args.tbptt:
                result = model(
                    chunks,
                    return_loss=True,
                    return_prev_hiddens=True,
                    prev_hiddens=hidden_state,
                )
            else:
                result = model(
                    chunks,
                    return_loss=True,
                )

            if isinstance(result, tuple):
                loss, (next_hidden, _) = result
            else:
                loss = result
                next_hidden = None

        health_check_every = max(1, int(args.distributed_health_check_every))
        check_health_now = (
            not dist_enabled
            or health_check_every == 1
            or step % health_check_every == 0
        )

        local_nonfinite_loss = not torch.isfinite(loss)
        if check_health_now:
            if status_collectives_enabled and distributed_any(local_nonfinite_loss, device, dist_enabled=True):
                if local_nonfinite_loss:
                    print(f"Non-finite loss at step {step}: {loss.item()}. Stopping before optimizer step.")
                else:
                    print(f"Peer reported non-finite loss at step {step}. Stopping before optimizer step.",
                          flush=True)
                stopped_nonfinite = True
                break
            elif not status_collectives_enabled and local_nonfinite_loss:
                if dist_enabled:
                    raise RuntimeError(
                        f"Non-finite loss at step {step}: {loss.item()} "
                        "(scalar status collectives disabled)"
                    )
                print(f"Non-finite loss at step {step}: {loss.item()}. Stopping before optimizer step.")
                stopped_nonfinite = True
                break
        elif local_nonfinite_loss:
            raise RuntimeError(
                f"Non-finite loss at step {step}: {loss.item()} "
                f"(distributed health check cadence={health_check_every})"
            )

        # Add orthogonality regularization for E79 if enabled
        if args.orth_reg > 0:
            orth_loss = 0.0
            for module in model.modules():
                if hasattr(module, 'orthogonality_loss'):
                    orth_loss = orth_loss + module.orthogonality_loss(
                        lambda_sep=args.orth_sep,
                        lambda_orth=args.orth_orth
                    )
            loss = loss + args.orth_reg * orth_loss

        # Scale for gradient accumulation
        scaled_loss = loss / args.grad_accum
        # DDP comm amortization: the all-reduce gradient hooks fire during
        # backward(). On the non-final micro-steps of a grad-accum window we
        # suppress the sync (model.no_sync()) so the 1.29B-param bf16 all-reduce
        # happens ONCE per optimizer step instead of once per micro-step — the
        # dominant cost on a PCIe (no-NVLink) box. grad_accum=1 always syncs
        # (last micro-step) => identical behavior to before.
        _is_last_micro = (accumulated_steps + 1) >= args.grad_accum
        if args._ddp_enabled and not _is_last_micro:
            with model.no_sync():
                scaled_loss.backward()
        else:
            scaled_loss.backward()

        # Update hidden state (only if TBPTT enabled)
        if args.tbptt and next_hidden is not None:
            hidden_state = [h.detach() if h is not None else None for h in next_hidden]

        accumulated_steps += 1
        running_loss += loss.item()
        tokens_processed += actual_lengths.sum().item()

        # Optimizer step
        if accumulated_steps >= args.grad_accum:
            # Gradient clipping
            if args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            else:
                grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5

            local_nonfinite_grad = not torch.isfinite(torch.as_tensor(grad_norm))
            if (check_health_now and status_collectives_enabled
                    and distributed_any(local_nonfinite_grad, device, dist_enabled=True)):
                grad_value = grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm)
                # Multi-rank (DiLoCo/DDP): a per-rank skip desyncs the collective merge
                # (ranks would diverge in step count). Keep the stop guard for safety.
                if local_nonfinite_grad:
                    print(f"Non-finite grad norm at step {step}: {grad_value}. Stopping before optimizer step (multi-rank).", flush=True)
                else:
                    print(f"Peer reported non-finite grad norm at step {step}. Stopping before optimizer step (multi-rank).", flush=True)
                optimizer.zero_grad(set_to_none=True)
                stopped_nonfinite = True
                break
            elif local_nonfinite_grad:
                grad_value = grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm)
                if dist_enabled:
                    raise RuntimeError(
                        f"Non-finite grad norm at step {step}: {grad_value} "
                        f"({'scalar status collectives disabled' if check_health_now else f'distributed health check cadence={health_check_every}'})"
                    )
                print(f"Non-finite grad norm at step {step}: {grad_value}. SKIPPING this step (transient overflow), continuing.", flush=True)
                optimizer.zero_grad(set_to_none=True)
                accumulated_steps = 0
                continue

            # Learning rate schedule (only for standard AdamW). Warmup + cosine decay
            # to min_lr_frac*lr over the full run; scales each group's base LR so
            # per-group ratios (--knob_lr_mult) are preserved. Schedule-free instead
            # applies its own warmup internally and is left at a constant base LR.
            if args.optimizer == 'adamw':
                scale = lr_scale_at(step, args.warmup_steps, args.steps, args.min_lr_frac)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['base_lr'] * scale
                lr = optimizer.param_groups[0]['lr']
            else:
                lr = args.lr  # Schedule-free handles LR internally

            optimizer.step()
            optimizer.zero_grad()

            step += 1
            total_tokens = advance_total_tokens(
                total_tokens,
                world_size,
                args.batch_size,
                args.chunk_size,
                args.grad_accum,
            )
            accumulated_steps = 0

            # DiLoCo inter-worker sync: every K local optimizer steps, average the
            # model weights across ranks (the periodic merge that replaces vanilla
            # DDP's per-step gradient all-reduce). ALL ranks reach this collective
            # at the same step (identical step counts under --steps), so it is a
            # natural barrier. The sync wall-clock is intentionally inside the
            # current log window's `elapsed`, so reported global_tok/s already
            # reflects the amortized communication cost.
            if use_diloco and step % args.diloco_k == 0:
                sync_s = diloco_merge(core_model, optimizer, args, world_size,
                                      outer_state, step=step,
                                      merge_index=diloco_merges + 1)
                diloco_merges += 1
                diloco_sync_total_s += sync_s
                if is_main:
                    print(f"  >>> [DiLoCo] merge #{diloco_merges} at step {step}: "
                          f"averaged model weights across {world_size} ranks in "
                          f"{sync_s*1000:.0f} ms (amortized over {args.diloco_k} steps)",
                          flush=True)

            # Logging
            if step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                elapsed = time.time() - start_time
                elapsed_total_h = (time.time() - train_start_time) / 3600.0
                wall_time = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
                # Per-rank tok/s; global tok/s = sum across the DDP world (each rank
                # processes a distinct batch of equal size, so global = per_rank*W).
                tokens_per_sec = tokens_processed / elapsed
                global_tps = tokens_per_sec * world_size

                if is_main:
                    print(f"step {step:6d} | loss {avg_loss:.4f} | lr {lr:.2e} | "
                          f"grad {grad_norm:.2f} | tok/s {tokens_per_sec:.0f} | "
                          f"global_tok/s {global_tps:.0f} | "
                          f"elapsed_h {elapsed_total_h:.3f} | time {wall_time}")
                    if getattr(optimizer, 'state_storage', None) == 'pinned-cpu':
                        offload_stats = optimizer.last_step_stats
                        print(f"  [sf-offload] update_s="
                              f"{offload_stats.get('seconds', 0.0):.3f} "
                              f"h2d_gib={offload_stats.get('h2d_bytes', 0) / (1024 ** 3):.3f} "
                              f"d2h_gib={offload_stats.get('d2h_bytes', 0) / (1024 ** 3):.3f}",
                              flush=True)

                # Track for last-100 average (each entry covers log_every steps)
                last_100_losses.append(avg_loss)
                last_train_loss = avg_loss

                running_loss = 0
                tokens_processed = 0
                start_time = time.time()

            # Final-stop coordination must run before any rank-0-only eval or
            # checkpoint work. Otherwise non-head ranks can enter finalization
            # while rank 0 is still validating/saving, which desynchronizes the
            # next distributed boundary.
            walltime_check_every = max(1, int(args.walltime_check_every))
            check_walltime_now = (
                step % walltime_check_every == 0
                or bool(getattr(final_ckpt, 'triggered', False))
            )
            if check_walltime_now:
                local_budget_done = train_end_time is not None and time.time() >= train_end_time
                extra_stop_reason = 'train_budget' if local_budget_done else None
                stop_for_final, final_reason, remaining_s = final_ckpt.maybe_request_stop(
                    step,
                    dist_enabled=dist_enabled,
                    peer_poll_s=(status_request_poll_s if not status_collectives_enabled else 0.0),
                    extra_local_reason=extra_stop_reason,
                )
                if status_collectives_enabled:
                    stop_for_final, final_reason, remaining_s = consensus_final_checkpoint_stop(
                        final_ckpt,
                        stop_for_final,
                        final_reason,
                        remaining_s,
                        device,
                        dist_enabled,
                    )
                if stop_for_final:
                    rem_text = 'unknown' if remaining_s is None else f"{remaining_s:.1f}"
                    print(f"[final-checkpoint] rank {rank}/{world_size} entering finalization "
                          f"at step={step} reason={final_reason} remaining_s={rem_text} "
                          f"model_variant={args._model_variant} is_head={is_main}",
                          flush=True)
                    break

                if status_collectives_enabled and distributed_any(local_budget_done, device, dist_enabled=True):
                    stop_training = True
                elif not status_collectives_enabled and local_budget_done:
                    stop_training = True

            # Fixed held-out curve (rank 0 only). For schedule-free y-mode this
            # intentionally keeps optimizer.train() weights and only flips the
            # module's eval/train flag around scoring to disable dropout.
            if heldout_curve and step % args.heldout_curve_every == 0 and is_main:
                prepare_schedulefree_eval_mode(optimizer, args)
                heldout_ce, heldout_bpb, heldout_tokens = score_heldout_tensor(
                    core_model,
                    heldout_curve['chunks'],
                    heldout_curve['bytes_per_token'],
                    device,
                    eval_bs=heldout_curve['eval_bs'],
                    bf16=args.bf16,
                )
                if args.optimizer == 'schedulefree':
                    optimizer.train()
                train_loss_for_curve = last_train_loss
                if train_loss_for_curve is None:
                    denom = max(1, step - start_step)
                    train_loss_for_curve = running_loss / denom if running_loss else float('nan')
                heldout_curve_total_tokens = total_tokens
                wall_time = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
                with open(heldout_curve_path, 'a') as f:
                    f.write(
                        f"{step},{heldout_curve_total_tokens},{train_loss_for_curve:.6f},"
                        f"{heldout_ce:.6f},{heldout_bpb:.6f},{heldout_tokens},"
                        f"{heldout_curve['bytes_per_token']:.6f},"
                        f"{heldout_eval_mode_label(args)},{wall_time}\n"
                    )
                print(f"  >>> heldout_curve mode={heldout_eval_mode_label(args)} step={step} "
                      f"tokens={heldout_curve_total_tokens} train_loss={train_loss_for_curve:.4f} "
                      f"heldout_bpb={heldout_bpb:.4f}", flush=True)

            # Validation (rank 0 only; uses unwrapped core_model, no DDP collectives)
            if val_loader and step % args.val_every == 0 and is_main:
                if args.optimizer == 'schedulefree':
                    optimizer.eval()  # Get averaged params for eval
                val_loss = validate(core_model, val_loader, device)
                print(f"  >>> validation loss: {val_loss:.4f}")
                if args.optimizer == 'schedulefree':
                    optimizer.train()  # Back to training mode

            # Checkpointing (rank 0 only; save the unwrapped core_model state_dict so
            # checkpoints load identically in single-GPU eval/generate paths)
            if step % args.save_every == 0 and is_main:
                if args.optimizer == 'schedulefree':
                    optimizer.eval()  # Get averaged params for checkpoint
                ckpt_path = save_checkpoint(
                    core_model, optimizer, step, avg_loss, output_dir,
                    args.keep_checkpoints, outer_state=outer_state,
                    total_tokens=total_tokens,
                    metadata=checkpoint_metadata_with_diloco_bootstrap({
                        'kind': 'periodic',
                        'model_variant': args._model_variant,
                        'model': args._model_metadata,
                        'run_label_prefix': args._model_run_label_prefix,
                        'total_params': args._model_total_params,
                        'trainable_params': args._model_trainable_params,
                        'rank': rank,
                        'world_size': world_size,
                        'is_head': is_main,
                        'sampler': dense_sampler_checkpoint_metadata(
                            args, world_size=world_size,
                            total_tokens=total_tokens),
                        'walltime_remaining_s': (
                            final_ckpt.deadline - time.time()
                            if final_ckpt.deadline is not None else None
                        ),
                    }, diloco_bootstrap_metadata),
                )
                print(f"  >>> saved checkpoint: {ckpt_path.name} "
                      f"step={step} total_tokens={total_tokens}")
                if args.optimizer == 'schedulefree':
                    optimizer.train()  # Back to training mode

    # Stop prefetch thread
    prefetch_stop.set()
    prefetch_thread.join(timeout=2.0)

    # Compute last-100 average (reliable metric)
    if len(last_100_losses) > 0:
        # Take last 100 log windows (or fewer if not enough data)
        # Each entry covers log_every steps, so 100 entries = 100*log_every steps
        last_n = min(100, len(last_100_losses))
        recent_losses = list(last_100_losses)[-last_n:]
        last_100_avg = sum(recent_losses) / len(recent_losses)
    else:
        last_100_avg = avg_loss  # Fallback if no logging happened

    if dist_enabled:
        final_ckpt.mark_finalization_ready(step)
        final_ckpt.wait_for_all_finalization_ready()

    # DiLoCo: one FINAL merge so the saved checkpoint is the cross-rank consensus
    # model (not whichever rank happens to be rank 0). All ranks participate in the
    # collective; skipped if training stopped non-finite (a broken rank would not
    # reach the collective and the others would hang).
    #
    # SKIP the final merge when the last step ALREADY merged (step % K == 0): the
    # periodic merge at that step has already produced the consensus, so a second
    # merge here is redundant. For outer momentum it is also HARMFUL — between the
    # periodic merge and this one there are ZERO local steps, so delta = mean_i(W) -
    # W_r = 0, yet the outer step still applies the leftover momentum buffer
    # (W <- W_r + outer_lr*beta*mom), a spurious extra step that DEGRADES the final
    # checkpoint (task diloco-loss-parity-longhorizon: observed BPB 2.19 -> 2.27 at
    # step 500, K=250, beta=0.5). beta=0 (local-SGD) is unaffected (no momentum), but
    # the guard is correct for all configs.
    last_step_already_merged = use_diloco and (step % args.diloco_k == 0)
    if use_diloco and not stopped_nonfinite and not last_step_already_merged:
        sync_s = diloco_merge(core_model, optimizer, args, world_size,
                              outer_state, step=step,
                              merge_index=diloco_merges + 1)
        diloco_merges += 1
        diloco_sync_total_s += sync_s
        if is_main:
            print(f"  >>> [DiLoCo] FINAL merge #{diloco_merges} at step {step}: "
                  f"consensus model averaged across {world_size} ranks "
                  f"({sync_s*1000:.0f} ms)", flush=True)
    elif use_diloco and last_step_already_merged and is_main:
        print(f"  >>> [DiLoCo] final merge SKIPPED at step {step}: last step already "
              f"merged (step % K == 0); checkpoint is already consensus "
              f"(avoids spurious outer-momentum double-step)", flush=True)

    # Under distributed training, only rank 0 runs the final checkpoint +
    # held-out/train evals (all ranks hold identical synchronized weights after the
    # final merge / DDP all-reduce). Other ranks skip to the barrier.
    _run_final = (args._is_main if args._dist_enabled else True)
    final_metadata = {
        'kind': 'final',
        'reason': final_ckpt.reason or 'training_complete',
        'model_variant': args._model_variant,
        'model': args._model_metadata,
        'run_label_prefix': args._model_run_label_prefix,
        'total_params': args._model_total_params,
        'trainable_params': args._model_trainable_params,
        'rank': rank,
        'world_size': world_size,
        'is_head': is_main,
        'sampler': dense_sampler_checkpoint_metadata(
            args, world_size=world_size, total_tokens=total_tokens),
        'walltime_deadline_source': final_ckpt.deadline_source,
        'walltime_remaining_s': (
            final_ckpt.deadline - time.time()
            if final_ckpt.deadline is not None else None
        ),
        'walltime_margin_s': final_ckpt.margin_s,
        'shutdown_signal': _SHUTDOWN_REQUEST['signal'],
    }

    # Final checkpoint - use last-100 average for reliable metric
    if not _run_final:
        pass
    elif stopped_nonfinite:
        print("Skipping final checkpoint because training stopped on non-finite loss/gradient.")
    else:
        if args.optimizer == 'schedulefree':
            optimizer.eval()  # Get averaged params for final checkpoint
        remaining = final_metadata['walltime_remaining_s']
        rem_text = 'unknown' if remaining is None else f"{remaining:.1f}"
        print(f"[final-checkpoint] START kind=final reason={final_metadata['reason']} "
              f"step={step} total_tokens={total_tokens} loss={last_100_avg:.4f} remaining_s={rem_text} "
              f"model_variant={args._model_variant} rank={rank}/{world_size} "
              f"is_head={is_main}", flush=True)
        ckpt_path = save_checkpoint(
            core_model, optimizer, step, last_100_avg, output_dir,
            args.keep_checkpoints, outer_state=outer_state,
            total_tokens=total_tokens,
            metadata=checkpoint_metadata_with_diloco_bootstrap(
                final_metadata, diloco_bootstrap_metadata),
        )
        print(f"[final-checkpoint] END path={ckpt_path} latest={output_dir / 'latest.pt'} "
              f"step={step} total_tokens={total_tokens} "
              f"model_variant={args._model_variant} rank={rank}/{world_size} "
              f"is_head={is_main}", flush=True)

    # within-layer LM screen (task e97-within-layer): ONE final held-out eval on the
    # schedule-free AVERAGED weights (leaderboard methodology) — distinct from the
    # periodic NON-averaged validation during training. Opt-in via --final_heldout_eval.
    if _run_final and args.final_heldout_eval and args.heldout_tensor and not stopped_nonfinite:
        # task lb-compare: score a FIXED pre-tokenized held-out tensor (byte-for-byte
        # identical slice across all models, tokenizer-correct CE). The default is
        # averaged x-mode for legacy leaderboard compatibility; long-reference runs
        # pass --heldout_eval_mode y/train to score schedule-free training weights.
        ho = torch.load(args.heldout_tensor, map_location='cpu')
        if 'chunks' not in ho or 'bytes_per_token' not in ho:
            raise ValueError("--heldout_tensor must contain 'chunks' and 'bytes_per_token'")
        ho_chunks = ho['chunks']                       # [N, chunk+1]
        bpt = float(ho['bytes_per_token'])
        # Eval batch is decoupled from the train batch: CE/BPB are batch-invariant
        # (every chunk scored identically), so use a larger fwd-only batch to cut
        # the number of (slow sequential-kernel) forward calls. Override via env.
        eval_bs = int(os.environ.get('HELDOUT_EVAL_BS', '8'))
        eval_bs = max(1, min(eval_bs, ho_chunks.shape[0]))

        # Diagnostic: also report NON-averaged (training) weights when requested,
        # to separate a schedule-free averaging artifact from real generalization.
        if os.environ.get('HELDOUT_REPORT_NONAVG') == '1' and args.optimizer == 'schedulefree':
            optimizer.train()  # raw (non-averaged) weights
            ce_raw, bpb_raw, _ = score_heldout_tensor(
                core_model, ho_chunks, bpt, device, eval_bs=eval_bs, bf16=args.bf16)
            print(f"FINAL_HELDOUT_CE_NONAVG: {ce_raw:.4f}")
            print(f"FINAL_HELDOUT_BPB_NONAVG: {bpb_raw:.4f}")
        prepare_schedulefree_eval_mode(optimizer, args)
        heldout_ce, heldout_bpb, tot_tok = score_heldout_tensor(
            core_model, ho_chunks, bpt, device, eval_bs=eval_bs, bf16=args.bf16)
        if args.optimizer == 'schedulefree':
            optimizer.train()
        print(f"FINAL_HELDOUT_CE: {heldout_ce:.4f}")
        print(f"FINAL_HELDOUT_BPB: {heldout_bpb:.4f}")
        print(f"FINAL_HELDOUT_TOKENS: {tot_tok}")
        print(f"FINAL_HELDOUT_BYTES_PER_TOKEN: {bpt:.4f}")
        print(f"FINAL_HELDOUT_MODE: {heldout_eval_mode_label(args)}")
    elif _run_final and args.final_heldout_eval and val_loader is not None and not stopped_nonfinite:
        if args.optimizer == 'schedulefree':
            optimizer.eval()  # averaged weights
        import math as _math
        heldout_ce = validate(core_model, val_loader, device, max_batches=args.final_val_batches)
        heldout_bpb = (heldout_ce / _math.log(2.0)) / args.heldout_bytes_per_token
        print(f"FINAL_HELDOUT_CE: {heldout_ce:.4f}")
        print(f"FINAL_HELDOUT_BPB: {heldout_bpb:.4f}")

    # task e97-audit2: clean eval on a TRAIN-distribution slice (--data) with the SAME
    # averaged weights + clean machinery, to isolate the train->held generalization gap
    # (same units, same weights) from the units/running-average measurement artifacts.
    if _run_final and args.final_train_eval and not stopped_nonfinite:
        if args.optimizer == 'schedulefree':
            optimizer.eval()  # averaged weights (same as held-out)
        import math as _math2
        train_eval_loader = create_dataloader(
            args.data,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size + 1,
            device=device,
        )
        train_ce = validate(core_model, train_eval_loader, device, max_batches=args.final_val_batches)
        train_bpb = (train_ce / _math2.log(2.0)) / args.heldout_bytes_per_token
        print(f"FINAL_TRAIN_CE: {train_ce:.4f}")
        print(f"FINAL_TRAIN_BPB: {train_bpb:.4f}")

    # Print final metrics in parseable format (per-rank peak memory always; main
    # prints the run-level summary). DDP teardown after a barrier so no rank exits
    # while another is still in a collective.
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        reserved_mb = torch.cuda.max_memory_reserved() / 1024 / 1024
        print(f"[rank {args._rank}] PEAK_MEMORY_MB: {peak_mb:.0f} | "
              f"RESERVED_MEMORY_MB: {reserved_mb:.0f}", flush=True)
    if _run_final:
        print(f"\nTraining complete! Final step: {step}")
        print(f"TOTAL_TOKENS: {total_tokens}")
        print(f"FINAL_LOSS_LAST100: {last_100_avg:.4f}")
        if torch.cuda.is_available():
            print(f"PEAK_MEMORY_MB: {peak_mb:.0f}")
            print(f"RESERVED_MEMORY_MB: {reserved_mb:.0f}")
        if use_diloco:
            avg_sync = diloco_sync_total_s / max(diloco_merges, 1)
            print(f"DILOCO_MERGES: {diloco_merges}")
            print(f"DILOCO_K: {args.diloco_k}")
            print(f"DILOCO_SYNC_TOTAL_S: {diloco_sync_total_s:.3f}")
            print(f"DILOCO_SYNC_AVG_MS: {avg_sync*1000:.1f}")

    if args._dist_enabled and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    args = parse_args()
    train(args)
