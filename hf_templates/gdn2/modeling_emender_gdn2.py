from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_emender_gdn2 import EmenderGDN2Config


class PortableRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)


class PortableShortConv(nn.Module):
    """Weight-compatible causal depthwise convolution used before q/k/v."""

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size))
        self.kernel_size = int(kernel_size)
        nn.init.normal_(self.weight, mean=0.0, std=self.kernel_size ** -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.conv1d(
            x.transpose(1, 2),
            self.weight,
            bias=None,
            padding=self.kernel_size - 1,
            groups=x.shape[-1],
        )
        return F.silu(y[:, :, : x.shape[1]].transpose(1, 2).contiguous())


class PortableGatedRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        y = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        y = y * self.weight.float() * F.silu(gate.float())
        return y.to(x.dtype)


class PortableGDN2Core(nn.Module):
    """Clean PyTorch implementation of the published GDN2 recurrence.

    Per token, with matrix state S[K,V]:
      S <- diag(exp(g)) S
      value_delta <- (w * v) - (b * k)^T S
      S <- S + k value_delta^T
      output <- S^T (q / sqrt(K))
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int = 128,
        expand_v: float = 1.0,
        conv_size: int = 4,
        allow_neg_eigval: bool = False,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.num_v_heads = int(num_heads)
        self.head_k_dim = int(head_dim)
        self.head_v_dim = int(head_dim * expand_v)
        self.key_dim = self.num_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.allow_neg_eigval = bool(allow_neg_eigval)

        self.A_log = nn.Parameter(torch.zeros(self.num_heads))
        self.dt_bias = nn.Parameter(torch.zeros(self.key_dim))
        self.q_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.q_conv1d = PortableShortConv(self.key_dim, conv_size)
        self.k_conv1d = PortableShortConv(self.key_dim, conv_size)
        self.v_conv1d = PortableShortConv(self.value_dim, conv_size)
        self.f_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.key_dim, bias=False),
        )
        self.b_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.w_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.g_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True),
        )
        self.o_norm = PortableGatedRMSNorm(self.head_v_dim, eps=1e-5)
        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        q = self.q_conv1d(self.q_proj(hidden_states))
        k = self.k_conv1d(self.k_proj(hidden_states))
        v = self.v_conv1d(self.v_proj(hidden_states))

        g = -self.A_log.float().exp().repeat_interleave(self.head_k_dim) * F.softplus(
            self.f_proj(hidden_states).float() + self.dt_bias.float()
        )
        erase = torch.sigmoid(self.b_proj(hidden_states))
        write = torch.sigmoid(self.w_proj(hidden_states))
        if self.allow_neg_eigval:
            erase = erase * 2.0

        q = q.view(bsz, seq_len, self.num_heads, self.head_k_dim)
        k = k.view(bsz, seq_len, self.num_heads, self.head_k_dim)
        v = v.view(bsz, seq_len, self.num_v_heads, self.head_v_dim)
        g = g.view(bsz, seq_len, self.num_heads, self.head_k_dim)
        erase = erase.view(bsz, seq_len, self.num_heads, self.head_k_dim)
        write = write.view(bsz, seq_len, self.num_v_heads, self.head_v_dim)

        qf = q.float() / torch.sqrt(q.float().square().sum(dim=-1, keepdim=True) + 1e-6)
        kf = k.float() / torch.sqrt(k.float().square().sum(dim=-1, keepdim=True) + 1e-6)
        qf = qf * (self.head_k_dim ** -0.5)
        vf, erasef, writef = v.float(), erase.float(), write.float()
        state = torch.zeros(
            bsz,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            device=hidden_states.device,
            dtype=torch.float32,
        )
        outputs = []
        for t in range(seq_len):
            state = state * torch.exp(g[:, t].float()).unsqueeze(-1)
            read_key = erasef[:, t] * kf[:, t]
            erased = torch.einsum("bhkv,bhk->bhv", state, read_key)
            value_delta = writef[:, t] * vf[:, t] - erased
            state = state + torch.einsum("bhk,bhv->bhkv", kf[:, t], value_delta)
            outputs.append(torch.einsum("bhkv,bhk->bhv", state, qf[:, t]))

        recurrent = torch.stack(outputs, dim=1).to(hidden_states.dtype)
        gate = self.g_proj(hidden_states).view(
            bsz, seq_len, self.num_v_heads, self.head_v_dim
        )
        recurrent = self.o_norm(recurrent, gate)
        return self.o_proj(recurrent.reshape(bsz, seq_len, self.value_dim))


class PortableGDN2Layer(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int, conv_size: int):
        super().__init__()
        # Double gdn2 is intentional: it matches the trained checkpoint names
        # layers.N.gdn2.gdn2.* (MLP layer -> wrapper -> external core).
        self.gdn2 = PortableGDN2Core(dim, num_heads, head_dim, 1.0, conv_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gdn2(x)


class PortableSwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class PortableGDN2MLPBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int, conv_size: int, mlp_hidden: int):
        super().__init__()
        self.gdn2 = PortableGDN2Layer(dim, num_heads, head_dim, conv_size)
        self.norm_2 = PortableRMSNorm(dim)
        self.mlp = PortableSwiGLU(dim, mlp_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = self.gdn2(x)
        return mixed + self.mlp(self.norm_2(x + mixed))


class PortableGDN2LM(nn.Module):
    def __init__(self, args: dict, vocab_size: int):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.dim = int(args["dim"])
        self.depth = int(args["depth"])
        num_heads = int(args["n_heads"])
        head_dim = int(args.get("head_dim", 128))
        conv_size = int(args.get("d_conv", 4))
        mlp_ratio = float(args["gdn2_mlp_ratio"])
        multiple = int(args.get("gdn2_mlp_multiple", 64))
        mlp_hidden = max(multiple, int(round(self.dim * mlp_ratio / multiple) * multiple))

        self.embedding = nn.Embedding(self.vocab_size, self.dim)
        self.layer_norms = nn.ModuleList([PortableRMSNorm(self.dim) for _ in range(self.depth)])
        self.layers = nn.ModuleList([
            PortableGDN2MLPBlock(self.dim, num_heads, head_dim, conv_size, mlp_hidden)
            for _ in range(self.depth)
        ])
        self.norm = PortableRMSNorm(self.dim)
        self.lm_head = nn.Linear(self.dim, self.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        residual = None
        for norm, layer in zip(self.layer_norms, self.layers):
            residual = x + residual if residual is not None else x
            x = norm(residual.to(norm.weight.dtype))
            residual = residual.float()
            x = layer(x)
        x = self.norm((x + residual).to(self.norm.weight.dtype))
        return self.lm_head(x)


class EmenderGDN2ForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = EmenderGDN2Config
    base_model_prefix = "model"
    main_input_name = "input_ids"
    _tied_weights_keys = {"model.lm_head.weight": "model.embedding.weight"}

    def __init__(self, config: EmenderGDN2Config):
        super().__init__(config)
        self.model = PortableGDN2LM(config.gdn2_args, config.vocab_size)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embedding

    def set_input_embeddings(self, value):
        self.model.embedding = value

    def get_output_embeddings(self):
        return self.model.lm_head

    def set_output_embeddings(self, value):
        self.model.lm_head = value

    def tie_weights(self, *args, **kwargs):
        del args, kwargs
        if hasattr(self, "model"):
            self.model.lm_head.weight = self.model.embedding.weight

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        del attention_mask, kwargs
        if input_ids is None:
            raise ValueError("input_ids are required")
        logits = self.model(input_ids)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        if return_dict is False:
            return (loss, logits) if loss is not None else (logits,)
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}
