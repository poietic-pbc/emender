"""E97 post-mixer mixture-of-experts conversion.

This module deliberately knows nothing about the E88 construction path.  It
converts the instantiated post-mixer ``SwiGLUMLP`` modules in an E97
``LadderLM`` while leaving mixers, recurrent state, norms, embeddings, and the
LM head untouched.

The first implementation is a dropless single-process reference.  Its expert
numbering and ownership functions are also the contract for the later
node-local all-to-all implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from typing import Iterable, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .ladder_lm import MixerMLPWrapper, SwiGLUMLP


@dataclass(frozen=True)
class E97MoEConfig:
    hidden_dim: int
    routed_experts: int = 64
    shared_experts: int = 1
    top_k: int = 3
    expert_parallel_size: int = 8
    router_init_std: float = 1.0e-3
    load_balance_coefficient: float = 1.0e-3
    z_loss_coefficient: float = 1.0e-4
    # Production expert GEMMs use the same tuned ROCm linear backend as dense
    # E97. Triton remains responsible for recurrence, routing and packing.
    expert_backend: str = "rocblas"

    def validate(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("MoE expert hidden width must be positive")
        if self.shared_experts != 1:
            raise ValueError("the E97 recipe requires exactly one shared expert")
        if not 0 < self.top_k <= self.routed_experts:
            raise ValueError("top_k must be in [1, routed_experts]")
        if self.routed_experts % self.expert_parallel_size:
            raise ValueError("routed experts must divide evenly over expert ranks")
        if self.router_init_std < 0:
            raise ValueError("router_init_std must be nonnegative")
        if self.expert_backend not in {"triton", "rocblas", "grouped"}:
            raise ValueError("expert_backend must be 'triton', 'rocblas', or 'grouped'")


def expert_owner(expert_index: int, *, routed_experts: int = 64,
                 expert_parallel_size: int = 8) -> int:
    """Return the deterministic contiguous owner rank for an expert index."""
    if not 0 <= expert_index < routed_experts:
        raise ValueError(f"expert index {expert_index} is outside [0, {routed_experts})")
    if routed_experts % expert_parallel_size:
        raise ValueError("routed_experts must divide evenly over expert_parallel_size")
    return expert_index // (routed_experts // expert_parallel_size)


def experts_for_rank(rank: int, *, routed_experts: int = 64,
                     expert_parallel_size: int = 8) -> tuple[int, ...]:
    if not 0 <= rank < expert_parallel_size:
        raise ValueError(f"expert rank {rank} is outside [0, {expert_parallel_size})")
    per_rank = routed_experts // expert_parallel_size
    if routed_experts % expert_parallel_size:
        raise ValueError("routed_experts must divide evenly over expert_parallel_size")
    start = rank * per_rank
    return tuple(range(start, start + per_rank))


def _new_expert_like(seed: SwiGLUMLP, hidden_dim: int) -> SwiGLUMLP:
    """Construct a widened expert on the seed parameter device and dtype."""
    if seed.extra_in:
        raise ValueError("the 35B E97 recipe does not support state-summary FFN inputs")
    weight = seed.w1.weight
    with torch.device(weight.device):
        expert = SwiGLUMLP(seed.dim, hidden_dim, dropout=0.0, extra_in=0)
    return expert.to(dtype=weight.dtype)


@torch.no_grad()
def widen_swiglu_function_preserving(
    seed: SwiGLUMLP,
    hidden_dim: int,
    *,
    new_channel_std: float = 0.02,
) -> SwiGLUMLP:
    """Widen ``seed`` without changing its function.

    Trained gate/up rows are copied.  New gate/up rows receive independent
    normal initialization, while every new down-projection column is zero.
    Consequently the added channels contribute exactly zero initially.
    """
    old_hidden = seed.w1.out_features
    if hidden_dim < old_hidden:
        raise ValueError(
            f"target hidden width {hidden_dim} would shrink seed width {old_hidden}")
    if seed.w2.out_features != old_hidden or seed.w3.in_features != old_hidden:
        raise ValueError("seed is not a structurally consistent bias-free SwiGLU expert")
    if any(layer.bias is not None for layer in (seed.w1, seed.w2, seed.w3)):
        raise ValueError("the E97 recipe requires bias-free SwiGLU experts")

    widened = _new_expert_like(seed, hidden_dim)
    widened.w1.weight.zero_()
    widened.w2.weight.zero_()
    widened.w3.weight.zero_()
    widened.w1.weight[:old_hidden].copy_(seed.w1.weight)
    widened.w2.weight[:old_hidden].copy_(seed.w2.weight)
    widened.w3.weight[:, :old_hidden].copy_(seed.w3.weight)
    if hidden_dim > old_hidden:
        nn.init.normal_(widened.w1.weight[old_hidden:], mean=0.0, std=new_channel_std)
        nn.init.normal_(widened.w2.weight[old_hidden:], mean=0.0, std=new_channel_std)
    return widened


class NodeLocalSharedRoutedMoE(nn.Module):
    """Production packed shard for one rank of an eight-GCD E97 MoE island."""

    def __init__(self, dim: int, config: E97MoEConfig, *,
                 local_expert_rank: int, expert_template: SwiGLUMLP,
                 expert_group=None):
        super().__init__()
        config.validate()
        if config.routed_experts != 64 or config.top_k != 3 or config.expert_parallel_size != 8:
            raise ValueError("production E97 MoE requires 64 experts, top-3, and EP size 8")
        if not 0 <= local_expert_rank < 8:
            raise ValueError("local expert rank must be in [0, 8)")
        if expert_template.w1.out_features != config.hidden_dim:
            raise ValueError("expert template hidden width mismatch")
        self.dim = int(dim)
        self.config = config
        self.local_expert_rank = int(local_expert_rank)
        self.expert_group = expert_group
        self.router = nn.Linear(
            dim, 64, bias=False, dtype=torch.float32,
            device=expert_template.w1.weight.device)
        nn.init.normal_(self.router.weight, mean=0.0, std=config.router_init_std)
        self.shared_expert = copy.deepcopy(expert_template)
        self.local_gate_weight = nn.Parameter(
            expert_template.w1.weight.detach().unsqueeze(0).expand(8, -1, -1).clone())
        self.local_up_weight = nn.Parameter(
            expert_template.w2.weight.detach().unsqueeze(0).expand(8, -1, -1).clone())
        self.local_down_weight = nn.Parameter(
            expert_template.w3.weight.detach().unsqueeze(0).expand(8, -1, -1).clone())
        self._load_balance_loss: torch.Tensor | None = None
        self._z_loss: torch.Tensor | None = None
        self.last_metrics: dict[str, torch.Tensor | int | str] = {}
        self._topology = None

    @classmethod
    def from_dense(cls, seed: SwiGLUMLP, config: E97MoEConfig, *,
                   local_expert_rank: int, expert_group=None):
        widened = widen_swiglu_function_preserving(seed, config.hidden_dim)
        with torch.no_grad():
            widened.w3.weight.mul_(0.5)
        return cls(seed.dim, config, local_expert_rank=local_expert_rank,
                   expert_template=widened, expert_group=expert_group)

    @property
    def auxiliary_loss(self) -> torch.Tensor:
        if self._load_balance_loss is None or self._z_loss is None:
            return self.router.weight.new_zeros(())
        return (self.config.load_balance_coefficient * self._load_balance_loss
                + self.config.z_loss_coefficient * self._z_loss)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("production node-local E97 MoE requires ROCm/HIP")
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected input width {self.dim}, got {x.shape[-1]}")
        from ndm.e97_moe_ep import assert_node_local_ep_group, node_local_fused_moe_autograd
        if self._topology is None:
            self._topology = assert_node_local_ep_group(self.expert_group)
        flat = x.reshape(-1, self.dim).contiguous()
        result = node_local_fused_moe_autograd(
            flat, self.router.weight,
            self.local_gate_weight, self.local_up_weight, self.local_down_weight,
            self.shared_expert.w1.weight, self.shared_expert.w2.weight,
            self.shared_expert.w3.weight,
            group=self.expert_group, topology=self._topology,
            expert_backend=self.config.expert_backend)
        self._load_balance_loss = result.load_balance_loss
        self._z_loss = result.z_loss
        self.last_metrics = {
            "expert_token_counts": result.expert_counts.detach(),
            "expert_traffic_scope": "one-node-eight-rank-group-only",
            "hostname": result.topology.hostname,
            "local_expert_rank": self.local_expert_rank,
            "dropped_tokens": 0,
        }
        return result.output.reshape_as(x)


class SharedRoutedMoE(nn.Module):
    """Dropless token-level top-k reference MoE for one E97 FFN.

    Router logits and probability operations are always FP32.  Expert compute
    follows the input/expert dtype.  ``last_metrics`` and the differentiable
    ``load_balance_loss``/``z_loss`` properties describe the latest forward.
    """

    def __init__(self, dim: int, config: E97MoEConfig, *,
                 expert_template: SwiGLUMLP | None = None):
        super().__init__()
        config.validate()
        self.dim = int(dim)
        self.config = config
        if expert_template is None:
            expert_template = SwiGLUMLP(dim, config.hidden_dim)
        if expert_template.w1.out_features != config.hidden_dim:
            raise ValueError("expert template hidden width differs from MoE configuration")
        self.router = nn.Linear(
            dim, config.routed_experts, bias=False, dtype=torch.float32,
            device=expert_template.w1.weight.device,
        )
        nn.init.normal_(self.router.weight, mean=0.0, std=config.router_init_std)

        self.shared_expert = copy.deepcopy(expert_template)
        self.routed_experts = nn.ModuleList(
            copy.deepcopy(expert_template) for _ in range(config.routed_experts)
        )
        self._load_balance_loss: torch.Tensor | None = None
        self._z_loss: torch.Tensor | None = None
        self.last_metrics: dict[str, torch.Tensor | int | float] = {}

    @classmethod
    def from_dense(cls, seed: SwiGLUMLP, config: E97MoEConfig) -> "SharedRoutedMoE":
        """Apply the recipe's widen, clone, and one-half down scaling."""
        widened = widen_swiglu_function_preserving(seed, config.hidden_dim)
        with torch.no_grad():
            widened.w3.weight.mul_(0.5)
        return cls(seed.dim, config, expert_template=widened)

    @property
    def load_balance_loss(self) -> torch.Tensor:
        if self._load_balance_loss is None:
            return self.router.weight.new_zeros(())
        return self._load_balance_loss

    @property
    def z_loss(self) -> torch.Tensor:
        if self._z_loss is None:
            return self.router.weight.new_zeros(())
        return self._z_loss

    @property
    def auxiliary_loss(self) -> torch.Tensor:
        return (self.config.load_balance_coefficient * self.load_balance_loss
                + self.config.z_loss_coefficient * self.z_loss)

    def forward(self, x: torch.Tensor, *,
                forced_topk_indices: torch.Tensor | None = None) -> torch.Tensor:
        if x.is_cuda:
            raise RuntimeError(
                "the Python MoE reference is forbidden on GPU; use the fail-closed "
                "fused Triton E97 MoE path")
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected input width {self.dim}, got {x.shape[-1]}")
        flat_x = x.reshape(-1, self.dim)
        token_count = flat_x.shape[0]

        # The router is an explicit FP32 island even when expert compute runs
        # under BF16 autocast.
        with torch.autocast(device_type=flat_x.device.type, enabled=False):
            logits = F.linear(flat_x.float(), self.router.weight.float())
            probs = torch.softmax(logits, dim=-1)
        if forced_topk_indices is None:
            top_indices = torch.topk(logits, self.config.top_k, dim=-1).indices
        else:
            expected = (*x.shape[:-1], self.config.top_k)
            if tuple(forced_topk_indices.shape) != expected:
                raise ValueError(
                    f"forced top-k shape must be {expected}, got {tuple(forced_topk_indices.shape)}")
            top_indices = forced_topk_indices.reshape(token_count, self.config.top_k).long()
            if top_indices.numel() and (
                int(top_indices.min()) < 0
                or int(top_indices.max()) >= self.config.routed_experts
            ):
                raise ValueError("forced top-k contains an invalid expert index")
            if self.config.top_k > 1:
                ordered = top_indices.sort(dim=-1).values
                if bool((ordered[:, 1:] == ordered[:, :-1]).any()):
                    raise ValueError("forced top-k experts must be distinct per token")
        selected_logits = logits.gather(-1, top_indices)
        top_weights = torch.softmax(selected_logits, dim=-1)

        routed = flat_x.new_zeros(flat_x.shape)
        for expert_index, expert in enumerate(self.routed_experts):
            token_slot = (top_indices == expert_index).nonzero(as_tuple=False)
            if token_slot.numel() == 0:
                continue
            token_indices = token_slot[:, 0]
            route_slots = token_slot[:, 1]
            expert_input = flat_x.index_select(0, token_indices)
            contribution = expert(expert_input)
            weights = top_weights[token_indices, route_slots].to(contribution.dtype).unsqueeze(-1)
            routed = routed.index_add(0, token_indices, contribution * weights)

        # Switch-style loss normalized to 1 for exactly uniform routing.
        counts = torch.bincount(
            top_indices.reshape(-1), minlength=self.config.routed_experts)
        assignment_fraction = counts.float() / max(token_count * self.config.top_k, 1)
        probability_fraction = probs.mean(dim=0)
        self._load_balance_loss = (
            self.config.routed_experts
            * torch.sum(assignment_fraction * probability_fraction)
        )
        self._z_loss = torch.logsumexp(logits, dim=-1).square().mean()

        one_hot = F.one_hot(top_indices, self.config.routed_experts).sum(dim=1).float()
        co_selection = one_hot.transpose(0, 1) @ one_hot
        entropy = -(probs * probs.clamp_min(torch.finfo(probs.dtype).tiny).log()).sum(-1)
        self.last_metrics = {
            "expert_token_counts": counts.detach(),
            "expert_token_fractions": assignment_fraction.detach(),
            "router_logits_dtype": logits.dtype,
            "router_entropy": entropy.mean().detach(),
            "router_max_probability": probs.max(dim=-1).values.mean().detach(),
            "selected_weight_sum_max_error": (
                top_weights.sum(dim=-1) - 1.0).abs().max().detach(),
            "expert_co_selection": co_selection.detach(),
            "dropped_tokens": 0,
            "token_count": token_count,
        }

        shared = self.shared_expert(flat_x)
        return (shared + routed).reshape_as(x)


@dataclass(frozen=True)
class E97MoEParameterRecipe:
    model_width: int
    layers: int
    seed_expert_hidden: int
    shared_non_moe_parameters: int
    routed_experts: int
    shared_experts: int
    top_k: int
    expert_hidden: int
    router_parameters: int
    total_parameters: int
    active_parameters: int

    def to_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


def _dense_ffns(model: nn.Module) -> list[SwiGLUMLP]:
    layers = getattr(model, "layers", None)
    if layers is None or not layers:
        raise ValueError("model has no recurrent layers")
    ffns: list[SwiGLUMLP] = []
    for index, layer in enumerate(layers):
        if not isinstance(layer, MixerMLPWrapper) or not isinstance(layer.mlp, SwiGLUMLP):
            raise ValueError(f"layer {index} is not an E97 post-mixer SwiGLU layer")
        ffns.append(layer.mlp)
    signatures = {
        (ffn.dim, ffn.extra_in, ffn.w1.out_features,
         tuple(ffn.w1.weight.shape), tuple(ffn.w2.weight.shape),
         tuple(ffn.w3.weight.shape))
        for ffn in ffns
    }
    if len(signatures) != 1:
        raise ValueError(f"E97 FFN layers differ unexpectedly: {sorted(signatures)}")
    return ffns


def calculate_e97_moe_recipe(
    model: nn.Module,
    *,
    target_parameters: int = 35_000_000_000,
    routed_experts: int = 64,
    shared_experts: int = 1,
    top_k: int = 3,
    multiple: int = 128,
) -> E97MoEParameterRecipe:
    """Solve width from the actual instantiated dense E97 parameter graph."""
    if multiple <= 0:
        raise ValueError("expert width multiple must be positive")
    ffns = _dense_ffns(model)
    dim = ffns[0].dim
    layers = len(ffns)
    seed_hidden = ffns[0].w1.out_features
    ffn_parameter_ids = {id(parameter) for ffn in ffns for parameter in ffn.parameters()}
    shared_non_moe = sum(
        parameter.numel() for parameter in model.parameters()
        if id(parameter) not in ffn_parameter_ids
    )
    router_parameters = layers * dim * routed_experts
    expert_factor = layers * (routed_experts + shared_experts) * 3 * dim
    ideal = (target_parameters - shared_non_moe - router_parameters) / expert_factor
    expert_hidden = max(multiple, int(math.floor(ideal / multiple + 0.5)) * multiple)
    total = shared_non_moe + router_parameters + expert_factor * expert_hidden
    active = (shared_non_moe + router_parameters
              + layers * (shared_experts + top_k) * 3 * dim * expert_hidden)
    return E97MoEParameterRecipe(
        model_width=dim,
        layers=layers,
        seed_expert_hidden=seed_hidden,
        shared_non_moe_parameters=shared_non_moe,
        routed_experts=routed_experts,
        shared_experts=shared_experts,
        top_k=top_k,
        expert_hidden=expert_hidden,
        router_parameters=router_parameters,
        total_parameters=total,
        active_parameters=active,
    )


def convert_e97_ffns_to_node_local_moe(
    model: nn.Module,
    config: E97MoEConfig,
    *,
    local_expert_rank: int,
    expert_group=None,
) -> nn.Module:
    """Replace only E97 FFNs with packed eight-expert production shards."""
    ffns = _dense_ffns(model)
    for layer, seed_ffn in zip(model.layers, ffns):
        layer.mlp = NodeLocalSharedRoutedMoE.from_dense(
            seed_ffn, config, local_expert_rank=local_expert_rank,
            expert_group=expert_group)
    return model


def convert_e97_ffns_to_moe(model: nn.Module, config: E97MoEConfig) -> nn.Module:
    """Replace only the instantiated post-mixer FFNs, in place."""
    ffns = _dense_ffns(model)  # validates the complete graph before mutation
    for layer, seed_ffn in zip(model.layers, ffns):
        layer.mlp = SharedRoutedMoE.from_dense(seed_ffn, config)
    return model


def iter_e97_moe_layers(model: nn.Module) -> Iterable[SharedRoutedMoE | NodeLocalSharedRoutedMoE]:
    for layer in getattr(model, "layers", ()):
        moe = getattr(layer, "mlp", None)
        if isinstance(moe, (SharedRoutedMoE, NodeLocalSharedRoutedMoE)):
            yield moe


def e97_moe_auxiliary_loss(model: nn.Module) -> torch.Tensor:
    layers = list(iter_e97_moe_layers(model))
    if not layers:
        parameter = next(model.parameters(), None)
        return torch.tensor(0.0, device=parameter.device if parameter is not None else None)
    return torch.stack([layer.auxiliary_loss for layer in layers]).sum()
