"""Fail-closed fused ScheduleFree optimizer for E97 MoE training."""
from __future__ import annotations

from typing import Iterable

import torch

from .triton.e97_moe_fused import (
    fused_schedulefree_adamw_update_,
    fused_schedulefree_lerp_,
)


class FusedScheduleFreeAdamW(torch.optim.Optimizer):
    """ScheduleFree AdamW using only fused Triton tensor updates.

    ``z`` and ``exp_avg_sq`` use each parameter's own BF16/FP32 dtype. There is
    no FP32 master-weight copy and no eager/foreach GPU update fallback.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float = 2.5e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1.0e-8,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
    ):
        if lr < 0 or eps < 0 or weight_decay < 0 or warmup_steps < 0:
            raise ValueError("invalid fused ScheduleFree hyperparameters")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("ScheduleFree betas must be in [0, 1)")
        defaults = dict(
            lr=float(lr), betas=betas, eps=float(eps),
            weight_decay=float(weight_decay), warmup_steps=int(warmup_steps),
            r=float(r), weight_lr_power=float(weight_lr_power),
            k=0, train_mode=False, weight_sum=0.0, lr_max=-1.0,
            scheduled_lr=0.0,
        )
        super().__init__(params, defaults)
        self.z_offloaded = False

    @staticmethod
    def _materialize_z(parameter, state):
        z = state["z"]
        return z if z.is_cuda else z.to(parameter.device, non_blocking=False)

    @staticmethod
    def _commit_z(state, z_work) -> None:
        if state["z"] is not z_work:
            state["z"].copy_(z_work, non_blocking=False)

    @torch.no_grad()
    def offload_z_(self) -> None:
        """Move ScheduleFree z to pinned host memory between GPU uses."""
        for parameter in self.param_groups[0]["params"]:
            state = self.state.get(parameter, {})
            z = state.get("z")
            if z is None or not z.is_cuda:
                continue
            host = torch.empty(
                z.shape, dtype=z.dtype, device="cpu", pin_memory=True)
            host.copy_(z, non_blocking=False)
            state["z"] = host
        self.z_offloaded = True
        torch.cuda.empty_cache()

    @torch.no_grad()
    def train(self) -> None:
        for group in self.param_groups:
            if group["train_mode"]:
                continue
            beta1 = group["betas"][0]
            for parameter in group["params"]:
                state = self.state[parameter]
                if "z" in state:
                    z_work = self._materialize_z(parameter, state)
                    fused_schedulefree_lerp_(
                        parameter, z_work, weight=1.0 - beta1)
                    self._commit_z(state, z_work)
            group["train_mode"] = True

    @torch.no_grad()
    def eval(self) -> None:
        for group in self.param_groups:
            if not group["train_mode"]:
                continue
            beta1 = group["betas"][0]
            for parameter in group["params"]:
                state = self.state[parameter]
                if "z" in state:
                    z_work = self._materialize_z(parameter, state)
                    fused_schedulefree_lerp_(
                        parameter, z_work, weight=1.0 - 1.0 / beta1)
                    self._commit_z(state, z_work)
            group["train_mode"] = False

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise RuntimeError("fused E97 ScheduleFree forbids eager closure execution")
        for group in self.param_groups:
            if not group["train_mode"]:
                raise RuntimeError("call fused ScheduleFree train() before step()")
            beta1, beta2 = group["betas"]
            k = group["k"]
            warmup = group["warmup_steps"]
            schedule = (k + 1) / warmup if warmup and k < warmup else 1.0
            lr = group["lr"] * schedule
            group["scheduled_lr"] = lr
            group["lr_max"] = max(lr, group["lr_max"])
            weight = ((k + 1) ** group["r"]) * (
                group["lr_max"] ** group["weight_lr_power"])
            group["weight_sum"] += weight
            ckp1 = weight / group["weight_sum"] if group["weight_sum"] else 0.0
            bias_correction2 = 1.0 - beta2 ** (k + 1)
            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("fused E97 ScheduleFree forbids sparse gradients")
                if (not parameter.is_cuda or parameter.dtype not in
                        (torch.bfloat16, torch.float32) or not parameter.is_contiguous()):
                    raise RuntimeError("fused ScheduleFree requires contiguous GPU BF16/FP32 parameters")
                if grad.dtype != parameter.dtype or not grad.is_contiguous():
                    raise RuntimeError("gradient dtype/layout must match its parameter")
                state = self.state[parameter]
                if "z" not in state:
                    state["z"] = parameter.detach().clone()
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                z_work = self._materialize_z(parameter, state)
                fused_schedulefree_adamw_update_(
                    parameter, grad, z_work, state["exp_avg_sq"],
                    lr=lr, beta1=beta1, beta2=beta2,
                    bias_correction2=bias_correction2, eps=group["eps"],
                    weight_decay=group["weight_decay"], ckp1=ckp1)
                self._commit_z(state, z_work)
            group["k"] = k + 1
        if self.z_offloaded:
            self.offload_z_()
        return None

    def assert_no_master_weights(self) -> None:
        for parameter, state in self.state.items():
            for name in ("z", "exp_avg_sq"):
                tensor = state.get(name)
                if tensor is not None and tensor.dtype != parameter.dtype:
                    raise RuntimeError(f"optimizer state {name} is a forbidden master dtype")
