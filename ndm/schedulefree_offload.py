"""Pinned-CPU state storage for Schedule-Free AdamW.

The live model, gradients, and arithmetic stay on the parameter device.  The
large Schedule-Free tensors (``z`` and ``exp_avg_sq``) live in pinned host
memory between uses and are streamed through fixed-size accelerator buckets.
This keeps peak accelerator memory bounded by the staging size rather than by
the whole optimizer state.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from itertools import chain
import time
from typing import Any, Callable, Iterable, Optional, Tuple, Union

import torch


class CPUOffloadAdamWScheduleFree(torch.optim.Optimizer):
    """Schedule-Free AdamW with pinned-CPU ``z`` and second moments.

    Parameters and gradients remain on their original device.  Optimizer state
    is initialized directly on the CPU, including the first step, so a fresh
    large model never transiently materializes the complete state in HBM.

    The implementation uses one bounded reusable staging bucket per
    device/dtype. Copies and elementwise updates are ordered on the accelerator
    stream and synchronized once per step, avoiding a host/device fence for
    every parameter while retaining a fixed HBM bound.
    """

    state_storage = "pinned-cpu"
    state_schema = "emender-schedulefree-cpu-offload-v1"

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: Union[float, torch.Tensor] = 0.0025,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1.0e-8,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        *,
        pin_memory: bool = True,
        release_gradients: bool = True,
        bucket_numel: int = 67_108_864,
    ) -> None:
        if float(lr) < 0 or eps < 0 or weight_decay < 0 or warmup_steps < 0:
            raise ValueError("invalid Schedule-Free CPU-offload hyperparameters")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("Schedule-Free betas must be in [0, 1)")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=float(eps),
            r=float(r),
            k=0,
            warmup_steps=int(warmup_steps),
            train_mode=False,
            weight_sum=0.0,
            lr_max=-1.0,
            scheduled_lr=0.0,
            weight_lr_power=float(weight_lr_power),
            weight_decay=float(weight_decay),
            foreach=False,
            state_storage=self.state_storage,
            state_schema=self.state_schema,
        )
        super().__init__(params, defaults)
        if int(bucket_numel) <= 0:
            raise ValueError("Schedule-Free CPU-offload bucket_numel must be positive")
        self.pin_memory = bool(pin_memory)
        self.release_gradients = bool(release_gradients)
        self.bucket_numel = int(bucket_numel)
        self.last_step_stats: dict[str, float | int] = {}

    def _new_host_tensor(self, parameter: torch.Tensor, *, source=None, zero=False):
        if parameter.layout != torch.strided:
            raise RuntimeError("Schedule-Free CPU offload supports strided parameters only")
        host = torch.empty_like(
            parameter,
            device="cpu",
            memory_format=torch.preserve_format,
            pin_memory=self.pin_memory,
        )
        if zero:
            host.zero_()
        elif source is not None:
            host.copy_(
                source.detach(),
                non_blocking=(self.pin_memory and source.device.type == "cuda"),
            )
        return host

    def _host_copy_for_parameter(self, parameter: torch.Tensor, source: torch.Tensor):
        if tuple(source.shape) != tuple(parameter.shape):
            raise ValueError(
                f"optimizer state shape {tuple(source.shape)} does not match "
                f"parameter shape {tuple(parameter.shape)}"
            )
        host = torch.empty_like(
            parameter,
            device="cpu",
            memory_format=torch.preserve_format,
            pin_memory=self.pin_memory,
        )
        host.copy_(source.detach().to(device="cpu", dtype=parameter.dtype),
                   non_blocking=False)
        return host

    @staticmethod
    def _work_buckets(parameters, bucket_numel):
        """Yield bounded flat slices without constructing a full-model flat tensor."""
        parameters = list(parameters)
        parameter_index = 0
        parameter_offset = 0
        while parameter_index < len(parameters):
            filled = 0
            work = []
            while filled < bucket_numel and parameter_index < len(parameters):
                parameter = parameters[parameter_index]
                available = int(parameter.numel()) - parameter_offset
                count = min(bucket_numel - filled, available)
                final_slice = parameter_offset + count == parameter.numel()
                work.append((
                    parameter, parameter_offset, filled, count, final_slice))
                filled += count
                parameter_offset += count
                if final_slice:
                    parameter_index += 1
                    parameter_offset = 0
            yield filled, work

    def _initialize_state(self, parameter: torch.Tensor) -> dict[str, torch.Tensor]:
        state = self.state[parameter]
        if "z" not in state:
            # Allocate on the host from the beginning.  Initializing these with
            # torch.clone(parameter) would make an 8B first step exceed HBM.
            state["z"] = self._new_host_tensor(parameter, source=parameter)
            state["exp_avg_sq"] = self._new_host_tensor(parameter, zero=True)
        return state

    @torch.no_grad()
    def initialize_state_(self) -> dict[str, float | int]:
        """Preallocate all host state before the first forward/backward step."""
        started = time.perf_counter()
        initialized_bytes = 0
        synchronize_devices = set()
        for group in self.param_groups:
            for parameter in group["params"]:
                if "z" in self.state.get(parameter, {}):
                    continue
                self._initialize_state(parameter)
                initialized_bytes += 2 * parameter.numel() * parameter.element_size()
                if parameter.device.type == "cuda":
                    synchronize_devices.add(parameter.device)
        for device in synchronize_devices:
            torch.cuda.synchronize(device)
        self.assert_state_offloaded()
        return {
            "seconds": time.perf_counter() - started,
            "state_bytes": initialized_bytes,
        }

    @torch.no_grad()
    def offload_state_(self) -> None:
        """Ensure every initialized tensor state is pinned CPU storage."""
        for group in self.param_groups:
            for parameter in group["params"]:
                state = self.state.get(parameter, {})
                for name in ("z", "exp_avg_sq"):
                    tensor = state.get(name)
                    if tensor is None:
                        continue
                    if tensor.device.type != "cpu" or (
                            self.pin_memory and not tensor.is_pinned()):
                        state[name] = self._host_copy_for_parameter(parameter, tensor)

    def offloaded_state_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for parameter, state in self.state.items()
            if isinstance(parameter, torch.Tensor)
            for name in ("z", "exp_avg_sq")
            for tensor in (state.get(name),)
            if tensor is not None
        )

    def assert_state_offloaded(self) -> None:
        for parameter, state in self.state.items():
            if not isinstance(parameter, torch.Tensor):
                continue
            for name in ("z", "exp_avg_sq"):
                tensor = state.get(name)
                if tensor is None:
                    continue
                if tensor.device.type != "cpu":
                    raise RuntimeError(f"optimizer state {name} is not CPU-offloaded")
                if self.pin_memory and not tensor.is_pinned():
                    raise RuntimeError(f"optimizer state {name} is not pinned")
                if tensor.dtype != parameter.dtype:
                    raise RuntimeError(
                        f"optimizer state {name} dtype {tensor.dtype} does not match "
                        f"parameter dtype {parameter.dtype}"
                    )

    @torch.no_grad()
    def _basis_lerp_(self, parameters, weight: float) -> None:
        by_device_dtype = defaultdict(list)
        for parameter in parameters:
            if "z" in self.state.get(parameter, {}):
                if not parameter.is_contiguous():
                    raise RuntimeError("Schedule-Free parameters must be contiguous")
                by_device_dtype[(parameter.device, parameter.dtype)].append(parameter)

        synchronize_devices = set()
        for (device, dtype), device_parameters in by_device_dtype.items():
            total = sum(parameter.numel() for parameter in device_parameters)
            bucket_numel = min(self.bucket_numel, total)
            staging = torch.empty(bucket_numel, dtype=dtype, device=device)
            async_copy = self.pin_memory and device.type == "cuda"
            if device.type == "cuda":
                synchronize_devices.add(device)
            for _filled, work in self._work_buckets(
                    device_parameters, bucket_numel):
                for parameter, parameter_offset, bucket_offset, count, _ in work:
                    staging.narrow(0, bucket_offset, count).copy_(
                        self.state[parameter]["z"].view(-1).narrow(
                            0, parameter_offset, count),
                        non_blocking=async_copy,
                    )
                for parameter, parameter_offset, bucket_offset, count, _ in work:
                    parameter.view(-1).narrow(0, parameter_offset, count).lerp_(
                        staging.narrow(0, bucket_offset, count), weight=weight)
            del staging
        for device in synchronize_devices:
            torch.cuda.synchronize(device)

    @torch.no_grad()
    def eval(self) -> None:
        for group in self.param_groups:
            if not group["train_mode"]:
                continue
            beta1 = group["betas"][0]
            self._basis_lerp_(group["params"], 1.0 - 1.0 / beta1)
            group["train_mode"] = False

    @torch.no_grad()
    def train(self) -> None:
        for group in self.param_groups:
            if group["train_mode"]:
                continue
            beta1 = group["betas"][0]
            self._basis_lerp_(group["params"], 1.0 - beta1)
            group["train_mode"] = True

    @torch.no_grad()
    def step(
        self,
        closure: Optional[Callable[[], float]] = None,
    ) -> Optional[float]:
        if not self.param_groups[0]["train_mode"]:
            raise RuntimeError(
                "optimizer was not in train mode; call optimizer.train() before step()"
            )
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        started = time.perf_counter()
        h2d_bytes = 0
        d2h_bytes = 0
        updated_tensors = 0
        synchronize_devices = set()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            k = int(group["k"])
            warmup_steps = int(group["warmup_steps"])
            schedule = ((k + 1) / warmup_steps
                        if k < warmup_steps and warmup_steps > 0 else 1.0)
            lr = group["lr"] * schedule
            group["scheduled_lr"] = lr
            group["lr_max"] = max(lr, group["lr_max"])
            weight = ((k + 1) ** group["r"]) * (
                group["lr_max"] ** group["weight_lr_power"])
            group["weight_sum"] += weight
            ckp1 = weight / group["weight_sum"] if group["weight_sum"] else 0.0
            bias_correction2 = 1.0 - beta2 ** (k + 1)

            by_device_dtype = defaultdict(list)
            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("Schedule-Free CPU offload forbids sparse gradients")
                if grad.device != parameter.device or grad.shape != parameter.shape:
                    raise RuntimeError("gradient device/shape must match its parameter")
                if grad.dtype != parameter.dtype:
                    raise RuntimeError("gradient dtype must match its parameter")
                if not parameter.is_contiguous() or not grad.is_contiguous():
                    raise RuntimeError(
                        "Schedule-Free parameters and gradients must be contiguous")
                self._initialize_state(parameter)
                by_device_dtype[(parameter.device, parameter.dtype)].append(parameter)

            for (device, dtype), device_parameters in by_device_dtype.items():
                total = sum(parameter.numel() for parameter in device_parameters)
                bucket_numel = min(self.bucket_numel, total)
                z_staging = torch.empty(bucket_numel, dtype=dtype, device=device)
                exp_staging = torch.empty(bucket_numel, dtype=dtype, device=device)
                async_copy = self.pin_memory and device.type == "cuda"
                if device.type == "cuda":
                    synchronize_devices.add(device)

                for _filled, work in self._work_buckets(
                        device_parameters, bucket_numel):
                    for parameter, parameter_offset, bucket_offset, count, _ in work:
                        state = self.state[parameter]
                        z_staging.narrow(0, bucket_offset, count).copy_(
                            state["z"].view(-1).narrow(
                                0, parameter_offset, count),
                            non_blocking=async_copy,
                        )
                        exp_staging.narrow(0, bucket_offset, count).copy_(
                            state["exp_avg_sq"].view(-1).narrow(
                                0, parameter_offset, count),
                            non_blocking=async_copy,
                        )
                        if device.type != "cpu":
                            h2d_bytes += 2 * count * parameter.element_size()

                    for parameter, parameter_offset, bucket_offset, count, final_slice in work:
                        parameter_work = parameter.view(-1).narrow(
                            0, parameter_offset, count)
                        grad_work = parameter.grad.view(-1).narrow(
                            0, parameter_offset, count)
                        z_work = z_staging.narrow(0, bucket_offset, count)
                        exp_work = exp_staging.narrow(0, bucket_offset, count)
                        exp_work.mul_(beta2).addcmul_(
                            grad_work, grad_work, value=1.0 - beta2)
                        denom = exp_work.div(bias_correction2).sqrt_().add_(
                            group["eps"])
                        grad_work.div_(denom)
                        if group["weight_decay"] != 0:
                            grad_work.add_(
                                parameter_work, alpha=group["weight_decay"])
                        parameter_work.lerp_(end=z_work, weight=ckp1)
                        parameter_work.add_(
                            grad_work,
                            alpha=lr * (beta1 * (1.0 - ckp1) - 1.0),
                        )
                        z_work.sub_(grad_work, alpha=lr)
                        state = self.state[parameter]
                        state["z"].view(-1).narrow(
                            0, parameter_offset, count).copy_(
                                z_work, non_blocking=async_copy)
                        state["exp_avg_sq"].view(-1).narrow(
                            0, parameter_offset, count).copy_(
                                exp_work, non_blocking=async_copy)
                        if device.type != "cpu":
                            d2h_bytes += 2 * count * parameter.element_size()
                        if final_slice:
                            if self.release_gradients:
                                parameter.grad = None
                            updated_tensors += 1
                        del parameter_work, grad_work, z_work, exp_work, denom
                del z_staging, exp_staging

            group["k"] = k + 1

        for device in synchronize_devices:
            torch.cuda.synchronize(device)
        self.assert_state_offloaded()
        self.last_step_stats = {
            "seconds": time.perf_counter() - started,
            "h2d_bytes": h2d_bytes,
            "d2h_bytes": d2h_bytes,
            "updated_tensors": updated_tensors,
        }
        return loss

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load without PyTorch's normal tensor-to-parameter-device cast.

        ``Optimizer.load_state_dict`` normally moves every tensor state to the
        parameter device.  For an 8B checkpoint that would materialize roughly
        30 GiB in HBM before it could be offloaded, defeating restart safety.
        """
        incoming = state_dict.copy()
        for hook in self._optimizer_load_state_dict_pre_hooks.values():
            result = hook(self, incoming)
            if result is not None:
                incoming = result

        groups = self.param_groups
        saved_groups = deepcopy(incoming["param_groups"])
        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of parameter groups")
        if any(len(group["params"]) != len(saved["params"])
               for group, saved in zip(groups, saved_groups)):
            raise ValueError(
                "loaded state dict contains a parameter group that does not match"
            )

        id_map = dict(zip(
            chain.from_iterable(group["params"] for group in saved_groups),
            chain.from_iterable(group["params"] for group in groups),
        ))
        restored = defaultdict(dict)
        for key, value in incoming["state"].items():
            if key not in id_map:
                restored[key] = deepcopy(value)
                continue
            parameter = id_map[key]
            if not isinstance(value, dict):
                raise ValueError("per-parameter optimizer state must be a mapping")
            item = {}
            for name, state_value in value.items():
                if isinstance(state_value, torch.Tensor):
                    if name not in ("z", "exp_avg_sq"):
                        raise ValueError(
                            f"unsupported tensor optimizer state {name!r} in CPU-offload restore"
                        )
                    item[name] = self._host_copy_for_parameter(parameter, state_value)
                else:
                    item[name] = deepcopy(state_value)
            if ("z" in item) != ("exp_avg_sq" in item):
                raise ValueError("Schedule-Free checkpoint must contain both z and exp_avg_sq")
            restored[parameter] = item

        updated_groups = []
        for current, saved in zip(groups, saved_groups):
            saved["params"] = current["params"]
            if "param_names" in current and "param_names" not in saved:
                saved["param_names"] = current["param_names"]
            saved["foreach"] = False
            saved["state_storage"] = self.state_storage
            saved["state_schema"] = self.state_schema
            updated_groups.append(saved)
        self.__setstate__({"state": restored, "param_groups": updated_groups})
        self.assert_state_offloaded()

        for hook in self._optimizer_load_state_dict_post_hooks.values():
            hook(self)
