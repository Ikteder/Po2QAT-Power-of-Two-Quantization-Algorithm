from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ExponentRange:
    minimum: int
    maximum: int

    @property
    def positive_levels(self) -> int:
        return self.maximum - self.minimum + 1


def positive_levels(num_bits: int) -> int:
    """Number of positive values in a signed Po2 codebook that also contains zero."""
    if num_bits < 2:
        raise ValueError("num_bits must be at least 2")
    return (2**num_bits - 1) // 2


def choose_exponent_range(weight: torch.Tensor, num_bits: int = 4) -> ExponentRange:
    """Choose a fixed per-tensor exponent window from the pre-QAT weights.

    Four bits gives 15 values: zero plus seven positive and seven negative powers
    of two. The window is frozen during QAT so the simulated deployment codebook
    does not move from step to step.
    """
    levels = positive_levels(num_bits)
    max_abs = float(weight.detach().abs().max().item())
    exp_max = 0 if max_abs == 0.0 else math.ceil(math.log2(max_abs))
    return ExponentRange(exp_max - levels + 1, exp_max)


def quantize_po2(weight: torch.Tensor, exp_min: int, exp_max: int) -> torch.Tensor:
    """Project values onto {0, +/-2**exp_min, ..., +/-2**exp_max}."""
    if exp_min > exp_max:
        raise ValueError("exp_min must not exceed exp_max")
    abs_weight = weight.abs()
    smallest = torch.tensor(2.0**exp_min, dtype=weight.dtype, device=weight.device)
    # Values below half the smallest level map to zero; the rest map to the
    # nearest base-2 exponent. This gives a useful zero code and exact Po2 values.
    nonzero = abs_weight >= smallest / 2
    safe = torch.clamp(abs_weight, min=torch.finfo(weight.dtype).tiny)
    exponents = torch.round(torch.log2(safe)).clamp(exp_min, exp_max)
    projected = torch.sign(weight) * torch.pow(torch.tensor(2.0, dtype=weight.dtype, device=weight.device), exponents)
    return torch.where(nonzero, projected, torch.zeros_like(projected))


def fake_quantize_po2(weight: torch.Tensor, exp_min: int, exp_max: int) -> torch.Tensor:
    """Po2 forward pass with a straight-through-estimator backward pass."""
    projected = quantize_po2(weight, exp_min, exp_max)
    return weight + (projected - weight).detach()


class Po2Linear(nn.Module):
    def __init__(self, source: nn.Linear, num_bits: int = 4) -> None:
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.weight_fp = nn.Parameter(source.weight.detach().clone())
        self.bias = nn.Parameter(source.bias.detach().clone()) if source.bias is not None else None
        window = choose_exponent_range(self.weight_fp, num_bits)
        self.exp_min = window.minimum
        self.exp_max = window.maximum
        self.num_bits = num_bits

    def quantized_weight(self) -> torch.Tensor:
        return quantize_po2(self.weight_fp, self.exp_min, self.exp_max)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, fake_quantize_po2(self.weight_fp, self.exp_min, self.exp_max), self.bias)


class Po2Conv2d(nn.Module):
    def __init__(self, source: nn.Conv2d, num_bits: int = 4) -> None:
        super().__init__()
        self.in_channels = source.in_channels
        self.out_channels = source.out_channels
        self.kernel_size = source.kernel_size
        self.stride = source.stride
        self.padding = source.padding
        self.dilation = source.dilation
        self.groups = source.groups
        self.padding_mode = source.padding_mode
        self.weight_fp = nn.Parameter(source.weight.detach().clone())
        self.bias = nn.Parameter(source.bias.detach().clone()) if source.bias is not None else None
        window = choose_exponent_range(self.weight_fp, num_bits)
        self.exp_min = window.minimum
        self.exp_max = window.maximum
        self.num_bits = num_bits

    def quantized_weight(self) -> torch.Tensor:
        return quantize_po2(self.weight_fp, self.exp_min, self.exp_max)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = fake_quantize_po2(self.weight_fp, self.exp_min, self.exp_max)
        return F.conv2d(inputs, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def _set_module(root: nn.Module, path: str, module: nn.Module) -> None:
    parent = root
    parts = path.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def prepare_po2_qat(model: nn.Module, num_bits: int = 4) -> tuple[nn.Module, list[str]]:
    """Return a copied model whose Linear and Conv2d weights use Po2 fake quantization."""
    prepared = copy.deepcopy(model)
    names: list[str] = []
    for name, module in list(prepared.named_modules()):
        if not name:
            continue
        if isinstance(module, nn.Linear):
            _set_module(prepared, name, Po2Linear(module, num_bits))
            names.append(name)
        elif isinstance(module, nn.Conv2d):
            _set_module(prepared, name, Po2Conv2d(module, num_bits))
            names.append(name)
    if not names:
        raise ValueError("Model has no Linear or Conv2d modules to quantize")
    return prepared, names


def materialize_model(model: nn.Module, quantized: bool) -> nn.Module:
    """Convert QAT wrappers back to ordinary PyTorch modules for portable checkpoints."""
    result = copy.deepcopy(model).cpu()
    for name, module in list(result.named_modules()):
        if isinstance(module, Po2Linear):
            native = nn.Linear(module.in_features, module.out_features, bias=module.bias is not None)
            source_weight = module.quantized_weight() if quantized else module.weight_fp
            native.weight.data.copy_(source_weight.detach())
            if module.bias is not None:
                native.bias.data.copy_(module.bias.detach())
            _set_module(result, name, native)
        elif isinstance(module, Po2Conv2d):
            native = nn.Conv2d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
                module.bias is not None,
                module.padding_mode,
            )
            source_weight = module.quantized_weight() if quantized else module.weight_fp
            native.weight.data.copy_(source_weight.detach())
            if module.bias is not None:
                native.bias.data.copy_(module.bias.detach())
            _set_module(result, name, native)
    return result


def quantized_module_metadata(model: nn.Module) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for name, module in model.named_modules():
        if isinstance(module, (Po2Linear, Po2Conv2d)):
            result[name] = {"num_bits": module.num_bits, "exp_min": module.exp_min, "exp_max": module.exp_max}
    return result

