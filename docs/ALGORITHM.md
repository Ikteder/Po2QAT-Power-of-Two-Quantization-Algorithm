# Algorithm note: signed power-of-two QAT

## Scope

This repository implements weight-only, signed power-of-two quantization-aware training for every `torch.nn.Conv2d` and `torch.nn.Linear` module. Biases, normalization parameters, positional parameters, and embedding lookup tables remain floating point. The export includes them, but labels them as unquantized.

The name **Po2QAT** in this repository is equivalent to the **PoT-QAT** terminology used in the cited literature.

## Codebook

For a bit width `b`, the number of positive levels is

```text
L = (2^b - 1) // 2.
```

For example, `b=4` gives seven positive levels, seven negative levels, and zero: 15 total values.

For each eligible tensor `W`, the exponent window is selected once, immediately before QAT:

```text
emax = ceil(log2(max(abs(W))))
emin = emax - L + 1.
```

The window stays fixed during QAT. A scalar weight `w` is projected as follows:

```text
Q(w) = 0                                      if |w| < 2^emin / 2
Q(w) = sign(w) * 2^clip(round(log2|w|),emin,emax) otherwise.
```

Therefore every nonzero projected weight is an exact signed integer power of two.

## Straight-through estimator

Rounding and exponent selection are not differentiable. The training forward pass uses:

```text
Wfake = W + stop_gradient(Q(W) - W).
```

Numerically, `Wfake == Q(W)` in the forward pass. In the backward pass, `dWfake/dW = 1`, which is the straight-through estimator. The optimizer updates the floating-point master tensor `W` while every training forward pass experiences the Po2 projection.

## Export semantics

- `initial_fp32.pt`: trained baseline just before wrappers are inserted;
- `qat_master_fp32.pt`: floating-point master tensors after QAT updates;
- `po2_quantized.pt`: the materialized `Q(W)` tensors used by QAT forward passes;
- `po2_sign_exponent.npz`: exact `sign * 2^exponent` encoding for eligible tensors.

This separation is important: the master checkpoint is not a deployable Po2 checkpoint until it is projected.

## Differences from a production deployment

- Activations are not quantized.
- PyTorch still evaluates materialized Po2 values with ordinary floating-point convolution/matrix multiplication.
- No FPGA, ASIC, CPU bit-shift kernel, or packed four-bit runtime is included.
- Exponent storage in the NPZ is intentionally simple and inspectable, not bit-packed to its theoretical minimum.
- The small models and short classroom schedules do not reproduce the headline metrics of the papers.

## Primary references

- Dominika Przewlocka-Rus et al., *Power-of-Two Quantization for Low Bitwidth and Hardware Compliant Neural Networks*, arXiv:2203.05025 (2022).
- Mahmoud Elgenedy, *Power-of-Two Quantization-Aware-Training (PoT-QAT) in Large Language Models (LLMs)*, arXiv:2601.02298 (2026).

