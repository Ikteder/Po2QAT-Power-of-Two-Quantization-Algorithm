# Decision 0002: exact tensor-level Po2 codebook

Date: 2026-08-19  
Status: accepted

## Decision

Use a fixed per-tensor exponent window and exact levels `0` or `±2^e`. Apply weight-only fake quantization to every convolution and linear layer with an STE backward pass.

## Rationale

Students can verify every nonzero exported value with `log2`, and the sign/exponent file reconstructs weights without an additional floating-point scale. Freezing each codebook before QAT makes the optimization target stable and makes before/after comparisons easier to explain.

## Tradeoffs

Per-channel scaling or learned scales may improve quality. Activation quantization would give a more complete integer pipeline. Both would add conceptual and implementation complexity that is outside this introductory lab.

