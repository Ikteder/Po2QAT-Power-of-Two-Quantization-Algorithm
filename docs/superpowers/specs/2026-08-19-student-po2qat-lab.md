# Student Po2QAT lab specification

Date: 2026-08-19  
Status: implemented from instructor request

## Objective

Deliver one cloneable repository in which students on Windows and macOS can apply the same power-of-two QAT algorithm to a small CNN, a small ViT, and a small language model, then inspect pre-QAT, post-QAT master, and final Po2 weights.

## Acceptance criteria

- One command per model and one command for all models.
- A no-download smoke profile and dataset-backed quick/full profiles.
- Exact signed-Po2 export verification.
- Full state dictionaries for initial, post-QAT master, and materialized Po2 states.
- Human-readable weight comparison and summary tables.
- Packed sign/exponent arrays.
- Windows and macOS setup instructions.
- Automated tests on Windows, macOS, and Linux.
- Documentation of datasets, models, algorithm choices, limitations, and experiments.

## Non-goals

- Production low-bit kernels.
- Activation quantization.
- Reproducing full ImageNet or GPT-2 paper results.
- Claiming speed or memory gains that were not measured by this project.

