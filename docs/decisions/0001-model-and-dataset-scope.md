# Decision 0001: small self-contained model families

Date: 2026-08-19  
Status: accepted

## Decision

Use a MobileNet-style CNN and compact ViT on CIFAR-10, plus a compact character-level GPT on Tiny Shakespeare. Implement the architectures locally instead of depending on changing model-hub APIs or large pretrained downloads.

## Rationale

- The three families expose convolution, attention, and autoregressive language modeling.
- Model sizes are feasible on ordinary student laptops.
- Local architectures keep checkpoints structurally stable across platforms.
- CIFAR-10 and Tiny Shakespeare are recognizable teaching datasets with small downloads.

## Tradeoffs

These are teaching-sized models trained with short schedules. Their quality is not comparable to full MobileNet/ViT/GPT benchmarks, and TinyGPT is not a modern large language model despite demonstrating the same transformer operations.

