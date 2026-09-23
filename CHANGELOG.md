# Changelog

All notable changes to this teaching project are documented here.

## [Unreleased]

### Added

- CIFAR-adapted VGG19 and ResNet50 model options with automatic conservative batch sizing.
- Interactive, command-line, notebook, model-card, and assignment guidance for both large models.

### Changed

- The research reference now cites the repository author's IEEE ICAD 2026 Po2QAT paper.
- `--model classroom` preserves the fast original three-model workflow; `--model all` includes both large models.

## [1.0.0] - 2026-08-19

### Added

- Reproducible Po2QAT experiments for SmallCNN, TinyViT, and TinyGPT.
- Interactive terminal menu with smoke, quick, and strong profiles.
- Initial FP32, QAT master, and final exact-Po2 checkpoints plus auditable weight tables.
- Accuracy/loss or perplexity/token-accuracy comparisons, training history, quality gates, and classifier confusion matrices.
- Colab-ready results notebook, measured reference charts, and a student assignment with grading rubric.
- Windows, macOS, Linux, CPU, NVIDIA CUDA, and Apple Silicon MPS guidance.

### Validation

- Automated tests cover quantization, checkpoint round trips, artifacts, all smoke pipelines, the interactive menu, and teaching assets.
- Documented strong-profile reference runs completed for all three model families with the configured quality gates passing.
