# Implementation notes

Date: 2026-08-19

- Confirmed that the literature commonly calls the method PoT-QAT; retained Po2QAT as the repository/student-facing name.
- Implemented one shared fake-quantization path for convolution and linear modules across all model families.
- Chose exact powers of two without a floating-point scale so students can inspect and reconstruct values directly.
- Added separate pre-QAT, post-QAT master, and projected checkpoint exports to prevent ambiguity about which weights are actually constrained.
- Added a synthetic no-download smoke profile. Results from that profile must not be reported as model-quality evidence.
- Cross-platform worker default is zero to avoid Windows process-spawn issues.
- Attempted a dataset-backed CIFAR-10 quick run. The canonical 170 MB archive began downloading correctly, but the local connection was about 0.1 MB/s, so the optional run was stopped after roughly 3.5 MB. The CIFAR-10 training path was therefore not completed in this environment; CI/unit validation and all three no-download end-to-end pipelines did complete.
- Expanded final evaluation on 2026-08-19: classifiers now export loss, top-1/top-5 accuracy, macro/weighted precision-recall-F1, balanced accuracy, per-class tables, and confusion matrices. TinyGPT exports cross-entropy, perplexity, bits per character, and next-character top-1/top-5 accuracy. All models export aggregate quantization error and theoretical storage metrics.
- Completed real-data strong runs for all three models. Each final Po2 checkpoint improved over its initial FP32 checkpoint on the primary metric. Added the measured `strong` profile and an automatic quality gate to flag future regressions.
- Added a terminal-first interactive launcher on 2026-08-19. Running `python -m po2qat` now asks students to choose CNN, ViT, or TinyGPT and then a profile. Training prints epoch/step progress so multi-minute runs do not appear stalled.
- Added a Colab-ready results notebook, accessible reference-result charts, and a 40-point assignment worksheet so the full experiment-to-analysis workflow is ready for students.
- Prepared the v1.0.0 changelog and release materials; reference charts use the measured single-seed strong-profile results and retain the documented uncertainty.
