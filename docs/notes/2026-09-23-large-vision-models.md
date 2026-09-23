# VGG19 and ResNet50 implementation note

Date: 2026-09-23

- Added CIFAR-adapted VGG19 and ResNet50 model builders without pretrained weights. VGG19 is built directly with its CIFAR head so constructing it never allocates the unused 4096-unit ImageNet classifier.
- Verified output shape `(1, 10)` for both models on `32x32` RGB input.
- Verified Po2QAT wrapping and quantized forward passes locally.
- VGG19: 20,292,170 parameters; 18 wrapped convolution/linear modules; all 18 materialized tensors passed exact signed-Po2 checks.
- ResNet50: 23,520,842 parameters; 54 wrapped convolution/linear modules; all 54 materialized tensors passed exact signed-Po2 checks.
- Completed no-training end-to-end CLI checks on synthetic data for both models. Each produced 20 expected metric, confusion-matrix, checkpoint, and weight-export artifacts; all eligible tensors passed the exact-Po2 audit.
- Full training was intentionally not run because these optional models are substantially larger. No accuracy or quality-gate claim is made for them.
- Updated the research citation to the repository author's IEEE ICAD 2026 paper and removed the two previously listed third-party Po2QAT papers.
