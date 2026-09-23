# Decision 0002: CIFAR-adapted VGG19 and ResNet50

Date: 2026-09-23
Status: accepted

## Decision

Add VGG19 and torchvision ResNet50 backbones as optional CIFAR-10 models. Both use random initialization and preserve the named backbone depth. VGG19 is constructed directly with adaptive `1x1` pooling and a `512 -> 512 -> 10` head, avoiding even transient allocation of the unused ImageNet classifier. ResNet50 uses the standard CIFAR `3x3`, stride-1 stem, removes the initial max pool, and uses a 10-class head.

The default profile batch size is capped at 16 for these models unless the user explicitly supplies `--batch-size`. The `classroom` model group retains the original three small models; `all` includes the two large models.

## Rationale

- Standard ImageNet VGG19's 4096-unit classifier is unnecessarily large for 32x32 CIFAR-10 and would make student checkpoints much larger without improving the teaching objective.
- The CIFAR ResNet stem avoids discarding most spatial detail before the residual stages.
- Random initialization keeps the comparison consistent with the existing from-scratch classroom experiments and avoids extra pretrained-weight downloads.
- Explicit small/large model groups prevent an innocent convenience command from unexpectedly launching two compute-intensive runs.

## Consequences

- VGG19 has approximately 20.3 million parameters and 18 eligible convolution/linear modules.
- ResNet50 has approximately 23.5 million parameters and 54 eligible convolution/linear modules.
- Both models use the same metrics, confusion matrices, exact-Po2 verification, checkpoint exports, and notebook analysis as the smaller vision models.
- No reference accuracy is claimed until a documented real-data schedule is run.
