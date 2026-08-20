# Model cards

## MobileNetTiny

- Intended use: classroom CIFAR-10 classification and Po2QAT inspection
- Family: MobileNet-style depthwise separable inverted residual CNN
- Input: RGB `32x32`
- Output: 10 logits
- Metrics: loss, top-1/top-5 accuracy, macro/weighted precision-recall-F1, balanced accuracy, per-class metrics, and confusion matrix
- Quantized: all convolution and linear weights
- Not quantized: batch-normalization parameters and biases
- Non-goal: matching an official pretrained MobileNet checkpoint

## TinyViT

- Intended use: classroom CIFAR-10 classification and transformer quantization inspection
- Family: patch embedding plus three pre-norm transformer blocks
- Patch size: `4x4`; embedding dimension: 96; attention heads: 3
- Input: RGB `32x32`
- Output: 10 logits from a class token
- Metrics: loss, top-1/top-5 accuracy, macro/weighted precision-recall-F1, balanced accuracy, per-class metrics, and confusion matrix
- Quantized: patch convolution and all linear weights
- Not quantized: layer norms, class token, positional embedding, and biases
- Non-goal: ImageNet-scale ViT quality

## TinyGPT

- Intended use: demonstrate Po2QAT in an autoregressive transformer on laptop hardware
- Family: decoder-only, character-level transformer
- Context: at most 96 characters; embedding dimension: 128; blocks: 3; heads: 4
- Quantized: attention, MLP, and output-head linear weights
- Not quantized: token/position embeddings, layer norms, and biases
- Metrics: validation cross-entropy, perplexity, bits per character, and next-character top-1/top-5 accuracy
- Non-goal: general-purpose text generation or comparison with production LLMs

## Shared risks and limitations

All models train from scratch with intentionally short schedules. A metric can vary across platforms, dependencies, seeds, and devices. The models are educational artifacts and are not intended for safety-critical, medical, surveillance, or production decision-making.
