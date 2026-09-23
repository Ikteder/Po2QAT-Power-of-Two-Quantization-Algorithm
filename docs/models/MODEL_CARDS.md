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

## VGG19

- Intended use: optional large-model CIFAR-10 Po2QAT exercise
- Family: VGG19 feature backbone with all 16 convolutional layers, constructed directly to avoid allocating the unused ImageNet classifier
- CIFAR adaptation: adaptive `1x1` pooling and a `512 -> 512 -> 10` classifier replace the ImageNet-specific `7x7/4096` head
- Input/output: RGB `32x32` images and 10 logits
- Initialization: random; no pretrained weights are downloaded
- Quantized: all convolution and linear weights
- Not quantized: biases
- Resource note: about 20.3 million parameters; use a GPU/MPS when possible and begin with `--batch-size 8`
- Non-goal: reproducing torchvision ImageNet accuracy or the original ImageNet classifier

## ResNet50

- Intended use: optional large-model CIFAR-10 Po2QAT exercise
- Family: torchvision ResNet50 bottleneck backbone
- CIFAR adaptation: `3x3`, stride-1 stem, no initial max pool, and a 10-class head
- Input/output: RGB `32x32` images and 10 logits
- Initialization: random; no pretrained weights are downloaded
- Quantized: all convolution and final linear weights
- Not quantized: batch-normalization parameters and biases
- Resource note: about 23.5 million parameters; use a GPU/MPS when possible and begin with `--batch-size 8`
- Non-goal: reproducing torchvision ImageNet accuracy

## TinyGPT

- Intended use: demonstrate Po2QAT in an autoregressive transformer on laptop hardware
- Family: decoder-only, character-level transformer
- Context: at most 96 characters; embedding dimension: 128; blocks: 3; heads: 4
- Quantized: attention, MLP, and output-head linear weights
- Not quantized: token/position embeddings, layer norms, and biases
- Metrics: validation cross-entropy, perplexity, bits per character, and next-character top-1/top-5 accuracy
- Non-goal: general-purpose text generation or comparison with production LLMs

## Shared risks and limitations

All models train from scratch with intentionally short schedules. VGG19 and ResNet50 are optional compute-intensive extensions and do not have measured reference scores in this repository. A metric can vary across platforms, dependencies, seeds, and devices. The models are educational artifacts and are not intended for safety-critical, medical, surveillance, or production decision-making.
