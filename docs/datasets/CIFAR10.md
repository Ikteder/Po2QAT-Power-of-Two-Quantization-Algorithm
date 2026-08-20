# Dataset card: CIFAR-10

## Source

The project downloads CIFAR-10 through `torchvision.datasets.CIFAR10`. The canonical dataset page is <https://www.cs.toronto.edu/~kriz/cifar.html>.

## Use

- Models: MobileNetTiny and TinyViT
- Task: 10-class image classification
- Quick profile: seeded 4,096-image train subset and 1,000-image test subset
- Full profile: standard 50,000 train and 10,000 test images
- Preprocessing: random crop with four-pixel padding and horizontal flip for training; tensor conversion for evaluation

## Leakage controls

Train examples come only from the canonical training split. Metrics use only the canonical test split. Quick subsets are chosen independently with deterministic seeds.

## License and responsible use

The CIFAR website describes the dataset and source corpus but does not provide an SPDX-style license in this repository. Users should review the canonical page and their institution's policy before redistribution. Downloaded data are ignored by Git and are not included in releases.

## Quality concerns

Images are only 32x32. Short, from-scratch training is noisy, and quick-subset metrics should not be compared with published full-training CIFAR-10 results.

