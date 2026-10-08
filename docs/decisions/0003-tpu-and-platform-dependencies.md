# Decision 0003: TPU execution and matched platform dependencies

Date: 2026-10-08
Status: accepted

## Decision

Support a single PyTorch/XLA device through the `tpu` and `xla` device aliases. Synchronize the XLA graph after every optimizer update. Use the matching PyTorch 2.9.0, torchvision 0.24.0, and PyTorch/XLA 2.9.0 packages for TPU. Install matching PyTorch 2.11.0 and torchvision 0.26.0 Linux wheels from the same PyTorch wheel index for ordinary CPU/CUDA setups.

Treat `--output-dir` as idempotent when its final directory already equals the selected model name. This keeps both `--output-dir runs` and `--output-dir runs/cnn` from nesting the model directory twice.

## Rationale

- A TPU is not a normal `torch.device("tpu")`; it requires `torch_xla.device()` and explicit graph synchronization.
- Matching torch and torchvision releases avoids Linux binary/operator mismatches.
- A single tested dependency pair is easier for students to reproduce than broad independent version ranges.
- Notebook code should distinguish the shared output root from the final model run directory.

## Consequences

- TPU support is single-device and educational; this release does not launch all TPU cores or claim TPU performance scaling.
- Students must select the Colab TPU runtime and set `DEVICE = "tpu"` before running the setup cell.
- Linux CUDA users can override the setup script's wheel index with `PO2QAT_TORCH_INDEX_URL`.
- TPU setup uses Colab's Python 3.12 runtime; the ordinary CPU/CUDA/MPS project remains compatible with Python 3.10 through 3.14.
