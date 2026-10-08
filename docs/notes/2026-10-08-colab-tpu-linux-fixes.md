# Colab, TPU, Linux, and output-path fixes

Date: 2026-10-08

- Traced the notebook's `runs/cnn/cnn` output to passing `runs/cnn` as the output root while the experiment runner also appended the model name.
- Changed the notebook to pass `OUTPUT_ROOT = Path("runs")` and read artifacts from `RUN_DIR = OUTPUT_ROOT / MODEL`.
- Made the Colab setup cell safe to rerun after it has already changed into the cloned repository, avoiding a nested second clone.
- Made runner output-path resolution idempotent for students who still pass a model-specific path.
- Added explicit `tpu` and `xla` device aliases using `torch_xla.device()` plus XLA synchronization after optimizer updates.
- Added a Colab TPU installation branch using matched PyTorch 2.9.0, torchvision 0.24.0, and PyTorch/XLA 2.9.0 packages.
- Changed Linux setup and CI to install the matched PyTorch 2.11.0 and torchvision 0.26.0 wheels from one official wheel index.
- Added regression tests for TPU resolution/synchronization, output paths, and notebook configuration.
- TPU hardware was not available locally; CPU behavior, package metadata, notebook syntax, and the mocked XLA integration path are covered by automated tests.
