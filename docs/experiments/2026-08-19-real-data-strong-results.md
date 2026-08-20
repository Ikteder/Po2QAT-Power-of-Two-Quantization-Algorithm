# Real-data strong-profile results

Date: 2026-08-19  
Status: measured reference run

## Environment

- OS: Windows 11
- Device: CPU, 10 PyTorch threads
- Python: 3.14.7
- PyTorch: 2.11.0+cpu
- NumPy: 2.4.4
- Seed: 42
- Quantization: four-bit, exact signed Po2 weight codebooks
- Vision data: fixed 4,096-train/1,000-test CIFAR-10 subsets
- Language data: first 200,000 Tiny Shakespeare characters, 90/10 sequential split

## Schedules

- CNN and ViT: 10 FP32 baseline epochs plus 5 Po2QAT epochs
- TinyGPT: 600 FP32 baseline steps plus 300 Po2QAT steps

## Main results

| Model | Metric | Initial FP32 | Final Po2 | Change | Po2 loss | Runtime |
|---|---|---:|---:|---:|---:|---:|
| MobileNetTiny | Accuracy | 53.3% | **63.8%** | +10.5 pp | 1.013 (from 1.326) | 276.6 s |
| TinyViT | Accuracy | 20.8% | **28.2%** | +7.4 pp | 1.927 (from 2.045) | 293.1 s |
| TinyGPT | Next-token accuracy | 50.75% | **51.73%** | +0.98 pp | CE 1.670 (from 1.709) | 309.7 s |

TinyGPT perplexity improved from 5.526 to **5.314**, and bits per character improved from 2.466 to **2.410**.

## Additional results

| Model | Initial macro F1 / top-5 token accuracy | Final Po2 | Eligible parameters | Theoretical eligible compression | Master-to-Po2 cosine |
|---|---:|---:|---:|---:|---:|
| MobileNetTiny | macro F1 0.507 | **0.632** | 61,832 | 8x | 0.9794 |
| TinyViT | macro F1 0.170 | **0.251** | 337,344 | 8x | 0.9812 |
| TinyGPT | top-5 token accuracy 81.87% | **82.24%** | 597,760 | 8x | 0.9781 |

All three final checkpoints passed the exact signed-Po2 tensor check. All three pass the repository quality gate, and none degraded relative to its own pre-QAT checkpoint on the measured primary metrics.

## Interpretation and limitations

The CNN and language-model results are strong for a short laptop-scale teaching run. TinyViT is quantization-safe—the Po2 result improves substantially over its starting checkpoint—but its absolute 28.2% accuracy is modest. Ten baseline epochs on only 4,096 images are insufficient for a transformer trained from scratch. This should be described as a data/training-budget limitation, not hidden or attributed to quantization.

These are single-seed results, not confidence intervals. QAT adds training exposure, so improvement over the pre-QAT checkpoint does not prove that quantization itself improves generalization. The result that matters for this project is that the final exact-Po2 model retained or improved task quality after Po2-aware optimization.

