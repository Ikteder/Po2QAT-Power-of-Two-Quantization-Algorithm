# Smoke validation

Date: 2026-08-19  
Purpose: structural end-to-end validation, not model-quality evaluation

## Environment

- OS: Windows
- Device: CPU
- Python: 3.14.7
- PyTorch: 2.11.0+cpu
- NumPy: 2.4.4
- Seed: 42

## Command

```text
python -m po2qat run --model all --profile smoke --device cpu --output-dir runs-smoke
```

## Observations

| Model | Parameters | Quantized modules | Initial metric | Po2 metric | Elapsed |
|---|---:|---:|---:|---:|---:|
| MobileNetTiny | 64,418 | 18 | accuracy 0.15625 | accuracy 0.15625 | 0.56 s |
| TinyViT | 347,722 | 14 | accuracy 0.12500 | accuracy 0.15625 | 0.35 s |
| TinyGPT | 612,992 | 13 | perplexity 9.067 | perplexity 7.794 | 0.66 s |

All 45 eligible matrix/kernel tensors across the three models passed the exact-Po2 check. The language checkpoint also included two unquantized embedding matrices, as intended. After adding full before/after evaluation reports and delta tables, unit tests passed (`7 passed`) and all three smoke pipelines were rerun successfully.

## Interpretation

Synthetic labels and two language-model update steps make these metrics scientifically meaningless. This run establishes that the three training paths complete, exports are created, and all eligible materialized weights obey the signed-Po2 representation.
