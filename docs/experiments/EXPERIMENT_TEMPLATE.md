# Po2QAT experiment report

Date:  
Student/group:  
Git commit:  
Model:  
Profile:  
Device:  
Python/PyTorch version:  
Seed:  
Bit width:  

## Configuration

Attach or summarize `config.json`.

## Metrics

| State | Loss | Primary metric | Secondary metric | Macro F1 or bits/character |
|---|---:|---:|---:|---:|
| Initial FP32 (pre-QAT) | | | | |
| QAT master FP32 | | | | |
| Materialized Po2 | | | | |

## Layer comparison

| Layer | Exponent range | Master-to-Po2 MAE | Observation |
|---|---:|---:|---|
| | | | |

## Exactness check

Explain how you verified that all eligible nonzero values are signed powers of two.

## Quantization metrics

Report master-to-Po2 MAE, RMSE, cosine similarity, zero fraction, levels used, and theoretical compression. For a classifier, identify the largest per-class F1 change and refer to its confusion matrix. For TinyGPT, compare perplexity, bits per character, and next-character accuracy.

## Conclusions

State what this run supports and what it does not support. Include at least two limitations.
