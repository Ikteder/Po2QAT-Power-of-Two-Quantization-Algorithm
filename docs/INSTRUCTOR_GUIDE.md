# Instructor guide

## Recommended sequence

1. Ask students to complete the smoke profile before class.
2. In class, inspect `weight_comparison.csv` and derive one row by hand.
3. Assign one quick-profile model to each group.
4. Combine accuracy, perplexity, and layer-error observations across groups.
5. Discuss why exported Po2 weights alone do not guarantee faster PyTorch inference.

## Minimum submission

- `config.json`, `metrics.json`, and `weight_summary.csv` from one quick run;
- a comparison of initial FP32 and final Po2 quality;
- verification that eligible exported tensors contain only zero or signed powers of two;
- one layer-level error observation; and
- at least two limitations.

## Suggested rubric (20 points)

| Criterion | Points |
|---|---:|
| Reproducible configuration and environment reported | 4 |
| Correct interpretation of the three checkpoints | 4 |
| Metric comparison and quantization-error analysis | 5 |
| Correct explanation of STE and Po2 encoding | 4 |
| Limitations stated without overstating hardware gains | 3 |

## Expected variability

The same CPU command and seed should be close across machines, but bit-for-bit equality is not guaranteed across PyTorch versions or hardware. MPS and CUDA can differ slightly from CPU. Grade the method and evidence, not an exact target score.

## Publishing checklist

Before sharing the repository URL with students:

- confirm the published clone URL in the README points to the course repository;
- protect the default branch if students will submit pull requests;
- confirm the GitHub Actions matrix is green;
- optionally create a release tag for the course term; and
- run the quick profile once on the lab machines and record approximate runtime.
