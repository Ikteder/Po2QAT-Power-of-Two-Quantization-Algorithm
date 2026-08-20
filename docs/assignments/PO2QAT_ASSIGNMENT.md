# Po2QAT Student Assignment

## Goal

Run power-of-two quantization-aware training on one model family, verify that the exported weights satisfy the Po2 constraint, and explain the resulting quality–compression trade-off with evidence.

## Learning outcomes

By the end of this assignment, you should be able to:

- distinguish an initial FP32 checkpoint, QAT master weights, and a deployable Po2 checkpoint;
- interpret task metrics, loss curves, confusion matrices, and weight distributions;
- verify a numerical invariant programmatically;
- discuss quantization results without overstating a single run.

## Procedure

1. Install the project by following the root \`README.md\`.
2. Run \`python -m po2qat\` and select CNN, ViT, or LLM.
3. Complete a \`smoke\` run to confirm your environment works.
4. Run your assigned model with the \`quick\` profile. Use \`strong\` only if your instructor requests it and your machine has enough time.
5. Open \`notebooks/po2qat_results_lab.ipynb\` locally or in Colab and generate all applicable plots.
6. Keep the complete output folder for your submission.

## Evidence to submit

- \`config.json\`, \`metrics.json\`, \`metrics_comparison.csv\`, \`training_history.csv\`, and \`weight_summary.csv\`;
- initial and final checkpoints;
- both confusion matrices for CNN/ViT, or the token-metric table for LLM;
- four labeled figures: task metrics, training loss, weight distribution, and confusion matrices when applicable;
- a 600–900 word analysis answering the questions below.

## Analysis questions

1. What model, profile, device, seed, dataset split, and evaluation subset did you use?
2. What changed between the initial FP32 model and the final Po2 model? Report every relevant metric and both absolute and relative changes.
3. Did the final model pass the configured quality gate? Explain what the gate protects against and what it does **not** prove.
4. Compare the float-finetune and QAT loss curves. Where do you see convergence, instability, or under-training?
5. For CNN/ViT: which classes are most often confused, and what evidence supports your answer? For LLM: relate loss, perplexity, and token accuracy.
6. How did the weight distribution change across initial FP32, QAT master, and final Po2 checkpoints?
7. Use \`weight_summary.csv\`, \`weight_comparison.csv\`, and \`po2_sign_exponent.npz\` to explain how you know the quantized weights are exactly zero or signed powers of two.
8. Why are biases, normalization parameters, embeddings, and the output head excluded by default? State one advantage and one limitation of that choice.
9. Can this run establish that Po2QAT always improves accuracy? Why or why not?
10. Propose one controlled follow-up experiment. Name the independent variable, constants, metrics, and number of seeds.

## Grading rubric (40 points)

| Criterion | Excellent | Partial | Points |
|---|---|---|---:|
| Reproducibility | Complete environment/run metadata, exact command or menu choices, seed, and required artifacts | Important setup details or artifacts missing | 6 |
| Metric analysis | Correct initial-vs-Po2 values, absolute/relative changes, and quality-gate interpretation | Values reported without careful comparison | 8 |
| Visual analysis | Clear, labeled plots with specific interpretation of loss and task behavior | Plots present but weakly explained or mislabeled | 8 |
| Weight verification | Correctly demonstrates the exact Po2 invariant and distinguishes master from exported weights | Relies only on the histogram or makes an incorrect claim | 8 |
| Critical reasoning | Addresses limitations, exclusions, stochastic variation, and avoids overgeneralization | Treats one run as universal evidence | 6 |
| Follow-up design | Controlled, measurable experiment with constants and multiple seeds | Vague experiment without controls | 4 |

## Submission checklist

- [ ] My plots include titles, axes, units, and model/profile labels.
- [ ] I report the seed and evaluation subset.
- [ ] I distinguish observed evidence from interpretation.
- [ ] I verify exact Po2 values numerically, not only visually.
- [ ] I note that results can vary across seeds, versions, and devices.
