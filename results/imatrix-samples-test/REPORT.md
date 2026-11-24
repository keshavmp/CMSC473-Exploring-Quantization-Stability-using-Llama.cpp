# Imatrix benchmark comparison

Measured scores only. No synthetic accuracies or forced-token probabilities.

## Evaluation

- GSM8K strict exact match (flexible extraction also saved)
- IFEval prompt-level strict accuracy (loose/instruction-level also saved)
- MMLU College CS **generative** exact match; NOT standard multiple-choice likelihood accuracy
- Scores are percentages; identical task sample indices and seed across models
- A short generation cap can truncate answers; smoke-test accuracy is not a research claim
- Model size is GGUF file bytes, not peak RAM or inference throughput

## Results

```text
     model  size_mib  gsm8k_strict  gsm8k_flexible  gsm8k_n  ifeval_strict  ifeval_loose  ifeval_instruction_strict  ifeval_n  mmlu_generative  mmlu_college_computer_science_generative_n  evaluation_seconds  completed_requests  reported_completion_tokens  seconds_per_request  tokens_per_second  mean_primary_accuracy
uniform-q4   770.277           0.0           100.0        1            0.0           0.0                     33.333         1              0.0                                           1             295.595                   3                        1026               90.722               3.77                    0.0
```

BF16 baseline not yet measured; no accuracy-loss or recovery conclusion is possible.
