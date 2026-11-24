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
   chat-q4   770.278           0.0            10.0       20           45.0          45.0                     50.000        20             20.0                                          20            1926.104                  60                       11512               31.815              6.031                 21.667
   code-q4   770.278           0.0            20.0       20           35.0          40.0                     43.333        20             25.0                                          20            2012.993                  60                       12079               33.285              6.048                 20.000
   math-q4   770.278           0.0            15.0       20           40.0          45.0                     50.000        20             25.0                                          20            2052.464                  60                       12164               33.923              5.976                 21.667
uniform-q4   770.277           0.0            25.0       20           40.0          40.0                     46.667        20             20.0                                          20            2724.775                  60                       13717               44.920              5.089                 20.000
```

BF16 baseline not yet measured; no accuracy-loss or recovery conclusion is possible.
