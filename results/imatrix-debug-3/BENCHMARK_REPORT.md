# Imatrix Benchmark Evaluation

Experiment: `imatrix-debug-3`

## Evaluation protocol

- GSM8K: strict exact match
- IFEval: prompt-level strict accuracy
- MMLU College CS: generative exact match
- Scores are percentages
- BF16 is the reference model when available
- Model size is GGUF file size, not runtime memory

## Results

```text
     model  size_mib  gsm8k  ifeval  mmlu_college_computer_science_generative
uniform-q4   770.277    0.0    50.0                                       0.0
```

## Interpretation

BF16 results are unavailable; baseline accuracy changes cannot yet be calculated.

These benchmark scores measure downstream model quality.
They do not independently establish layer-level sensitivity.

Generative MMLU is not equivalent to standard log-likelihood MMLU.

Small evaluation subsets are exploratory. Use larger fixed subsets for research conclusions.
