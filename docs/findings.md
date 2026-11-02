# Quantization stability: compression, task quality, and local error

**How far can Llama 3.2 1B Instruct be compressed before quality degrades, and where
does local quantization error concentrate?**

These runs compare one BF16 checkpoint with pure Q2/Q3/Q4/Q5/Q6/Q8 variants.
They combine benchmark scores, output-distribution KL, and activation-weighted
local weight error. Each measure answers a different part of the question.

## 1. Compression has a task-dependent cost

| Precision | File reduction vs BF16 | GSM8K retained | IFEval retained | HellaSwag retained | Local squared-error ratio |
|---|---:|---:|---:|---:|---:|
| Q2 | 83.3% | 2.0% | 22.0% | 54.8% | 9.289% |
| Q3 | 78.3% | 36.1% | 70.3% | 91.5% | 2.274% |
| Q4 | 71.6% | 78.2% | 83.2% | 95.9% | 0.530% |
| Q5 | 65.4% | 82.2% | 93.1% | 99.7% | 0.136% |
| Q6 | 58.8% | 104.0% | 94.4% | 99.6% | 0.0331% |
| Q8 | 46.7% | 90.1% | 95.7% | 100.2% | 0.00304% |

Retained scores are ratios to the observed BF16 score on the same benchmark,
not absolute accuracies. HellaSwag uses answer-length-normalized accuracy.
BF16 scores are GSM8K 20.2%, IFEval prompt-strict 42.884%, and HellaSwag 56.15%.

![Precision overview](../results/important%20plots/top%20graphs/01_precision_overview.png)

Q2 suffers severe degradation. Q3 recovers much of HellaSwag while losing most
GSM8K performance relative to BF16, illustrating why a single benchmark can give
an incomplete picture. Q5-Q6 offer useful measured storage/quality tradeoffs.
Q8 most closely preserves local weights and output distributions, but the task
scores do not improve monotonically with precision in this sample.

The Q6 GSM8K score is only 0.8 percentage points above BF16; this does not establish
that Q6 is intrinsically better. Score retention also inherits uncertainty in the BF16 baseline.

## 2. Dataset differences depend on what is measured

![Dataset overview](../results/important%20plots/top%20graphs/02_dataset_comparison.png)

Code, math and chat are calibration/text domains, distinct from evaluation
benchmarks GSM8K, IFEval and HellaSwag. Local fixed-activation error ratios are
similar across these domains, while output KL differs. Code has the highest KL
at Q2-Q3; chat has the highest at Q4-Q8. Neither result identifies a universally
most sensitive domain.

Energy drift compares a quantized tensor against the same BF16 tensor on the
same domain, using median absolute percentage changes. Entropy drift uses mean
absolute percentage-point changes. Absolute changes avoid signed cancellation.

## 3. Precision-wide scales hide differences between blocks and tensor types

The original block and tensor plots divide local squared error by matching BF16
reference energy. Their shared logarithmic color scales preserve the large
precision effect. The additional column-normalized views instead show
`100 * (relative error / mean of displayed rows in that precision - 1)`.

![Blocks within precision](../results/important%20plots/top%20graphs/03b_block_sensitivity_within_precision.png)

Block 1 is 48.1% above the block average at Q2 but 40.0% below it at Q3.
This is a relative ranking within each precision, not evidence that Q3 has
negative quantization error or better-than-BF16 reconstruction.

![Tensor types within precision](../results/important%20plots/top%20graphs/04b_tensor_comparison_within_precision.png)

FFN-down is 52.5% above the type average at Q2 but 50.1% below it at Q3.
Attention-value has above-average proportional error across all tested precisions.
These are local-error observations, not proofs of downstream task sensitivity.

Original magnitude-preserving views:
[block errors](../results/important%20plots/top%20graphs/03_block_sensitivity.png) and
[tensor-type errors](../results/important%20plots/top%20graphs/04_tensor_comparison.png).

## 4. Precision-averaged dataset views summarize a similar overall pattern

The requested dataset heatmaps average relative errors equally across
Q2/Q3/Q4/Q5/Q6/Q8, excluding BF16. They then normalize displayed rows within each
dataset column, following the same format as the precision-column views.

- [Mean block errors by dataset](../results/important%20plots/top%20graphs/05_blocks_by_dataset.png)
- [Column-normalized block errors by dataset](../results/important%20plots/top%20graphs/05_blocks_by_dataset_normalized.png)
- [Mean tensor-type errors by dataset](../results/important%20plots/top%20graphs/06_tensors_by_dataset.png)
- [Column-normalized tensor-type errors by dataset](../results/important%20plots/top%20graphs/06_tensors_by_dataset_normalized.png)

Block 1 and FFN-down stand out in these averages across all three domains.
Because Q2 errors are much larger, Q2 contributes strongly even with equal
arithmetic weights. These views summarize the chosen precision mixture and can
hide precision-specific reversals visible above. They do not establish a
precision-independent ranking of layers or datasets.

## Method and limits

For input column second moments `a_j = E[x_j^2]`, local error and reference energy are:

```text
weighted_error  = sum_j a_j * sum_i (Wq[i,j] - W[i,j])^2
reference_energy = sum_j a_j * sum_i W[i,j]^2
relative_error_pct = 100 * weighted_error / reference_energy
```

`W` is the original BF16 weight matrix converted to floating point; `Wq` is the
dequantized stored quantized matrix. BF16 input moments are held fixed across
precisions and normalized by observation counts. Domain averages are balanced;
block/type ratios pool errors and references before division.

- This diagonal local approximation omits cross-dimension correlations and
  downstream propagation. Relative squared error is not percent accuracy lost.
- Coverage is 16 blocks and seven projection types: 112 tensors. Embeddings,
  norms and uncollected output tensors are outside the measurement.
- Calibration uses code/math/chat lengths of 49/97/699 chunks of 512 tokens.
  Count normalization does not equalize sampling uncertainty.
- KL is `KL(BF16 || quantized)` in nats, using only 510 scored tokens per domain.
- Evaluation uses 1,000 GSM8K, 541 IFEval and 2,000 HellaSwag examples, zero-shot.
  BF16 standard errors are about 1.27/2.13/1.11 percentage points respectively.
- No paired significance test or independent calibration-repeat study is claimed.
- Q7 is unavailable. Quantization labels denote families, not exact whole-file
  bits per weight. File-size savings are not measured VRAM or speed savings.

The measurements identify candidates for controlled higher-precision block or
tensor experiments. Testing whether those changes improve held-out accuracy/KL
is a separate step.

## Evidence and verification

[Six compact result tables](../results/published/README.md) retain 232 rows, formulas,
counts and hashes. All ten figures above are PNG files. Existing benchmark JSONs
remain in `results/Pure_Sweep/`.

```bash
python src/published_results.py
python -m pytest tests/test_imatrix_analysis.py tests/test_imatrix_report.py tests/test_important_imatrix_plots.py tests/test_important_precision_plots.py -q
```

The verification command checks table/figure hashes, row counts, relative-error
formulas and column normalization without model downloads. Full recomputation
requires the original models, calibration inputs and llama.cpp tools; these are
not bundled with the concise published tables. See the root README for the analysis workflow.
