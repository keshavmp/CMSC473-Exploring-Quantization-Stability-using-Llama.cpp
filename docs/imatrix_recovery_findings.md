# How much quality does imatrix quantization recover?

**Imatrix calibration helps most at low precision, but local reconstruction gains do
not guarantee better output distributions or every benchmark score.**

This experiment addresses the practical half of the project: whether using activation
importance during quantization preserves quality at essentially the same model size.
It is separate from predicting sensitivity through controlled block/tensor interventions.

All **24 configurations completed**: Q2/Q3/Q4/Q5/Q6/Q8, each with a no-imatrix control
and BF16-derived code/math/chat imatrices. All models were quantized directly from the
same BF16 checkpoint using standard recipes without `--pure`. Within each precision,
stored tensor types match across the four variants. Imatrix metadata adds only 448 bytes.

The evaluation uses the same 1,000 GSM8K, 541 IFEval and 2,000 HellaSwag questions for every
model, plus 1,020 scored KL tokens in each held-out code/math/chat domain. Raw output
hashes, selected IDs, prompt/question hashes, model provenance and sample-score means were
checked against the saved completion records. Recorded evaluation binaries, pipeline and
held-out text hashes match across the 24 runs.

## 1. Benchmark recovery is task dependent

![Benchmark scores](../results/important%20plots/imatrix%20recovery/01_benchmark_scores.png)

![Gains over matched controls](../results/important%20plots/imatrix%20recovery/02_benchmark_gains.png)

| Result | No-imatrix score | Calibrated score | Gain |
|---|---:|---:|---:|
| Q2, code imatrix, HellaSwag | 46.70% | 50.15% | +3.45 pp |
| Q2, chat imatrix, HellaSwag | 46.70% | 50.45% | +3.75 pp |
| Q3, math imatrix, GSM8K | 14.50% | 19.90% | +5.40 pp |
| Q4, math imatrix, GSM8K | 17.20% | 24.20% | +7.00 pp |

These four gains pass a two-sided exact paired McNemar test with Holm correction over
all 54 benchmark/precision/calibration comparisons. Q4 math-calibrated GSM8K has a
pointwise paired-bootstrap 95% interval of approximately +4.0 to +10.1 percentage points.
These tests are conditional on the saved model outputs and question-resampling assumptions;
they do not measure variability between model executions or independently collected imatrices.

IFEval has mixed results. At Q3 every imatrix variant has a lower observed IFEval score
than its control. Q4 code calibration improves all three observed benchmark scores,
while math calibration has the largest GSM8K gain and a small HellaSwag decrease.
There is no universal best calibration domain across all tasks and precisions.

Historical BF16 scores are shown for context. A calibrated model scoring above the
historical BF16 score does not establish that it is intrinsically better than BF16.
The paired tests above compare each model only with its matched no-imatrix control.

## 2. Output recovery is strongest at Q2-Q4

![Output KL recovery](../results/important%20plots/imatrix%20recovery/03_kl_recovery_by_dataset.png)

| Precision | Mean held-out KL reduction across calibration variants |
|---|---|
| Q2 | 34.6% to 39.1% lower |
| Q3 | 31.9% to 33.2% lower |
| Q4 | 13.1% to 16.0% lower |
| Q5 | Mixed: 3.8% worse to 9.5% lower |
| Q6 | 5.4% to 11.0% worse |
| Q8 | No change |

Mean KL gives code/math/chat equal weight before computing recovery against the matched
control. It measures similarity to BF16 output distributions, not task accuracy.
The per-domain heatmaps show transfer: calibration on one domain can help other domains.
They also show failure cases, including worse math-text KL for Q5 code/chat calibration.
The short fixed text evaluation does not establish the population-wide domain ordering.

## Calibration transfer: its related task and other tasks

![Calibration to task improvement](../results/important%20plots/imatrix%20recovery/07_calibration_to_task_improvement.png)

Each panel is one precision. Rows are code/math/chat calibration imatrices; columns
are GSM8K, IFEval and HellaSwag. Cells show percentage-point gains against the matched
no-imatrix control. Red means improvement, white no change, and blue worsening in all recovery heatmaps.
The transfer CSV includes control scores, calibrated scores and paired uncertainty.

At Q4, the pattern is:

| Imatrix | GSM8K gain | IFEval gain | HellaSwag gain |
|---|---:|---:|---:|
| Code | +1.70 pp | +2.40 pp | +0.15 pp |
| Math | +7.00 pp | +1.48 pp | -0.45 pp |
| Chat | +4.50 pp | -0.92 pp | -0.10 pp |

Math calibration has a related-domain benchmark in GSM8K. We did not evaluate a code
benchmark, so code's own-task accuracy recovery is unavailable. IFEval measures instruction
following and HellaSwag commonsense completion; neither is a direct chat benchmark.

![Own-domain and cross-domain recovery](../results/important%20plots/imatrix%20recovery/08_own_vs_other_domain_recovery.png)

For held-out code/math/chat KL, all three calibration domains have a direct counterpart.
Outlined diagonal cells show own-domain recovery; other cells show cross-domain transfer.
For example, Q3 math calibration reduces math-text KL by 42.7%, code-text KL by 32.4%,
and chat-text KL by 24.6%. These percentages measure output-distribution preservation,
not benchmark gains. Domain-matched calibration does not always give the largest improvement.

## 3. Better local reconstruction does not guarantee better output behavior

![Local versus output recovery](../results/important%20plots/imatrix%20recovery/04_local_vs_output_recovery.png)

Local weighted reconstruction improves by roughly 33% at Q2, 18% at Q3 and 15-16% at
Q4-Q6. Yet all Q6 variants have worse mean held-out KL than their control.
This is a central result: the calibration-local diagonal objective omits cross-dimension
correlations and downstream propagation. Optimizing it is useful but is not equivalent
to optimizing final output KL or benchmark accuracy.

## 4. Where is local computation recovered?

![Block recovery](../results/important%20plots/imatrix%20recovery/05_block_recovery.png)

![Tensor-type recovery](../results/important%20plots/imatrix%20recovery/06_tensor_type_recovery.png)

The heatmaps show error reduction relative to each same-recipe control, retaining every
block and pooling projections by type across all 16 blocks. Domains are averaged equally
before pooling error/reference quantities; ratios are not averages of per-tensor percentages.

FFN-down has particularly large proportional recovery: about 86% at Q3 and 31% at Q4.
Attention-query/key projections also improve substantially, while FFN gate/up recovery
is smaller at Q4-Q6. Block patterns depend on precision and calibration.
These are reconstruction-recovery maps, not measured block sensitivity rankings and not
the effect of selectively changing those blocks alone.

## 5. Q8 is a useful negative control

The four Q8 models have **identical stored tensor payloads**, verified by hashing each
tensor's name, type, shape and bytes, excluding metadata. Their local errors and output
KL are identical. Nevertheless, observed benchmark gains range from approximately
-0.92 to +0.60 percentage points.

Those Q8 score differences cannot be attributed to changed stored weights. They show
execution/scoring variability or another uncontrolled execution factor; this analysis
does not establish its cause. Small benchmark changes should therefore be treated
cautiously, and repeated executions are needed to estimate run-to-run variability.

## Reading the normalizations

- Benchmark recovery is shown primarily as **percentage-point score gain** over the
  same-precision no-imatrix control. This avoids unstable ratios to small BF16 score gaps.
- KL recovery is `100 * (1 - imatrix_KL / control_KL)`. Positive improves; negative worsens.
- Local recovery uses the same formula with weighted reconstruction errors. It uses
  calibration activations, not the held-out KL texts or benchmark questions.
- All figures are PNG. No new model inference was required to generate this report.

Calibration budgets are unequal: code/math/chat use 49/97/699 chunks of 512 tokens.
The experiment measures the effect of these available imatrices. It cannot isolate
domain content from calibration budget or establish a universal domain-specific advantage.

## Evidence and rebuilding

[Compact tables and audit manifest](../results/published/imatrix_recovery/README.md)
preserve 882 rows across six core CSVs and two transfer CSVs. Raw generated answers, dataset text, weight files,
logits and intermediate caches are excluded from the proposed publication.

Rebuild plots from the compact published tables:

```powershell
python -m src.report_imatrix_recovery --render-only
```

Repeat the full audit from the original local results and Q8 models:

```powershell
python -m src.report_imatrix_recovery
```

The full sweep instructions are in [imatrix_full_sweep.md](imatrix_full_sweep.md).
