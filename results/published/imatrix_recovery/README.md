# Imatrix recovery: compact published results

Six core tables and two transfer tables support eight PNG figures and the [findings narrative](../../../docs/imatrix_recovery_findings.md).

| Table | Rows | Content |
|---|---:|---|
| `model_metrics.csv` | 24 | Raw task scores, standard errors, model sizes, mean KL, runtimes and recorded model hashes |
| `benchmark_gains.csv` | 54 | Paired gains, 95% bootstrap intervals, wins/losses/ties, exact paired p-values and Holm adjustment |
| `kl_by_domain.csv` | 72 | KL means, tool uncertainty, token counts and recovery versus matched controls |
| `local_errors_by_domain.csv` | 72 | Pooled local weighted errors, references and control-relative reductions |
| `block_local_recovery.csv` | 384 | 16 blocks x six precisions x four variants, pooling equally averaged domain errors |
| `tensor_type_local_recovery.csv` | 168 | Seven projection types x six precisions x four variants |
| `calibration_task_transfer.csv` | 54 | Imatrix x benchmark x precision: control/calibrated scores, gains, paired uncertainty and domain relationship |
| `calibration_domain_transfer.csv` | 54 | Imatrix x held-out text domain x precision, marking same-domain versus cross-domain KL |

Total: **882 rows**. No generated answers or benchmark/calibration text are included.
`experiment_manifest.json` records source/output hashes, calibration coverage, scoring
rules, statistical methods, the historical BF16 reference and the Q8 tensor-payload check.
It contains relative source paths and hashes rather than local usernames/absolute paths.

## Units and formulas

`*_score_pct` uses a 0-100 scale. GSM8K uses flexible extraction, IFEval prompt-strict
accuracy, and HellaSwag answer-length-normalized accuracy. `*_stderr_pp` and benchmark
`gain_pp`/intervals are percentage points. Sizes are MiB of GGUF files, not VRAM use.

Recovery percentages equal `100 * (1 - calibrated_error / control_error)`.
Positive means lower error, negative means higher error. In the figures, positive
gains/reductions are red, zero is white and negative values are blue.
KL is `KL(BF16 || model)` in nats, with 1,020 scored tokens/domain. Its supplied uncertainty is not a paired
significance test and is not interpreted as such.

Local error is a diagonal second-moment approximation using fixed BF16 calibration
activations. Block/type rows average domain errors equally, then pool tensors before
division. The denominator in `relative_error_pct` is corresponding BF16 diagonal
reference energy. These are squared-error ratios, not accuracy losses.

The no-imatrix control is matched to the same standard quantization recipe and target
precision. None of the models uses `--pure`; historical pure-sweep models are not the
matched controls. Each recipe's stored tensor precisions match across its variants.

## Paired analysis and limits

Question IDs and document/prompt hashes were checked. GSM8K selects only the
`flexible-extract` sample records, excluding duplicate strict-match filter records.
Gain intervals use 10,000 paired-question bootstrap samples with seed 473, implemented
by multinomial sampling of wins/losses/ties. These are pointwise 95% intervals.
Two-sided exact McNemar/binomial discordance p-values are Holm-adjusted across all
54 task/precision/calibration comparisons. Tests are conditional on recorded outputs
and question-level assumptions; repeated execution and calibration variability are not estimated.

The four Q8 variants have identical tensor payloads and identical KL/local errors,
but their benchmark scores differ slightly. Treat those changes as a negative-control
warning, not as imatrix benefits. Their cause has not been established.
Calibration lengths differ between code/math/chat, so calibration budget is a confound
when comparing domain-specific imatrices. The historical BF16 scores are context,
not a newly paired reference for claims of surpassing BF16.

Rebuild plots without raw answers or model weights:

```powershell
python -m src.report_imatrix_recovery --render-only
```

Full source/model verification requires the original local sweep folder and Q8 GGUFs:

```powershell
python -m src.report_imatrix_recovery
```
