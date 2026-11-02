# Published result tables

Six tables preserve 232 rows supporting the findings. No model weights, calibration text,
generated answers, raw collection dumps, or intermediate plotting caches are included here.

| File | Rows | Meaning |
|---|---:|---|
| `precision_summary.csv` | 7 | BF16 and six quantizations: size, task scores, standard errors, retention, local error, KL |
| `dataset_summary.csv` | 18 | Six quantizations by code/math/chat: local error, KL, BF16-matched activation/entropy drift |
| `block_summary.csv` | 96 | 16 blocks by six precisions: original relative errors and within-precision deviations |
| `tensor_type_summary.csv` | 42 | Seven projection types by six precisions: original relative errors and within-precision deviations |
| `block_dataset_summary.csv` | 48 | 16 blocks by three domains, averaged across six precisions |
| `tensor_type_dataset_summary.csv` | 21 | Seven types by three domains, averaged across six precisions |

`relative_error_pct` is a **squared-error** ratio, not accuracy loss. The denominator
is the diagonal approximation of BF16 local output energy. Block/type ratios pool
errors and references before division. Model-wide error is also a ratio of sums.

`column_mean_error_pct` gives each displayed row equal weight; it is a different
aggregate from the model-wide ratio. `deviation_from_column_mean_pct` equals
`100 * (relative_error_pct / column_mean_error_pct - 1)`.

Dataset summaries average Q2/Q3/Q4/Q5/Q6/Q8 equally before column normalization.
Large Q2 errors contribute strongly to this arithmetic average. These tables describe
the tested mixture of precisions, not a single model or a precision-independent property.

Task `_pct` columns are scores on a 0-100 scale; `_delta_pp` and `_stderr_pp` use
percentage points. `_retained_pct` is a ratio to the matching observed BF16 score.
HellaSwag raw and answer-length-normalized scores are separate. Figures use the latter.
KL columns use nats. Sizes are MiB of GGUF files, not VRAM measurements.

`experiment_manifest.json` retains hashes, settings, counts, formulas and figure/table hashes,
with machine-specific filesystem paths omitted. The existing benchmark JSONs in
`results/Pure_Sweep/` retain more evaluation detail.

Verify the bundle from the repository root without downloading any model:

```bash
python src/published_results.py
```

Export it again from the complete **local** analysis outputs:

```bash
python src/published_results.py --export
```

The published summaries verify the displayed ratios and preserve exact plotted values.
Recomputing measurements from weights/activations requires the excluded original inputs
and the tools described in the root README. Hashes alone cannot reconstruct those inputs.
