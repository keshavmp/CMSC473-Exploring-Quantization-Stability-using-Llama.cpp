# Imatrix recovery data for correlation analysis

All three models (Llama, Qwen, Gemma), six precisions (Q2/Q3/Q4/Q5/Q6/Q8),
and three calibration variants (code/math/chat) are included. Model source
identities and SHA256 hashes are recorded in `manifest.json`.

| CSV | Rows | Unit of analysis |
|---|---:|---|
| `local_tensor_detail.csv` | 26,460 | Model / precision / calibration / measurement domain / individual projection tensor |
| `precision_by_calibration.csv` | 54 | Model / precision / calibration; all blocks and projection types pooled |
| `block_by_calibration.csv` | 1,260 | Model / precision / calibration / native transformer block |
| `layer_type_by_calibration.csv` | 378 | Model / precision / calibration / projection type; all blocks pooled |
| `precision_recovery.csv` | 18 | Model / precision; calibrations pooled |
| `block_recovery.csv` | 420 | Model / precision / native block; calibrations pooled |
| `layer_type_recovery.csv` | 126 | Model / precision / projection type; calibrations pooled |
| `output_by_domain.csv` | 162 | Held-out output KL by model / precision / calibration / domain |
| `output_recovery.csv` | 18 | Held-out output KL pooled by model / precision |
| `benchmark_by_task.csv` | 162 | Benchmark score and gain by model / precision / calibration / task |

`layer` is the zero-based **transformer block index**: Llama has 16, Qwen 28,
Gemma 26. `tensor_type` identifies a projection inside the block: `attn_q`,
`attn_k`, `attn_v`, `attn_output`, `ffn_gate`, `ffn_up`, `ffn_down`.
`variant` is the calibration imatrix used during quantization; `domain` is
the activation measurement domain. They are separate dimensions.

## Error and recovery columns

- `control_error`: activation-weighted squared weight error for the matched
  no-imatrix model at the same model, precision, tensor and measurement domain.
- `weighted_error`: corresponding error after imatrix quantization.
- `error_recovered = control_error - weighted_error`: absolute error removed.
- `recovery_pct = 100 * error_recovered / control_error`: percent of control
  error removed. Negative means worse; zero control yields an empty value.
- `reference_energy`: source-model diagonal output-energy approximation, in
  the detailed tensor table.
- `relative_error_pct = 100 * weighted_error / reference_energy`: remaining
  error relative to source energy, **not** percent recovered.
- `error_reduction_vs_control_pct`: historical name for `recovery_pct` in
  the detailed tensor table; both columns contain the same values.

The local error is `sum_j E[x_j^2] * sum_i (W_quant[i,j] - W_source[i,j])^2`,
using fixed source-model calibration moments. It approximates local output
reconstruction with a diagonal second moment; it is not a measured activation
drift after error has propagated through the network. Qwen uses an F16 source;
Llama and Gemma use BF16.

Pooled local tables **sum** constituent errors before computing recovery.
They pool all three measurement domains; tables without `by_calibration` also
pool all three calibration choices. Absolute errors in these tables are sums,
not domain or calibration averages. For an equal mean across domains divide
by three; when calibrations are also pooled divide by another three. Recovery
percentages are unchanged by these common factors. Blocks/types pool tensors
by summing errors, not averaging tensor recovery percentages.

KL tables use `error` rather than `weighted_error`, in nats per scored token
for individual domain rows. Their `error_recovered` is KL reduction. Pooled KL
rows sum nine calibration/domain values; divide by nine to obtain their mean.

## Joining to activation-error results

For individual projections, join on `model`, `precision`, `variant`, `domain`,
and `tensor`. For blocks, use `model`, `precision`, `variant`, `layer`, and
match the same pooling over measurement domains. Projection-type summaries
use `tensor_type` instead of `layer`. Keep models separate when comparing
native block indices; `depth_pct` provides relative depth without interpolation.
An activation-error experiment with different calibration or measurement data
needs an explicit matching decision before joining; do not silently cross-join.

```python
import pandas as pd

data = "results/imatrix_recovery_tables/"
blocks = pd.read_csv(data + "block_by_calibration.csv")
# Your activation-error table must have one row per matching key and use
# the same measurement-domain pooling as blocks.
keys = ["model", "precision", "variant", "layer"]
joined = blocks.merge(activation_errors, on=keys, validate="one_to_one")
print(joined[["recovery_pct", "activation_error"]].corr(method="spearman"))
```

Check unmatched keys before interpreting a correlation. Native block rows and
calibration choices are related observations, not independent repeated trials.
Q8 is a negative control that can have unchanged weights and zero recovery.
Raw errors have architecture-dependent scales; report within-model results
and the precise aggregation used.

## Rebuilding and file sizes

These CSVs are small enough for regular Git; model weights, generated answers,
and multi-gigabyte raw sweep outputs are unnecessary to read or join them.
`manifest.json` records table row counts and SHA256 hashes.

To re-export from the original local sweeps without redrawing plots:

```powershell
python -m src.compare_imatrix_recovery --tables-only
```

Reading these published CSVs requires only pandas. Re-exporting additionally
requires the original sweep folders and the project's analysis dependencies.
