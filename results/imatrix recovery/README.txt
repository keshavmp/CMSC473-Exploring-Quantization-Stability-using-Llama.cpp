IMATRIX RECOVERY: LLAMA, QWEN AND GEMMA

Start with 01_precision_comparison.png. All figures are saved as PNG files.

01_precision_comparison: local reconstruction recovery and held-out output KL
recovery, by precision. Each model has one consistent color throughout line plots.
02_block_comparison: one panel per precision, showing local recovery at each
native transformer block. Horizontal position is relative depth (first block 0%,
last block 100%). Llama has 16 blocks, Qwen 28, Gemma 26; no interpolation.
03_layer_type_comparison: one panel per precision; seven projection types within
transformer blocks, compared across models. Values are printed in the cells.
04_benchmark_comparison: task score changes relative to each matching control.
05_calibration_comparison: code, math and chat imatrix results shown separately.
06_llama_benchmark_gains: grouped code/math/chat bars for each task and precision.
07_qwen_benchmark_gains: the same grouped-bar chart for Qwen.
08_gemma_benchmark_gains: the same grouped-bar chart for Gemma.
09_all_models_benchmark_gains: all three models, with model rows and task columns.
Figures 06-09 match the reference chart in percentage-point units and use one
shared y-axis scale across models. Plain bars only; no intervals or significance
markers. Gain = calibrated score (%) minus matching control score (%).
For example, a score of 20% becoming 22% is a +2 percentage-point gain.

NORMALIZATION
Local and output recovery (%) = 100 * (1 - calibrated_error / control_error).
Each denominator matches model, precision and the measured domain/unit.
0% = unchanged; 25% = one quarter of control error removed; 100% = no error left;
-25% = one quarter more error than the control. Negative values are retained.
Zero denominators are missing (N/A), never set to zero or clipped.

Overview plots pool code/math/chat calibrations and the three measurement domains
with equal coefficients: sum calibrated errors and matched control errors FIRST,
then take the ratio. This weights tensors by their control error, so tiny-error
tensors cannot dominate through large percentages. It is not the arithmetic mean
of tensor recovery percentages. Details are in ../imatrix_recovery_tables/.

Benchmark gain (%) = 100 * (calibrated_score - control_score) / control_score;
the overview averages these gains across the three calibrations. A gain of 10%
means a control score of 20 becomes 22, not 30. This is a relative score increase,
not the fraction of the source-model score gap recovered. No significance claims
or repeated-trial uncertainty are inferred from these descriptive averages.

TERMINOLOGY AND LIMITS
A transformer block is often called a transformer layer. Figure 02 compares those
units. Figure 03 instead compares layer/projection types inside blocks: attention
Q/K/V/output and feed-forward gate/up/down. Native block indices and all individual
tensor rows are retained in the CSVs. These are local reconstruction proxies,
not measured block interventions or downstream sensitivity.

These are Llama-3.2-1B-Instruct (BF16), Qwen2.5-1.5B-Instruct (F16), and
Gemma-3-1B-IT (BF16). Sources, sizes, tokenizers and architectures differ. Ratios
compare improvement relative to each model's own control, not absolute quality.
KL uses 1020 scored tokens per held-out domain; tokenizers cover different text
spans. Calibration activations and held-out output KL measure different things.
Q8 is a negative control; imatrix can leave its stored weights unchanged. Small
benchmark/output differences at Q8 can reflect execution variation.

DETAILS AND REPRODUCIBILITY
by model/: all native blocks and precisions, split by calibration, for each model.
../imatrix_recovery_tables/: local tensor rows, pooled values, KL, benchmark gains and
manifest.json with input/output SHA256 hashes, model sources and protocol records.
Start with ../imatrix_recovery_tables/README.md for keys, joins and error units.
Local CSVs include error_recovered (control minus calibrated error) and
recovery_pct at precision, block and projection-type levels for all three models.
The generator checks saved aggregate hashes, sample counts/document IDs, tensor
layouts, local CSV hashes and matching model/control provenance. It does not
rerun inference or redo the legacy report's per-question audit/payload inspection.
llama legacy/: original eight Llama figures and their notes, preserved in full.
Their compact audited data remains in results/published/imatrix_recovery/.

Rebuild all cross-model figures and tables from saved sweeps:
  python -m src.compare_imatrix_recovery
Rebuild the original Llama figures from compact audited tables:
  python -m src.report_imatrix_recovery --render-only
