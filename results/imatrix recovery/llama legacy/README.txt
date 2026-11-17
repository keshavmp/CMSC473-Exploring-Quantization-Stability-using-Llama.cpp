IMATRIX RECOVERY: TOP GRAPHS

01_benchmark_scores.png
Scores normalized to each benchmark's historical BF16 score = 100%; values above 100% are retained.

02_benchmark_gains.png
Main task-quality figure: relative gain = 100 * (calibrated score - control score) / control score.
Paired 95% difference intervals are scaled by the fixed observed same-precision control score.
These intervals do not resample denominator uncertainty. Zero means no change.
Stars mark exact paired comparisons passing Holm correction across all 54 comparisons.

03_kl_recovery_by_dataset.png
Annotated precision-by-calibration heatmaps for mean KL and separate held-out domains.
Positive/red reduces KL; negative/blue worsens it; white means no change. Every panel shares a symmetric scale.

04_local_vs_output_recovery.png
Main mechanism/limitation figure: local calibration error improves even when held-out output KL worsens.
The two panels share a color scale. These measure different texts and different stages of computation.

05_block_recovery.png
Where local error is recovered, by all 16 blocks and precision, with separate imatrix panels.

06_tensor_type_recovery.png
Where local error is recovered, by the seven projection types pooled across 16 layers.

07_calibration_to_task_improvement.png
Each precision: calibration imatrix rows vs benchmark columns, showing relative gains (%) vs matched controls.
Math/GSM8K is domain-related. No direct code/chat benchmark was evaluated.

08_own_vs_other_domain_recovery.png
Each precision: calibration imatrix rows vs held-out code/math/chat columns, showing KL reduction.
Outlined diagonal cells are own-domain matches; other cells show transfer to another domain.

ALL RECOVERY HEATMAPS: red = improvement, white = no change, blue = worsening.

KL/local recovery already use 100 * (1 - calibrated error / matched control error).
No ratios use the BF16-minus-control score gap; gains can exceed 100% and are not clipped.
All are PNG, preserving the previous top-graphs style. Raw values used for plotting are in
results/published/imatrix_recovery/. A detailed narrative is docs/imatrix_recovery_findings.md.
Rebuild with: python -m src.report_imatrix_recovery --render-only

Most useful figures for the main story: 02, 03 and 04. Figures 05/06 explain local recovery.
Use 07/08 to inspect each calibration imatrix's related-domain and cross-domain effects.
Do not present local-recovery heatmaps as measured block/tensor intervention sensitivity.
Q8 stored tensor payloads are identical across variants; small task-score changes are not weight improvements.
