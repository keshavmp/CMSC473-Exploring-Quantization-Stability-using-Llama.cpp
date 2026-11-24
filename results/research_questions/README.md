# Three research questions: results guide

Open the PNGs in numerical order. All quantities come from saved results; this build launches no inference.

1. **Sensitivity:** `01` shows accuracy lost against BF16; `02` shows domain KL; `04` locates block sensitivity.
2. **Recovery:** the three `03` figures show imatrix gains against matched controls. Positive means improvement.
3. **Blocks:** `05` shows recovery and its concentration within a model; `06` compares sensitive blocks with the rest.

## Benchmark results passing global Holm correction

Tests cover all 162 model/precision/calibration/task comparisons. This is stricter than the earlier Llama-only family of 54.

- Llama Q2, code calibration, hellaswag: +3.45 pp (pointwise paired 95% interval +1.75 to +5.15); adjusted p=0.01793.
- Llama Q2, chat calibration, hellaswag: +3.75 pp (pointwise paired 95% interval +2.00 to +5.50); adjusted p=0.005366.
- Llama Q4, math calibration, gsm8k: +7.00 pp (pointwise paired 95% interval +4.00 to +10.10); adjusted p=0.001715.
- Qwen Q3, math calibration, gsm8k: +7.10 pp (pointwise paired 95% interval +3.60 to +10.60); adjusted p=0.01127.
- Qwen Q4, code calibration, gsm8k: -19.60 pp (pointwise paired 95% interval -23.00 to -16.20); adjusted p=5.157e-26.
- Qwen Q4, math calibration, gsm8k: -21.00 pp (pointwise paired 95% interval -24.40 to -17.70); adjusted p=5.719e-31.
- Qwen Q4, chat calibration, gsm8k: -23.10 pp (pointwise paired 95% interval -26.20 to -19.80); adjusted p=1.979e-40.
- Qwen Q6, math calibration, gsm8k: +3.50 pp (pointwise paired 95% interval +1.70 to +5.30); adjusted p=0.02287.
- Qwen Q6, chat calibration, gsm8k: +4.40 pp (pointwise paired 95% interval +2.70 to +6.10); adjusted p=0.0001254.
- Gemma Q3, code calibration, gsm8k: -6.50 pp (pointwise paired 95% interval -10.00 to -3.20); adjusted p=0.02705.
- Gemma Q3, math calibration, gsm8k: +11.60 pp (pointwise paired 95% interval +8.10 to +15.00); adjusted p=1.464e-08.
- Gemma Q4, code calibration, gsm8k: -11.10 pp (pointwise paired 95% interval -14.50 to -7.70); adjusted p=6.809e-08.
- Gemma Q4, chat calibration, gsm8k: -9.00 pp (pointwise paired 95% interval -12.40 to -5.50); adjusted p=3.865e-05.
- Gemma Q5, math calibration, gsm8k: -13.50 pp (pointwise paired 95% interval -16.90 to -10.20); adjusted p=2.736e-12.
- Gemma Q5, chat calibration, gsm8k: -9.70 pp (pointwise paired 95% interval -13.00 to -6.50); adjusted p=1.005e-06.
- Gemma Q6, chat calibration, gsm8k: -6.60 pp (pointwise paired 95% interval -9.50 to -3.70); adjusted p=0.001413.

## Interpretation

- Each task has its own difficulty and baseline; percentage-point losses describe these particular benchmarks, not universal task difficulty.
- Bootstrap intervals resample matched questions. They do not capture execution or calibration-sample variability.
- Question identities include document, prompt and target hashes. Binary sample means are checked against aggregate metrics.
- Precision effects use pure models; imatrix gains use same-recipe mixed controls. Do not subtract across those studies.
- Historical calibration budgets are unequal. Domain-specific recovery is descriptive until the equal-budget study finishes.
- Calibration-local weighted error omits correlations and downstream propagation. It is not accuracy loss.
- Block recovery above whole-model recovery subtracts each calibration/measurement pair's pooled recovery. It reveals concentration beyond overall improvement.
- Blocks and domains are related observations. No significance test treats block rows as independent repeated experiments.
- Block sensitivity provenance has cross-experiment limitations; the copied input manifest is hashed below.
- New equal-budget and repeated-inference results, when completed, are under `results/controlled_q4_study/`.

Rebuild: `python -m src.research_questions`. CSVs contain the plotted values and correctness transitions (wins/losses/ties).
