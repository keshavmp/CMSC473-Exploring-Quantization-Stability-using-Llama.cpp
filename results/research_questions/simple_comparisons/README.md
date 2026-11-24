# Simple comparison plots

**One metric per figure.** Model-family colors stay blue (Llama), orange (Qwen), green (Gemma).
Figure 06 instead colors calibration domains, explicitly labeled in its legend.

| Figure | One quantity measured | What to compare |
|---|---|---|
| 01 | Accuracy (%) | Precisions and families, separated by benchmark task |
| 02 | Accuracy lost from source (pp) | Precisions and families, separated by benchmark task |
| 03 | Accuracy gained from imatrix (pp) | Precisions and families, separated by benchmark task |
| 04 | Output KL removed (%) | Precisions and families, separated by measurement domain |
| 05 | Local reconstruction error removed (%) | Blocks and families, separated by precision |
| 06 | Local reconstruction error removed (%) | Blocks and calibration datasets, separated by model family and measurement domain (Q4) |

At Q4, block-recovery rankings are similar across calibration choices: mean pairwise Spearman agreement is approximately 0.994 (Llama), 0.914 (Qwen), and 0.971 (Gemma). These are descriptive averages across related groups, not significance tests or proof of dataset-specific targeting.
These figures summarize saved historical results. They do not include the unfinished equal-budget calibration or execution-repeat study.
Source checkpoints differ in size, architecture, and training; family differences cannot be assigned solely to architecture.
Figures 01/02 use standard mixed-recipe no-imatrix controls; they are a separate series from the pure Llama sweep.
Question-resampling intervals are in the parent results folder. Simple averages here are descriptive and hide calibration-specific variation.
All CSVs preserve plotted values. Rebuild: `python -m src.simple_research_plots`.
