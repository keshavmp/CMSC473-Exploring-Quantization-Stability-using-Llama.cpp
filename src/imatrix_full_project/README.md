# Extended imatrix research pack (P1 + P2)

This **adds** analysis and experiment modules alongside your team's existing `src/imatrix.py` and `run_evaluation.py`. It produces CSV, PNG, and Markdown outputs; JSON output is not required.

## Environment

```bash
python -m pip install gguf numpy pandas matplotlib pytest
cd ~/llama.cpp
cmake --build build-cpu --target llama-quantize -j 4
cd ~/CMSC473-Exploring-Quantization-Stability-using-Llama.cpp
```

## Step 1: Analyze imatrix results

Your existing imatrix script must have already generated `models/imatrix/chat.imatrix.gguf`, `math.imatrix.gguf`, and `code.imatrix.gguf`.

```bash
python src/imatrix_research.py analyze
python src/imatrix_research.py recommend --high-fraction .25
```

Output: tensor/layer features, domain rankings, correlation matrices, three figures, preliminary recommendations and `REPORT.md`.

## Step 2: Plan P1 and P2 quantization experiments

```bash
python src/imatrix_ablations.py plan
```

This writes `results/imatrix_analysis/experiment_plan.csv` and `overrides/*.txt` for:
- uncalibrated uniform Q4 and Q6 model controls
- imatrix-guided uniform Q4 for chat, math, code
- domain-guided Q4/Q6 mixed-precision models
- P1 all-high control and one-layer-low ablations for every measured transformer block

**Note:** The optimizer uses fractions of layers, not an exact byte budget. Actual model sizes must be measured. Some tensor types may be adjusted or rejected by llama.cpp depending on tensor shape.

## Step 3: Quantize models (start with one)

```bash
python src/imatrix_ablations.py build --only mixed-chat --dry-run
python src/imatrix_ablations.py build --only mixed-chat
```

Build the full design only if you have compute and storage budget:

```bash
python src/imatrix_ablations.py build
```

Model outputs remain in `models/imatrix_experiments/` and are intentionally excluded from Git. Quantization logs are written in `results/imatrix_analysis/logs/`.

## Step 4: Evaluate real models with your repo's existing evaluator

Your team runner (`run_evaluation.py`) reads `Config` objects in code. There is **no verified CLI interface** to automatically pass arbitrary GGUF paths to it. For each experiment you choose to test, set its `name` and model path in the team's `Config`, run `python run_evaluation.py`, then collate the benchmark measurements into a CSV. Keep prompt templates, task splits, sample counts and seed fixed.

Sample **header only** (no made-up scores):

```csv
model,gsm8k,ifeval,mmlu,size_mib
```

Use model names from `experiment_plan.csv`, exactly. Sizes are also recorded there after quantization. Include P1's `p1-all-high` and `p1-layer-XX` evaluations to compute P1 sensitivity; include uniform and mixed model evaluations to compute the quality–size Pareto frontier.

## Step 5: Analyze benchmark outputs

```bash
python src/imatrix_results.py --scores-csv results/benchmark_scores.csv
```

Creates `p1_layer_sensitivity.csv` and `p1_layer_ablation.png` when controlled ablation data exist; `p2_model_comparison.csv` and `p2_pareto.png` when uniform/mixed runs and sizes exist; plus `EXPERIMENT_REPORT.md`.

Validate the imatrix predictor against independently measured P1 layer sensitivity:

```bash
python src/imatrix_research.py validate --p1-csv results/imatrix_analysis/p1_layer_sensitivity.csv
```

Also available:

```bash
python src/imatrix_research.py compare --scores-csv results/benchmark_scores.csv
```

## Git

```bash
git switch viharika/imatrix-analysis
git add src/imatrix_research.py src/imatrix_ablations.py src/imatrix_results.py
git add -f results/imatrix_analysis/*.csv results/imatrix_analysis/*.png results/imatrix_analysis/*.md
git commit -m "Add P1 ablation design and P2 mixed-precision experiment analysis"
git push origin viharika/imatrix-analysis
```

Add only files that exist; missing wildcard matches may cause a Git error. GGUF weights/imatrices are intentionally not copied into normal Git history.

## Scientific scope

`imatrix_research.py` builds an **activation-energy ranking heuristic**, not proven sensitivity. P1 measures accuracy loss after lowering one block's precision. These are two fundamentally different quantities. This package prepares and analyzes both; it cannot produce real benchmarks without the actual GGUF runs and your team's evaluation configuration.
