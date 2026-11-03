# Mixed-Q4 imatrix comparison

The experiment compares four models created directly from the same BF16 checkpoint:

| Variant | Recipe | Calibration importance |
|---|---|---|
| control | Q4_K_M, mixed | None |
| code | Q4_K_M, mixed | Verified BF16 code imatrix |
| math | Q4_K_M, mixed | Verified BF16 math imatrix |
| chat | Q4_K_M, mixed | Verified BF16 chat imatrix |

No variant passes `--pure`. The runner checks that all four models have identical
stored tensor-type layouts, so an imatrix difference is not confounded with a
different precision allocation. The prior pure-Q4 results provide context, not the matched control.

## Smoke test

```powershell
python run_imatrix_comparison.py --limit 5 --kl-chunks 2
```

This evaluates the same leading **five examples per benchmark** (GSM8K, IFEval,
HellaSwag), or 60 model/question combinations across all four variants. It retains
the original prompt/scoring rules and four request workers. Benchmark generation
limits are unchanged; only example counts are reduced.

KL runs separately on all three existing held-out texts with context 512 and two
chunks: 510 scored tokens per domain. The runner also measures local reconstruction
error against BF16 for all 112 projections, using all three BF16 imatrices. Every
calibrated model is thus evaluated under all three domains, not only its own.

The smoke test establishes that quantization, loading, scoring, KL, result recording,
and reconstruction analysis work. **Five examples cannot establish a reliable accuracy
improvement or select a scientifically defensible winning calibration domain.**

The existing imatrices have code/math/chat coverage of 49/97/699 chunks of 512 tokens.
They are reused to avoid a new collection run during this smoke test. For a controlled
calibration-domain study, collect a new set with equal token budgets; use a new model
directory/protocol identifier so those inputs are not confused with this pilot.

## Files and resuming

- Models and quantization logs/manifests: `models/q4-imatrix-comparison/` (git-ignored).
- Smoke results: `results/Q4_Imatrix_Comparison_smoke5_kl2/`.
- `comparison_plan.json`: sample caps, calibration coverage, source hashes and limitations.
- `quantization_manifest.json`: actual tensor precisions and model/input hashes.
- `summary.csv`: benchmark scores, file sizes and per-domain KL.
- `local_errors.csv`: per-tensor/domain errors and improvement over the mixed control.
- `local_error_summary.csv`: pooled local errors by model and domain.
- Per-model `metadata.json`: selected document IDs, observed sample counts and scoring configuration.
- Per-model `comparison_complete.json`: verified evaluation identity and output hashes.

Rerunning the identical command reuses verified model files and completed evaluations.
Partially failed evaluations may rerun that model's evaluation; this is not per-question resume.
A completed run with changed settings is rejected: use a new `--experiment NAME`.
Existing unrelated models and published findings are preserved.

For individual steps:

```powershell
python run_imatrix_comparison.py --stage quantize
python run_imatrix_comparison.py --stage evaluate
python run_imatrix_comparison.py --stage analyze
```

All stages validate/reuse the model set first. The experiment requires the local
BF16 weights, verified imatrix inventory and input files, llama.cpp tools, and the
installed evaluation datasets. It does not download models automatically.

## Next stages

After the smoke test succeeds, increase KL coverage on matching texts for all four
models, then use a larger matched benchmark subset. A full evaluation is explicit:

```powershell
python run_imatrix_comparison.py --full --kl-chunks 4
```

`--full` restores 1,000 GSM8K, all 541 IFEval, and 2,000 HellaSwag examples per model.
It creates a separate result folder. Do not mix different KL chunk counts or benchmark
sample sizes in the same comparison. A model is successful only when final metrics,
sample metadata and KL all complete. `run_status.json` records individual failures.

Interpret preservation in two ways: the stored tensor type shows bit allocation,
while lower imatrix-weighted reconstruction error shows better local preservation
at that allocation. Neither by itself proves an accuracy improvement. Compare
per-tensor improvements with sensitivity rankings at Q4, not precision-averaged rankings.
