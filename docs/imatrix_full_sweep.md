# Full imatrix recovery sweep

From the repository folder, use the existing Python environment:

```powershell
& "..\Capstone Project\Scripts\python.exe" -u run_imatrix_sweep.py --full --kl-chunks 4 --local-errors
```

This runs **24 model configurations**: Q2/Q3/Q4/Q5/Q6/Q8, each with no imatrix
and BF16-derived code/math/chat imatrices. Every model is created directly from
the original BF16 checkpoint, using the standard recipe without `--pure`.
Actual stored tensor-type layouts must match the no-imatrix control at that precision.
Q6/Q8 can be uniform despite omitting `--pure`; mixtures depend on the recipe.
Existing verified Q4 comparison models are reused. Other files are not overwritten.

Each model uses the established full benchmark protocol:

- GSM8K: first 1,000 test questions; flexible-extraction exact match.
- IFEval: all 541 examples; prompt-level strict accuracy.
- HellaSwag: first 2,000 validation questions; answer-length-normalized accuracy.
- Same document IDs, prompt settings, zero-shot configuration and four request workers.
- KL against BF16 on each of code/math/chat: context 512, four chunks, 1,020 scored tokens per domain.
- With `--local-errors`, reconstruction estimates for all 112 measured projections and all three domains.

Here, full means the existing project evaluation counts, not every available benchmark example.
An existing BF16 reference is reused only after verifying its model hash, harness version,
selected IDs, sample counts and key evaluation settings. Historical pure quantizations
are not used as the matched controls; this sweep measures calibration within each standard recipe.

Results go under `results/Imatrix_Recovery_Sweep_full_kl4/`:

| File | Purpose |
|---|---|
| `sweep_plan.json` | Recipes, counts, source/calibration identities and BF16 reference |
| `sweep_status.json` | Current invocation's completed/failed precision groups |
| `run_status_q*.json` | Per-model evaluation status at each precision |
| `summary.csv` | Combined scores, gains, recovered-loss percentages and KL reductions |
| `summary_q*.csv` | Individual precision summaries |
| `summary_notes.txt` | Formula definitions and limits |
| `mixed-q*-*/` | Metrics, samples, metadata, KL and verified completion marker |
| `local_errors/q*/` | Optional per-tensor reconstruction estimates |

Models and quantization manifests are saved under `models/q*-imatrix-comparison/`.
Model files remain git-ignored. Calibration text and raw logits are not published.

The primary effect is the score gain relative to the **same-recipe, same-precision
no-imatrix control**, expressed in percentage points. Recovered BF16 score loss is
`100 * (imatrix_score - control_score) / (BF16_score - control_score)`.
That percentage is undefined when the control matches or exceeds BF16; small gaps
make it unstable. Always examine the raw scores and percentage-point gains first.
KL recovery is reported as a percentage reduction relative to the matched control.

## Running and resuming

Keep the terminal and computer running for this long evaluation. Models run sequentially
on one GPU; do not start another server on port 8080 at the same time.
Rerun the exact same command to resume. Verified models and completed model evaluations
are reused. If interrupted mid-evaluation, that model restarts; this is not per-question resume.
Changed completed-run settings are rejected; use a new `--experiment NAME` when changing the protocol.
Failed configurations are recorded and the remaining groups continue. The command exits
nonzero if any group fails. Review `sweep_status.json` before treating the sweep as complete.

Preview the plan without quantizing or evaluating:

```powershell
& "..\Capstone Project\Scripts\python.exe" run_imatrix_sweep.py --full --kl-chunks 4 --plan
```

For a subset, choose the desired precisions and a distinct scope:

```powershell
& "..\Capstone Project\Scripts\python.exe" -u run_imatrix_sweep.py --full --precisions q2 q3 q4 --experiment Imatrix_Recovery_Q2_Q3_Q4 --kl-chunks 4
```

Use a normal user terminal with access to the existing Hugging Face dataset cache.
The runner never downloads model weights automatically. With all datasets already cached,
optional offline mode can avoid Hugging Face network lookups:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"
```

## Two terminals

Partition precisions so the terminals do not write the same model files. Give each
terminal a distinct port and experiment directory. From the repository folder:

Terminal 1:

```powershell
& "..\Capstone Project\Scripts\python.exe" -u run_imatrix_sweep.py --full --precisions q2 q3 q4 --port 8080 --experiment Imatrix_Recovery_low --kl-chunks 4 --local-errors
```

Terminal 2:

```powershell
& "..\Capstone Project\Scripts\python.exe" -u run_imatrix_sweep.py --full --precisions q5 q6 q8 --port 8081 --experiment Imatrix_Recovery_high --kl-chunks 4 --local-errors
```

These write separate `summary.csv` files under `results/Imatrix_Recovery_low/` and
`results/Imatrix_Recovery_high/`. Do not also run the unpartitioned sweep at the same time.
If the unpartitioned sweep is already running, leave it alone; use this split for a fresh run.
Restart each terminal with its original command to reuse its completed evaluations.
Changing experiment folders does not import completed evaluations from other result folders.
Ports only separate HTTP servers; all processes still share the same GPU, CPU and memory.
On the 8 GB GPU, start with at most two and compare throughput/memory use. Parallel runs
can be slower or run out of memory; a single sweep already uses four concurrent requests.

## Interpretation limits

Existing calibration lengths are unequal: code/math/chat 49/97/699 x 512 tokens.
This tests each available imatrix against its matched control; domain differences
cannot be attributed purely to calibration content or treated as statistically established.
Q8_0 may ignore imatrix importance weights; include it as a negative control rather
than expecting improvement. Four KL chunks are still a short text evaluation.
Standard errors are retained in per-model metrics. Accuracy improvements need paired
example-level analysis before claiming significance. Local error recovery is a calibration
measure and does not itself prove held-out accuracy recovery.
