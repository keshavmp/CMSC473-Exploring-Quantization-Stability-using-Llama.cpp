# Quantization Stability with llama.cpp

Compare how different quantizations of the same Llama checkpoint hold up on
three tasks - GSM8K, IFEval, and HellaSwag - when served
with llama.cpp.

## Project layout

All generated files live in this repository, never in the parent folder:

```text
<repo>/
├── data/               # calibration files for llama-imatrix (git-ignored)
├── models/             # GGUF checkpoints (git-ignored)
├── results/            # one folder per run + shared summary CSV / plots
├── src/                # config, server, quantize, evaluate, report, ...
├── tests/              # pytest unit tests
└── run_evaluation.py   # main entry point
```

Default paths resolve from the source file location, so they point to this
repository even when a script is launched from another working directory.
`Config.project_root` can override the destination for isolated experiments.

## Setup

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
winget install llama.cpp
```

### macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install llama.cpp
```

## GGUF backend

The evaluator loads `src/gguf.py` directly; no installed backend patch is needed.
For llama-server builds without prompt logprobs, HellaSwag choices are scored
one token at a time using pre-sampling probabilities. This is slower than the
legacy echo API, but preserves actual answer likelihoods.

Local evaluation uses four concurrent requests, persistent HTTP connections per
worker, and four server slots with continuous batching. `request_workers` in
`run_evaluation.py` controls concurrency; use 1 for sequential evaluation or 2
if GPU memory is limited. Each slot retains `context_size` tokens (the server's
total context allocation is multiplied by the worker count). Results remain in
input order. Generation limits, question selection, and scoring are unchanged.
Batching can introduce small numerical differences, so use the same worker
count for every model in a comparison. Benchmark representative questions before
choosing a worker count; speedup depends on the GPU and workload.
The backend also skips subsequent echo probes once prompt logprobs are found
to be unsupported. Request read timeout defaults to 300 seconds and server log
verbosity to 3 (info); both can be adjusted in `Config`.

## Download Models

Models are saved to the repo's `models/` folder, which is created if missing.
GGUF files are not committed to Git.

The BF16 source has a fixed local path:
`models/Llama-3.2-1B-Instruct-BF16.gguf`. If it is missing, the evaluation
script asks whether to download it from `unsloth/Llama-3.2-1B-Instruct-GGUF`.
The repo and filename are set near the top of `run_evaluation.py`.

## Run an Evaluation

Activate your Python environment, then run:

```powershell
python run_evaluation.py --limit 10  # first 10 questions per task, skips KL
python run_evaluation.py             # use the configured task counts
```

Edit `BASE` and `CONFIGS` in `run_evaluation.py` to change the experiments.
The BF16 reference and pure quantized variants are listed there. If a quantized
model is missing, the script stops before evaluation and tells you to edit
`src/quantize.py` and run it yourself. It never creates variants automatically.
Model hashes, task settings, counts, and per-question samples are still saved.

There was no complete Q2–Q8 mapping in the repository. The supported sweep uses
Q2=`Q2_K`, Q3=`Q3_K_M`, Q4=`Q4_K_M` (the existing recipe), Q5=`Q5_K_M`,
Q6=`Q6_K`, and Q8=`Q8_0`. **Q7 is unavailable in the installed llama.cpp**;
the runner reports this omission and does not substitute another type.
`--pure` disables recipe mixtures; ineligible tensors may retain floating types.

Per the updated sample policy, every model uses the same leading document IDs:
GSM8K first **1,000** of 1,319 `test` examples, IFEval all
(actual **541**, harness `test_split: train`), HellaSwag first **2,000** of
10,042 labeled `validation` examples (the installed task has no test split).
MBPP is excluded. Each evaluation records the resolved task configurations and
actual sample counts in its metadata.

The sweep keeps harness batch size 1, four concurrent HTTP requests,
context 2048 per slot, zero shots
(IFEval enforces zero shots), no chat template or extra system prompt, and
greedy generation (`temperature=0`, `do_sample=False`). The inspected lm-eval
0.4.13 API supports Python seed 0 and NumPy, Torch, and few-shot seeds 1234.
Task stop sequences and generation lengths come from the installed YAMLs.
Existing KL metrics remain enabled for the sweep and disabled for limited runs.

Each model writes separate `metrics.json`, `samples.json`, `config.json`, and
`metadata.json` under `results/<model>-sweep/` (limited runs add `-limitN`). Metadata
retains harness task versions/configs, `n-samples`, unique evaluated document
counts, selected IDs, model filename/hash, quantization type, pure setting, and evaluation settings.
It fails clearly if logged unique sample counts differ from the selection.

Each run starts llama-server, evaluates the three tasks, and stops the server.
Every run `<name>` writes:

- `results/<name>/metrics.json` - task metrics
- `results/<name>/samples.json` - per-question records (git-ignored)
- `results/<name>/config.json` - effective configuration
- `results/<name>/server.log` - server log (git-ignored)
- `results/<name>/kl.json` - domain KL metrics, when enabled
- `results/<name>/kl-<domain>.log` - raw KL output (git-ignored)

Use a different name when changing settings. Reusing a name replaces that run's
outputs. Historical results retain their original metrics; no configuration or
KL values are invented for them.

To serve a model yourself instead, leave this running in its own terminal:

```powershell
llama-server -m models\Llama-3.2-1B-Instruct-BF16.gguf --port 8080
```

## Quantize

Edit these settings at the bottom of `src/quantize.py`:

```python
SOURCE_MODEL = cfg.source_model
OUTPUT_MODEL = cfg.models_dir / "pure-q4.gguf"
QUANT_TYPE = "Q4_K_M"
PURE = True
```

Change the output filename and recipe to create each variant in the mapping
above. Every variant should use the same original BF16 source. Then run:

```bash
python src/quantize.py
```

The destination folder is created automatically. Always generate every
quantized variant from the same BF16 source.

Optional - build the calibration data and importance matrices used by
imatrix-guided quantization:

```bash
python src/imatrix.py
```

## Compare Results

After one or more evaluations, compare every run side by side:

```bash
python src/report.py                     # every results/<name>/metrics.json
python src/report.py chat-q4 uniform-q4  # only these runs, in this order
```

This prints one row per run and one column per task, and saves
`results/summary.csv` and `results/summary.png` (task score bar chart).
KL statistics appear in the table and CSV in nats, with a separate
`results/kl_summary.png` chart for domain means (lower is better).
Older runs without KL have blank KL cells. Legacy flat result files are also
accepted, with nested results taking precedence for duplicate names.
New result files and new tasks are picked up automatically - no edits needed.

## Size and stability comparisons

The summary includes `size_mib` (GGUF file size, not runtime memory). New runs
record the size in `config.json`; older runs use their local GGUF when present.
Missing model files leave the size blank. `results/size_quality.png` plots each
task score and available mean KL against size, with one point per run.

`results/stability.csv` compares each run with the BF16 baseline on matching
saved samples. Columns count `compared`, `correct_to_incorrect`,
`incorrect_to_correct`, and `unchanged` (same correctness, even if the answer
text differs). Only matching document, prompt, and target hashes are compared.
GSM8K uses strict exact match, IFEval uses prompt-level strict accuracy, and
MMLU uses accuracy. Alternate filters are excluded to avoid counting a
question twice. Counts refer only to the shared questions, not the full run.
Missing baseline samples produce a CSV with headers and no comparison rows.

To choose another reference run:

```bash
python src/report.py --baseline uniform-q4
```

## Domain KL divergence

The runner enables KL with `BASE = Config(kl_enabled=True)`. Set it to `False`
for task-only evaluation. Point `llama_perplexity` at your local
`llama-perplexity` executable if it differs from the default Windows build.

After each task evaluation stops its server, llama-perplexity compares the
model against `source_model` (BF16) on the same fixed token sequences. Use only
variants of that checkpoint with the same tokenizer. The reported direction
is BF16 || evaluated model. Each domain records mean, estimated uncertainty,
median, 95th percentile, and evaluated token count. These are corpus metrics,
not KL scores for the benchmark questions.

By default, `data/kl/chat.txt`, `math.txt`, and `code.txt` are generated once
from UltraChat train_sft, GSM8K train, and MBPP train. Each dataset is shuffled
with seed 42; examples 300 through 331 are used, skipping the 300 examples
selected by the current imatrix script. These examples are also separate from
the benchmark test splits. If calibration settings change, choose a new
non-overlapping KL subset. Saved text is reused so every model sees the same
text. For your own held-out corpus, set `kl_text_dir=Path("...")`: each `.txt`
file becomes a domain. Preserve the text files with the experiment records.

The default budget is two 512-token chunks per domain. llama.cpp scores the
second half of each chunk (510 tokens per domain); this is a quick comparison,
so increase `kl_chunks` for final experiments. Custom text needs at least two
full context chunks. Settings must score at least 100 tokens for final statistics. Text that cannot fill the requested budget fails clearly.

Reference distributions are stored in git-ignored `results/_kl_reference/`
and reused across runs. Cache identity includes text contents, source-model
and executable paths/sizes/modification times, context, chunk count, and GPU
settings. Text uses plain role labels rather than tokenizer-specific chat tokens. Replacing a model in place while preserving its size and timestamp
requires deleting its reference cache. Default Llama 3 references use roughly
125 MiB per domain; larger context or chunk counts increase storage. The
reference format compresses probabilities, so BF16 compared against its own
saved reference can have a small nonzero KL floor. The tool's uncertainty is
an estimate across tokens, not a confidence interval across independent runs.

KL failure marks the configuration as failed while keeping any completed task
metrics. Rerun the configuration after correcting the tool, corpus, or model.

## Block and tensor sensitivity

Looks at which parts of Llama 3.2 1B are most sensitive to quantization. For each precision combination (background -> target), two separate sweeps lower one part of the model to the target precision, keep everything else at the background precision, and score the result with the domain KL tool above (BF16 || variant on the chat, math, and code text).

block sweep, 16 variants: one whole block (blk.N, its seven weight matrices), N = 0 to 15
tensor sweep, 7 variants: one tensor type (attn_q, attn_k, attn_v, attn_output, ffn_gate, ffn_up, ffn_down) in every block

Combinations: BF16 -> Q2, Q4, Q6 | Q8 -> Q2, Q4, Q6 | Q6 -> Q2, Q4 | Q4 -> Q2 | Q3 -> Q2 (10 combinations, 230 variants). Types follow the pure sweep (Q2_K, Q3_K_M, Q4_K_M, Q6_K, Q8_0, applied with --pure). The embedding table is never varied; it stays at the background precision. A variant's sensitivity is the KL it adds over the pure background model: KL(variant) - KL(pure background).

python -m src.sensitivity --backgrounds bf16 --levels q4_k --sweep block --blocks 3   # one variant, to time it
python -m src.sensitivity --backgrounds q8_0 --sweep tensor                           # one background, tensor sweep
python -m src.sensitivity                                                             # everything
python -m src.report_blocks                                                           # tables, heatmaps, CSVs

Needs pip install gguf and the llama.cpp paths in src/config.py. Run it with python -m; running python src/sensitivity.py makes src/gguf.py shadow the installed package. Runs that already have a kl.json are skipped, so an interrupted sweep resumes. Variant GGUF files are deleted after scoring unless --keep is given (BF16-background variants are 2.4 GB while they exist). KL uses KL_CHUNKS = 4 (about 1,020 scored tokens per domain), so values from the default two chunks are not directly comparable; pure_baselines.csv re-scores the BF16 source and every pure model with the same setting.

Every variant is built from the BF16 checkpoint with llama-quantize --pure and --tensor-type overrides. llama-quantize ignores --tensor-type when asked for BF16, so a BF16 background uses the Q8_0 recipe and pins every weight outside the target back to BF16. After each build the script reads the new file and checks the type of every weight, embedding included; it stops if anything other than the intended weights changed.

Results go to results/Block_Sensitivity_Differences/:

<run>/kl.json - one scored model
block_sensitivity.csv, tensor_sensitivity.csv, pure_baselines.csv - all results, merged across batches
<sweep>_sensitivity_summary.csv - sensitivity, uncertainty, and a clear/not-clear flag per variant and domain
<sweep>_sensitivity_normalized.csv - the heatmap values
<sweep>_sensitivity_heatmap.png - one heatmap per sweep (--lines also draws per-combination line plots)

Reading the heatmaps: each row is one combination, divided by its own mean sensitivity (negative values count as 0), so 1.0 is an average block or tensor type. Without this, Q2 targets (about 1000x the KL of Q6 targets) hide everything else. Cells average chat, math, and code, and the last row averages all combinations. Row labels give the raw peak KL in nats; hatched cells are within 2x uncertainty of zero in most domains. report_blocks also prints each combination's ranking, the most/least sensitive ratio, and the Spearman rank correlation between domains (1 = same ranking).

Findings so far (first full sweep): block 1 is the most sensitive block in every combination and domain, about 10 to 11x an average block at Q2 targets and 6 to 7x at Q4 and Q6 targets. ffn_down is the most sensitive tensor type in every combination, about 4x an average tensor type at Q4 and Q6 targets and 5.5x at Q2 targets; attn_q and attn_k are the least sensitive.

These sweeps measure quality loss directly, so they can be compared with the block-level imatrix statistics above to test whether activation statistics predict sensitivity. Limits: one model, about 1,020 scored tokens per domain per run, and three text domains. Uncertainties are combined as independent. Q8 and Q3 backgrounds are noisier, with many small effects inside 2x uncertainty; the BF16 background is the cleanest. Each result changes one part at a time.

## Tests

```bash
python -m pytest tests -q
```

Unit tests for the main functions in `src/`. They need no GPU, server, or
network access - subprocess calls, HTTP, and lm-eval are mocked.

