# Quantization Stability with llama.cpp

Compare how different quantizations of the same Llama checkpoint hold up on
three tasks - GSM8K, IFEval, and MMLU (college computer science) - when served
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

## Important: Patch lm-eval

After installing dependencies, replace the installed lm-eval GGUF backend with the `src/gguf.py` included in this repo.

Windows:

```powershell
$GGUF_PATH = python -c "import lm_eval.models.gguf as g; print(g.__file__)"
Copy-Item ".\src\gguf.py" $GGUF_PATH -Force
```

macOS:

```bash
GGUF_PATH=$(python -c "import lm_eval.models.gguf as g; print(g.__file__)")
cp src/gguf.py "$GGUF_PATH"
```

If `lm-eval` is reinstalled or upgraded, run this step again.

## Download Models

Models are saved to the repo's `models/` folder, which is created if missing.
GGUF files are not committed to Git.

Either place a GGUF there yourself as `models/<name>.gguf`, or set `hf_repo`
and `hf_filename` on the `Config` in `run_evaluation.py` - the server
downloads the file automatically the first time it is needed.

## Run an Evaluation

Edit `run_evaluation.py`: one `Config` per run, each starting from `BASE` and
overriding only what differs.

```python
BASE = Config(kl_enabled=True)

CONFIGS = [
    BASE.with_overrides(name="chat-q4", num_examples=50),
    BASE.with_overrides(name="uniform-q4", num_examples=50),
]
```

Then:

```bash
python run_evaluation.py
```

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

Edit the `quantize_model(...)` call at the bottom of `src/quantize.py`, then:

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

## Tests

```bash
python -m pytest tests -q
```

Unit tests for the main functions in `src/`. They need no GPU, server, or
network access - subprocess calls, HTTP, and lm-eval are mocked.

