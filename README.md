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
├── results/            # one JSON report per run + summary.csv / summary.png
├── src/                # config, server, quantize, evaluate, report, ...
├── tests/              # pytest unit tests
└── run_evaluation.py   # main entry point
```

Every script resolves these folders from its own location, and refuses to run
from anywhere else - so running a script by mistake can never create a second
copy of `models/` or `results/` in a parent directory.

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
BASE = Config()

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

- `results/<name>.json` - task metrics
- `results/<name>_samples.json` - per-question records
- `results/<name>-server.log` - server log (git-ignored)

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
python src/report.py                     # every results/*.json
python src/report.py chat-q4 uniform-q4  # only these runs, in this order
```

This prints one row per run and one column per task, and saves
`results/summary.csv` and `results/summary.png` (grouped bar chart).
New result files and new tasks are picked up automatically - no edits needed.

## Tests

```bash
python -m pytest tests -q
```

Unit tests for the main functions in `src/`. They need no GPU, server, or
network access - subprocess calls, HTTP, and lm-eval are mocked.

