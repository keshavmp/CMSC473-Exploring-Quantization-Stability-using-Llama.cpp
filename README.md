# Quantization Stability with llama.cpp

## Project layout

All generated files live in this repository, never in the parent folder:

```text
<repo>/
├── models/     # GGUF checkpoints (git-ignored)
└── results/    # one JSON report per model
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

After installing dependencies, replace the installed lm-eval GGUF backend with the `gguf.py` included in this repo.

Windows:

```powershell
$GGUF_PATH = python -c "import lm_eval.models.gguf as g; print(g.__file__)"
Copy-Item ".\gguf.py" $GGUF_PATH -Force
```

macOS:

```bash
GGUF_PATH=$(python -c "import lm_eval.models.gguf as g; print(g.__file__)")
cp gguf.py "$GGUF_PATH"
```

If `lm-eval` is reinstalled or upgraded, run this step again.

## Download Models

```bash
python download_models.py
```

Models are saved to the repo's `models/` folder, which is created if missing.
GGUF files are not committed to Git.

## Run a Model

Pass the GGUF to serve. Relative paths resolve against this repo, not your
current directory. Both scripts default to the BF16 checkpoint.

### Windows

```powershell
.\llm_server.ps1
.\llm_server.ps1 -Model "models\Llama-3.2-1B-Instruct-Q4_K_M.gguf"
```

If `llama-server` is not on your PATH, point at it explicitly:

```powershell
.\llm_server.ps1 -Server "C:\tools\llama.cpp\llama-server.exe"
```

### macOS

```bash
chmod +x llm_server_mac.sh
./llm_server_mac.sh
./llm_server_mac.sh models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

Leave the server running.

## Quantize

Edit `CONFIG` in `quantize_model.py` (source, output, recipe, tensor
overrides), then:

```bash
python quantize_model.py
```

The destination folder is created automatically. Always generate every
quantized variant from the same BF16 source, and never reuse the source
filename as the output - the script refuses to do so.

## Evaluate

Open another terminal, activate the virtual environment, then run:

```bash
python evaluate.py
```

Results are saved to the repo's `results/` folder as
`results/<model-name>.json`, where `<model-name>` is the served GGUF's
filename without its extension.
