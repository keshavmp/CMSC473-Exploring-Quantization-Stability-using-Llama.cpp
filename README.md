# Quantization Stability with llama.cpp


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

Models are saved in:

```text
models/
```

GGUF files are not committed to Git.

## Run a Model

Choose the model inside the server script.

### Windows

```powershell
.\llm_server.ps1
```

### macOS

```bash
chmod +x llm_server.sh
./llm_server.sh
```

Leave the server running.

## Evaluate

Open another terminal, activate the virtual environment, then run:

```bash
python evaluate.py
```

Results are saved in:

```text
results/
```
