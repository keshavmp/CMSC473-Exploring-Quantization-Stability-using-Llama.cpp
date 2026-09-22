# Quantization Stability with llama.cpp

This project evaluates how GGUF quantization affects Llama 3.2 1B performance.

Pipeline:

```text
GGUF Model → llama-server → lm-eval → Results
```

## Setup

Clone the repo:

```bash
git clone https://github.com/keshavmp/CMSC473-Exploring-Quantization-Stability-using-Llama.cpp.git
cd CMSC473-Exploring-Quantization-Stability-using-Llama.cpp
```

### Windows

Run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
```

Activate the environment later with:

```powershell
.\.venv\Scripts\Activate.ps1
```

### macOS

Make the setup script executable and run it:

```bash
chmod +x setup.sh
./setup.sh
```

Activate the environment later with:

```bash
source .venv/bin/activate
```

The setup scripts:

* create `.venv`
* install Python dependencies
* install/check llama.cpp
* overwrite the installed `lm_eval/models/gguf.py` with the patched `gguf.py` included in this repo

> If `lm-eval` is reinstalled or upgraded, run the setup script again so the patched `gguf.py` is copied back.

## Download Models

```bash
python download_models.py
```

Models are stored locally in:

```text
models/
```

GGUF model files are not committed to Git.

## Run a Model

Choose the model by editing the `MODEL` line in the server script.

### Windows

```powershell
.\llm_server.ps1
```

### macOS / Linux

```bash
chmod +x llm_server.sh
./llm_server.sh
```

Leave the server running.

## Evaluate

Open another terminal, activate `.venv`, then run:

```bash
python evaluate.py
```

The evaluator automatically gets the model name from the running llama-server.

Results are saved in:

```text
results/
```
