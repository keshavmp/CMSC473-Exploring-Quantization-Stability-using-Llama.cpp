# CMSC473-Exploring-Quantization-Stability-using-Llama.cpp
How stable is LLM quantization sensitivity across workloads, calibration data, context lengths, and hardware, and can llama.cpp’s importance-matrix statistics predict this sensitivity well enough to guide efficient weight and KV-cache precision choices?

## Setup

```powershell
pip install -r requirements.txt
winget install llama.cpp
```

### Important: Replace lm-eval `gguf.py`

The installed `lm-eval` GGUF backend is currently incompatible with the version of `llama-server`.

This repo contains the updated `gguf.py`. After installing requirements, overwrite the installed version:

```powershell
$GGUF_PATH = python -c "import lm_eval.models.gguf as g; print(g.__file__)"
Copy-Item ".\gguf.py" $GGUF_PATH -Force
```

This must be done after installing/updating `lm-eval`.

## Run

Download the GGUF models with:

```text
download_models.ipynb
```

Select the model in:

```text
llm_server.ps1
```

Start the server:

```powershell
.\llm_server.ps1
```

Then in another terminal:

```powershell
python evaluate.py
```

Results are saved to `results/`.

## Notes

* We use standalone `llama.cpp`, not `llama-cpp-python`.
* `.gguf` model files go in `models/` and should not be committed.
* The model only needs to be changed in `llm_server.ps1`.
