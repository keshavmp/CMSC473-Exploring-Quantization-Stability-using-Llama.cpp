# Pure quantization sweeps across model families (Intel Mac, CPU)

This folder repeats the pure Q2–Q8 sweep from `results/Pure_Sweep/` for more
model families, using the shared code in `src/` unchanged. Everything new
(code, models, results) stays in `rohan/`.

| Family | Model | Source GGUF | Results |
|---|---|---|---|
| `qwen` | Qwen2.5-1.5B-Instruct | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` (F16; no BF16 is published) | `results/Qwen_Pure_Sweep/` |
| `gemma` | Gemma-3-1B-it | `unsloth/gemma-3-1b-it-GGUF` (BF16) | `results/Gemma_Pure_Sweep/` |
| `llama` | Llama-3.2-1B-Instruct | `unsloth/Llama-3.2-1B-Instruct-GGUF` (BF16) | `results/Llama_Pure_Sweep/` |

Only the source GGUF is downloaded from Hugging Face. The six variants
(Q2=`Q2_K`, Q3=`Q3_K_M`, Q4=`Q4_K_M`, Q5=`Q5_K_M`, Q6=`Q6_K`, Q8=`Q8_0`, all
`--pure`) are made locally with `llama-quantize`, the same way as the original
Llama sweep. The baseline run is always named `BF16-pure-sweep`, so that
`src/report.py` detects it. For Qwen this baseline is actually the F16 source.

## Settings

These match `run_evaluation.py` except for the sample caps and the device:

- GSM8K 100, IFEval 50, HellaSwag 200: the first N document IDs of each task.
  The original Llama sweep used 1000 / 541 / 2000.
- Zero-shot, greedy decoding, 4 server slots of 2048 context each.
- KL against the source model on chat/math/code text, 2 × 512-token chunks.
- CPU only (`gpu_layers="0"`). The Iris Plus iGPU shares the CPU's memory
  bandwidth, and llama.cpp's Metal backend targets Apple silicon.

## Setup (once)

On this Intel Mac, `brew install llama.cpp` compiles LLVM, Rust and Node from
source, which takes hours. Build llama.cpp itself instead, next to this repo
(`../llama.cpp/build/bin`, where `src/config.py` looks on macOS):

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git ../llama.cpp
cmake -S ../llama.cpp -B ../llama.cpp/build -DGGML_METAL=OFF -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build ../llama.cpp/build -j 8 \
  --target llama-server llama-quantize llama-perplexity llama-bench
```

Then set up Python:

```bash
/usr/local/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -r rohan/requirements-mac-intel.txt
python -m pytest rohan/tests -q
```

`torch==2.2.2` is the last release built for Intel Macs. It prints a harmless
"Failed to initialize NumPy" warning: the pipeline only uses it for
`torch.manual_seed`.

## Prepare models and data (before an overnight run)

```bash
python -m rohan.run_family_sweep --family qwen gemma llama --prepare   # download sources, datasets, KL texts and references
python -m rohan.quantize_family --family qwen gemma llama              # 18 pure GGUFs
python -m rohan.run_family_sweep --family qwen gemma llama --limit 3   # smoke test
```

Smoke-test runs go to separate `*-limit3` folders and never mix with full results.

## Overnight runs

Plug in the Mac, keep the lid open (closing it forces sleep), and quit heavy
apps. `caffeinate -i` stops the Mac from idle-sleeping, and the display can
still turn off.

```bash
# Night 1: Qwen, then Gemma (~9–11 h)
nohup caffeinate -i python -m rohan.run_family_sweep --family qwen gemma --resume \
  > rohan/results/overnight-1.log 2>&1 &

# Night 2: any unfinished Gemma runs, then Llama (~4–5 h)
nohup caffeinate -i python -m rohan.run_family_sweep --family gemma llama --resume \
  > rohan/results/overnight-2.log 2>&1 &
```

- **Progress:** `tail -f rohan/results/overnight-1.log`.
- **Stop:** `pkill -f rohan.run_family_sweep`.
- **Resume:** `--resume` skips every run that already has `metrics.json` and
  `kl.json`, so an interrupted run is simply redone next time.

## Reports

```bash
python -m rohan.report_families
```

- **Per family:** writes `summary.csv`, `summary.png`, `kl_summary.png`,
  `size_quality.png` and `stability.csv` into each family's results folder.
- **Across families:** writes `results/family_comparison.csv` and
  `results/family_comparison.png`, one panel per task plus mean KL. The
  original GPU Llama sweep appears as a gray dashed reference line. Its sample
  sizes and hardware differ, so compare its trend rather than exact numbers.

To add a family (e.g. Mistral), add an entry to `FAMILIES` in
`run_family_sweep.py` with its Hugging Face repo and source filename.
