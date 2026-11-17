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

For the full write-up (figures, re-scored GSM8K, tensor-type check, comparison
with `results/Pure_Sweep/` and `results/task21_kv_workloads/`):

```bash
python -m rohan.analyze_findings   # writes rohan/results/findings/
```

To add a family (e.g. Mistral), add an entry to `FAMILIES` in
`run_family_sweep.py` with its Hugging Face repo and source filename.

# Context length × weight quantization (Llama-3.2-1B)

`run_long_context.py` asks whether weight-quantization damage grows with
context length, and records time, speed and memory for every run.

It differs from the teammates' `results/task15_long_context_results/` in what
it varies:

| | Task15 | This study |
|---|---|---|
| Weights | Fixed at mixed Q2_K | Varied: BF16, pure Q2–Q8 |
| KV cache | Varied | Fixed at f16 |
| Hardware | Apple-silicon GPU | This Intel CPU |

| Experiment | What it measures |
|---|---|
| E1 `kl-*` | KL from BF16 with 4K, 8K and 16K windows over the same 32K tokens of essays. Each window scores its second half, so larger windows measure later positions. 32K is skipped: `llama-perplexity` keeps n_ctx × vocab float logits, about 17 GB at 32K, more than this 16 GB Mac has |
| E2 `ruler-*` (`niah_multikey_2`, `ruler_vt`) | Retrieval among many distractor keys, and multi-hop variable tracking, at 4K, 8K and 16K |
| E2b `ruler-*-niah_single_1-*` | Simple retrieval for every quant level. Also runs task15's exact mixed Q2_K model, so its f16-cache results can be checked on this CPU |
| E3 `gsm8k-*-padNk` | The pure sweep's GSM8K questions behind N tokens of essay padding: 100 at 4K, the first 50 at 16K. Padding 0 is `Llama_Pure_Sweep` |
| E4 | `niah_multikey_2` at 32K for BF16 and Q4 |

Every run's `stats.json` records:

- **Accuracy**
- **Time:** wall time, load time, seconds per question, mean time to first
  token
- **Speed:** prompt and generation tok/s, prompt-cache hit rate
- **Memory:** baseline and peak memory, KV-cache size, model file size

All values come from llama-server's per-request `timings`, and from
psutil memory sampling every 0.1 s.

```bash
python -m rohan.run_long_context --prepare                 # tokenizer, essays, GSM8K, KL text
python -m rohan.run_long_context --part pilot              # small timing and sanity run
python -m rohan.run_long_context --part night1 --list      # runs and done/todo status
nohup caffeinate -i python -m rohan.run_long_context --part night1 --resume \
  > rohan/results/long-context-night1.log 2>&1 &
```

Estimates come from the pilot. Seconds per question:

- BF16: about 48 at 4K and 285 at 16K
- Q4: about 30 at 4K and 200 at 16K

| Part | Runs | Estimate |
|---|---|---|
| `night1` | E2 for BF16 and Q4 | ~8.5 h |
| `night2` | E2 for Q2 (4K only), Q6 and Q3 | ~8 h |
| `night3` | E3 | ~6.5 h |
| `night4` | E1, E2b at 4K, E2b at 16K for Q2 and mixed Q2_K | ~6.5–7 h |
| `night5` | E2b at 16K for BF16 and Q4, then E4 | ~7 h |

After each night, `python -m rohan.analyze_long_context` refreshes the tables
and figures in `results/Llama_Long_Context/findings/`. The interim write-up is
`LONG_CONTEXT_FINDINGS.md` in the same folder.

Runs are ordered by length, so an interrupted night leaves the short lengths
complete. `--resume` skips any run that has a `stats.json`. Pilot runs go to
`results/Llama_Long_Context-pilot/`, and everything else to
`results/Llama_Long_Context/`.
