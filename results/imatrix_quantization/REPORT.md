# imatrix Quantization Experiments

Source checkpoint: `/Users/viharikapenmetsa/CMSC473-Exploring-Quantization-Stability-using-Llama.cpp/models/Llama-3.2-1B-Instruct-BF16.gguf`
Quantizer: `/Users/viharikapenmetsa/llama.cpp/build-cpu/bin/llama-quantize`
Generated: 2026-11-03T17:09:16+00:00

| Variant | Status | Size (MiB) | Imatrix |
|---|---|---:|---|
| uniform_q4 | skipped_existing | 770.28 | none |
| chat_q4_imatrix | skipped_existing | 770.28 | chat |
| math_q4_imatrix | skipped_existing | 770.28 | math |
| code_q4_imatrix | skipped_existing | 770.28 | code |
| chat_mixed_q4_q6 | completed | 795.03 | chat |

These are **file size** measurements, not model accuracy or inference memory.
For performance conclusions, evaluate every produced GGUF using the same GSM8K, IFEval, MMLU settings.
Mixed precision depends on a separately generated tensor_types.txt recommendation and should be validated.
