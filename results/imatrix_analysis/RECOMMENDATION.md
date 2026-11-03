# Mixed-precision preliminary allocation

Layers: 16. High-precision layers: 4 (25% budget).

Heuristic top layers: 10, 13, 14, 15.

Low type: `Q4_K`. High type: `Q6_K`.

This is a **fraction of layers**, not an exact memory budget. Actual GGUF size and accuracy must be measured; tensors without matching imatrix entries retain llama.cpp default behavior. Verify quantization type fallbacks and the output model tensor types.

To build, run `quantize`; evaluate the resulting GGUF with your existing `run_evaluation.py`, then compare its CSV task scores with uniform baselines.
