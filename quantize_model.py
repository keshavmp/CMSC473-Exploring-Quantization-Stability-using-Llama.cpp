"""Edit CONFIG below, then run: python quantize_model.py
On macOS, use python3 if needed. Requires llama.cpp installed separately.
Relative CONFIG paths resolve against this repo, not the current directory.
"""

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if Path.cwd() != ROOT:
    raise SystemExit(f"Run this from the project folder:\n  {ROOT}")


CONFIG = {
    #llama-quantize PATHH
    # Windows example: r"C:\tools\llama.cpp\llama-quantize.exe"
    # macOS example: "/Users/yourname/llama.cpp/build/bin/llama-quantize"
    "quantize_bin": "llama-quantize",

    # Original GGUF checkpoint
    # Generate EVERY quantized variant from this file
    "source": "models/Llama-3.2-1B-Instruct-BF16.gguf",

    # Destination GGUF checkpoint, written inside the repo's models/ folder.
    # The parent directory is created automatically.
    # Use a different filename for each experiment: llama-quantize may overwrite
    # an existing output, and the script refuses to quantize a model onto itself.
    # Keep the naming aligned with results/ so each report maps 1:1 to a checkpoint.
    "output": "models/Llama-3.2-1B-Instruct-Q4_K_S.gguf",

    # Model-wide quantization recipe.
    # Examples: Q4_K_S, Q4_K_M, Q8_0.
    # A recipe can assign different storage types to different tensors.
    # In particular, Q4_K_M is a mixed-precision recipe, not a tensor type.
    # Run llama-quantize --help to see what your installed build supports.
    "recipe": "Q4_K_S",

    # True: Disable automatic precision mixtures (passes --pure to llama.cpp).
    # False: let llama.cpp use its usual recipe-specific tensor choices.
    # This does NOT force every tensor to be quantized: eligibility checks and
    # shape-dependent fallbacks still apply. Keep this identical across runs.
    "pure": True,

    # Optional per-tensor overrides: each entry is REGEX=STORAGE_TYPE.
    # [] means no overrides, which is the baseline configuration.
    # For an ablation, change output above and uncomment an example below.
    # Actual types include q8_0, q4_k, and f16; Q4_K_M is NOT a tensor type.
    # Confirm tensor names in your GGUF before selecting them.
    #
    # Regex notes:
    #   ^ and $ anchor the beginning/end; \. matches a literal dot.
    #   .* matches any sequence; [0-9]+ matches one or more digits.
    #   r"..." preserves backslashes in Python. Do not add shell quotes inside it.
    # Current upstream uses the first matching override, so avoid overlap.
    "tensor_types": [
        # One tensor in block 0:
        # r"^blk\.0\.attn_v\.weight$=q8_0",

        # All eligible weight tensors in block 0 (not normalization vectors):
        # r"^blk\.0\..*\.weight$=q8_0",

        # Attention value projection weights across all blocks:
        # r"^blk\.[0-9]+\.attn_v\.weight$=q8_0",
    ],

    # CPU threads used during quantization. None uses llama.cpp's default.
    # Set a positive integer such as 8 to choose explicitly.
    # This controls quantization work, not inference GPU offloading.
    "threads": None,

    # Optional importance-matrix file produced from calibration data for this
    # checkpoint. None means no --imatrix argument is passed.
    # It guides quantization error optimization; it does not select layer types.
    # Example: "calibration/llama32-1b.imatrix"
    # Hold this fixed when comparing tensor overrides in a single experiment.
    "imatrix": None,

    # Optional actual storage type for output.weight (the vocabulary projection).
    # None leaves its treatment to the recipe. Example: "f16" or "q8_0".
    # Models with tied embeddings may not have a separate output.weight tensor.
    "output_tensor_type": None,

    # Optional actual storage type for token embeddings. Example: "q8_0".
    # None leaves their treatment to the recipe. Keep fixed across ablations.
    "token_embedding_type": None,
}


def main():
    source, output = ROOT / CONFIG["source"], ROOT / CONFIG["output"]

    if source.resolve() == output.resolve():
        raise SystemExit("Refusing to quantize a model onto itself. Change CONFIG['output'].")
    if not source.is_file():
        found = sorted(p.name for p in (ROOT / "models").glob("*.gguf"))
        raise SystemExit(f"Source GGUF not found: {source}\nAvailable: {found or '(none)'}")

    bin_ = shutil.which(CONFIG["quantize_bin"]) or CONFIG["quantize_bin"]
    if not Path(bin_).is_file():
        raise SystemExit(f"llama-quantize not found: {bin_}. Set CONFIG['quantize_bin'].")

    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [bin_]
    if CONFIG["pure"]:
        cmd.append("--pure")
    for override in CONFIG["tensor_types"]:
        cmd += ["--tensor-type", override]
    if CONFIG["imatrix"]:
        cmd += ["--imatrix", str(ROOT / CONFIG["imatrix"])]
    if CONFIG["output_tensor_type"]:
        cmd += ["--output-tensor-type", CONFIG["output_tensor_type"]]
    if CONFIG["token_embedding_type"]:
        cmd += ["--token-embedding-type", CONFIG["token_embedding_type"]]

    # Options first, then the positionals llama-quantize requires.
    cmd += [str(source), str(output), CONFIG["recipe"]]
    if CONFIG["threads"]:
        cmd.append(str(CONFIG["threads"]))

    print(f"Source: {source}\nOutput: {output}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

