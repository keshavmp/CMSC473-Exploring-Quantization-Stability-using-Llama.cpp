import subprocess
from pathlib import Path

try:
    from src.config import Config
except ModuleNotFoundError:  # running directly: python src/quantize.py
    from config import Config


# Supported sweep recipes. llama.cpp does not provide a Q7 quantization.
# Preserve the project's Q4_K_M recipe; --pure disables its automatic mixtures.
PURE_QUANT_TYPES = {
    "q2": "Q2_K", "q3": "Q3_K_M", "q4": "Q4_K_M",
    "q5": "Q5_K_M", "q6": "Q6_K", "q8": "Q8_0",
}


def quantize_model(
    quantize_tool: Path,
    source_model: Path,
    output_model: Path,
    quant_type: str = "Q4_K_M",
    imatrix_file: Path | None = None,
    pure: bool = False, # This makes it truly uniform instead of llama.cpp's default K-quant mix.
    tensor_types: list[str] | None = None,  # e.g. ["attn_q=q8_0"]
    output_type: str | None = None,         # type for output.weight, e.g. "q8_0"
) -> None:
    """Quantize the BF16 source model into one GGUF model."""
    if not quantize_tool.exists():
        raise FileNotFoundError(f"llama-quantize not found: {quantize_tool}")

    if not source_model.exists():
        raise FileNotFoundError(f"Source model not found: {source_model}")

    output_model.parent.mkdir(parents=True, exist_ok=True)

    command = [str(quantize_tool)]

    # Makes the baseline truly uniform instead of llama.cpp's default K-quant mix.
    if pure:
        command.append("--pure")

    # Uses task-specific calibration importance during quantization.
    if imatrix_file is not None:
        command += ["--imatrix", str(imatrix_file)]

    if tensor_types is not None:
        for tensor_type in tensor_types or []:
            command += ["--tensor-type", tensor_type]

    if output_type is not None:
        command += ["--output-type", output_type]

    command += [
        str(source_model),
        str(output_model),
        quant_type,
    ]

    print(f"\nCreating: {output_model.name}")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    cfg = Config()

    # Edit these settings, then run: python src/quantize.py
    # Always use the original BF16 source, never another quantized model.
    SOURCE_MODEL = cfg.source_model
    OUTPUT_MODEL = cfg.models_dir / "pure-q4.gguf"
    QUANT_TYPE = "Q4_K_M"
    PURE = True

    # Recipes: q2=Q2_K, q3=Q3_K_M, q4=Q4_K_M,
    #          q5=Q5_K_M, q6=Q6_K, q8=Q8_0. There is no Q7.
    quantize_model(
        cfg.quantize_tool,
        SOURCE_MODEL,
        OUTPUT_MODEL,
        QUANT_TYPE,
        pure=PURE,
    )
