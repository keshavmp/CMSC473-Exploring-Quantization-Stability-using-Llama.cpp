import subprocess
from pathlib import Path

try:
    from src.config import Config
except ModuleNotFoundError:  # running directly: python src/quantize.py
    from config import Config


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
    # uniform Q4 baseline: no imatrix.
if __name__ == "__main__":
    cfg = Config()
    quantize_model(cfg.quantize_tool, cfg.source_model, cfg.models_dir / "chat-q4-nonPure.gguf", "Q4_K_M", pure=False,imatrix_file='CMSC473-Exploring-Quantization-Stability-using-Llama.cpp\models\imatrix\chat.imatrix.gguf')
    quantize_model(cfg.quantize_tool, cfg.source_model, cfg.models_dir / "code-q4-nonPure.gguf", "Q4_K_M", pure=False,imatrix_file='CMSC473-Exploring-Quantization-Stability-using-Llama.cpp\models\imatrix\code.imatrix.gguf')
    quantize_model(cfg.quantize_tool, cfg.source_model, cfg.models_dir / "math-q4-nonPure.gguf", "Q4_K_M", pure=False,imatrix_file='CMSC473-Exploring-Quantization-Stability-using-Llama.cpp\models\imatrix\math.imatrix.gguf')
    # # All start as uniform Q4, then one tensor family goes to Q8.
    # quantize_model(
    #     cfg.quantize_tool, cfg.source_model, cfg.models_dir / "q4-attn-qkv-q8.gguf", "Q4_K_M",
    #     tensor_types=["attn_q=q8_0", "attn_k=q8_0", "attn_v=q8_0"],
    # )
    # quantize_model(
    #     cfg.quantize_tool, cfg.source_model, cfg.models_dir / "q4-attn-out-q8.gguf", "Q4_K_M",
    #     tensor_types=["attn_output=q8_0"],
    # )
    # quantize_model(
    #     cfg.quantize_tool, cfg.source_model, cfg.models_dir / "q4-ffn-up-gate-q8.gguf", "Q4_K_M",
    #     tensor_types=["ffn_up=q8_0", "ffn_gate=q8_0"],
    # )
    # quantize_model(
    #     cfg.quantize_tool, cfg.source_model, cfg.models_dir / "q4-ffn-down-q8.gguf", "Q4_K_M",
    #     tensor_types=["ffn_down=q8_0"],
    # )

