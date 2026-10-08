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
    pure: bool = True, # This makes it truly uniform instead of llama.cpp's default K-quant mix.
) -> None:
    """Quantize the BF16 source model into one GGUF model."""
    if not quantize_tool.exists():
        raise FileNotFoundError(f"llama-quantize not found: {quantize_tool}")

    if not source_model.exists():
        raise FileNotFoundError(f"Source model not found: {source_model}")

    if imatrix_file is not None and not imatrix_file.exists():
        raise FileNotFoundError(f"Imatrix file not found: {imatrix_file}")

    output_model.parent.mkdir(parents=True, exist_ok=True)

    command = [str(quantize_tool)]

    # Makes the baseline truly uniform instead of llama.cpp's default K-quant mix.
    if pure:
        command.append("--pure")

    # Uses task-specific calibration importance during quantization.
    if imatrix_file is not None:
        command += ["--imatrix", str(imatrix_file)]

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
    quantize_model(
        quantize_tool=cfg.quantize_tool,
        source_model=cfg.source_model,
        output_model=cfg.models_dir / "math-q4.gguf",
        imatrix_file=cfg.models_dir / "imatrix" / "math.imatrix.gguf"
    )