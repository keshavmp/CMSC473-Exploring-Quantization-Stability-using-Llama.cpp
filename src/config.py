"""Settings for a single evaluation run.

Each Config instance describes one experiment: which model to serve, which
tasks to evaluate, and where results are written. Build as many as you need
and pass the list to src.experiment.run_experiments().
"""

from dataclasses import dataclass, field, replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LLAMA_BUILD_DIR = Path(r"C:\llama.cpp\build\bin\Release")


@dataclass
class Config:
    """All settings for one evaluation run."""

    # --- Identity ---------------------------------------------------------
    # Used for result and log filenames so runs never overwrite each other.
    # The served model defaults to models/<name>.gguf (see resolve_model_file).
    name: str = "uniform-q4"

    # --- Repo layout ------------------------------------------------------
    project_root: Path = PROJECT_ROOT

    # --- llama.cpp binaries ----------------------------------------------
    llama_server: Path = LLAMA_BUILD_DIR / "llama-server.exe"   # CUDA build (GPU)
    llama_imatrix: Path = LLAMA_BUILD_DIR / "llama-imatrix.exe"
    quantize_tool: Path = LLAMA_BUILD_DIR / "llama-quantize.exe"

    # --- Server -----------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8080
    context_size: int = 2048
    gpu_layers: str = "all"
    server_start_timeout: int = 120    # seconds to wait for /health

    # --- Model ------------------------------------------------------------
    # None -> models/<name>.gguf
    model_file: Path | None = None
    # BF16 checkpoint that every quantized variant is generated from.
    source_model: Path = PROJECT_ROOT / "models" / "Llama-3.2-1B-Instruct-BF16.gguf"
    # Used only when the GGUF does not already exist locally.
    hf_repo: str | None = None        # Example: "your-huggingface-org/your-gguf-repository"
    hf_filename: str | None = None    # Example: "Llama-3.2-1B-Instruct-f16.gguf"

    # --- Evaluation -------------------------------------------------------
    tasks: list[str] = field(
        default_factory=lambda: [
            "gsm8k",
            "ifeval",
            "mmlu_college_computer_science",
        ]
    )
    num_examples: int = 300   # examples per task
    num_fewshot: int = 5
    batch_size: int = 1

    # --- Derived paths ----------------------------------------------------
    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def results_dir(self) -> Path:
        return self.project_root / "results"

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def log_file(self) -> Path:
        return self.results_dir / f"{self.name}-server.log"

    @property
    def results_file(self) -> Path:
        return self.results_dir / f"{self.name}.json"

    @property
    def samples_file(self) -> Path:
        return self.results_dir / f"{self.name}_samples.json"

    # --- Helpers ----------------------------------------------------------
    def resolve_model_file(self) -> Path:
        """The GGUF to serve: explicit model_file, else models/<name>.gguf."""
        if self.model_file is not None:
            return self.model_file
        return self.models_dir / f"{self.name}.gguf"

    def with_overrides(self, **kwargs) -> "Config":
        """Return a copy of this config with some fields replaced."""
        return replace(self, **kwargs)