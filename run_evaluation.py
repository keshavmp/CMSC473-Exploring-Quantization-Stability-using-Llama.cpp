import os
import time
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

from src.config import Config
from src.experiment import run_experiments


BASE = Config(
    num_examples=200,
    num_fewshot=5,
    context_size=2048,
    kl_enabled=True,
)

MODELS = [
    "chat-q4-nonPure",
    "math-q4-nonPure",
    "code-q4-nonPure",
]

CONFIGS = [
    BASE.with_overrides(
        name=f"{model}-pilot200",
        model_file=BASE.models_dir / f"{model}.gguf",
    )
    for model in MODELS
]


if __name__ == "__main__":
    start_time = time.time()
    print(f"Running {len(CONFIGS)} experiments...")
    run_experiments(CONFIGS)
    elapsed_time = time.time() - start_time
    print(f"All experiments completed in {elapsed_time:.2f} seconds.")
    print(f"Avg time per experiment: {elapsed_time / len(CONFIGS):.2f} seconds.")