import json
from pathlib import Path

import lm_eval

from src.config import Config


def evaluate_model(config: Config, limit: int | None = None) -> Path:
    """Evaluate the model served by config and save the results to JSON."""
    if limit is None:
        limit = config.num_examples

    model_name = config.resolve_model_file().stem
    print(f"Evaluating: {model_name}")
    print(f"Tasks: {', '.join(config.tasks)}")

    results = lm_eval.simple_evaluate(
        model="gguf",
        model_args=f"base_url={config.base_url},timeout=30",
        tasks=config.tasks,
        num_fewshot=config.num_fewshot,
        torch_random_seed=None,
        batch_size=config.batch_size,
        gen_kwargs={"temperature": 0, "do_sample": False},
        limit=limit,
        log_samples=True,
        verbosity="INFO",
    )

    config.results_dir.mkdir(parents=True, exist_ok=True)
    with open(config.results_file, "w", encoding="utf-8") as file:
        json.dump(results["results"], file, indent=2)

    # per-question records, used to compare models question by question
    with open(config.samples_file, "w", encoding="utf-8") as file:
        json.dump(results["samples"], file, indent=2, default=str)

    print(f"Saved: {config.results_file}")
    print(f"Saved: {config.samples_file}")
    return config.results_file