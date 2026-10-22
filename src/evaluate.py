import json
import hashlib
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import lm_eval
from lm_eval.tasks import TaskManager

from src.config import Config


def file_hash(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def resolve_tasks(config: Config):
    """Resolve installed YAMLs and select the same leading IDs for every model."""
    task_root = Path(lm_eval.__file__).parent / "tasks"
    manager = TaskManager(
        include_defaults=False,
        include_path=[task_root / name for name in config.tasks],
    )
    tasks = manager.load(config.tasks)["tasks"]
    selection, details = {}, {}
    for name, task in tasks.items():
        available = len(task.eval_docs)
        cap = config.task_sample_caps.get(name, config.num_examples)
        count = available if cap is None else min(cap, available)
        selection[name] = list(range(count))
        resolved = task.dump_config()
        split = resolved.get("test_split") or resolved.get("validation_split")
        details[name] = {
            "split": split, "available": available, "requested_cap": cap,
            "selected": count, "subset": count < available,
            "configuration": resolved,
        }
        print(f"{name}: {count}/{available} from {split} (requested cap: {cap})")
        if cap is not None and cap > available:
            print(f"WARNING: {name} has only {available} available examples, fewer than {cap} requested.")
    return manager, tasks, selection, details


def evaluate_model(config: Config, limit: int | None = None) -> Path:
    """Evaluate the model served by config and save the results to JSON."""
    if limit is None:
        limit = config.num_examples

    model_name = config.resolve_model_file().stem
    print(f"Evaluating: {model_name}")
    print(f"Tasks: {', '.join(config.tasks)}")

    # Use the repository backend directly, avoiding an out-of-date installed patch.
    from src.gguf import GGUFLM

    selection, task_details = None, None
    manager = None
    tasks = config.tasks
    if config.task_sample_caps:
        manager, resolved, selection, task_details = resolve_tasks(config)
        tasks = list(resolved.values())
        limit = None

    with GGUFLM(
        base_url=config.base_url, max_length=config.context_size,
        request_workers=config.request_workers, timeout=config.request_timeout,
    ) as model:
        results = lm_eval.simple_evaluate(
            model=model,
            tasks=tasks,
            task_manager=manager,
            num_fewshot=config.num_fewshot,
            random_seed=0,
            numpy_random_seed=1234,
            torch_random_seed=1234,
            fewshot_random_seed=1234,
            batch_size=config.batch_size,
            gen_kwargs={"temperature": 0, "do_sample": False},
            limit=limit,
            samples=selection,
            log_samples=True,
            verbosity="INFO",
        )

    config.run_dir.mkdir(parents=True, exist_ok=True)
    with open(config.results_file, "w", encoding="utf-8") as file:
        json.dump(results["results"], file, indent=2)

    # per-question records, used to compare models question by question
    with open(config.samples_file, "w", encoding="utf-8") as file:
        json.dump(results["samples"], file, indent=2, default=str)

    metadata = {key: value for key, value in results.items() if key not in ("results", "samples")}
    model = config.resolve_model_file()
    metadata.update(
        model_filename=model.name,
        model_sha256=file_hash(model) if model.is_file() else None,
        quantization_type=config.quantization_type,
        pure_quantization=config.pure_quantization,
        lm_eval_version=version("lm_eval"),
        project_configuration=asdict(config),
        resolved_tasks=task_details,
        selected_doc_ids=selection,
    )
    # Harness n-samples and logged records are both retained; filters may log a doc twice.
    metadata["actual_unique_samples"] = {
        name: len({sample["doc_id"] for sample in records})
        for name, records in results["samples"].items()
    } if isinstance(results["samples"], dict) else {}
    (config.run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    if selection is not None:
        for name, ids in selection.items():
            if metadata["actual_unique_samples"].get(name) != len(ids):
                raise RuntimeError(f"{name}: evaluated sample count differs from selected count; see metadata.json")

    print(f"Saved: {config.results_file}")
    print(f"Saved: {config.samples_file}")
    return config.results_file
