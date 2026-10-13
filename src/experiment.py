import json
from dataclasses import asdict
from pathlib import Path

from src.config import Config
from src.evaluate import evaluate_model
from src.kl import evaluate_kl
from src.server import start_server, stop_server


def run_experiment(config: Config) -> Path:
    """Start the server for one config, run the evaluation, stop the server."""
    config.run_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("metrics.json", "samples.json", "kl.json"):
        (config.run_dir / filename).unlink(missing_ok=True)
    (config.run_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2, default=str), encoding="utf-8"
    )
    process, log = start_server(config)
    try:
        model = config.resolve_model_file()
        if model.is_file():
            settings = asdict(config)
            settings.update(model_file=str(model.resolve()), model_size_bytes=model.stat().st_size)
            (config.run_dir / "config.json").write_text(
                json.dumps(settings, indent=2, default=str), encoding="utf-8"
            )
        output = evaluate_model(config)
    finally:
        stop_server(process, log)
    if config.kl_enabled:
        evaluate_kl(config)
    return output


def run_experiments(configs: list[Config]) -> list[dict]:
    """Run several configurations one after another (one GPU, one port).

    A failing config is recorded and the sweep continues with the next one.
    Returns one summary entry per config: name, status, and output/error.
    """
    summaries = []

    for config in configs:
        print(f"\n=== Running: {config.name} ===")
        try:
            output_file = run_experiment(config)
            summaries.append(
                {"name": config.name, "status": "ok", "output": str(output_file)}
            )
        except Exception as error:
            print(f"[{config.name}] failed: {error}")
            summaries.append(
                {"name": config.name, "status": "failed", "error": str(error)}
            )

    print("\n=== Summary ===")
    for summary in summaries:
        detail = summary.get("output") or summary.get("error")
        print(f"  {summary['status']:>6}  {summary['name']}: {detail}")

    return summaries