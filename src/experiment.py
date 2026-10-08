from pathlib import Path

from src.config import Config
from src.evaluate import evaluate_model
from src.server import start_server, stop_server


def run_experiment(config: Config) -> Path:
    """Start the server for one config, run the evaluation, stop the server."""
    process, log = start_server(config)
    try:
        return evaluate_model(config)
    finally:
        stop_server(process, log)


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