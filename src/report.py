"""Compare every run in results/ on one table, CSV, and bar chart.

Each results/<name>.json becomes one row; every task found in the files
becomes one column, so new models and new tasks are picked up automatically.

    python src/report.py                     # compare every run
    python src/report.py chat-q4 uniform-q4  # only these runs, in this order
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures without a display
import matplotlib.pyplot as plt
import pandas as pd

try:
    from src.config import Config
except ModuleNotFoundError:  # running directly: python src/report.py
    from config import Config

# Preferred metric per task; tasks not listed use the first numeric,
# non-stderr metric found in their results.
PRIMARY_METRICS = {
    "gsm8k": "exact_match,strict-match",
    "ifeval": "prompt_level_strict_acc,none",
    "mmlu_college_computer_science": "acc,none",
}


def find_result_files(results_dir: Path) -> list[Path]:
    """One JSON per run, skipping the per-question *_samples.json files."""
    return sorted(
        path
        for path in results_dir.glob("*.json")
        if not path.name.endswith("_samples.json")
    )


def score_for(task: str, task_results: dict) -> float | None:
    """Pick the primary metric for this task, skipping stderr and "N/A" values."""
    primary = PRIMARY_METRICS.get(task)
    if primary is not None and isinstance(task_results.get(primary), (int, float)):
        return task_results[primary]

    for key, value in task_results.items():
        if "stderr" in key or not isinstance(value, (int, float)):
            continue
        return value
    return None


def build_table(files: list[Path]) -> pd.DataFrame:
    """Rows are run names, columns are tasks, values are scores out of 100."""
    runs: dict[str, dict[str, float | None]] = {}
    tasks: list[str] = []

    for path in files:
        results = json.loads(path.read_text(encoding="utf-8"))
        runs[path.stem] = {}
        for task, task_results in results.items():
            if task not in tasks:
                tasks.append(task)
            score = score_for(task, task_results)
            runs[path.stem][task] = None if score is None else round(100 * score, 1)

    return pd.DataFrame.from_dict(runs, orient="index").reindex(columns=tasks)


def save_csv(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, na_rep="")


def save_plot(table: pd.DataFrame, path: Path) -> None:
    """One group of bars per task, one bar per run."""
    ax = table.plot(kind="bar", figsize=(10, 5), ylim=(0, 100))
    ax.set_ylabel("score (%)")
    ax.set_xlabel("")
    ax.legend(title="run", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(ax.figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "names",
        nargs="*",
        help="run names to include (default: every results/*.json)",
    )
    args = parser.parse_args()

    cfg = Config()
    files = find_result_files(cfg.results_dir)
    if args.names:
        by_name = {path.stem: path for path in files}
        missing = [name for name in args.names if name not in by_name]
        if missing:
            parser.error(f"no results for: {', '.join(missing)}")
        files = [by_name[name] for name in args.names]

    if not files:
        parser.error(f"no result files in {cfg.results_dir}")

    table = build_table(files)

    print("\n" + table.to_string(na_rep="-") + "\n")

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    csv_file = cfg.results_dir / "summary.csv"
    plot_file = cfg.results_dir / "summary.png"
    save_csv(table, csv_file)
    save_plot(table, plot_file)

    print(f"Saved: {csv_file}")
    print(f"Saved: {plot_file}")


if __name__ == "__main__":
    main()
