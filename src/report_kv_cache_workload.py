
"""Compare KV-cache experiment runs in results/.

Each experiment JSON becomes one row in the CSV. Quality is summarized
by task, while performance and memory metrics are included as columns.

Examples:
    python -m src.report_kv_cache_workload
    python -m src.report_kv_cache_workload f16_f16_arc_easy_4096
    python -m src.report_kv_cache_workload --results-dir results
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    from src.config import Config
except ModuleNotFoundError:
    from config import Config


# Preferred quality metric per task.
# If unavailable, score_for() falls back to another numeric metric.
PRIMARY_METRICS = {
    "arc_easy": (
        "acc_norm,none",
        "acc_norm",
        "acc,none",
        "acc",
    ),
    "gsm8k": (
        "exact_match,strict-match",
        "exact_match,flexible-extract",
        "exact_match",
    ),
    "ifeval": (
        "inst_level_strict_acc,none",
        "prompt_level_strict_acc,none",
        "prompt_level_loose_acc,none",
        "inst_level_loose_acc,none",
    ),
    "mmlu_college_computer_science": (
        "acc,none",
        "acc_norm,none",
        "acc",
        "acc_norm",
    ),
    "mathqa": (
        "acc_norm,none",
        "acc_norm",
        "acc,none",
        "acc",
    ),
}


def find_result_files(results_dir: Path) -> list[Path]:
    """Find experiment JSON files, excluding per-question sample files."""
    return sorted(
        path
        for path in results_dir.rglob("*.json")
        if not path.name.endswith("_samples.json")
    )


def score_for(task: str, task_results: dict) -> tuple[str | None, float | None]:
    """Return the selected metric and its score, or (None, None)."""
    if not isinstance(task_results, dict):
        return None, None

    for metric in PRIMARY_METRICS.get(task, ()):
        value = task_results.get(metric)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return metric, float(value)

    # Fallback: first numeric metric that is not a standard error.
    for metric, value in task_results.items():
        if "stderr" in metric.lower():
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return metric, float(value)

    return None, None


def load_result(path: Path) -> dict | None:
    """Load one experiment result, skipping invalid or unrelated JSON files."""
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Skipping {path.name}: {exc}")
        return None

    # Experiment files from the KV runner have a nested quality section.
    if not isinstance(result, dict) or not isinstance(result.get("quality"), dict):
        return None

    return result


def build_table(files: list[Path]) -> pd.DataFrame:
    """Build one row per run, with scores and performance/memory metrics."""
    rows = []
    all_tasks = []

    for path in files:
        result = load_result(path)
        if result is None:
            continue

        quality = result["quality"]
        performance = result.get("performance", {})
        memory = result.get("memory", {})

        # Use the declared task list when available; otherwise infer it
        # from the quality dictionary.
        tasks = result.get("tasks") or list(quality.keys())

        row = {
            "run": path.stem,
            "model": Path(result.get("model", "")).name,
            "weight_quant": result.get("weight_quant"),
            "k_type": result.get("k_type"),
            "v_type": result.get("v_type"),
            "context_limit": result.get("context_limit"),
            "backend": result.get("backend"),
            "sample_limit": result.get("sample_limit"),
            "prompt_tokens_per_second": performance.get(
                "prompt_tokens_per_second"
            ),
            "generation_tokens_per_second": performance.get(
                "generation_tokens_per_second"
            ),
            "evaluation_wall_seconds": performance.get(
                "evaluation_wall_seconds"
            ),
            "prompt_tokens": performance.get("prompt_tokens"),
            "generation_tokens": performance.get("generation_tokens"),
            "max_observed_sequence_length": performance.get(
                "max_observed_sequence_length"
            ),
            "baseline_rss_mb": memory.get("baseline_rss_mb"),
            "peak_rss_mb": memory.get("peak_rss_mb"),
            "peak_rss_delta_mb": memory.get("peak_rss_delta_mb"),
        }

        for task in tasks:
            if task not in all_tasks:
                all_tasks.append(task)

            metric, score = score_for(task, quality.get(task, {}))
            # Keep raw scores on a 0-1 scale for consistency across tasks.
            row[f"{task}_quality"] = score
            row[f"{task}_metric"] = metric

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    # Quality columns are arranged by task; all other columns are retained.
    columns = list(rows[0].keys())
    for task in all_tasks:
        for suffix in ("quality", "metric"):
            column = f"{task}_{suffix}"
            if any(column in row for row in rows) and column not in columns:
                columns.append(column)

    return pd.DataFrame(rows).reindex(columns=columns)


def save_csv(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, index=False, na_rep="")

def plot_context(table: pd.DataFrame) -> str:
    """Describe the backend and sample count for plot titles."""
    backends = table["backend"].dropna().unique()
    samples = table["sample_limit"].dropna().unique()

    backend_label = (
        str(backends[0]).upper() if len(backends) == 1 else "Mixed backends"
    )
    sample_label = (
        f"{int(samples[0])} samples"
        if len(samples) == 1
        else "Mixed sample counts"
    )

    return f"{backend_label} | {sample_label}"



def save_task_quality_plot(
    table: pd.DataFrame,
    task: str,
    path: Path,
) -> None:
    """Plot quality scores for one task across experiment runs."""
    column = f"{task}_quality"
    if column not in table.columns:
        return

    plot_data = table[["run", column]].dropna()
    if plot_data.empty:
        print(f"Skipping quality plot for {task}: no values available.")
        return

    ax = plot_data.plot(
        x="run",
        y=column,
        kind="bar",
        figsize=(max(8, len(plot_data) * 0.7), 5),
        legend=False,
    )
    ax.set_ylabel("Quality score (0–1)")
    ax.set_xlabel("Run")
    ax.set_title(f"Quality: {task} ({plot_context(table)})")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(ax.figure)


def save_metric_plot(
    table: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    """Create a bar chart for a numeric metric, skipping missing values."""
    if metric not in table.columns:
        return

    plot_data = table[["run", metric]].dropna()
    if plot_data.empty:
        print(f"Skipping {title}: no values available.")
        return

    ax = plot_data.plot(
        x="run",
        y=metric,
        kind="bar",
        figsize=(max(10, len(plot_data) * 0.6), 5),
        legend=False,
    )
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Run")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(ax.figure)


def sort_runs_by_cache_type(table: pd.DataFrame) -> pd.DataFrame:
    """Order F16 runs before Q8 runs, and Q8 before Q4 runs."""
    order = {"f16": 0, "q8_0": 1, "q4_0": 2}

    sorted_table = table.copy()
    sorted_table["_k_order"] = (
        sorted_table["k_type"].map(order).fillna(99)
    )
    sorted_table["_v_order"] = (
        sorted_table["v_type"].map(order).fillna(99)
    )

    sorted_table = sorted_table.sort_values(
        ["_k_order", "_v_order", "run"],
        kind="stable",
    )

    return sorted_table.drop(columns=["_k_order", "_v_order"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "names",
        nargs="*",
        help="result filenames without .json; default is all valid runs",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="directory containing experiment JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory for the CSV and charts (default: results directory)",
    )
    args = parser.parse_args()

    cfg = Config()
    results_dir = args.results_dir or Path(cfg.results_dir)
    output_dir = args.output_dir or results_dir

    files = find_result_files(results_dir)

    if args.names:
        by_name = {path.stem: path for path in files}
        missing = [name for name in args.names if name not in by_name]
        if missing:
            parser.error(f"no result files for: {', '.join(missing)}")
        files = [by_name[name] for name in args.names]

    table = build_table(files)
    if table.empty:
        parser.error(f"no valid KV experiment results in {results_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Print the main results table.
    display_columns = [
        column for column in table.columns
        if column.endswith("_quality")
    ]
    display_columns = [
        "run", "k_type", "v_type", "context_limit",
        *display_columns,
        "prompt_tokens_per_second",
        "generation_tokens_per_second",
        "peak_rss_delta_mb",
    ]
    display_columns = [
        column for column in display_columns if column in table.columns
    ]

    print("\n" + table[display_columns].to_string(index=False, na_rep="-") + "\n")

    csv_file = output_dir / "kv_cache_workload_summary.csv"
    save_csv(table, csv_file)

    
    # Generate separate plots for each task.
    quality_columns = [
        column for column in table.columns
        if column.endswith("_quality")
    ]

    plots_dir = output_dir / "plots"

    for column in quality_columns:
        task = column.removesuffix("_quality")

        # Include only runs with a quality score for this task.
        task_table = table[table[column].notna()].copy()
        task_table = sort_runs_by_cache_type(task_table)
        if task_table.empty:
            continue

        task_dir = plots_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)

        save_task_quality_plot(
            task_table,
            task,
            task_dir / "quality.png",
        )

        save_metric_plot(
            task_table,
            "prompt_tokens_per_second",
            "Prompt tokens/second",
            f"Prompt throughput: {task} ({plot_context(task_table)})",
            task_dir / "prompt_throughput.png",
        )

        save_metric_plot(
            task_table,
            "peak_rss_delta_mb",
            "Peak RSS increase (MB)",
            f"Peak memory increase: {task} ({plot_context(task_table)})",
            task_dir / "peak_rss_delta.png",
        )

    print(f"Saved: {csv_file}")
    print(f"Task-specific charts saved in: {plots_dir}")


    print(f"Saved: {csv_file}")
    print(f"Task-specific charts saved in: {plots_dir}")


if __name__ == "__main__":
    main()
