"""Generate the Task 15 long-context retrieval summary table and graphs.

Task 15 runs the needle-in-a-haystack (NIAH) benchmark at several input
lengths and KV-cache precisions, then compares retrieval accuracy, memory
use, runtime, and throughput.
"""

import argparse
import json
import math
from pathlib import Path

# Headless backend: renders to files with no display. Must be set before
# pyplot is imported.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Works both as a module and as a plain script.
try:
    from src.config import Config
except ModuleNotFoundError:
    from config import Config


# Sort keys for cache precision, highest first, so lines and legend entries
# appear in a consistent f16 -> q8_0 -> q4_0 order.
CACHE_ORDER = {"f16": 0, "q8_0": 1, "q4_0": 2}


def find_result_files(results_dir: Path) -> list[Path]:
    """Find Task 15 experiment JSON files, excluding per-question sample files.

    Files from other tasks are filtered out later, in load_result().
    """
    return sorted(
        path
        for path in results_dir.rglob("*.json")
        if not path.name.endswith("_samples.json")
    )


def as_number(value) -> float | None:
    """Convert a JSON value to a finite float.

    Result files can hold numbers, numeric strings, nulls, or NaN/inf. This
    turns anything unusable into None so later code only has to check for one
    kind of "missing".
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def get_quality(data: dict, context_length: int) -> float | None:
    """Get the NIAH score for the target input length used in this run.

    A result file can hold scores for several lengths, but each run only
    evaluates one, so only the entry matching this run's length is read.
    """
    results = data.get("evaluation", {}).get("results", {}).get("niah_single_1", {})
    if not isinstance(results, dict):
        return None

    # lm-eval stores the score under keys such as '16384,none'.
    score = as_number(results.get(f"{context_length},none"))

    # -1 means that this run did not evaluate this sequence length.
    if score is None or score < 0:
        return None

    return score


def load_result(path: Path) -> dict | None:
    """Load and normalize one Task 15 JSON result.

    Applies the validity checks (right task, no error, clean server exit),
    then flattens the nested JSON into one row, so a failed run can't
    contribute misleading numbers.
    """
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        print(f"Skipping {path.name}: {error}")
        return None

    # Avoid accidentally including Task 21 or unrelated JSON files.
    if data.get("task") != "niah_single_1":
        return None

    if data.get("error") is not None:
        print(f"Skipping {path.name}: run contains an error")
        return None

    if data.get("server_exit_code") not in (None, 0):
        print(f"Skipping {path.name}: server exit code was not zero")
        return None

    context_length = as_number(data.get("target_input_length"))
    if context_length is None:
        print(f"Skipping {path.name}: missing target_input_length")
        return None
    context_length = int(context_length)

    k_type = str(data.get("k_type", "unknown")).lower()
    v_type = str(data.get("v_type", "unknown")).lower()
    metric_deltas = data.get("metric_deltas") or {}

    return {
        "run": path.stem,
        "k_type": k_type,
        "v_type": v_type,
        # Combined label used as the legend entry and grouping key in plots.
        "cache_config": f"K={k_type.upper()} / V={v_type.upper()}",
        "context_length": context_length,
        "quality": get_quality(data, context_length),
        "sample_limit": as_number(data.get("sample_limit")),
        "backend": data.get("backend"),
        "server_context_capacity": as_number(data.get("server_context_capacity")),
        "baseline_server_rss_mb": as_number(data.get("baseline_server_rss_mb")),
        "peak_server_rss_mb": as_number(data.get("peak_server_rss_mb")),
        "server_rss_increase_mb": as_number(data.get("server_rss_increase_mb")),
        "evaluation_seconds": as_number(data.get("evaluation_seconds")),
        "elapsed_seconds": as_number(data.get("elapsed_seconds")),
        # Throughput comes from llama.cpp's server metrics.
        "prompt_tokens_per_second": as_number(
            metric_deltas.get("llamacpp:prompt_tokens_seconds")
        ),
        "generation_tokens_per_second": as_number(
            metric_deltas.get("llamacpp:predicted_tokens_seconds")
        ),
    }


def build_table(files: list[Path]) -> pd.DataFrame:
    """Build a DataFrame from valid Task 15 result files.

    Returns an empty frame if nothing is usable, so main() can report it.
    """
    rows = []
    for path in files:
        row = load_result(path)
        if row is not None:
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    return sort_runs_by_cache_type(pd.DataFrame(rows))


def sort_runs_by_cache_type(table: pd.DataFrame) -> pd.DataFrame:
    """Sort by context length, then F16, Q8_0, and Q4_0.

    Gives the CSV and console table a predictable layout, with the
    precision levels grouped within each context length.
    """
    table = table.copy()
    # Unrecognized cache types get 99 so they sort last.
    table["_k_order"] = table["k_type"].map(CACHE_ORDER).fillna(99)
    table["_v_order"] = table["v_type"].map(CACHE_ORDER).fillna(99)
    return (
        table.sort_values(
            ["context_length", "_k_order", "_v_order", "run"],
            kind="stable",
        )
        .drop(columns=["_k_order", "_v_order"])
        .reset_index(drop=True)
    )


def save_csv(table: pd.DataFrame, path: Path) -> None:
    """Write the table to a CSV file, without the index."""
    table.to_csv(path, index=False)


def plot_context(table: pd.DataFrame) -> str:
    """List the context lengths covered, for use in plot titles."""
    contexts = sorted(table["context_length"].dropna().unique())
    return "contexts: " + ", ".join(f"{int(value):,}" for value in contexts)


def save_metric_plot(
    table: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    *,
    quality_axis: bool = False,
) -> None:
    """Plot one metric against context length, with one line per K/V config.

    Every chart in the report is this same shape, so one helper handles all of
    them. `quality_axis` fixes the y-axis to 0-1 for accuracy plots so
    differences aren't visually exaggerated.
    """
    plot_table = table.dropna(subset=[metric]).copy()
    if plot_table.empty:
        print(f"Skipping {output_path.name}: no values for {metric}")
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    markers = {
        "f16/f16": "o",
        "q8_0/q8_0": "s",
        "q4_0/q4_0": "^",
    }

   

    # sort=False keeps the order produced by sort_runs_by_cache_type, so the
    # legend reads F16 -> Q8_0 -> Q4_0.
    for cache_config, group in plot_table.groupby("cache_config", sort=False):
        group = group.sort_values("context_length")

        marker_key = (
            cache_config.lower()
            .replace("k=", "")
            .replace("v=", "")
            .replace(" ", "")
        )

        ax.plot(
            group["context_length"],
            group[metric],
            marker=markers.get(marker_key, "o"),
            markersize=7,
            linewidth=2,
            label=cache_config,
        )

    ax.set_xlabel("Target input length (tokens)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # Tick only at the lengths actually tested, in plain (non-scientific) form.
    ax.set_xticks(sorted(plot_table["context_length"].unique()))
    ax.ticklabel_format(axis="x", style="plain")
    ax.grid(True, alpha=0.3)
    ax.legend(title="KV-cache precision")

    if quality_axis:
        ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    # Close the figure so repeated calls don't accumulate memory.
    plt.close(fig)
    print(f"Saved: {output_path}")


def save_quality_memory_plot(table: pd.DataFrame, output_path: Path) -> None:
    """Plot retrieval accuracy against total llama-server RSS increase.

    Shows the accuracy/memory trade-off directly. Each point is annotated
    with its context length, since length affects both axes.
    """
    plot_table = table.dropna(
        subset=["quality", "server_rss_increase_mb"]
    ).copy()

    if plot_table.empty:
        print("Skipping quality-vs-memory plot: no paired values")
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for cache_config, group in plot_table.groupby("cache_config", sort=False):
        ax.scatter(
            group["server_rss_increase_mb"],
            group["quality"],
            s=75,
            label=cache_config,
        )
        # Label each point with its context length.
        for _, row in group.iterrows():
            ax.annotate(
                f'{int(row["context_length"]):,}',
                (row["server_rss_increase_mb"], row["quality"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_xlabel("llama-server RSS increase (MB)")
    ax.set_ylabel("NIAH retrieval accuracy")
    ax.set_title("Task 15: Retrieval accuracy vs. server memory increase")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(title="KV-cache precision")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    """
    Run the complete Task 15 reporting pipeline.

    This function:
    1. Finds the Task 15 experiment JSON files.
    2. Optionally select specific experiment runs.
    3. Load and organize the results into a pandas DataFrame.
    4. Print the summary table.
    5. Save the results as a CSV file.
    6. Generate and save all seven plots.

    """

    parser = argparse.ArgumentParser(
        description="Generate Task 15 long-context retrieval graphs."
    )
    parser.add_argument(
        "names",
        nargs="*",
        help="result filenames without .json; default is all valid Task 15 runs",
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

    # Fall back to the configured results directory if none is given.
    cfg = Config()
    results_dir = (
        args.results_dir
        or Path(cfg.results_dir) / "task15_long_context_results" / "statistics"
    )
    output_dir = args.output_dir or results_dir

    files = find_result_files(results_dir)

    # Optionally narrow to specific runs, failing early on unknown names.
    if args.names:
        by_name = {path.stem: path for path in files}
        missing = [name for name in args.names if name not in by_name]
        if missing:
            parser.error(f"no result files for: {', '.join(missing)}")
        files = [by_name[name] for name in args.names]

    table = build_table(files)
    if table.empty:
        parser.error(f"no valid Task 15 results in {results_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Condensed console view of the table.
    display_columns = [
        "run",
        "k_type",
        "v_type",
        "context_length",
        "quality",
        "server_rss_increase_mb",
        "evaluation_seconds",
        "elapsed_seconds",
        "prompt_tokens_per_second",
        "generation_tokens_per_second",
    ]
    display_columns = [
        column for column in display_columns if column in table.columns
    ]

    print("\n" + table[display_columns].to_string(index=False, na_rep="-") + "\n")

    csv_file = output_dir / "task15_long_context_summary.csv"
    save_csv(table, csv_file)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Main Task 15 comparisons: quality, memory, runtime, and throughput.
    save_metric_plot(
        table,
        "quality",
        "NIAH retrieval accuracy",
        f"Retrieval accuracy vs. context length ({plot_context(table)})",
        plots_dir / "quality_vs_context.png",
        quality_axis=True,
    )

    save_metric_plot(
        table,
        "server_rss_increase_mb",
        "llama-server RSS increase (MB)",
        f"Server memory increase vs. context length ({plot_context(table)})",
        plots_dir / "rss_increase_vs_context.png",
    )

    save_metric_plot(
        table,
        "evaluation_seconds",
        "Evaluation time (seconds)",
        f"Evaluation time vs. context length ({plot_context(table)})",
        plots_dir / "evaluation_time_vs_context.png",
    )

    save_metric_plot(
        table,
        "elapsed_seconds",
        "Total elapsed time (seconds)",
        f"Total elapsed time vs. context length ({plot_context(table)})",
        plots_dir / "elapsed_time_vs_context.png",
    )

    save_metric_plot(
        table,
        "prompt_tokens_per_second",
        "Prompt processing throughput (tokens/second)",
        f"Prompt throughput vs. context length ({plot_context(table)})",
        plots_dir / "prompt_throughput_vs_context.png",
    )

    save_metric_plot(
        table,
        "generation_tokens_per_second",
        "Generation throughput (tokens/second)",
        f"Generation throughput vs. context length ({plot_context(table)})",
        plots_dir / "generation_throughput_vs_context.png",
    )

    save_quality_memory_plot(
        table,
        plots_dir / "quality_vs_rss_increase.png",
    )

    print(f"Saved: {csv_file}")
    print(f"Task 15 charts saved in: {plots_dir}")


if __name__ == "__main__":
    main()