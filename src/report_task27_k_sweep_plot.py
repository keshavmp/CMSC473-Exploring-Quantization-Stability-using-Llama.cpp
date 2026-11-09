"""
Generate Task 27 long-context K/V isolation summary tables and graphs.

Task 27 compares isolated K-cache and V-cache quantization against the
F16/F16 baseline using the NIAH (Needle-in-a-Haystack) long-context
retrieval benchmark.

The experiment isolates the effects of K-cache and V-cache quantization:

    F16/F16  -> baseline
    Q8_0/F16 -> K-only Q8 quantization
    Q4_0/F16 -> K-only Q4 quantization
    F16/Q8_0 -> V-only Q8 quantization
    F16/Q4_0 -> V-only Q4 quantization

The report includes:
    - NIAH retrieval accuracy
    - Observed llama-server RSS memory increase
    - Accuracy change relative to the F16/F16 baseline
    - Memory savings relative to the F16/F16 baseline
    - Prompt processing throughput
    - Generation throughput
    - Evaluation time

All baseline comparisons are matched by context length. For example, a
Q4_0/F16 run at 8192 tokens is compared with the F16/F16 baseline at
8192 tokens, not with a baseline at a different context length.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

# Use a non-interactive backend so the script can save plots without opening
# graphical windows. This is useful when running the report from a terminal.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

# Support running the script from the repository root or from the src folder.
# The fallback handles the case where "src" is not treated as a package.
try:
    from src.config import Config
except ModuleNotFoundError:
    from config import Config


# Establish a consistent ordering for cache precisions.
# This is used when sorting rows so the output is easier to read:
# F16 first, then Q8_0, then Q4_0.
CACHE_ORDER = {
    "f16": 0,
    "q8_0": 1,
    "q4_0": 2,
}


# These are the three configurations included in Task 27.
#
# The goal is to isolate K-cache and V-cache quantization:
#   - F16/F16 is the baseline.
#   - Q8_0/F16 and Q4_0/F16 quantize only the K cache.
#   
# The configurations where both K and V are quantized are intentionally
# excluded because this experiment focuses on their individual effects.
# Only include the completed K-cache sweep.
# V is held at F16 so the comparison isolates K-cache precision.
TASK27_CONFIGS = {
    ("f16", "f16"),   # Baseline
    ("q8_0", "f16"),  # K-only Q8
    ("q4_0", "f16"),  # K-only Q4
}

# Human-readable labels for the three K-sweep series.
CONFIG_LABELS = {
    ("f16", "f16"): "K=F16 (baseline; V=F16)",
    ("q8_0", "f16"): "K=Q8_0 (V=F16)",
    ("q4_0", "f16"): "K=Q4_0 (V=F16)",
}

# Assign a distinct marker to each configuration.
# This helps distinguish the configurations in plots, including when
# the plot is viewed in grayscale or printed.
# Distinct markers for the baseline and K-cache quantization levels.
MARKERS = {
    ("f16", "f16"): "o",
    ("q8_0", "f16"): "s",
    ("q4_0", "f16"): "^",
}

def find_result_files(results_dir: Path) -> list[Path]:
    """
    Find JSON result files recursively.

    The statistics directory may contain JSON files in nested folders,
    so rglob() searches through the entire directory tree.

    Per-question sample files are excluded because the report expects
    one JSON file containing the summary statistics for each experiment.
    """
    if not results_dir.exists():
        # Returning an empty list lets main() handle the missing-results
        # situation with a more informative error message.
        return []

    return sorted(
        path
        for path in results_dir.rglob("*.json")
        if not path.name.endswith("_samples.json")
    )


def as_number(value) -> float | None:
    """
    Convert a value to a finite float.

    JSON values may be missing, strings, or otherwise unsuitable for
    numerical calculations. This helper safely converts valid values.

    None is returned when conversion fails or when the value is NaN
    or infinity. Those values should not be used in the report's metrics.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    # math.isfinite() rejects NaN and positive/negative infinity.
    return number if math.isfinite(number) else None


def get_quality(data: dict, context_length: int) -> float | None:
    """
    Extract the NIAH retrieval score for a particular context length.

    The lm_eval JSON stores task results under:
        evaluation -> results -> niah_single_1

    The metric key includes the context length, such as:
        "8192,none"

    The context_length argument determines which score is extracted.
    """
    results = (
        data.get("evaluation", {})
        .get("results", {})
        .get("niah_single_1", {})
    )

    # The expected task result should be a dictionary.
    # If the JSON structure is different, the score is unavailable.
    if not isinstance(results, dict):
        return None

    # Look up the metric corresponding to this run's target context length.
    score = as_number(results.get(f"{context_length},none"))

    # A missing or negative score is treated as unavailable.
    if score is None or score < 0:
        return None

    return score


def load_result(path: Path) -> dict | None:
    """
    Load one result JSON and convert it into a normalized dictionary.

    Normalization means every accepted run uses the same field names,
    regardless of how the original JSON is structured.

    Invalid, failed, or unrelated runs are skipped.
    """
    try:
        # Explicitly specify UTF-8 so the JSON is read consistently.
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

    # Skip files that cannot be opened or parsed as valid JSON.
    except (OSError, json.JSONDecodeError) as error:
        print(f"Skipping {path.name}: {error}")
        return None

    # This report is specifically for the NIAH retrieval task.
    # Ignore JSON files generated for other evaluation tasks.
    if data.get("task") != "niah_single_1":
        return None

    # A recorded error means the experiment did not complete successfully.
    if data.get("error") is not None:
        print(f"Skipping {path.name}: run contains an error")
        return None

    # A nonzero server exit code indicates that llama-server did not
    # terminate successfully. Such a run should not be included.
    if data.get("server_exit_code") not in (None, 0):
        print(f"Skipping {path.name}: server exit code was not zero")
        return None

    # Read the target input length from the experiment metadata.
    # This is the context length used to group and compare experiments.
    context_length = as_number(data.get("target_input_length"))

    if context_length is None:
        print(f"Skipping {path.name}: missing target_input_length")
        return None

    # Context lengths are token counts, so store them as integers.
    context_length = int(context_length)

    # Normalize the precision identifiers to lowercase strings.
    # The default "unknown" ensures missing values do not accidentally
    # match one of the supported configurations.
    k_type = str(data.get("k_type", "unknown")).lower()
    v_type = str(data.get("v_type", "unknown")).lower()

    # Only include the three configurations defined for Task 27.
    # Other K/V combinations are ignored.
    if (k_type, v_type) not in TASK27_CONFIGS:
        return None

    # Throughput measurements are stored in the metric_deltas dictionary.
    # If the dictionary is absent, use an empty dictionary so that
    # the subsequent lookups safely return None.
    metric_deltas = data.get("metric_deltas") or {}

    # Return a standardized row for the pandas DataFrame.
    return {
        # Use the filename, without its extension, as the run identifier.
        "run": path.stem,

        # Record the individual K and V precisions.
        "k_type": k_type,
        "v_type": v_type,

        # Store the human-readable configuration label.
        "cache_config": CONFIG_LABELS[(k_type, v_type)],

        # Record the input length used in this experiment.
        "context_length": context_length,

        # Retrieval accuracy for the target context length.
        "quality": get_quality(data, context_length),

        # Number of evaluation samples used in the run.
        "sample_limit": as_number(data.get("sample_limit")),

        # Record backend and server capacity metadata when available.
        "backend": data.get("backend"),
        "server_context_capacity": as_number(
            data.get("server_context_capacity")
        ),

        # Memory measurements collected from llama-server.
        "baseline_server_rss_mb": as_number(
            data.get("baseline_server_rss_mb")
        ),
        "peak_server_rss_mb": as_number(
            data.get("peak_server_rss_mb")
        ),
        "server_rss_increase_mb": as_number(
            data.get("server_rss_increase_mb")
        ),

        # Timing measurements from the evaluation.
        "evaluation_seconds": as_number(
            data.get("evaluation_seconds")
        ),
        "elapsed_seconds": as_number(
            data.get("elapsed_seconds")
        ),

        # Prompt processing throughput, measured in tokens per second.
        "prompt_tokens_per_second": as_number(
            metric_deltas.get("llamacpp:prompt_tokens_seconds")
        ),

        # Generation throughput, also measured in tokens per second.
        "generation_tokens_per_second": as_number(
            metric_deltas.get("llamacpp:predicted_tokens_seconds")
        ),
    }


def sort_runs(table: pd.DataFrame) -> pd.DataFrame:
    """
    Sort runs by context length and then K/V precision.

    Sorting makes the printed table and CSV easier to inspect.
    The temporary ordering columns are removed before returning the table.
    """
    # Work on a copy so the original DataFrame is not modified unexpectedly.
    table = table.copy()

    # Map the precision labels to numerical ordering values.
    # Unknown labels receive 99 so they sort after known precisions.
    table["_k_order"] = table["k_type"].map(CACHE_ORDER).fillna(99)
    table["_v_order"] = table["v_type"].map(CACHE_ORDER).fillna(99)

    return (
        table.sort_values(
            # Sort first by context length, then K precision, then V precision.
            # The run name provides a deterministic final ordering.
            ["context_length", "_k_order", "_v_order", "run"],
            kind="stable",
        )
        # These columns were only needed for sorting.
        .drop(columns=["_k_order", "_v_order"])
        .reset_index(drop=True)
    )


def build_table(files: list[Path]) -> pd.DataFrame:
    """
    Build the complete DataFrame from valid Task 27 result files.

    Each valid JSON file contributes one row.
    Invalid files are skipped by load_result().
    """
    rows = []

    # Load and normalize every discovered result file.
    for path in files:
        row = load_result(path)

        # Only append runs that passed all validation checks.
        if row is not None:
            rows.append(row)

    # If no valid runs were found, return an empty DataFrame.
    # main() will handle this case.
    if not rows:
        return pd.DataFrame()

    # Convert the list of dictionaries into a DataFrame and sort it.
    return sort_runs(pd.DataFrame(rows))


def add_baseline_comparisons(table: pd.DataFrame) -> pd.DataFrame:
    """
    Add baseline-relative quality and server-RSS metrics.

    The baseline is F16/F16 at the same context length.

    If multiple F16/F16 runs exist at a context length, their available
    quality and memory values are averaged to form that context's baseline.

    Two derived metrics are added:

        accuracy_delta_from_baseline
            Run accuracy minus baseline accuracy.

        rss_savings_percent_from_baseline
            Percentage reduction in observed RSS increase compared
            with the baseline.
    """
    # Work on a copy because this function adds several columns.
    table = table.copy()

    # Select only the F16/F16 baseline rows.
    # Grouping by context length ensures each experiment is compared
    # with a baseline at the same input length.
    baseline = (
        table[
            (table["k_type"] == "f16")
            & (table["v_type"] == "f16")
        ]
        .groupby("context_length", as_index=True)[
            ["quality", "server_rss_increase_mb"]
        ]
        # Rename the columns so they are clearly identified as baselines.
        .mean()
        .rename(
            columns={
                "quality": "baseline_quality",
                "server_rss_increase_mb": (
                    "baseline_server_rss_increase_mb"
                ),
            }
        )
    )

    # Join the baseline values onto every row using context_length.
    # Each run now has the baseline values for its own context length.
    table = table.join(baseline, on="context_length")

    # Calculate the absolute accuracy difference from the baseline.
    #
    # Example:
    #   run accuracy      = 0.70
    #   baseline accuracy = 0.80
    #   delta             = -0.10
    #
    # A negative result means accuracy is lower than the baseline.
    # A positive result means accuracy is higher than the baseline.
    table["accuracy_delta_from_baseline"] = (
        table["quality"] - table["baseline_quality"]
    )

    # Retrieve the baseline RSS increase for each row.
    baseline_rss = table["baseline_server_rss_increase_mb"]

    # Calculate percentage RSS savings.
    #
    # Formula:
    #   (baseline RSS - current RSS) / baseline RSS * 100
    #
    # Positive values indicate lower observed memory increase than baseline.
    # Negative values indicate higher observed memory increase than baseline.
    #
    # where(baseline_rss > 0) prevents division by zero or negative values.
    table["rss_savings_percent_from_baseline"] = (
        (baseline_rss - table["server_rss_increase_mb"])
        / baseline_rss.where(baseline_rss > 0)
        * 100
    )

    return table


def save_csv(table: pd.DataFrame, path: Path) -> None:
    """
    Save the summary DataFrame as a CSV file.

    index=False prevents pandas from adding its own row-number column.
    """
    table.to_csv(path, index=False)


def plot_context(table: pd.DataFrame) -> str:
    """
    Format the tested context lengths for use in chart titles.

    The returned string might look like:
        contexts: 4,096, 8,192, 16,384
    """
    # Drop missing values, find unique context lengths, and sort them.
    contexts = sorted(table["context_length"].dropna().unique())

    # Format each context length with commas for readability.
    return "contexts: " + ", ".join(
        f"{int(value):,}" for value in contexts
    )


def save_metric_plot(
    table: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    *,
    zero_line: bool = False,
    quality_axis: bool = False,
) -> None:
    """
    Plot one metric against context length.

    Each K/V configuration receives its own line.
    The function is reused for accuracy, memory, throughput, and timing plots.

    Parameters:
        table:
            DataFrame containing the experiment results.

        metric:
            Name of the DataFrame column to plot on the y-axis.

        ylabel:
            Label displayed on the y-axis.

        title:
            Chart title.

        output_path:
            Destination path for the saved PNG file.

        zero_line:
            If True, draw a horizontal line at y=0.
            This is useful for baseline-relative metrics.

        quality_axis:
            If True, constrain the y-axis to the accuracy range.
    """
    # Remove rows where the requested metric is missing.
    # This allows the same function to handle metrics that may not exist
    # for every experiment.
    plot_table = table.dropna(subset=[metric]).copy()

    # Do not create an empty chart if no runs have this metric.
    if plot_table.empty:
        print(f"Skipping {output_path.name}: no values for {metric}")
        return

    # Create a figure and axes object.
    # The figure size is measured in inches.
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # Group rows by K and V precision.
    # Each group corresponds to one configuration and becomes one line.
    for (k_type, v_type), group in plot_table.groupby(
        ["k_type", "v_type"], sort=False
    ):
        # Sort each configuration's data by context length so the line
        # connects points in increasing context-length order.
        group = group.sort_values("context_length")

        # Construct the tuple used to look up its marker and legend label.
        config = (k_type, v_type)

        # Draw the line for this configuration.
        ax.plot(
            # X-axis: target input length.
            group["context_length"],

            # Y-axis: the selected metric.
            group[metric],

            # Use a unique marker for each configuration.
            marker=MARKERS.get(config, "o"),

            # Make markers visible at each measured data point.
            markersize=7,

            # Set the line thickness.
            linewidth=2,

            # Display the human-readable configuration in the legend.
            label=CONFIG_LABELS.get(
                config,
                f"K={k_type} / V={v_type}",
            ),
        )

    # Add axis labels and the chart title.
    ax.set_xlabel("Target input length (tokens)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Show ticks only at context lengths that were actually tested.
    ax.set_xticks(
        sorted(plot_table["context_length"].unique())
    )

    # Avoid scientific notation on the x-axis.
    ax.ticklabel_format(axis="x", style="plain")

    # Add a subtle grid to make values easier to compare.
    ax.grid(True, alpha=0.3)

    # Display the legend with the K/V configuration labels.
    ax.legend(title="K-cache configuration")

    # Accuracy is expected to range from 0 to 1.
    # A small amount of headroom above 1 keeps the top of the plot readable.
    if quality_axis:
        ax.set_ylim(0, 1.05)

    # Draw a horizontal zero reference when requested.
    # This is particularly useful for changes relative to the baseline.
    if zero_line:
        ax.axhline(
            0,
            linewidth=1,
            linestyle="--",
        )

    # Adjust spacing so titles, labels, and legends fit in the figure.
    fig.tight_layout()

    # Save the figure as a high-resolution PNG.
    fig.savefig(output_path, dpi=180)

    # Close the figure to release memory.
    # This matters because the script generates several charts.
    plt.close(fig)

    print(f"Saved: {output_path}")


def save_quality_memory_plot(
    table: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot NIAH accuracy against observed llama-server RSS increase.

    This scatter plot helps visualize the tradeoff between retrieval quality
    and memory usage.

    Each point represents one experiment.
    The point's marker indicates its K/V configuration.
    The context length is annotated beside each point.
    """
    # A scatter point requires both a quality value and a memory value.
    # Rows missing either measurement cannot be plotted.
    plot_table = table.dropna(
        subset=["quality", "server_rss_increase_mb"]
    ).copy()

    # Skip the plot if there are no complete pairs of measurements.
    if plot_table.empty:
        print("Skipping accuracy-vs-memory plot: no paired values")
        return

    # Create the figure and axes.
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # Draw one scatter series per K/V configuration.
    for (k_type, v_type), group in plot_table.groupby(
        ["k_type", "v_type"], sort=False
    ):
        config = (k_type, v_type)

        # X is memory increase; Y is retrieval accuracy.
        ax.scatter(
            group["server_rss_increase_mb"],
            group["quality"],

            # Use a reasonably large marker so the points are visible.
            s=75,

            # Use the same marker mapping as the other charts.
            marker=MARKERS.get(config, "o"),

            # Label this configuration in the legend.
            label=CONFIG_LABELS.get(
                config,
                f"K={k_type} / V={v_type}",
            ),
        )

        # Annotate each point with its context length.
        # This is helpful because multiple context lengths may share
        # the same K/V configuration.
        for _, row in group.iterrows():
            ax.annotate(
                f'{int(row["context_length"]):,}',

                # Position the annotation at the data point.
                (
                    row["server_rss_increase_mb"],
                    row["quality"],
                ),

                # Offset the text slightly so it does not sit directly
                # on top of the scatter marker.
                xytext=(5, 5),
                textcoords="offset points",

                # Keep annotations small so they do not dominate the plot.
                fontsize=8,
            )

    # Label the axes and title.
    ax.set_xlabel("Observed llama-server RSS increase (MB)")
    ax.set_ylabel("NIAH retrieval accuracy")
    ax.set_title(
        "Task 27: Retrieval accuracy vs. observed server RSS increase"
    )

    # Accuracy is a fraction between 0 and 1.
    ax.set_ylim(0, 1.05)

    # Add a grid and legend.
    ax.grid(True, alpha=0.3)
    ax.legend(title="K/V-cache configuration")

    # Fit the plot elements within the figure.
    fig.tight_layout()

    # Save and close the figure.
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"Saved: {output_path}")


def main() -> None:
    """
    Run the complete Task 27 reporting pipeline.

    This function:
        1. Parses command-line arguments.
        2. Locates the Task 27 result JSON files.
        3. Loads and validates the results.
        4. Calculates baseline-relative metrics.
        5. Prints and saves the summary table.
        6. Generates the plots.
    """
    parser = argparse.ArgumentParser(
        description="Generate Task 27 K-cache sweep report."
    )

    # Optional positional arguments let the user select specific runs.
    # Names should be filenames without the .json extension.
    parser.add_argument(
        "names",
        nargs="*",
        help=(
            "result filenames without .json; "
            "default is all valid Task 27 runs"
        ),
    )

    # Allow the user to override the default results directory.
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="directory containing NIAH experiment JSON files",
    )

    # Allow the user to override the output directory.
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="report output directory (default: configured Task 27 folder)",
    )

    args = parser.parse_args()

    # Load the project configuration.
    # Config provides the configured results directory.
    cfg = Config()

    # Use the user-provided results directory if available.
    # Otherwise, use the Task 15 long-context statistics directory,
    # which contains the JSON results used for this Task 27 report.
    results_dir = (
        args.results_dir
        or Path(cfg.results_dir)
        / "task15_long_context_results"
        / "statistics"
    )

    # Store Task 27's report separately from the Task 15 report.
    # This prevents the generated CSV and plots from overwriting Task 15.
    # Store the K-sweep report separately from Task 15 and the broader
    # Task 27 folder so the focused analysis has its own outputs.
    output_dir = (
        args.output_dir
        or Path(cfg.results_dir)
        / "task27_k_sweep"
        / "long_context"
    )

    # Find all candidate JSON result files.
    files = find_result_files(results_dir)

    # If the user supplied run names, restrict the file list to those runs.
    if args.names:
        # Build a lookup from filename stem to full Path.
        by_name = {
            path.stem: path
            for path in files
        }

        # Check that every requested run exists.
        missing = [
            name
            for name in args.names
            if name not in by_name
        ]

        # Stop with a clear message if any requested run is missing.
        if missing:
            parser.error(
                f"no result files for: {', '.join(missing)}"
            )

        # Keep only the explicitly requested runs.
        files = [
            by_name[name]
            for name in args.names
        ]

    # Load all selected files and build the summary DataFrame.
    table = build_table(files)

    # A valid report needs at least one accepted NIAH result.
    # If none are found, explain what was expected.
    if table.empty:
        parser.error(
            f"no valid Task 27 NIAH K-sweep results found in {results_dir}; "
            "expected F16/F16, Q8_0/F16, and Q4_0/F16 configurations"
        )

    # Add the baseline accuracy and memory comparisons.
    # This adds the baseline columns and derived comparison metrics.
    table = add_baseline_comparisons(table)

    # Create the output directory if it does not already exist.
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Select the columns that are most useful for the console summary.
    # The full DataFrame is still saved to CSV below.
    display_columns = [
        "run",
        "k_type",
        "v_type",
        "context_length",
        "quality",
        "baseline_quality",
        "accuracy_delta_from_baseline",
        "server_rss_increase_mb",
        "baseline_server_rss_increase_mb",
        "rss_savings_percent_from_baseline",
        "evaluation_seconds",
        "prompt_tokens_per_second",
        "generation_tokens_per_second",
    ]

    # Keep only columns that actually exist in the DataFrame.
    # This prevents a missing optional field from causing a KeyError.
    display_columns = [
        column
        for column in display_columns
        if column in table.columns
    ]

    # Print the summary table.
    # na_rep="-" makes missing values easier to recognize in the terminal.
    print(
        "\n"
        + table[display_columns].to_string(
            index=False,
            na_rep="-",
        )
        + "\n"
    )

    # Save the complete results table to CSV.
    # This includes the original metrics and derived baseline comparisons.
    csv_file = output_dir / "task27_long_context_summary.csv"
    save_csv(table, csv_file)

    # Store all plots in a dedicated subdirectory.
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Format the context lengths for chart titles.
    contexts = plot_context(table)

    # ---------------------------------------------------------------
    # Plot 1: Retrieval accuracy versus context length
    # ---------------------------------------------------------------
    # Shows the absolute NIAH score for each K/V configuration.
    save_metric_plot(
        table,
        "quality",
        "NIAH retrieval accuracy",
        f"Task 27: Retrieval accuracy vs. context length ({contexts})",
        plots_dir / "accuracy_vs_context.png",
        quality_axis=True,
    )

    # ---------------------------------------------------------------
    # Plot 2: Accuracy difference from the F16/F16 baseline
    # ---------------------------------------------------------------
    # Values below zero indicate reduced accuracy relative to baseline.
    # Values above zero indicate higher accuracy relative to baseline.
    save_metric_plot(
        table,
        "accuracy_delta_from_baseline",
        "Accuracy difference from F16/F16 (absolute points)",
        (
            "Task 27: Accuracy difference from baseline "
            f"vs. context length ({contexts})"
        ),
        plots_dir / "accuracy_delta_vs_context.png",
        zero_line=True,
    )

    # ---------------------------------------------------------------
    # Plot 3: Observed server RSS increase versus context length
    # ---------------------------------------------------------------
    # This is the measured increase in llama-server RSS, not an estimate
    # of theoretical KV-cache memory usage.
    save_metric_plot(
        table,
        "server_rss_increase_mb",
        "Observed llama-server RSS increase (MB)",
        (
            "Task 27: Server RSS increase "
            f"vs. context length ({contexts})"
        ),
        plots_dir / "rss_increase_vs_context.png",
    )

    # ---------------------------------------------------------------
    # Plot 4: Percentage memory savings versus baseline
    # ---------------------------------------------------------------
    # Positive values indicate memory savings.
    # Negative values indicate greater RSS increase than the baseline.
    save_metric_plot(
        table,
        "rss_savings_percent_from_baseline",
        "Observed server RSS savings vs. F16/F16 (%)",
        (
            "Task 27: RSS savings vs. baseline "
            f"by context length ({contexts})"
        ),
        plots_dir / "rss_savings_vs_context.png",
        zero_line=True,
    )

    # ---------------------------------------------------------------
    # Plot 5: Prompt processing throughput
    # ---------------------------------------------------------------
    # Higher values mean the model processed the prompt tokens faster.
    save_metric_plot(
        table,
        "prompt_tokens_per_second",
        "Prompt processing throughput (tokens/second)",
        (
            "Task 27: Prompt throughput "
            f"vs. context length ({contexts})"
        ),
        plots_dir / "prompt_throughput_vs_context.png",
    )

    # ---------------------------------------------------------------
    # Plot 6: Generation throughput
    # ---------------------------------------------------------------
    # Higher values mean generated tokens were produced faster.
    save_metric_plot(
        table,
        "generation_tokens_per_second",
        "Generation throughput (tokens/second)",
        (
            "Task 27: Generation throughput "
            f"vs. context length ({contexts})"
        ),
        plots_dir / "generation_throughput_vs_context.png",
    )

    # ---------------------------------------------------------------
    # Plot 7: Evaluation time
    # ---------------------------------------------------------------
    # This measures the time spent evaluating the run.
    # Lower values indicate a shorter evaluation.
    save_metric_plot(
        table,
        "evaluation_seconds",
        "Evaluation time (seconds)",
        (
            "Task 27: Evaluation time "
            f"vs. context length ({contexts})"
        ),
        plots_dir / "evaluation_time_vs_context.png",
    )

    # ---------------------------------------------------------------
    # Plot 8: Retrieval accuracy versus observed RSS increase
    # ---------------------------------------------------------------
    # This scatter plot presents the quality-memory tradeoff directly.
    # Each point is one run, and its annotation indicates context length.
    save_quality_memory_plot(
        table,
        plots_dir / "accuracy_vs_rss_increase.png",
    )

    # Print the final output locations so the user knows where to find
    # the CSV file and generated plots.
    print(f"Saved: {csv_file}")
    print(f"Task 27 long-context charts saved in: {plots_dir}")


# Run main() only when this file is executed directly.
# Importing this module from another Python file will not automatically
# start the reporting pipeline.
if __name__ == "__main__":
    main()