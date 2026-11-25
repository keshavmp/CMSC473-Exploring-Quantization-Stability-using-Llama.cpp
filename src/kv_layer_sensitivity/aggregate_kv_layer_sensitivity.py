"""Aggregate workload-level KV-cache sensitivity results for Task 31."""

from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "task31_kv_layer_sensitivity"
)

WORKLOADS_DIR = RESULTS_DIR / "workloads"
OUTPUT_DIR = RESULTS_DIR / "aggregated"

TASKS = [
    "arc_easy",
    "mathqa",
    "mmlu_college_computer_science",
]

N_LAYERS = 16
EXPECTED_SAMPLES = 50

TENSOR_TYPES = ["K", "V"]
QUANT_TYPES = ["q8_0", "q4_0"]

EXPECTED_CONFIGS = set(
    product(
        range(N_LAYERS),
        TENSOR_TYPES,
        QUANT_TYPES,
    )
)

METRIC_COLUMNS = [
    "mse",
    "normalized_mse_percent",
    "max_absolute_error",
]


# ============================================================
# 2. VALIDATE INDIVIDUAL SAMPLE
# ============================================================

def validate_sample(df, csv_path):
    required_columns = {
        "layer",
        "tensor",
        "quantization",
        *METRIC_COLUMNS,
    }

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Missing required columns in {csv_path}"
        )

    if len(df) != len(EXPECTED_CONFIGS):
        raise ValueError(
            f"Expected 64 rows in {csv_path}, found {len(df)}"
        )

    actual_configs = set(
        df[
            ["layer", "tensor", "quantization"]
        ].itertuples(index=False, name=None)
    )

    if actual_configs != EXPECTED_CONFIGS:
        raise ValueError(
            f"Unexpected or duplicate configurations in {csv_path}"
        )

    metrics = df[METRIC_COLUMNS].to_numpy(dtype=float)

    if not np.isfinite(metrics).all():
        raise ValueError(
            f"Non-finite metric values in {csv_path}"
        )

    if (metrics < 0).any():
        raise ValueError(
            f"Negative error values in {csv_path}"
        )


# ============================================================
# 3. LOAD WORKLOAD RESULTS
# ============================================================

def load_workload(task_name):
    task_dir = WORKLOADS_DIR / task_name

    csv_paths = sorted(
        task_dir.glob(
            "sample_*/layer_reconstruction_errors.csv"
        )
    )

    if len(csv_paths) != EXPECTED_SAMPLES:
        raise ValueError(
            f"{task_name}: expected {EXPECTED_SAMPLES} "
            f"samples, found {len(csv_paths)}"
        )

    frames = []

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)

        validate_sample(df, csv_path)

        df.insert(0, "workload", task_name)
        df.insert(1, "sample", csv_path.parent.name)

        frames.append(df)

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    print(
        f"{task_name}: "
        f"{len(csv_paths)} samples, "
        f"{len(combined)} measurements"
    )

    return combined


# ============================================================
# 4. AGGREGATE RESULTS
# ============================================================

def aggregate_results(combined):
    summary = (
        combined.groupby(
            [
                "workload",
                "layer",
                "tensor",
                "quantization",
            ]
        )
        .agg(
            sample_count=(
                "normalized_mse_percent", "count"
            ),
            mean_nmse_percent=(
                "normalized_mse_percent", "mean"
            ),
            std_nmse_percent=(
                "normalized_mse_percent", "std"
            ),
            median_nmse_percent=(
                "normalized_mse_percent", "median"
            ),
            min_nmse_percent=(
                "normalized_mse_percent", "min"
            ),
            max_nmse_percent=(
                "normalized_mse_percent", "max"
            ),
            mean_mse=(
                "mse", "mean"
            ),
            mean_max_absolute_error=(
                "max_absolute_error", "mean"
            ),
        )
        .reset_index()
    )

    if not (
        summary["sample_count"] == EXPECTED_SAMPLES
    ).all():
        raise ValueError(
            "Unexpected sample count in aggregated results"
        )

    return summary


# ============================================================
# 5. MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    frames = []

    for task in TASKS:
        frames.append(load_workload(task))

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    summary = aggregate_results(combined)

    combined_path = (
        OUTPUT_DIR / "all_sample_results.csv"
    )

    summary_path = (
        OUTPUT_DIR / "workload_summary.csv"
    )

    combined.to_csv(
        combined_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("\n" + "=" * 60)
    print("AGGREGATION COMPLETE")
    print("=" * 60)

    print(f"Workloads: {len(TASKS)}")
    print(f"Total samples: {len(TASKS) * EXPECTED_SAMPLES}")
    print(f"Total measurements: {len(combined)}")
    print(f"Aggregated configurations: {len(summary)}")

    print(f"\nCombined results: {combined_path}")
    print(f"Aggregated results: {summary_path}")


if __name__ == "__main__":
    main()