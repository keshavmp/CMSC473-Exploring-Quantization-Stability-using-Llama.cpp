"""Generate KV-cache sensitivity heatmaps for Task 31."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "task31_kv_layer_sensitivity"
    / "aggregated"
)

INPUT_CSV = RESULTS_DIR / "workload_summary.csv"

TASKS = {
    "arc_easy": "ARC Easy",
    "mathqa": "MathQA",
    "mmlu_college_computer_science": "MMLU CS",
}

CONFIGS = [
    ("K", "q4_0"),
    ("K", "q8_0"),
    ("V", "q4_0"),
    ("V", "q8_0"),
]

N_LAYERS = 16
METRIC = "mean_nmse_percent"


def plot_heatmap(
    matrix,
    row_labels,
    col_labels,
    title,
    ylabel,
    filename,
    normalized=False,
):

    fig, ax = plt.subplots(
        figsize=(
            max(7, len(col_labels) * 1.6),
            max(3.5, len(row_labels) * 0.48),
        )
    )

    if normalized:
        bound = max(np.abs(matrix).max(), 1)

        norm = TwoSlopeNorm(
            vmin=-bound,
            vcenter=0,
            vmax=bound,
        )

        cmap = "RdBu_r"
        colorbar_label = "Error above (+) or below (-) mean (%)"

    else:
        norm = LogNorm(
            vmin=matrix.min(),
            vmax=matrix.max(),
        )

        cmap = "magma"
        colorbar_label = "Mean normalized MSE (%)"

    im = ax.imshow(
        matrix,
        cmap=cmap,
        norm=norm,
        aspect="auto",
    )

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)

    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    ax.set_title(title, pad=15)
    ax.set_ylabel(ylabel)

    # Annotate every cell.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):

            value = matrix[i, j]

            if normalized:
                label = f"{value:+.1f}%"
            else:
                label = f"{value:.4g}%"

            r, g, b, _ = im.cmap(im.norm(value))

            luminance = (
                0.299 * r
                + 0.587 * g
                + 0.114 * b
            )

            color = "black" if luminance > 0.55 else "white"

            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                color=color,
                fontsize=10,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

    fig.tight_layout()

    output_path = RESULTS_DIR / filename

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


def normalize_columns(matrix):

    column_means = matrix.mean(axis=0)

    return 100 * (
        matrix / column_means - 1
    )


def get_layer_values(df, tensor, quantization, task=None):

    subset = df[
        (df["tensor"] == tensor)
        & (df["quantization"] == quantization)
    ]

    if task is not None:
        subset = subset[
            subset["workload"] == task
        ]

    values = (
        subset.groupby("layer")[METRIC]
        .mean()
        .reindex(range(N_LAYERS))
        .to_numpy()
    )

    if not np.isfinite(values).all():
        raise ValueError(
            f"Missing layer values: "
            f"{tensor}, {quantization}, {task}"
        )

    return values


def main():

    df = pd.read_csv(INPUT_CSV)

    layers = list(range(N_LAYERS))

    # --------------------------------------------------------
    # 1. Layer sensitivity across configurations
    # --------------------------------------------------------

    layer_matrix = np.column_stack([
        get_layer_values(df, tensor, quantization)
        for tensor, quantization in CONFIGS
    ])

    config_labels = [
        f"{tensor}\n{quantization.upper()}"
        for tensor, quantization in CONFIGS
    ]

    plot_heatmap(
        layer_matrix,
        layers,
        config_labels,
        "KV Layer Sensitivity\nEqual mean across three workloads",
        "Transformer Layer",
        "01_kv_layer_sensitivity.png",
    )

    # --------------------------------------------------------
    # 2. Normalized layer sensitivity
    # --------------------------------------------------------

    plot_heatmap(
        normalize_columns(layer_matrix),
        layers,
        config_labels,
        "KV Layer Sensitivity Within Each Configuration",
        "Transformer Layer",
        "02_kv_layer_sensitivity_normalized.png",
        normalized=True,
    )

    # --------------------------------------------------------
    # 3–4. Layer sensitivity by workload
    # --------------------------------------------------------

    workload_labels = [
        f"{label}\n{quant.upper()}"
        for quant in ["q4_0", "q8_0"]
        for label in TASKS.values()
    ]

    for tensor in ["K", "V"]:

        workload_matrix = np.column_stack([
            get_layer_values(
                df,
                tensor,
                quant,
                task,
            )
            for quant in ["q4_0", "q8_0"]
            for task in TASKS
        ])

        plot_heatmap(
            workload_matrix,
            layers,
            workload_labels,
            f"{tensor} Cache Layer Sensitivity by Workload",
            "Transformer Layer",
            f"03_{tensor.lower()}_layers_by_workload.png",
        )

    # --------------------------------------------------------
    # 5. K/V tensor comparison
    # --------------------------------------------------------

    tensor_matrix = np.array([
        [
            get_layer_values(df, tensor, quant).mean()
            for quant in ["q4_0", "q8_0"]
        ]
        for tensor in ["K", "V"]
    ])

    plot_heatmap(
        tensor_matrix,
        ["K", "V"],
        ["Q4_0", "Q8_0"],
        "KV Tensor Sensitivity\nEqual mean across layers and workloads",
        "Cache Tensor",
        "04_kv_tensor_comparison.png",
    )

    # --------------------------------------------------------
    # 6. K/V tensor comparison by workload
    # --------------------------------------------------------

    tensor_workload_matrix = np.array([
        [
            get_layer_values(
                df,
                tensor,
                quant,
                task,
            ).mean()
            for quant in ["q4_0", "q8_0"]
            for task in TASKS
        ]
        for tensor in ["K", "V"]
    ])

    plot_heatmap(
        tensor_workload_matrix,
        ["K", "V"],
        workload_labels,
        "KV Tensor Sensitivity by Workload",
        "Cache Tensor",
        "05_kv_tensors_by_workload.png",
    )

    print("\nAll six heatmaps generated.")


if __name__ == "__main__":
    main()