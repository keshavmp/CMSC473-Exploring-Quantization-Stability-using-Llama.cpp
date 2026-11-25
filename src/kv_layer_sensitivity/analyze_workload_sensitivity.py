"""Statistical analysis of KV-cache layer sensitivity for Task 31."""

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "task31_kv_layer_sensitivity"
    / "aggregated"
)

INPUT_CSV = RESULTS_DIR / "workload_summary.csv"

TASKS = [
    "arc_easy",
    "mathqa",
    "mmlu_college_computer_science",
]

TASK_LABELS = [
    "ARC Easy",
    "MathQA",
    "MMLU CS",
]

TENSORS = ["K", "V"]
QUANT_TYPES = ["q4_0", "q8_0"]

N_LAYERS = 16
TOP_K = 3
METRIC = "mean_nmse_percent"


def get_layer_values(df, tensor, quantization, task=None):

    subset = df[
        (df["tensor"] == tensor)
        & (df["quantization"] == quantization)
    ]

    if task is not None:
        subset = subset[subset["workload"] == task]

    values = (
        subset.groupby("layer")[METRIC]
        .mean()
        .reindex(range(N_LAYERS))
    )

    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(
            f"Invalid layer values: "
            f"{tensor}, {quantization}, {task}"
        )

    return values


def spearman(a, b):
    return a.rank().corr(b.rank())


def top_layers(values):
    return set(values.nlargest(TOP_K).index)


def compare_profiles(a, b):

    overlap = top_layers(a) & top_layers(b)

    return {
        "spearman": spearman(a, b),
        "top_k_overlap": len(overlap),
        "top_k_overlap_percent": (
            100 * len(overlap) / TOP_K
        ),
        "shared_layers": ",".join(
            str(layer) for layer in sorted(overlap)
        ),
    }


# ============================================================
# 1. CROSS-WORKLOAD CONSISTENCY
# ============================================================

def analyze_workload_consistency(df):

    results = []

    for tensor in TENSORS:
        for quant in QUANT_TYPES:

            for task_a, task_b in combinations(TASKS, 2):

                a = get_layer_values(
                    df, tensor, quant, task_a
                )

                b = get_layer_values(
                    df, tensor, quant, task_b
                )

                results.append({
                    "tensor": tensor,
                    "quantization": quant,
                    "workload_a": task_a,
                    "workload_b": task_b,
                    **compare_profiles(a, b),
                })

    return pd.DataFrame(results)


# ============================================================
# 2. CROSS-PRECISION CONSISTENCY
# ============================================================

def analyze_precision_consistency(df):

    results = []

    for tensor in TENSORS:

        for task in TASKS + [None]:

            a = get_layer_values(
                df, tensor, "q4_0", task
            )

            b = get_layer_values(
                df, tensor, "q8_0", task
            )

            results.append({
                "tensor": tensor,
                "workload": task or "all_workloads",
                **compare_profiles(a, b),
            })

    return pd.DataFrame(results)


# ============================================================
# 3. K VS V CONSISTENCY
# ============================================================

def analyze_tensor_consistency(df):

    results = []

    for quant in QUANT_TYPES:

        for task in TASKS + [None]:

            a = get_layer_values(
                df, "K", quant, task
            )

            b = get_layer_values(
                df, "V", quant, task
            )

            results.append({
                "quantization": quant,
                "workload": task or "all_workloads",
                **compare_profiles(a, b),
            })

    return pd.DataFrame(results)


# ============================================================
# 4. LAYER RANKINGS
# ============================================================

def analyze_layer_rankings(df):

    results = []

    for tensor in TENSORS:
        for quant in QUANT_TYPES:

            values = get_layer_values(
                df, tensor, quant
            )

            ranked = values.sort_values(
                ascending=False
            )

            for rank, (layer, value) in enumerate(
                ranked.items(), start=1
            ):

                results.append({
                    "tensor": tensor,
                    "quantization": quant,
                    "rank": rank,
                    "layer": layer,
                    "mean_nmse_percent": value,
                    "relative_sensitivity_percent": (
                        100 * (value / values.mean() - 1)
                    ),
                })

    return pd.DataFrame(results)


# ============================================================
# 5. EARLY VS MIDDLE VS LATE LAYERS
# ============================================================

def analyze_layer_depth(df):

    groups = {
        "Early": list(range(0, 5)),
        "Middle": list(range(5, 11)),
        "Late": list(range(11, 16)),
    }

    results = []

    for tensor in TENSORS:
        for quant in QUANT_TYPES:

            for task in TASKS + [None]:

                values = get_layer_values(
                    df, tensor, quant, task
                )

                for group, layers in groups.items():

                    results.append({
                        "tensor": tensor,
                        "quantization": quant,
                        "workload": task or "all_workloads",
                        "depth_group": group,
                        "layer_count": len(layers),
                        "mean_nmse_percent": (
                            values.loc[layers].mean()
                        ),
                    })

    return pd.DataFrame(results)


# ============================================================
# 6. CORRELATION HEATMAPS
# ============================================================

def plot_workload_correlations(df):

    for tensor in TENSORS:
        for quant in QUANT_TYPES:

            profiles = pd.DataFrame({
                label: get_layer_values(
                    df, tensor, quant, task
                )
                for task, label in zip(
                    TASKS, TASK_LABELS
                )
            })

            matrix = profiles.rank().corr()

            fig, ax = plt.subplots(figsize=(6, 5))

            im = ax.imshow(
                matrix,
                cmap="RdBu_r",
                vmin=-1,
                vmax=1,
            )

            ax.set_xticks(range(len(TASKS)))
            ax.set_xticklabels(TASK_LABELS)

            ax.set_yticks(range(len(TASKS)))
            ax.set_yticklabels(TASK_LABELS)

            for i in range(len(TASKS)):
                for j in range(len(TASKS)):

                    value = matrix.iloc[i, j]

                    ax.text(
                        j,
                        i,
                        f"{value:.3f}",
                        ha="center",
                        va="center",
                        color=(
                            "white"
                            if abs(value) > 0.6
                            else "black"
                        ),
                    )

            ax.set_title(
                f"{tensor} Cache — {quant.upper()}\n"
                "Cross-Workload Layer Rank Correlation"
            )

            fig.colorbar(
                im,
                ax=ax,
                label="Spearman Correlation",
            )

            fig.tight_layout()

            output_path = (
                RESULTS_DIR
                / f"{tensor.lower()}_{quant}_correlation.png"
            )

            fig.savefig(
                output_path,
                dpi=300,
                bbox_inches="tight",
            )

            plt.close(fig)


# ============================================================
# 7. MAIN
# ============================================================

def main():

    df = pd.read_csv(INPUT_CSV)

    analyses = {
        "workload_consistency": (
            analyze_workload_consistency(df)
        ),
        "precision_consistency": (
            analyze_precision_consistency(df)
        ),
        "tensor_consistency": (
            analyze_tensor_consistency(df)
        ),
        "layer_rankings": (
            analyze_layer_rankings(df)
        ),
        "layer_depth_comparison": (
            analyze_layer_depth(df)
        ),
    }

    for name, results in analyses.items():

        output_path = RESULTS_DIR / f"{name}.csv"

        results.to_csv(output_path, index=False)

        print(f"\n{'=' * 60}")
        print(name.upper())
        print("=" * 60)

        if name == "layer_rankings":
            print(
                results[
                    results["rank"] <= TOP_K
                ].to_string(index=False)
            )

        elif name == "layer_depth_comparison":
            print(
                results[
                    results["workload"] == "all_workloads"
                ].to_string(index=False)
            )

        else:
            print(results.to_string(index=False))

        print(f"\nSaved: {output_path}")

    plot_workload_correlations(df)

    print("\nStatistical analysis complete.")


if __name__ == "__main__":
    main()