"""Plot workload-level KV-cache sensitivity for Task 31."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

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
    "mmlu_college_computer_science": "MMLU College CS",
}

TENSOR_TYPES = ["K", "V"]
QUANT_TYPES = ["q8_0", "q4_0"]

N_LAYERS = 16


# ============================================================
# 2. PLOT LAYER SENSITIVITY
# ============================================================

def plot_layer_sensitivity(df, tensor, quantization):

    fig, ax = plt.subplots(figsize=(10, 6))

    for task, label in TASKS.items():

        subset = df[
            (df["workload"] == task)
            & (df["tensor"] == tensor)
            & (df["quantization"] == quantization)
        ].sort_values("layer")

        if len(subset) != N_LAYERS:
            raise ValueError(
                f"Expected {N_LAYERS} layers for "
                f"{task}, {tensor}, {quantization}"
            )

        ax.plot(
            subset["layer"],
            subset["mean_nmse_percent"],
            marker="o",
            linewidth=2,
            markersize=5,
            label=label,
        )

    ax.set_title(
        f"{tensor} Cache Layer Sensitivity — "
        f"{quantization.upper()}"
    )

    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Mean Normalized MSE (%)")

    ax.set_xticks(range(N_LAYERS))

    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    output_path = (
        RESULTS_DIR
        / f"{tensor.lower()}_{quantization}_workload_nmse.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


# ============================================================
# 3. MAIN
# ============================================================

def main():

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Aggregated results not found: {INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    for tensor in TENSOR_TYPES:
        for quantization in QUANT_TYPES:
            plot_layer_sensitivity(
                df,
                tensor,
                quantization,
            )

    print("\nAll four workload graphs generated.")


if __name__ == "__main__":
    main()