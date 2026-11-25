from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument(
    "--results-dir",
    type=Path,
    default=Path("results/task31_kv_layer_sensitivity/pilot")
)

args = parser.parse_args()
RESULTS_DIR = args.results_dir
CSV_PATH = RESULTS_DIR / "layer_reconstruction_errors.csv"


def main():
    df = pd.read_csv(CSV_PATH)

    for tensor in ["K", "V"]:
        for quantization in ["q8_0", "q4_0"]:
            subset = df[
                (df["tensor"] == tensor)
                & (df["quantization"] == quantization)
            ].sort_values("layer")

            fig, ax = plt.subplots(figsize=(9, 5))

            ax.plot(
                subset["layer"],
                subset["normalized_mse_percent"],
                marker="o",
                linewidth=2,
            )

            ax.set_title(
                f"{tensor} Cache: {quantization.upper()} "
                "Reconstruction Error by Layer"
            )
            ax.set_xlabel("Transformer Layer")
            ax.set_ylabel("Normalized MSE (%)")
            ax.set_xticks(range(16))
            ax.grid(True, alpha=0.3)

            fig.tight_layout()

            output = (
                RESULTS_DIR
                / f"{tensor.lower()}_{quantization}_layer_nmse.png"
            )

            fig.savefig(output, dpi=200)
            plt.close(fig)

            print(f"Saved {output}")


if __name__ == "__main__":
    main()
