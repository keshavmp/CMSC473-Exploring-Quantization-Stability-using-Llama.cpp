from pathlib import Path
import numpy as np

import argparse
DEFAULT_RESULTS_DIR = Path(
    "results/task31_kv_layer_sensitivity/pilot"
)

RESULTS_DIR = DEFAULT_RESULTS_DIR
LAYER = 0
VALUES_PER_TOKEN = 512


def load_kv_tensor(layer: int, tensor_type: str) -> np.ndarray:
    path = RESULTS_DIR / f"layer{layer}_{tensor_type}_f16.bin"

    values = np.fromfile(path, dtype=np.float16)

    if values.size == 0 or values.size % VALUES_PER_TOKEN != 0:
        raise ValueError(f"Unexpected tensor size: {values.size}")

    return values.reshape(-1, VALUES_PER_TOKEN).astype(np.float32)


def analyze_tensor(name: str, values: np.ndarray) -> None:
    print(f"\n{name} tensor")
    print(f"  Shape: {values.shape}")
    print(f"  Nonzero values: {np.count_nonzero(values)}")
    print(f"  All finite: {np.isfinite(values).all()}")
    print(f"  Mean: {values.mean():.6f}")
    print(f"  Standard deviation: {values.std():.6f}")
    print(f"  Minimum: {values.min():.6f}")
    print(f"  Maximum: {values.max():.6f}")

def load_reconstructed_tensor(layer, tensor_type, quant_type):
    path = RESULTS_DIR / (
        f"layer{layer}_{tensor_type}_{quant_type}_reconstructed_f32.bin"
    )
    return np.fromfile(path, dtype=np.float32)


def calculate_reconstruction_error(original, reconstructed):
    original = original.ravel().astype(np.float64)
    reconstructed = reconstructed.ravel().astype(np.float64)

    if original.shape != reconstructed.shape:
        raise ValueError("Original and reconstructed shapes do not match")

    error = original - reconstructed
    mse = np.mean(error ** 2)
    normalized_mse = mse / np.mean(original ** 2)
    max_absolute_error = np.max(np.abs(error))

    return mse, normalized_mse, max_absolute_error




def main():
    global RESULTS_DIR

    parser = argparse.ArgumentParser(
        description="Analyze layer-wise KV reconstruction errors"
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing original and reconstructed KV tensors"
    )

    args = parser.parse_args()
    RESULTS_DIR = args.results_dir


    n_layers = 16
    results = []

    for layer in range(n_layers):
        for tensor_type in ["k", "v"]:
            original = load_kv_tensor(layer, tensor_type)

            for quant_type in ["q8_0", "q4_0"]:
                reconstructed = load_reconstructed_tensor(
                    layer, tensor_type, quant_type
                )

                mse, nmse, max_error = calculate_reconstruction_error(
                    original, reconstructed
                )

                results.append({
                    "layer": layer,
                    "tensor": tensor_type.upper(),
                    "quantization": quant_type,
                    "mse": mse,
                    "normalized_mse_percent": nmse * 100,
                    "max_absolute_error": max_error,
                })

    print(
        f"{'Layer':<8} {'Tensor':<8} {'Format':<10} "
        f"{'MSE':<16} {'Normalized MSE (%)':<22}"
    )

    for result in results:
        print(
            f"{result['layer']:<8} "
            f"{result['tensor']:<8} "
            f"{result['quantization']:<10} "
            f"{result['mse']:<16.8f} "
            f"{result['normalized_mse_percent']:<22.6f}"
        )

    import csv

    output_path = RESULTS_DIR / "layer_reconstruction_errors.csv"

    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved results to {output_path}")






if __name__ == "__main__":
    main()
