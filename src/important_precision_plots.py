"""Compare quantization precision using task-balanced tensor and block statistics."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

try:
    from src.important_imatrix_plots import TASKS, TYPES, LABELS, save_figure, sha256, tensor_key, weighted_error
except ModuleNotFoundError:
    from important_imatrix_plots import TASKS, TYPES, LABELS, save_figure, sha256, tensor_key, weighted_error


def precision_averages(stats, errors, baseline):
    models = set(errors["model"]) | {baseline}
    stats = stats[stats["model"].isin(models)].copy()
    if set(stats["model"]) != models:
        raise ValueError("Missing precision-model statistics")
    if stats.duplicated(["model", "task", "tensor"]).any():
        raise ValueError("Ambiguous repeated model/domain/tensor statistics")
    for frame in (stats, errors):
        coverage = frame.groupby(["model", "tensor"])["task"].agg(list)
        if not coverage.map(lambda tasks: sorted(tasks) == sorted(TASKS)).all():
            raise ValueError("Precision comparisons require all three datasets for every tensor")
    stats["precision"] = stats["model"].map(lambda name: "BF16" if name == baseline else name.removeprefix("pure-").upper())
    stats["tensor_type"] = stats["tensor"].map(lambda name: TYPES[tensor_key(name)[1]])
    coverage = stats.groupby(["model", "task", "layer"])["tensor_type"].agg(list)
    if not coverage.map(lambda kinds: len(kinds) == len(TYPES) and set(kinds) == set(TYPES)).all():
        raise ValueError("Each precision/block must have the same seven tensor types")
    tensor = stats.groupby(["model", "precision", "tensor", "tensor_type", "layer"], as_index=False).agg(
        energy=("energy", "mean"), entropy_pct=("entropy_pct", "mean"), kurtosis=("kurtosis", "mean"),
        n_tasks=("task", "nunique"))
    block = tensor.groupby(["model", "precision", "layer"], as_index=False).agg(
        energy=("energy", "mean"), entropy_pct=("entropy_pct", "mean"), kurtosis=("kurtosis", "mean"),
        n_tensors=("tensor", "size"))
    averaged_errors = errors.groupby(["model", "precision", "tensor", "tensor_type", "layer"], as_index=False).agg(
        weighted_error=("weighted_error", "mean"), reference_energy=("reference_energy", "mean"),
        n_tasks=("task", "nunique"))
    block_errors = averaged_errors.groupby(["model", "precision", "layer"], as_index=False).agg(
        weighted_error=("weighted_error", "mean"), reference_energy=("reference_energy", "mean"),
        n_tensors=("tensor", "size"))
    for frame in (averaged_errors, block_errors):
        frame["relative_error_pct"] = 100 * frame["weighted_error"].div(frame["reference_energy"].where(frame["reference_energy"] > 0))
    return tensor, block, averaged_errors, block_errors


def reference_energies(reference, vectors, gguf, folder, provenance):
    output = folder / "bf16_diagonal_reference_energy.csv"
    manifest = folder / "reference_energy_provenance.json"
    key = {"reference": provenance["reference"], "imatrices": provenance["imatrices"],
           "dequantizer_sha256": provenance["dequantizer_sha256"], "formula_version": 1}
    if output.exists() and manifest.exists():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved.get("inputs") == key and saved.get("csv_sha256") == sha256(output):
            return pd.read_csv(output)
    reader = gguf.GGUFReader(reference)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    rows = []
    for name in sorted(vectors[TASKS[0]], key=tensor_key):
        tensor = tensors[name]
        n_inputs, n_outputs = map(int, tensor.shape)
        column_energy = np.zeros(n_inputs, dtype=np.float64)
        for start in range(0, n_outputs, 256):
            values = gguf.dequantize(tensor.data[start:start + 256], tensor.tensor_type).astype(np.float64)
            column_energy += np.einsum("ij,ij->j", values, values, dtype=np.float64)
        for task in TASKS:
            rows.append({"task": task, "tensor": name, "reference_energy": weighted_error(vectors[task][name], column_energy)})
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    manifest.write_text(json.dumps({"inputs": key, "csv_sha256": sha256(output)}, indent=2), encoding="utf-8")
    return frame


def error_heatmap(frame, folder, block, relative):
    metric = "relative_error_pct" if relative else "weighted_error"
    unit = "layer" if block else "tensor"
    grid = frame.pivot(index=unit, columns="precision", values=metric)
    names = sorted(grid.index) if block else sorted(grid.index, key=tensor_key)
    precisions = sorted(grid.columns, key=lambda name: int(name[1:]))
    grid = grid.reindex(index=names, columns=precisions)
    positive = grid.to_numpy()[grid.to_numpy() > 0]
    norm = LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))
    fig, ax = plt.subplots(figsize=(10, 8 if block else max(14, len(names) * 0.19)), layout="constrained")
    image = ax.imshow(np.ma.masked_less_equal(grid.to_numpy(), 0), aspect="auto", cmap="magma", norm=norm)
    ax.set_xticks(range(len(precisions)), precisions)
    labels = names if block else [f"L{tensor_key(name)[0]:02d} {TYPES[tensor_key(name)[1]]}" for name in names]
    ax.set_yticks(range(len(names)), labels, fontsize=9 if block else 6)
    ax.set_xlabel("Weight precision")
    ax.set_ylabel("Transformer block" if block else "Tensor: layer and type")
    title = "Relative activation-weighted quantization error" if relative else "Absolute activation-weighted quantization error"
    ax.set_title(title + "\nEqual mean across code, math, chat; approximate local measure")
    if block:
        for row, column in np.ndindex(grid.shape):
            value = grid.iloc[row, column]
            if value > 0:
                ax.text(column, row, f"{value:.2g}", ha="center", va="center", fontsize=8,
                        color="black" if norm(value) > 0.65 else "white")
    else:
        for boundary in range(6, len(names) - 1, 7):
            ax.axhline(boundary + 0.5, color="white", alpha=0.45, linewidth=0.6)
    label = "Squared error / BF16 diagonal reference energy (%)" if relative else "Mean weighted squared error" if block else "Weighted squared error"
    fig.colorbar(image, ax=ax, label=label + " (logarithmic colors)")
    name = "01_relative_weighted_error_by_precision" if relative else "03_absolute_weighted_error_by_precision"
    save_figure(fig, folder, name)


def concentration(frame, folder, block):
    precisions = sorted(frame["precision"].unique(), key=lambda name: -1 if name == "BF16" else int(name[1:]))
    panels = [None] if block else list(TYPES)
    fig, axes = plt.subplots(2, len(panels), figsize=(10 if block else 23, 8), sharex=True,
                             sharey="row", squeeze=False, layout="constrained")
    palette = dict(zip(precisions, plt.get_cmap("tab10").colors))
    palette["BF16"] = "black"
    for column, tensor_type in enumerate(panels):
        subset = frame if block else frame[frame["tensor_type"] == tensor_type]
        for precision in precisions:
            values = subset[subset["precision"] == precision].sort_values("layer")
            for row, metric in enumerate(("entropy_pct", "kurtosis")):
                axes[row, column].plot(values["layer"], values[metric], label=precision,
                    color=palette[precision], linestyle="--" if precision == "BF16" else "-", marker="o", markersize=3)
        if not block:
            axes[0, column].set_title(LABELS[column])
        axes[0, column].set_ylim(0, 100)
        axes[1, column].set_yscale("symlog", linthresh=1)
        axes[1, column].set_xlabel("Transformer block" if block else "Layer")
        for row in range(2):
            axes[row, column].grid(alpha=0.2)
            axes[row, column].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True, nbins=6))
    axes[0, 0].set_ylabel("Mean tensor normalized entropy (%)" if block else "Normalized energy entropy (%)")
    axes[1, 0].set_ylabel("Mean tensor excess kurtosis" if block else "Mean-activation excess kurtosis")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(precisions), title="Model precision")
    fig.suptitle("2. Importance concentration and outliers across precision\nEqual average of code, math and chat; descriptive diagnostics")
    save_figure(fig, folder, "02_importance_concentration_by_precision")


def figure_precision_comparisons(input_folder, output_folder, reference, vectors, gguf, provenance):
    folder = output_folder / "by precision"
    folder.mkdir(parents=True, exist_ok=True)
    error_path = output_folder / "activation_weighted_errors.csv"
    error_manifest = json.loads((output_folder / "weight_error_provenance.json").read_text(encoding="utf-8"))
    if sha256(error_path) != error_manifest["csv_sha256"] or error_manifest["inputs"] != provenance:
        raise ValueError("Weighted-error data do not match verified input provenance")
    errors = pd.read_csv(error_path)
    reference_energy = reference_energies(reference, vectors, gguf, folder, provenance)
    errors = errors.merge(reference_energy, on=["task", "tensor"], validate="many_to_one")
    if errors["reference_energy"].isna().any():
        raise ValueError("Missing BF16 reference energies")
    inventory = json.loads((input_folder / "inventory.json").read_text(encoding="utf-8"))
    hashes = {item["name"]: item["sha256"] for item in provenance["models"]}
    hashes[reference.stem] = provenance["reference"]["sha256"]
    selected = [item for item in inventory["matrices"] if item["model"] in hashes]
    controls = set()
    for item in selected:
        collection = item["collection"]
        if collection["model_sha256"] != hashes[item["model"]]:
            raise ValueError("Model statistics and weighted errors use different weight files")
        controls.add(json.dumps({key: collection[key] for key in (
            "context", "chunks", "batch_size", "parse_special", "tool_sha256", "tool_version")}, sort_keys=True))
    if len(controls) != 1:
        raise ValueError("Precision statistics have incompatible collection settings")
    for task in TASKS:
        task_hashes = {item["collection"]["calibration_sha256"] for item in selected if item["task"] == task}
        if task_hashes != {provenance["imatrices"][task]["calibration_sha256"]}:
            raise ValueError("Precision statistics have mismatched calibration text")
    stats = pd.read_csv(input_folder / "tensor_statistics.csv")
    stats = stats[stats["matrix"].isin({item["matrix"] for item in selected})]
    tensor, block, tensor_errors, block_errors = precision_averages(stats, errors, reference.stem)
    for name, data in (("tensors", tensor), ("by block", block)):
        destination = folder / name
        destination.mkdir(parents=True, exist_ok=True)
        error_data = block_errors if name == "by block" else tensor_errors
        data.to_csv(destination / "precision_statistics.csv", index=False)
        error_data.to_csv(destination / "precision_weighted_errors.csv", index=False)
        error_heatmap(error_data, destination, name == "by block", relative=True)
        concentration(data, destination, name == "by block")
        error_heatmap(error_data, destination, name == "by block", relative=False)
    (folder / "README.txt").write_text(
        "Precision comparisons: three figures at tensor level and three at block level.\n\n"
        "Every model averages code/math/chat equally. Blocks average their seven measured tensor types.\n"
        "Figure 1 is relative activation-weighted quantization error, replacing raw activation energy as the main sensitivity plot.\n"
        "Relative error (%) = 100 * weighted_error / sum_j E[x_j^2] sum_i W[i,j]^2.\n"
        "The denominator is a diagonal approximation of BF16 local output energy, not activation input energy.\n"
        "At block level, the ratio uses mean tensor error divided by mean tensor reference energy (ratio of means).\n"
        "It is NOT the arithmetic mean of per-tensor error percentages or a true residual-block reconstruction error.\n"
        "This squared-error ratio is not a relative amplitude error or a percentage of accuracy lost.\n"
        "Figure 2 compares each model's own entropy/kurtosis diagnostics; block values average tensor statistics, not pooled distributions.\n"
        "Figure 3 retains absolute weighted error; both error figures use fixed BF16 activations across precisions.\n"
        "BF16 has zero quantization error and is omitted from logarithmic error plots. Q7 is unavailable.\n"
        "Energy can identify large activations but does not measure their quantization damage.\n"
        "Local weighted error is more directly relevant; neither local error nor entropy/kurtosis proves downstream task loss.\n"
        "The diagonal approximation omits cross-dimension correlations and downstream propagation.\n"
        "Confirm candidates using held-out accuracy/KL and controlled higher-precision tensor experiments.\n"
        "Reference for local reconstruction objectives: https://arxiv.org/abs/2210.17323\n"
        "Reference for activation-aware importance: https://arxiv.org/abs/2306.00978\n", encoding="utf-8")
    (folder / "plot_provenance.json").write_text(json.dumps({
        "tensor_statistics_sha256": sha256(input_folder / "tensor_statistics.csv"),
        "inventory_sha256": sha256(input_folder / "inventory.json"),
        "weighted_errors_sha256": sha256(error_path), "reference_energies_sha256": sha256(folder / "bf16_diagonal_reference_energy.csv"),
        "dataset_average": "equal arithmetic mean over code, math, chat",
        "block_average": "equal arithmetic mean over seven measured tensor types",
    }, indent=2), encoding="utf-8")
    print(f"Saved tensor/block precision comparisons to {folder}", flush=True)
